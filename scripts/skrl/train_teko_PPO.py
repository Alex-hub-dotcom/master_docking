#!/usr/bin/env python3
"""
TEKO - 16-STAGE CURRICULUM TRAINING (STABLE BASELINE + FRAME STACKING)
======================================================================

Author: Alexandre Schleier Neves da Silva
Contact: alexandre.schleiernevesdasilva@uni-hohenheim.de

Features:
- Multi-stage curriculum for docking difficulty
- Strict success thresholds (no relaxation with time-in-stage)
- Stage mixing before advancement
- Overlapping curriculum stages fixed
- 20–40% replay from previous stage (handled in curriculum_manager)
- Stage-dependent entropy (more exploration in early / hard stages)
- Clamped log_std to prevent entropy explosion
- CPU-based rollout storage + GPU mini-batch PPO update
  -> works with frame stacking without blowing VRAM
"""

import argparse
import os
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from isaaclab.app import AppLauncher


# =============================================================================
# Hyperparameters
# =============================================================================

HYPERPARAMS = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.15,
    "value_clip": 0.2,
    "entropy_coef": 0.05,   # base value (overridden by stage-dependent schedule)
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "min_stage_steps": 15_000,
}

# Safety valve per stage (only used if SSR never reaches threshold)
MAX_STAGE_STEPS = 400_000

# Checkpoint interval (in environment steps, across all envs)
CHECKPOINT_INTERVAL = 10_000

# Will be set inside main()
args = None


def get_stage_threshold(level: int) -> float:
    """
    Strict success-rate thresholds per curriculum stage.

    These are fixed per stage and DO NOT relax with time.
    The agent must truly master each stage before advancing
    (unless the hard safety valve is hit).
    """
    if level <= 0:
        return 0.80   # Stage 0: 80% SSR
    elif level <= 4:
        return 0.70   # Stages 1–4: 70% SSR
    elif level <= 9:
        return 0.60   # Stages 5–9: 60% SSR
    else:
        return 0.50   # Stages 10–15: 50% SSR


def get_entropy_coef(level: int) -> float:
    """
    Stage-dependent entropy coefficient.

    Idea:
    - High exploration in early stages and around S3 (where learning can stall).
    - Slightly higher entropy again on hard jumps (S8, S14).
    - Lower entropy in later stages to refine behaviour.
    """
    if level <= 1:
        return 0.08   # S0–S1: Initial learning
    elif level <= 3:
        return 0.07   # S2–S3: Building basics
    elif level <= 7:
        return 0.06   # S4–S7: Refining approach
    elif level == 8:
        return 0.08   # S8: Hard offset jump, need more exploration
    elif level <= 11:
        return 0.07   # S9–S11: Large offsets
    elif level <= 13:
        return 0.07   # S12–S13: 180° turn
    elif level == 14:
        return 0.08   # S14: Arena search (new behaviour)
    else:
        return 0.06   # S15: Full autonomy


# =============================================================================
# GAE + PPO
# =============================================================================

def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                dones: torch.Tensor,
                gamma: float,
                lam: float):
    """
    Generalized Advantage Estimation (GAE-Lambda).

    Args:
        rewards: [T, N]
        values:  [T, N]
        dones:   [T, N] (1.0 if done, 0.0 otherwise)
        gamma:   discount factor
        lam:     GAE lambda

    Returns:
        advantages: [T, N]
        returns:    [T, N]
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = 0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(policy,
               optimizer: torch.optim.Optimizer,
               obs: torch.Tensor,
               actions: torch.Tensor,
               logp_old: torch.Tensor,
               advantages: torch.Tensor,
               returns: torch.Tensor,
               epochs: int = 4,
               batch_size: int = 64,
               clip_ratio: float = 0.15,
               value_clip: float | None = 0.2,
               entropy_coef: float = 0.10,
               value_coef: float = 0.5,
               max_grad_norm: float = 0.5):
    """
    PPO update with CPU rollout storage and GPU mini-batches.

    All input tensors are expected to live on CPU:
    - obs:      [T, N, C, H, W]
    - actions:  [T, N, 2]
    - logp_old: [T, N]
    - advantages, returns: [T, N]

    This function:
    - Flattens time and env dimensions.
    - Keeps data on CPU.
    - For each mini-batch, moves only that slice to GPU.
    """
    device = next(policy.parameters()).device

    # obs shape: [T, N, C, H, W]
    T, N, C, H, W = obs.shape
    total_samples = T * N

    # Flatten time and env dims
    obs_flat = obs.view(total_samples, C, H, W)
    actions_flat = actions.view(total_samples, 2)
    logp_old_flat = logp_old.view(-1)
    advantages_flat = advantages.view(-1)
    returns_flat = returns.view(-1)

    # Normalize advantages on CPU
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (
        advantages_flat.std() + 1e-8
    )

    policy_losses, value_losses, entropies = [], [], []

    for _ in range(epochs):
        # Random permutation on CPU
        indices = torch.randperm(total_samples)

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            mb_idx = indices[start:end]

            # Move only this mini-batch to GPU
            mb_obs = obs_flat[mb_idx].to(device)
            mb_actions = actions_flat[mb_idx].to(device)
            mb_logp_old = logp_old_flat[mb_idx].to(device)
            mb_adv = advantages_flat[mb_idx].to(device)
            mb_ret = returns_flat[mb_idx].to(device)

            logp, value, entropy = policy.evaluate(mb_obs, mb_actions)

            ratio = (logp - mb_logp_old).exp()
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            if value_clip is not None:
                value_pred = torch.clamp(
                    value,
                    mb_ret - value_clip,
                    mb_ret + value_clip,
                )
            else:
                value_pred = value

            value_loss = F.mse_loss(value_pred, mb_ret)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())

    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)


def do_training_step(env,
                     policy,
                     device: torch.device,
                     rollout_len: int | None = None):
    """
    Short adaptation rollout used during stage mixing.

    Important:
    - Keeps current observation on GPU for acting.
    - Stores rollout data on CPU (same pattern as main loop).
    """
    global args
    if rollout_len is None:
        rollout_len = args.rollout_len

    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []

    obs_dict, _ = env.reset()
    obs_gpu = obs_dict["rgb"].to(device)

    for _ in range(rollout_len):
        with torch.no_grad():
            action, logp, value = policy.act(obs_gpu)

        obs_dict, reward, term, trunc, info = env.step(action)
        next_obs_gpu = obs_dict["rgb"].to(device)
        done = term | trunc

        # Store CPU copies
        obs_buf.append(obs_gpu.cpu())
        act_buf.append(action.cpu())
        rew_buf.append(reward.cpu())
        val_buf.append(value.cpu())
        logp_buf.append(logp.cpu())
        done_buf.append(done.float().cpu())

        obs_gpu = next_obs_gpu

    return (
        torch.stack(obs_buf),   # [T, N, C, H, W]
        torch.stack(act_buf),   # [T, N, 2]
        torch.stack(rew_buf),   # [T, N]
        torch.stack(val_buf),   # [T, N]
        torch.stack(logp_buf),  # [T, N]
        torch.stack(done_buf),  # [T, N]
    )


# =============================================================================
# Arg parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel environments")
    parser.add_argument("--steps", type=int, default=60_000_000, help="Total training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--rollout_len", type=int, default=64, help="Rollout length (per env)")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (.pt) to resume from")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size for PPO (lower is safer for frame stacking)")

    # Isaac Lab app launcher args
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    global args
    args = parse_args()
    args.enable_cameras = True

    print("\n" + "=" * 70)
    print("🎓 TEKO - 16-STAGE CURRICULUM (STABLE BASELINE + FRAME STACKING)")
    print("=" * 70)
    print(f"Environments: {args.num_envs}")
    print(f"Total steps:  {args.steps:,}")
    print(f"LR:           {args.lr}")
    print(f"Rollout len:  {args.rollout_len}")
    print(f"PPO epochs:   {args.epochs}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Safety valve: {MAX_STAGE_STEPS:,} steps/stage")
    print("=" * 70 + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    # -------------------------------------------------------------------------
    # Launch Isaac (SimulationApp) *before* importing Teko / Isaac modules
    # -------------------------------------------------------------------------
    app = AppLauncher(args)
    sim = app.app

    # Now it's safe to import anything that touches Omniverse / pxr
    from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
    from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder
    from teko.tasks.direct.teko.curriculum.curriculum_manager import STAGE_NAMES

    # -------------------------------------------------------------------------
    # Policy definition (needs create_visual_encoder)
    # -------------------------------------------------------------------------
    class Policy(nn.Module):
        """
        Vision-based policy using a CNN encoder and MLP actor-critic head.

        - Encoder: create_visual_encoder (frame-stacking aware)
        - Actor: outputs mean actions in [-1, 1] via Tanh
        - Critic: outputs scalar state-value
        - log_std: global learnable log standard deviation for Gaussian policy
        """
        LOG_STD_MIN = -1.5
        LOG_STD_MAX = 0.2   # tighter upper bound for more stable actions

        def __init__(self):
            super().__init__()

            # Visual encoder returns feature vector of size 256
            self.encoder = create_visual_encoder("simple", 256, False)

            # Actor head
            self.actor = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Tanh(),  # actions in [-1, 1]
            )

            # Critic head
            self.critic = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

            # Start moderately exploratory (std = exp(0) = 1)
            self.log_std = nn.Parameter(torch.ones(2) * 0.0)

        def forward(self, obs: torch.Tensor):
            """
            Forward pass through encoder and heads.

            Args:
                obs: images [B, C, H, W]; C can be 3 or 3 * frame_stack.

            Returns:
                mean:   [B, 2]
                value:  [B, 1]
                log_std (clamped): [2]
            """
            feat = self.encoder(obs)
            mean = self.actor(feat)
            value = self.critic(feat)
            log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
            return mean, value, log_std

        def act(self, obs: torch.Tensor):
            """
            Sample an action given observations.

            Args:
                obs: [B, C, H, W] on the same device as the policy.

            Returns:
                action: [B, 2]
                logp:   [B]
                value:  [B]
            """
            mean, value, log_std = self.forward(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
            return action, logp, value.squeeze(-1)

        def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
            """
            Evaluate log-probabilities, values and entropy for given actions.

            Args:
                obs:     [B, C, H, W]
                actions: [B, 2]

            Returns:
                logp:    [B]
                value:   [B]
                entropy: [B]
            """
            mean, value, log_std = self.forward(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
            return logp, value.squeeze(-1), entropy

    # -------------------------------------------------------------------------
    # Training setup
    # -------------------------------------------------------------------------
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_curriculum = True
    env = None
    writer = None

    try:
        env = TekoEnv(cfg=cfg)

        policy = Policy().to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

        start_step = 0
        steps_in_current_stage = 0
        last_ckpt_step = 0

        # Optional checkpoint loading
        if args.checkpoint is not None:
            print(f"🔁 Loading checkpoint: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=device)
            policy.load_state_dict(ckpt["policy"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0)
            restored_level = ckpt.get("curriculum_level", 0)
            steps_in_current_stage = ckpt.get("steps_in_stage", 0)
            env.set_curriculum_level(restored_level)
            last_ckpt_step = start_step
            print(f"Resumed: step {start_step}, stage {env.curriculum_level}")

        # Logging directory
        log_dir = f"teko_curriculum/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

        print(f"✓ Policy: {sum(p.numel() for p in policy.parameters()):,} params")
        print(f"📊 Logs:   {log_dir}\n")

        # Initial reset
        obs_dict, _ = env.reset()
        obs = obs_dict["rgb"].to(device)

        # Rolling windows for statistics
        episode_rewards = deque(maxlen=100)
        episode_lengths = deque(maxlen=100)
        episode_successes = deque(maxlen=100)
        stage_success_window = deque(maxlen=200)

        # Per-env episode tracking on device
        current_episode_reward = torch.zeros(args.num_envs, device=device)
        current_episode_length = torch.zeros(args.num_envs, dtype=torch.int32, device=device)

        step = start_step
        print(f"[CURRICULUM] {STAGE_NAMES[env.curriculum_level]}\n")

        # ---------------------------------------------------------------------
        # Main training loop
        # ---------------------------------------------------------------------
        try:
            while step < args.steps:
                # Rollout buffers (CPU)
                obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []

                # Collect one rollout of length args.rollout_len
                for _ in range(args.rollout_len):
                    with torch.no_grad():
                        action, logp, value = policy.act(obs)

                    obs_dict, reward, term, trunc, info = env.step(action)
                    next_obs = obs_dict["rgb"].to(device)
                    done = term | trunc

                    # Update per-env episode stats (on device)
                    current_episode_reward += reward
                    current_episode_length += 1

                    # If an episode ends, log stats into rolling windows
                    for i in range(args.num_envs):
                        if done[i]:
                            episode_rewards.append(current_episode_reward[i].item())
                            episode_lengths.append(current_episode_length[i].item())

                            # Simple success heuristic based on big terminal reward
                            success = reward[i] > 50.0
                            episode_successes.append(1.0 if success else 0.0)
                            stage_success_window.append(1.0 if success else 0.0)

                            current_episode_reward[i] = 0.0
                            current_episode_length[i] = 0

                    # Store rollout data on CPU
                    obs_buf.append(obs.cpu())
                    act_buf.append(action.cpu())
                    rew_buf.append(reward.cpu())
                    val_buf.append(value.cpu())
                    logp_buf.append(logp.cpu())
                    done_buf.append(done.float().cpu())

                    obs = next_obs
                    step += args.num_envs
                    steps_in_current_stage += args.num_envs

                # Stack rollout into tensors on CPU
                obs_t = torch.stack(obs_buf)       # [T, N, C, H, W]
                act_t = torch.stack(act_buf)       # [T, N, 2]
                rew_t = torch.stack(rew_buf)       # [T, N]
                val_t = torch.stack(val_buf)       # [T, N]
                logp_t = torch.stack(logp_buf)     # [T, N]
                done_t = torch.stack(done_buf)     # [T, N]

                # Statistics BEFORE update
                mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
                mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
                success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0
                stage_success = float(np.mean(stage_success_window)) if stage_success_window else 0.0

                current_stage = env.curriculum_level
                stage_threshold = get_stage_threshold(current_stage)

                # Stage-dependent entropy coefficient (strict thresholds stay the same)
                entropy_coef_used = get_entropy_coef(current_stage)

                # Compute GAE on CPU
                with torch.no_grad():
                    advantages, returns = compute_gae(
                        rew_t, val_t, done_t,
                        HYPERPARAMS["gamma"], HYPERPARAMS["gae_lambda"]
                    )

                # PPO update (rollout on CPU, mini-batches on GPU)
                policy_loss, value_loss, entropy = ppo_update(
                    policy, optimizer,
                    obs_t, act_t, logp_t,
                    advantages, returns,
                    epochs=args.epochs, batch_size=args.batch_size,
                    clip_ratio=HYPERPARAMS["clip_ratio"],
                    value_clip=HYPERPARAMS["value_clip"],
                    entropy_coef=entropy_coef_used,
                    value_coef=HYPERPARAMS["value_coef"],
                    max_grad_norm=HYPERPARAMS["max_grad_norm"],
                )

                # TensorBoard logging
                writer.add_scalar("train/reward", mean_reward, step)
                writer.add_scalar("train/episode_length", mean_length, step)
                writer.add_scalar("train/success_rate", success_rate, step)
                writer.add_scalar("train/stage_success", stage_success, step)
                writer.add_scalar("train/curriculum_stage", current_stage, step)
                writer.add_scalar("train/stage_threshold", stage_threshold, step)
                writer.add_scalar("train/policy_loss", policy_loss, step)
                writer.add_scalar("train/value_loss", value_loss, step)
                writer.add_scalar("train/entropy", entropy, step)
                writer.add_scalar("train/steps_in_stage", steps_in_current_stage, step)
                writer.add_scalar("train/log_std_v", policy.log_std[0].item(), step)
                writer.add_scalar("train/log_std_w", policy.log_std[1].item(), step)
                writer.add_scalar("train/entropy_coef_used", entropy_coef_used, step)

                # Detailed reward component logging
                if hasattr(env, 'reward_components') and env.reward_components:
                    for comp_name, comp_values in env.reward_components.items():
                        if comp_values:
                            mean_val = float(np.mean(comp_values))
                            writer.add_scalar(f"rewards/{comp_name}", mean_val, step)
                    for key in env.reward_components:
                        env.reward_components[key].clear()

                # Action statistics
                writer.add_scalar("train/action_v_mean", env.actions[:, 0].mean().item(), step)
                writer.add_scalar("train/action_w_mean", env.actions[:, 1].mean().item(), step)
                writer.add_scalar("train/action_v_std", env.actions[:, 0].std().item(), step)
                writer.add_scalar("train/action_w_std", env.actions[:, 1].std().item(), step)

                # Distance statistics
                _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
                writer.add_scalar("train/min_distance", surface_xy.min().item(), step)
                writer.add_scalar("train/mean_distance", surface_xy.mean().item(), step)

                # CNN feature health check (every 1000 env-steps)
                if step % 1000 == 0:
                    with torch.no_grad():
                        sample_obs = obs[:min(4, obs.shape[0])]
                        features = policy.encoder(sample_obs)
                        writer.add_scalar("train/feature_mean", features.mean().item(), step)
                        writer.add_scalar("train/feature_std", features.std().item(), step)
                        writer.add_scalar("train/feature_max", features.abs().max().item(), step)

                print(
                    f"[{step:7d}] S{current_stage:02d} | "
                    f"R={mean_reward:6.1f} | Len={mean_length:4.0f} | "
                    f"SR={success_rate * 100:4.1f}% | SSR={stage_success * 100:4.1f}% | "
                    f"Thr={stage_threshold * 100:4.1f}% | Steps={steps_in_current_stage:6d} | "
                    f"EntCoef={entropy_coef_used:.3f}"
                )

                # ------------------------------
                # Curriculum advancement logic
                # ------------------------------
                if steps_in_current_stage >= HYPERPARAMS["min_stage_steps"]:
                    advance = False
                    enough_episodes = len(stage_success_window) >= 50

                    if enough_episodes and stage_success >= stage_threshold:
                        advance = True
                        print(f"✓ Stage {current_stage} mastered! (SSR={stage_success:.1%})")
                    elif steps_in_current_stage >= MAX_STAGE_STEPS:
                        advance = True
                        print(
                            f"⚠ Safety valve: advancing from S{current_stage} "
                            f"after {steps_in_current_stage:,} steps (SSR={stage_success:.1%})"
                        )

                    if advance and current_stage < len(STAGE_NAMES) - 1:
                        print(f"🔄 Mixing stages {current_stage} and {current_stage + 1}...")

                        # Shorter rollout for mixing to save time and memory
                        MIX_ROLLOUT_LEN = 16

                        for mix_iter in range(50):
                            if mix_iter % 2 == 0:
                                env.set_curriculum_level(current_stage)
                            else:
                                env.set_curriculum_level(current_stage + 1)

                            obs_t2, act_t2, rew_t2, val_t2, logp_t2, done_t2 = do_training_step(
                                env, policy, device, rollout_len=MIX_ROLLOUT_LEN
                            )

                            with torch.no_grad():
                                adv2, ret2 = compute_gae(
                                    rew_t2, val_t2, done_t2,
                                    HYPERPARAMS["gamma"], HYPERPARAMS["gae_lambda"]
                                )

                            # PPO update for mixing (still CPU rollout, GPU mini-batch)
                            mix_entropy_coef = get_entropy_coef(env.curriculum_level)
                            ppo_update(
                                policy, optimizer,
                                obs_t2, act_t2, logp_t2,
                                adv2, ret2,
                                epochs=args.epochs, batch_size=args.batch_size,
                                clip_ratio=HYPERPARAMS["clip_ratio"],
                                value_clip=HYPERPARAMS["value_clip"],
                                entropy_coef=mix_entropy_coef,
                                value_coef=HYPERPARAMS["value_coef"],
                                max_grad_norm=HYPERPARAMS["max_grad_norm"],
                            )

                        print(f"➡️  Advancing to Stage {current_stage + 1}")
                        env.set_curriculum_level(current_stage + 1)
                        stage_success_window.clear()
                        steps_in_current_stage = 0

                # ------------------------------
                # Checkpointing (robust)
                # ------------------------------
                if step - last_ckpt_step >= CHECKPOINT_INTERVAL:
                    ckpt_path = f"{log_dir}/ckpt_{step}.pt"
                    torch.save(
                        {
                            "policy": policy.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step,
                            "curriculum_level": env.curriculum_level,
                            "steps_in_stage": steps_in_current_stage,
                        },
                        ckpt_path,
                    )
                    last_ckpt_step = step
                    print(f"💾 Checkpoint: {ckpt_path}")

        except KeyboardInterrupt:
            # Emergency save on manual interrupt
            print("\n⏹ Training interrupted by user. Saving emergency checkpoint...")
            interrupt_path = f"{log_dir}/interrupt_{step}.pt"
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "curriculum_level": env.curriculum_level,
                    "steps_in_stage": steps_in_current_stage,
                },
                interrupt_path,
            )
            print(f"💾 Emergency checkpoint: {interrupt_path}")
        else:
            # Final save only if we reached the target number of steps
            final_path = f"{log_dir}/final.pt"
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "curriculum_level": env.curriculum_level,
                    "steps_in_stage": steps_in_current_stage,
                },
                final_path,
            )

            print("\n" + "=" * 70)
            print("✅ TRAINING COMPLETE!")
            print(f"Final stage: {STAGE_NAMES[env.curriculum_level]}")
            print(f"💾 Model: {final_path}")
            print("=" * 70 + "\n")

    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        sim.close()


if __name__ == "__main__":
    main()
