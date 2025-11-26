#!/usr/bin/env python3
"""
TEKO - 16-STAGE CURRICULUM TRAINING (v2.0 - TURN-FIRST + REHEARSAL)
===================================================================

Author: Alexandre Schleier Neves da Silva
Contact: alexandre.schleiernevesdasilva@uni-hohenheim.de

v2.0 Changes:
- Separate log_std for linear (v) and angular (w) velocity
- Wider exploration bounds for angular velocity
- Proper observation resync after stage mixing AND rehearsal
- Periodic rehearsal on mastered stages to prevent catastrophic forgetting
- Compatible with reward_functions v8.8 (turn-first curriculum)

Rehearsal mechanism:
- Once stages are truly mastered (SSR >= threshold), they're added to a list
- Every REHEARSAL_INTERVAL_STEPS, we briefly train on a random mastered stage
- This maintains performance on easy stages while learning harder ones
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
    "entropy_coef": 0.05,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "min_stage_steps": 15_000,
}

MAX_STAGE_STEPS = 400_000
CHECKPOINT_INTERVAL = 10_000

# -----------------------------------------------------------------------------
# Rehearsal configuration
# -----------------------------------------------------------------------------
REHEARSAL_ENABLED = True
REHEARSAL_MIN_STAGE = 2           # Start rehearsal only when at stage >= 2
REHEARSAL_MAX_HISTORY = 3         # Rehearse among the last N mastered stages
REHEARSAL_INTERVAL_STEPS = 50_000 # Time between rehearsals (global env-steps)
REHEARSAL_ROLLOUT_LEN = 32        # Rollout length per env during rehearsal
REHEARSAL_UPDATES = 3             # Number of short rollouts per rehearsal event

args = None


def get_stage_threshold(level: int) -> float:
    """Strict success-rate thresholds per curriculum stage."""
    if level <= 0:
        return 0.80
    elif level <= 4:
        return 0.70
    elif level <= 9:
        return 0.60
    else:
        return 0.50


def get_entropy_coef(level: int) -> float:
    """
    Stage-dependent entropy coefficient.
    
    Slightly higher for offset stages to prevent one-direction turning trap.
    """
    if level <= 1:
        return 0.05   # S0-S1: Easy forward stages
    elif level <= 3:
        return 0.05   # S2-S3: Still forward
    elif level <= 5:
        return 0.07   # S4-S5: First offsets - need exploration for BOTH directions
    elif level <= 7:
        return 0.07   # S6-S7: More offsets
    elif level == 8:
        return 0.08   # S8: Harder offsets
    elif level <= 11:
        return 0.07   # S9-S11: Large offsets
    elif level <= 13:
        return 0.08   # S12-S13: 180° turn - new behavior
    elif level == 14:
        return 0.08   # S14: Arena search - new behavior
    else:
        return 0.05   # S15: Final stage, exploit


# =============================================================================
# GAE + PPO
# =============================================================================

def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                dones: torch.Tensor,
                gamma: float,
                lam: float):
    """Generalized Advantage Estimation (GAE-Lambda)."""
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
    """PPO update with CPU rollout storage and GPU mini-batches."""
    device = next(policy.parameters()).device

    T, N, C, H, W = obs.shape
    total_samples = T * N

    obs_flat = obs.view(total_samples, C, H, W)
    actions_flat = actions.view(total_samples, 2)
    logp_old_flat = logp_old.view(-1)
    advantages_flat = advantages.view(-1)
    returns_flat = returns.view(-1)

    advantages_flat = (advantages_flat - advantages_flat.mean()) / (
        advantages_flat.std() + 1e-8
    )

    policy_losses, value_losses, entropies = [], [], []

    for _ in range(epochs):
        indices = torch.randperm(total_samples)

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            mb_idx = indices[start:end]

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
                value_pred = torch.clamp(value, mb_ret - value_clip, mb_ret + value_clip)
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


def do_training_step(env, policy, device: torch.device, rollout_len: int | None = None):
    """
    Short adaptation rollout used during stage mixing and rehearsal.
    
    NOTE: This function changes the environment state. After calling it,
    you MUST resync the observation in the main loop.
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

        obs_buf.append(obs_gpu.cpu())
        act_buf.append(action.cpu())
        rew_buf.append(reward.cpu())
        val_buf.append(value.cpu())
        logp_buf.append(logp.cpu())
        done_buf.append(done.float().cpu())

        obs_gpu = next_obs_gpu

    return (
        torch.stack(obs_buf),
        torch.stack(act_buf),
        torch.stack(rew_buf),
        torch.stack(val_buf),
        torch.stack(logp_buf),
        torch.stack(done_buf),
    )


# =============================================================================
# Arg parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=60_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout_len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)

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
    print("🎓 TEKO - 16-STAGE CURRICULUM (v2.0 - TURN-FIRST + REHEARSAL)")
    print("=" * 70)
    print(f"Environments: {args.num_envs}")
    print(f"Total steps:  {args.steps:,}")
    print(f"LR:           {args.lr}")
    print(f"Rollout len:  {args.rollout_len}")
    print(f"PPO epochs:   {args.epochs}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Safety valve: {MAX_STAGE_STEPS:,} steps/stage")
    if REHEARSAL_ENABLED:
        print(f"Rehearsal:    every {REHEARSAL_INTERVAL_STEPS:,} steps "
              f"(last {REHEARSAL_MAX_HISTORY} mastered stages)")
    print("=" * 70 + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    app = AppLauncher(args)
    sim = app.app

    from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
    from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder
    from teko.tasks.direct.teko.curriculum.curriculum_manager import STAGE_NAMES

    # -------------------------------------------------------------------------
    # Policy with SEPARATE log_std for v and w
    # -------------------------------------------------------------------------
    class Policy(nn.Module):
        """
        Vision-based policy with separate exploration parameters for
        linear velocity (v) and angular velocity (w).
        """
        LOG_STD_V_MIN = -1.5
        LOG_STD_V_MAX = 0.2
        LOG_STD_W_MIN = -1.0   # Allow some determinism when learned
        LOG_STD_W_MAX = 0.6    # WIDER - robot needs to explore both turn directions

        def __init__(self):
            super().__init__()

            self.encoder = create_visual_encoder("simple", 256, False)

            self.actor = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Tanh(),
            )

            self.critic = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

            self.log_std_v = nn.Parameter(torch.tensor(0.0))
            self.log_std_w = nn.Parameter(torch.tensor(0.0))

        def forward(self, obs: torch.Tensor):
            feat = self.encoder(obs)
            mean = self.actor(feat)
            value = self.critic(feat)
            
            log_std_v = self.log_std_v.clamp(self.LOG_STD_V_MIN, self.LOG_STD_V_MAX)
            log_std_w = self.log_std_w.clamp(self.LOG_STD_W_MIN, self.LOG_STD_W_MAX)
            log_std = torch.stack([log_std_v, log_std_w])
            
            return mean, value, log_std

        def act(self, obs: torch.Tensor):
            mean, value, log_std = self.forward(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
            return action, logp, value.squeeze(-1)

        def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
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
        last_rehearsal_step = 0

        # Track which stages have been truly mastered (SSR >= threshold)
        mastered_stages: list[int] = []

        if args.checkpoint is not None:
            print(f"🔁 Loading checkpoint: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=device)
            policy.load_state_dict(ckpt["policy"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0)
            restored_level = ckpt.get("curriculum_level", 0)
            steps_in_current_stage = ckpt.get("steps_in_stage", 0)
            mastered_stages = ckpt.get("mastered_stages", [])
            env.set_curriculum_level(restored_level)
            last_ckpt_step = start_step
            last_rehearsal_step = start_step
            print(f"Resumed: step {start_step}, stage {env.curriculum_level}")
            if mastered_stages:
                print(f"🧠 Restored mastered stages: {mastered_stages}")

        log_dir = f"teko_curriculum/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

        print(f"✓ Policy: {sum(p.numel() for p in policy.parameters()):,} params")
        print(f"📊 Logs:   {log_dir}\n")

        obs_dict, _ = env.reset()
        obs = obs_dict["rgb"].to(device)

        episode_rewards = deque(maxlen=100)
        episode_lengths = deque(maxlen=100)
        episode_successes = deque(maxlen=100)
        stage_success_window = deque(maxlen=200)

        current_episode_reward = torch.zeros(args.num_envs, device=device)
        current_episode_length = torch.zeros(args.num_envs, dtype=torch.int32, device=device)

        step = start_step
        print(f"[CURRICULUM] {STAGE_NAMES[env.curriculum_level]}\n")

        # ---------------------------------------------------------------------
        # Main training loop
        # ---------------------------------------------------------------------
        try:
            while step < args.steps:
                obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []

                for _ in range(args.rollout_len):
                    with torch.no_grad():
                        action, logp, value = policy.act(obs)

                    obs_dict, reward, term, trunc, info = env.step(action)
                    next_obs = obs_dict["rgb"].to(device)
                    done = term | trunc

                    current_episode_reward += reward
                    current_episode_length += 1

                    for i in range(args.num_envs):
                        if done[i]:
                            episode_rewards.append(current_episode_reward[i].item())
                            episode_lengths.append(current_episode_length[i].item())

                            success = reward[i] > 50.0
                            episode_successes.append(1.0 if success else 0.0)
                            stage_success_window.append(1.0 if success else 0.0)

                            current_episode_reward[i] = 0.0
                            current_episode_length[i] = 0

                    obs_buf.append(obs.cpu())
                    act_buf.append(action.cpu())
                    rew_buf.append(reward.cpu())
                    val_buf.append(value.cpu())
                    logp_buf.append(logp.cpu())
                    done_buf.append(done.float().cpu())

                    obs = next_obs
                    step += args.num_envs
                    steps_in_current_stage += args.num_envs

                obs_t = torch.stack(obs_buf)
                act_t = torch.stack(act_buf)
                rew_t = torch.stack(rew_buf)
                val_t = torch.stack(val_buf)
                logp_t = torch.stack(logp_buf)
                done_t = torch.stack(done_buf)

                mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
                mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
                success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0
                stage_success = float(np.mean(stage_success_window)) if stage_success_window else 0.0

                current_stage = env.curriculum_level
                stage_threshold = get_stage_threshold(current_stage)
                entropy_coef_used = get_entropy_coef(current_stage)

                with torch.no_grad():
                    advantages, returns = compute_gae(
                        rew_t, val_t, done_t,
                        HYPERPARAMS["gamma"], HYPERPARAMS["gae_lambda"]
                    )

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
                writer.add_scalar("train/log_std_v", policy.log_std_v.item(), step)
                writer.add_scalar("train/log_std_w", policy.log_std_w.item(), step)
                writer.add_scalar("train/entropy_coef_used", entropy_coef_used, step)
                writer.add_scalar("train/num_mastered_stages", len(mastered_stages), step)

                # Reward component logging
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

                # CNN feature health
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

                # ----------------------------------------------------------
                # Curriculum advancement
                # ----------------------------------------------------------
                if steps_in_current_stage >= HYPERPARAMS["min_stage_steps"]:
                    advance = False
                    mastered_now = False
                    enough_episodes = len(stage_success_window) >= 50

                    if enough_episodes and stage_success >= stage_threshold:
                        advance = True
                        mastered_now = True
                        print(f"✓ Stage {current_stage} mastered! (SSR={stage_success:.1%})")
                    elif steps_in_current_stage >= MAX_STAGE_STEPS:
                        advance = True
                        print(
                            f"⚠ Safety valve: advancing from S{current_stage} "
                            f"after {steps_in_current_stage:,} steps (SSR={stage_success:.1%})"
                        )

                    if advance and current_stage < len(STAGE_NAMES) - 1:
                        # Record truly mastered stages for rehearsal
                        if mastered_now and current_stage not in mastered_stages:
                            mastered_stages.append(int(current_stage))
                            mastered_stages = sorted(set(mastered_stages))
                            print(f"🧠 Mastered stages so far: {mastered_stages}")

                        print(f"🔄 Mixing stages {current_stage} and {current_stage + 1}...")

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

                        # CRITICAL: Resync observation after mixing
                        obs_dict, _ = env.reset()
                        obs = obs_dict["rgb"].to(device)
                        current_episode_reward.zero_()
                        current_episode_length.zero_()

                        stage_success_window.clear()
                        steps_in_current_stage = 0

                # ----------------------------------------------------------
                # Periodic rehearsal on mastered stages
                # ----------------------------------------------------------
                if (
                    REHEARSAL_ENABLED
                    and mastered_stages
                    and env.curriculum_level >= REHEARSAL_MIN_STAGE
                    and (step - last_rehearsal_step) >= REHEARSAL_INTERVAL_STEPS
                ):
                    current_stage = env.curriculum_level
                    lower_bound = max(0, current_stage - REHEARSAL_MAX_HISTORY)

                    # Restrict rehearsal to recent mastered stages below current
                    candidate_stages = [
                        s for s in mastered_stages
                        if lower_bound <= s < current_stage
                    ]

                    if candidate_stages:
                        rehearsal_stage = int(np.random.choice(candidate_stages))
                        original_stage = current_stage

                        print(
                            f"🧪 Rehearsal: training briefly on Stage {rehearsal_stage} "
                            f"(current: S{original_stage})"
                        )

                        env.set_curriculum_level(rehearsal_stage)

                        for _ in range(REHEARSAL_UPDATES):
                            obs_t2, act_t2, rew_t2, val_t2, logp_t2, done_t2 = do_training_step(
                                env, policy, device, rollout_len=REHEARSAL_ROLLOUT_LEN
                            )

                            with torch.no_grad():
                                adv2, ret2 = compute_gae(
                                    rew_t2, val_t2, done_t2,
                                    HYPERPARAMS["gamma"], HYPERPARAMS["gae_lambda"]
                                )

                            rehearsal_entropy_coef = get_entropy_coef(rehearsal_stage)
                            ppo_update(
                                policy, optimizer,
                                obs_t2, act_t2, logp_t2,
                                adv2, ret2,
                                epochs=args.epochs, batch_size=args.batch_size,
                                clip_ratio=HYPERPARAMS["clip_ratio"],
                                value_clip=HYPERPARAMS["value_clip"],
                                entropy_coef=rehearsal_entropy_coef,
                                value_coef=HYPERPARAMS["value_coef"],
                                max_grad_norm=HYPERPARAMS["max_grad_norm"],
                            )

                        # Return to main stage
                        env.set_curriculum_level(original_stage)
                        
                        # CRITICAL: Resync observation after rehearsal
                        obs_dict, _ = env.reset()
                        obs = obs_dict["rgb"].to(device)
                        current_episode_reward.zero_()
                        current_episode_length.zero_()
                        
                        last_rehearsal_step = step
                        print(f"✅ Rehearsal complete, back to S{original_stage}")

                        # Log rehearsal event
                        writer.add_scalar("train/rehearsal_stage", rehearsal_stage, step)

                # ----------------------------------------------------------
                # Checkpointing
                # ----------------------------------------------------------
                if step - last_ckpt_step >= CHECKPOINT_INTERVAL:
                    ckpt_path = f"{log_dir}/ckpt_{step}.pt"
                    torch.save(
                        {
                            "policy": policy.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step,
                            "curriculum_level": env.curriculum_level,
                            "steps_in_stage": steps_in_current_stage,
                            "mastered_stages": mastered_stages,
                        },
                        ckpt_path,
                    )
                    last_ckpt_step = step
                    print(f"💾 Checkpoint: {ckpt_path}")

        except KeyboardInterrupt:
            print("\n⏹ Training interrupted. Saving emergency checkpoint...")
            interrupt_path = f"{log_dir}/interrupt_{step}.pt"
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "curriculum_level": env.curriculum_level,
                    "steps_in_stage": steps_in_current_stage,
                    "mastered_stages": mastered_stages,
                },
                interrupt_path,
            )
            print(f"💾 Emergency checkpoint: {interrupt_path}")
        else:
            final_path = f"{log_dir}/final.pt"
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "curriculum_level": env.curriculum_level,
                    "steps_in_stage": steps_in_current_stage,
                    "mastered_stages": mastered_stages,
                },
                final_path,
            )

            print("\n" + "=" * 70)
            print("✅ TRAINING COMPLETE!")
            print(f"Final stage: {STAGE_NAMES[env.curriculum_level]}")
            print(f"Mastered stages: {mastered_stages}")
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