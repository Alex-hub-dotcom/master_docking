#!/usr/bin/env python3
"""
TEKO - 16-STAGE ULTRA-GRADUAL CURRICULUM TRAINING (FINAL)
=========================================================
Author: Alexandre Schleier Neves da Silva
Contact: alexandre.schleiernevesdasilva@uni-hohenheim.de

- Vision-only PPO with your current visual encoder (create_visual_encoder("simple", 256, True))
- 16-stage ultra-gradual curriculum (see curriculum_manager.py)
- 2D continuous action space [v_cmd, w_cmd] mapped to wheel torques
- Default 3,000,000 environment steps (can be changed via --steps)

Curriculum advancement logic:
- Per-stage minimum:          15k steps (HYPERPARAMS["min_stage_steps"])
- Stage-dependent success thresholds:

    Stage  0–2   ->  0.90   (easy, should be almost perfect)
    Stage  3–7   ->  0.80
    Stage  8–11  ->  0.70
    Stage 12–15  ->  0.60

- Anti-stall safety:
    If a stage has already seen more than MAX_STAGE_STEPS (default 250k steps)
    it will advance even if success rate is below the target threshold.

This makes it *very likely* that training will eventually go through all
stages if you run for enough steps (e.g. 3M+), while still forcing the
policy to learn something useful at each stage.
"""

import argparse
import os
from datetime import datetime
from collections import deque

from isaaclab.app import AppLauncher

# -----------------------------
# CLI arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16, help="Parallel environments")
parser.add_argument("--steps", type=int, default=3_000_000, help="Total training steps")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--rollout_len", type=int, default=64, help="Rollout length")
parser.add_argument("--epochs", type=int, default=8, help="PPO epochs per update")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint (.pt) to resume from")
parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # important for vision-only

app = AppLauncher(args)
sim = app.app

# -----------------------------
# Imports after simulator start
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder
from teko.tasks.direct.teko.curriculum.curriculum_manager import STAGE_NAMES

# =============================================================================
# Central hyperparameters
# =============================================================================

HYPERPARAMS = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_clip": 0.2,
    "entropy_coef": 0.02,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "min_stage_steps": 15_000,    # minimum steps in a stage before checking advance
}

# Max steps allowed in a single stage before we *force* advancement
MAX_STAGE_STEPS = 400_000


def get_stage_threshold(level: int) -> float:
    """
    Stage-dependent success threshold.
    Returns the success rate required to move from this stage to the next.
    """
    if level <= 2:
        return 0.80   # very easy baby-forward stages -> almost perfect
    elif level <= 7:
        return 0.70   # offset stages
    elif level <= 11:
        return 0.60   # larger offsets / lateral
    else:
        return 0.50   # 180°, search, full autonomy


# =============================================================================
# Policy network: CNN encoder + actor / critic heads
# =============================================================================

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        # Use your existing visual encoder configuration
        # (do not change this unless you intentionally want to)
        self.encoder = create_visual_encoder("simple", 256, False)

        self.actor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2),   nn.Tanh()   # actions in [-1, 1]
        )

        self.critic = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Log-std for Gaussian policy (shared across batch)
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, obs):
        # obs: [B, 3, 480, 640]
        feat = self.encoder(obs)
        mean = self.actor(feat)
        value = self.critic(feat)
        return mean, value, self.log_std

    def act(self, obs):
        mean, value, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value.squeeze(-1)

    def evaluate(self, obs, actions):
        mean, value, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, value.squeeze(-1), entropy


# =============================================================================
# GAE + PPO update
# =============================================================================

def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards, values, dones: [T, N]
    returns:
        advantages: [T, N]
        returns:    [T, N]
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = 0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(policy, optimizer, obs, actions, logp_old, advantages, returns,
               epochs=8, batch_size=256,
               clip_ratio=0.2, value_clip=0.2,
               entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5):
    """
    Standard clipped PPO update over flattened [T*N] data.
    """
    T, N = obs.shape[0], obs.shape[1]
    total_samples = T * N

    obs_flat = obs.view(total_samples, 3, 480, 640)
    actions_flat = actions.view(total_samples, 2)
    logp_old_flat = logp_old.view(-1)
    advantages_flat = advantages.view(-1)
    returns_flat = returns.view(-1)

    # Normalize advantages
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

    policy_losses, value_losses, entropies = [], [], []

    for _ in range(epochs):
        indices = torch.randperm(total_samples, device=obs.device)

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            mb_idx = indices[start:end]

            logp, value, entropy = policy.evaluate(obs_flat[mb_idx], actions_flat[mb_idx])

            # PPO policy loss
            ratio = (logp - logp_old_flat[mb_idx]).exp()
            unclipped = ratio * advantages_flat[mb_idx]
            clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages_flat[mb_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()

            # Value loss with optional clipping
            if value_clip is not None:
                value_pred = torch.clamp(
                    value,
                    returns_flat[mb_idx] - value_clip,
                    returns_flat[mb_idx] + value_clip,
                )
            else:
                value_pred = value

            value_loss = F.mse_loss(value_pred, returns_flat[mb_idx])

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())

    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)


# =============================================================================
# Main training loop
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("🎓 TEKO - 16-STAGE ULTRA-GRADUAL CURRICULUM (FINAL)")
    print("=" * 70)
    print(f"Environments: {args.num_envs}")
    print(f"Total steps: {args.steps:,}")
    print(f"Stages: {len(STAGE_NAMES)} (ultra-gradual)")
    print(f"Advancement: stage-dependent thresholds + anti-stall, "
          f"min {HYPERPARAMS['min_stage_steps']:,} steps/stage")
    print("=" * 70 + "\n")

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    # ------------------------------------------------------------------
    # Environment + Policy
    # ------------------------------------------------------------------
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_curriculum = True
    env = TekoEnv(cfg=cfg)

    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # Optional checkpoint loading (resume training)
    # ------------------------------------------------------------------
    start_step = 0
    steps_in_current_stage = 0

    if args.checkpoint is not None:
        print(f"🔁 Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        restored_level = ckpt.get("curriculum_level", 0)
        steps_in_current_stage = ckpt.get("steps_in_stage", 0)
        env.set_curriculum_level(restored_level)
        print(f"Resumed from step {start_step}, stage {env.curriculum_level}")

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    log_dir = f"teko_curriculum/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"✓ Policy: {sum(p.numel() for p in policy.parameters()):,} params")
    print(f"📊 TensorBoard: tensorboard --logdir teko_curriculum")
    print(f"💾 Checkpoints: {log_dir}\n")

    # Initial reset (uses current env.curriculum_level)
    obs_dict, _ = env.reset()
    obs = obs_dict["rgb"].to(device)

    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_successes = deque(maxlen=100)
    stage_success_window = deque(maxlen=200)   # rolling success for current stage

    current_episode_reward = torch.zeros(args.num_envs, device=device)
    current_episode_length = torch.zeros(args.num_envs, dtype=torch.int32, device=device)

    step = start_step  # continue from checkpoint step if resuming

    print(f"[CURRICULUM] {STAGE_NAMES[env.curriculum_level]}\n")

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

            # Episode bookkeeping for stats
            for i in range(args.num_envs):
                if done[i]:
                    episode_rewards.append(current_episode_reward[i].item())
                    episode_lengths.append(current_episode_length[i].item())
                    # Heuristic: a successful dock gives a big positive reward at the end
                    success = reward[i] > 50.0
                    episode_successes.append(1.0 if success else 0.0)
                    stage_success_window.append(1.0 if success else 0.0)
                    current_episode_reward[i] = 0.0
                    current_episode_length[i] = 0

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value)
            logp_buf.append(logp)
            done_buf.append(done.float())

            obs = next_obs
            step += args.num_envs
            steps_in_current_stage += args.num_envs

        # Stack rollout tensors: [T, N, ...]
        obs_t = torch.stack(obs_buf)
        act_t = torch.stack(act_buf)
        rew_t = torch.stack(rew_buf)
        val_t = torch.stack(val_buf)
        logp_t = torch.stack(logp_buf)
        done_t = torch.stack(done_buf)

        # GAE + returns
        with torch.no_grad():
            advantages, returns = compute_gae(
                rew_t,
                val_t,
                done_t,
                gamma=HYPERPARAMS["gamma"],
                lam=HYPERPARAMS["gae_lambda"],
            )

        # PPO update
        policy_loss, value_loss, entropy = ppo_update(
            policy,
            optimizer,
            obs_t.to(device),
            act_t.to(device),
            logp_t.to(device),
            advantages.to(device),
            returns.to(device),
            epochs=args.epochs,
            batch_size=args.batch_size,
            clip_ratio=HYPERPARAMS["clip_ratio"],
            value_clip=HYPERPARAMS["value_clip"],
            entropy_coef=HYPERPARAMS["entropy_coef"],
            value_coef=HYPERPARAMS["value_coef"],
            max_grad_norm=HYPERPARAMS["max_grad_norm"],
        )

        # Stats
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
        success_rate = np.mean(episode_successes) if episode_successes else 0.0
        stage_success = np.mean(stage_success_window) if stage_success_window else 0.0

        current_stage = env.curriculum_level
        stage_threshold = get_stage_threshold(current_stage)

        writer.add_scalar("train/reward",           mean_reward,   step)
        writer.add_scalar("train/episode_length",   mean_length,   step)
        writer.add_scalar("train/success_rate",     success_rate,  step)
        writer.add_scalar("train/stage_success",    stage_success, step)
        writer.add_scalar("train/curriculum_stage", current_stage, step)
        writer.add_scalar("train/stage_threshold",  stage_threshold, step)
        writer.add_scalar("train/policy_loss",      policy_loss,   step)
        writer.add_scalar("train/value_loss",       value_loss,    step)
        writer.add_scalar("train/entropy",          entropy,       step)
        writer.add_scalar("train/steps_in_stage",   steps_in_current_stage, step)

        print(f"[{step:7d}] S{current_stage:02d} | "
              f"R={mean_reward:6.1f} | Len={mean_length:4.0f} | "
              f"SR={success_rate*100:4.1f}% | SSR={stage_success*100:4.1f}% | "
              f"Thr={stage_threshold*100:4.1f}% | "
              f"StageSteps={steps_in_current_stage:6d}")

        # ------------------------------------------------------------------
        # Curriculum advancement: stage-dependent threshold + anti-stall
        # ------------------------------------------------------------------
        if steps_in_current_stage >= HYPERPARAMS["min_stage_steps"]:
            advance = False

            # Only trust stage_success if we have a reasonable number of episodes
            enough_episodes = len(stage_success_window) >= 50

            if enough_episodes and stage_success >= stage_threshold:
                advance = True
            elif steps_in_current_stage >= MAX_STAGE_STEPS:
                # Safety valve: don't get stuck forever in one stage
                print(f"[CURRICULUM] Forcing advance from stage {current_stage} "
                      f"after {steps_in_current_stage} steps (SSR={stage_success:.3f})")
                advance = True

            if advance and current_stage < len(STAGE_NAMES) - 1:
                env.set_curriculum_level(current_stage + 1)
                stage_success_window.clear()
                steps_in_current_stage = 0

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if step % 50_000 == 0 and step > start_step:
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
            print(f"💾 Checkpoint: {ckpt_path}")

    # Final save
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

    print(f"\n{'=' * 70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"Final stage: {STAGE_NAMES[env.curriculum_level]}")
    print(f"💾 Model: {final_path}")
    print(f"{'=' * 70}\n")

    writer.close()
    env.close()
    sim.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sim.close()
