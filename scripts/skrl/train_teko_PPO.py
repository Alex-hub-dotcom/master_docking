#!/usr/bin/env python3
"""
TEKO - 16-STAGE CURRICULUM TRAINING (STABLE BASELINE)
=====================================================
Author: Alexandre Schleier Neves da Silva
Contact: alexandre.schleiernevesdasilva@uni-hohenheim.de

Features:
- Stage mixing before advancement
- Overlapping curriculum stages fixed
- 20% replay from previous stage (in curriculum_manager)
- Proper advancement logic
- Clamped log_std to prevent entropy explosion
- FIXED entropy coefficient (no peaks / scheduling)
"""

import argparse
import os
from datetime import datetime
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16, help="Parallel environments")
parser.add_argument("--steps", type=int, default=60_000_000, help="Total training steps")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--rollout_len", type=int, default=64, help="Rollout length")
parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per update")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint (.pt) to resume from")
parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args)
sim = app.app

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder
from teko.tasks.direct.teko.curriculum.curriculum_manager import STAGE_NAMES

# =============================================================================
# Hyperparameters
# =============================================================================

HYPERPARAMS = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.15,
    "value_clip": 0.2,
    "entropy_coef": 0.05,  # single fixed entropy coefficient
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "min_stage_steps": 15_000,
}

MAX_STAGE_STEPS = 400_000  # safety valve per stage


def get_stage_threshold(level: int) -> float:
    if level <= 0:
        return 0.80
    elif level <= 4:
        return 0.70
    elif level <= 9:
        return 0.60
    else:
        return 0.50


# =============================================================================
# Policy
# =============================================================================

class Policy(nn.Module):
    """
    Policy with SOFT clamping on log_std.
    Allows exploration but prevents explosion.
    """
    LOG_STD_MIN = -1.5
    LOG_STD_MAX = 0.2   # tighter upper bound for more stable actions

    def __init__(self):
        super().__init__()

        self.encoder = create_visual_encoder("simple", 256, False)

        self.actor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        # start moderately exploratory (std = exp(0) = 1)
        self.log_std = nn.Parameter(torch.ones(2) * 0.0)

    def forward(self, obs):
        feat = self.encoder(obs)
        mean = self.actor(feat)
        value = self.critic(feat)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, value, log_std

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
# GAE + PPO
# =============================================================================

def compute_gae(rewards, values, dones, gamma, lam):
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
               epochs=4, batch_size=256, clip_ratio=0.15, value_clip=0.2,
               entropy_coef=0.10, value_coef=0.5, max_grad_norm=0.5):
    T, N = obs.shape[0], obs.shape[1]
    total_samples = T * N

    obs_flat = obs.view(total_samples, 3, 480, 640)
    actions_flat = actions.view(total_samples, 2)
    logp_old_flat = logp_old.view(-1)
    advantages_flat = advantages.view(-1)
    returns_flat = returns.view(-1)

    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

    policy_losses, value_losses, entropies = [], [], []

    for _ in range(epochs):
        indices = torch.randperm(total_samples, device=obs.device)

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            mb_idx = indices[start:end]

            logp, value, entropy = policy.evaluate(obs_flat[mb_idx], actions_flat[mb_idx])

            ratio = (logp - logp_old_flat[mb_idx]).exp()
            unclipped = ratio * advantages_flat[mb_idx]
            clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages_flat[mb_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()

            if value_clip is not None:
                value_pred = torch.clamp(
                    value,
                    returns_flat[mb_idx] - value_clip,
                    returns_flat[mb_idx] + value_clip,
                )
            else:
                value_pred = value

            value_loss = F.mse_loss(value_pred, returns_flat[mb_idx])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())

    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)


def do_training_step(env, policy, device, rollout_len=None):
    """
    Short adaptation rollout used during stage mixing.

    Important:
    - Keeps 'current obs' on GPU for acting.
    - Stores all rollout buffers on CPU to avoid large GPU stacks.
    """
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

        # Store CPU copies to avoid large GPU allocations when stacking
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
# Main
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("🎓 TEKO - 16-STAGE CURRICULUM (STABLE BASELINE)")
    print("=" * 70)
    print(f"Environments: {args.num_envs}")
    print(f"Total steps: {args.steps:,}")
    print(f"LR: {args.lr}")
    print(f"Fixed entropy coef: {HYPERPARAMS['entropy_coef']}")
    print(f"Safety valve: {MAX_STAGE_STEPS:,} steps/stage")
    print("=" * 70 + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0")

    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_curriculum = True
    env = TekoEnv(cfg=cfg)

    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    start_step = 0
    steps_in_current_stage = 0

    if args.checkpoint is not None:
        print(f"🔁 Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        restored_level = ckpt.get("curriculum_level", 0)
        steps_in_current_stage = ckpt.get("steps_in_stage", 0)
        env.set_curriculum_level(restored_level)
        print(f"Resumed: step {start_step}, stage {env.curriculum_level}")

    log_dir = f"teko_curriculum/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"✓ Policy: {sum(p.numel() for p in policy.parameters()):,} params")
    print(f"📊 Logs: {log_dir}\n")

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

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value)
            logp_buf.append(logp)
            done_buf.append(done.float())

            obs = next_obs
            step += args.num_envs
            steps_in_current_stage += args.num_envs

        obs_t = torch.stack(obs_buf)
        act_t = torch.stack(act_buf)
        rew_t = torch.stack(rew_buf)
        val_t = torch.stack(val_buf)
        logp_t = torch.stack(logp_buf)
        done_t = torch.stack(done_buf)

        # --- stats BEFORE update ---
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
        success_rate = np.mean(episode_successes) if episode_successes else 0.0
        stage_success = np.mean(stage_success_window) if stage_success_window else 0.0

        current_stage = env.curriculum_level
        stage_threshold = get_stage_threshold(current_stage)

        # Fixed entropy coefficient
        entropy_coef_used = HYPERPARAMS["entropy_coef"]
        entropy_peak_prob_used = 0.0  # no peaks

        # Compute GAE
        with torch.no_grad():
            advantages, returns = compute_gae(
                rew_t, val_t, done_t,
                HYPERPARAMS["gamma"], HYPERPARAMS["gae_lambda"]
            )

        # PPO update
        policy_loss, value_loss, entropy = ppo_update(
            policy, optimizer,
            obs_t.to(device), act_t.to(device), logp_t.to(device),
            advantages.to(device), returns.to(device),
            epochs=args.epochs, batch_size=args.batch_size,
            clip_ratio=HYPERPARAMS["clip_ratio"],
            value_clip=HYPERPARAMS["value_clip"],
            entropy_coef=entropy_coef_used,
            value_coef=HYPERPARAMS["value_coef"],
            max_grad_norm=HYPERPARAMS["max_grad_norm"],
        )

        # Logging
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
        writer.add_scalar("train/entropy_peak_prob_used", entropy_peak_prob_used, step)

        print(
            f"[{step:7d}] S{current_stage:02d} | "
            f"R={mean_reward:6.1f} | Len={mean_length:4.0f} | "
            f"SR={success_rate*100:4.1f}% | SSR={stage_success*100:4.1f}% | "
            f"Thr={stage_threshold*100:4.1f}% | Steps={steps_in_current_stage:6d} | "
            f"EntCoef={entropy_coef_used:.3f}"
        )

        # Curriculum advancement
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

                # Use a SHORTER rollout (e.g., 16) for mixing to save VRAM
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

                    # For mixing we keep the same fixed entropy
                    ppo_update(
                        policy, optimizer,
                        obs_t2.to(device), act_t2.to(device), logp_t2.to(device),
                        adv2.to(device), ret2.to(device),
                        epochs=args.epochs, batch_size=args.batch_size,
                        clip_ratio=HYPERPARAMS["clip_ratio"],
                        value_clip=HYPERPARAMS["value_clip"],
                        entropy_coef=HYPERPARAMS["entropy_coef"],
                        value_coef=HYPERPARAMS["value_coef"],
                        max_grad_norm=HYPERPARAMS["max_grad_norm"],
                    )

                print(f"➡️  Advancing to Stage {current_stage + 1}")
                env.set_curriculum_level(current_stage + 1)
                stage_success_window.clear()
                steps_in_current_stage = 0

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

    writer.close()
    env.close()
    sim.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        sim.close()
