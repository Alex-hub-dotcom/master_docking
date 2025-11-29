#!/usr/bin/env python3
"""
TEKO - 28-STAGE CURRICULUM TRAINING (v8.0 - ULTRA MICRO-STEPS)
===============================================================

v8.0: Ultra micro-steps for guaranteed convergence.
- 28 stages (S0-S27)
- Max +2° yaw OR +1cm lateral per stage
- Never increase both simultaneously

Usage:
    python train_teko_PPO.py --num_envs 16 --steps 100000000 --headless
    python train_teko_PPO.py --num_envs 16 --headless --checkpoint path/to/ckpt.pt
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
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "min_stage_steps": 50_000,
}

MAX_STAGE_STEPS = 1000_000
CHECKPOINT_INTERVAL = 30_000

# Rehearsal: always enabled, conservative settings
REHEARSAL_ENABLED = True
REHEARSAL_MIN_STAGE = 2
REHEARSAL_MAX_HISTORY = 3
REHEARSAL_INTERVAL_STEPS = 40_000
REHEARSAL_ROLLOUT_LEN = 32
REHEARSAL_UPDATES = 4

args = None


def get_stage_threshold(level: int) -> float:
    """Success thresholds per stage."""
    if level <= 0:
        return 0.80
    elif level <= 4:
        return 0.75
    elif level <= 6:
        return 0.70
    elif level <= 12:
        return 0.60
    elif level <= 22:
        return 0.58  # Advanced offset (ultra micro-steps)
    else:
        return 0.55  # 180° turns


def get_entropy_coef(level: int) -> float:
    if level <= 6:
        return 0.05   # S0-S6: low
    elif level <= 12:
        return 0.06   # S7-S12: micro-steps
    elif level <= 22:
        return 0.05   # S13-S22: ultra micro-steps
    else:
        return 0.05   # S23+: 180° turns

# =============================================================================
# PPO Functions
# =============================================================================

def compute_gae(rewards, values, dones, gamma, lam):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = 0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def ppo_update(policy, optimizer, obs, actions, logp_old, advantages, returns,
               epochs=4, batch_size=64, clip_ratio=0.15, value_clip=0.2,
               entropy_coef=0.05, value_coef=0.5, max_grad_norm=0.5):
    device = next(policy.parameters()).device
    T, N, C, H, W = obs.shape
    total = T * N

    obs_flat = obs.view(total, C, H, W)
    actions_flat = actions.view(total, 2)
    logp_flat = logp_old.view(-1)
    adv_flat = advantages.view(-1)
    ret_flat = returns.view(-1)
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    p_losses, v_losses, entropies = [], [], []

    for _ in range(epochs):
        idx = torch.randperm(total)
        for start in range(0, total, batch_size):
            mb = idx[start:start + batch_size]
            mb_obs = obs_flat[mb].to(device)
            mb_act = actions_flat[mb].to(device)
            mb_logp = logp_flat[mb].to(device)
            mb_adv = adv_flat[mb].to(device)
            mb_ret = ret_flat[mb].to(device)

            logp, value, entropy = policy.evaluate(mb_obs, mb_act)
            ratio = (logp - mb_logp).exp()
            p_loss = -torch.min(ratio * mb_adv,
                               torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * mb_adv).mean()
            
            if value_clip:
                value = torch.clamp(value, mb_ret - value_clip, mb_ret + value_clip)
            v_loss = F.mse_loss(value, mb_ret)
            
            loss = p_loss + value_coef * v_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(entropy.mean().item())

    return np.mean(p_losses), np.mean(v_losses), np.mean(entropies)


def do_training_step(env, policy, device, rollout_len):
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    obs_dict, _ = env.reset()
    obs = obs_dict["rgb"].to(device)

    for _ in range(rollout_len):
        with torch.no_grad():
            action, logp, value = policy.act(obs)
        obs_dict, reward, term, trunc, _ = env.step(action)
        next_obs = obs_dict["rgb"].to(device)
        done = term | trunc

        obs_buf.append(obs.cpu())
        act_buf.append(action.cpu())
        rew_buf.append(reward.cpu())
        val_buf.append(value.cpu())
        logp_buf.append(logp.cpu())
        done_buf.append(done.float().cpu())
        obs = next_obs

    return (torch.stack(obs_buf), torch.stack(act_buf), torch.stack(rew_buf),
            torch.stack(val_buf), torch.stack(logp_buf), torch.stack(done_buf))


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout_len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    args.enable_cameras = True

    print("\n" + "=" * 70)
    print("🎓 TEKO - 28-STAGE CURRICULUM (v8.0 - ULTRA MICRO-STEPS)")
    print("=" * 70)
    print(f"Envs: {args.num_envs} | Steps: {args.steps:,} | LR: {args.lr}")
    print("=" * 70 + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    app = AppLauncher(args)
    sim = app.app

    from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
    from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder
    from teko.tasks.direct.teko.curriculum.curriculum_manager import STAGE_NAMES

    class Policy(nn.Module):
        LOG_STD_V_MIN, LOG_STD_V_MAX = -1.5, 0.2
        LOG_STD_W_MIN, LOG_STD_W_MAX = -1.0, 0.6

        def __init__(self):
            super().__init__()
            self.encoder = create_visual_encoder("simple", 256, False)
            self.actor = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 2), nn.Tanh())
            self.critic = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1))
            self.log_std_v = nn.Parameter(torch.tensor(0.0))
            self.log_std_w = nn.Parameter(torch.tensor(0.0))

        def forward(self, obs):
            feat = self.encoder(obs)
            mean = self.actor(feat)
            value = self.critic(feat)
            log_std = torch.stack([
                self.log_std_v.clamp(self.LOG_STD_V_MIN, self.LOG_STD_V_MAX),
                self.log_std_w.clamp(self.LOG_STD_W_MIN, self.LOG_STD_W_MAX)])
            return mean, value, log_std

        def act(self, obs):
            mean, value, log_std = self.forward(obs)
            dist = torch.distributions.Normal(mean, log_std.exp())
            action = dist.sample()
            return action, dist.log_prob(action).sum(-1), value.squeeze(-1)

        def evaluate(self, obs, actions):
            mean, value, log_std = self.forward(obs)
            dist = torch.distributions.Normal(mean, log_std.exp())
            return dist.log_prob(actions).sum(-1), value.squeeze(-1), dist.entropy().sum(-1)

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
        steps_in_stage = 0
        last_ckpt = 0
        last_rehearsal = 0
        mastered = []

        if args.checkpoint:
            print(f"🔁 Loading: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=device)
            policy.load_state_dict(ckpt["policy"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0)
            env.set_curriculum_level(ckpt.get("curriculum_level", 0))
            steps_in_stage = ckpt.get("steps_in_stage", 0)
            mastered = ckpt.get("mastered_stages", [])
            last_ckpt = start_step
            last_rehearsal = start_step
            print(f"Resumed: step {start_step}, S{env.curriculum_level}, mastered: {mastered}")

        log_dir = f"teko_curriculum/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"📊 Logs: {log_dir}\n")

        obs_dict, _ = env.reset()
        obs = obs_dict["rgb"].to(device)

        ep_rewards = deque(maxlen=100)
        ep_lengths = deque(maxlen=100)
        ep_successes = deque(maxlen=100)
        stage_successes = deque(maxlen=200)

        cur_reward = torch.zeros(args.num_envs, device=device)
        cur_length = torch.zeros(args.num_envs, dtype=torch.int32, device=device)

        step = start_step

        try:
            while step < args.steps:
                obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []

                for _ in range(args.rollout_len):
                    with torch.no_grad():
                        action, logp, value = policy.act(obs)

                    obs_dict, reward, term, trunc, _ = env.step(action)
                    next_obs = obs_dict["rgb"].to(device)
                    done = term | trunc

                    cur_reward += reward
                    cur_length += 1

                    for i in range(args.num_envs):
                        if done[i]:
                            ep_rewards.append(cur_reward[i].item())
                            ep_lengths.append(cur_length[i].item())
                            success = reward[i] > 50.0
                            ep_successes.append(1.0 if success else 0.0)
                            stage_successes.append(1.0 if success else 0.0)
                            cur_reward[i] = 0.0
                            cur_length[i] = 0

                    obs_buf.append(obs.cpu())
                    act_buf.append(action.cpu())
                    rew_buf.append(reward.cpu())
                    val_buf.append(value.cpu())
                    logp_buf.append(logp.cpu())
                    done_buf.append(done.float().cpu())

                    obs = next_obs
                    step += args.num_envs
                    steps_in_stage += args.num_envs

                obs_t = torch.stack(obs_buf)
                act_t = torch.stack(act_buf)
                rew_t = torch.stack(rew_buf)
                val_t = torch.stack(val_buf)
                logp_t = torch.stack(logp_buf)
                done_t = torch.stack(done_buf)

                mean_r = np.mean(ep_rewards) if ep_rewards else 0
                mean_len = np.mean(ep_lengths) if ep_lengths else 0
                ssr = np.mean(stage_successes) if stage_successes else 0

                stage = env.curriculum_level
                threshold = get_stage_threshold(stage)
                ent_coef = get_entropy_coef(stage)

                with torch.no_grad():
                    adv, ret = compute_gae(rew_t, val_t, done_t,
                                          HYPERPARAMS["gamma"], HYPERPARAMS["gae_lambda"])

                p_loss, v_loss, entropy = ppo_update(
                    policy, optimizer, obs_t, act_t, logp_t, adv, ret,
                    epochs=args.epochs, batch_size=args.batch_size,
                    clip_ratio=HYPERPARAMS["clip_ratio"],
                    value_clip=HYPERPARAMS["value_clip"],
                    entropy_coef=ent_coef,
                    value_coef=HYPERPARAMS["value_coef"],
                    max_grad_norm=HYPERPARAMS["max_grad_norm"])

                # Logging
                writer.add_scalar("train/reward", mean_r, step)
                writer.add_scalar("train/stage_success", ssr, step)
                writer.add_scalar("train/curriculum_stage", stage, step)
                writer.add_scalar("train/entropy_coef", ent_coef, step)

                if hasattr(env, 'reward_components') and env.reward_components:
                    for name, vals in env.reward_components.items():
                        if vals:
                            writer.add_scalar(f"rewards/{name}", np.mean(vals), step)
                    for k in env.reward_components:
                        env.reward_components[k].clear()

                print(f"[{step:8d}] S{stage:02d} | R={mean_r:6.1f} | "
                      f"SSR={ssr*100:4.1f}% | Thr={threshold*100:.0f}% | "
                      f"Steps={steps_in_stage:6d} | Ent={ent_coef:.3f}")

                # Advancement
                if steps_in_stage >= HYPERPARAMS["min_stage_steps"]:
                    advance = False
                    mastered_now = False

                    if len(stage_successes) >= 50 and ssr >= threshold:
                        advance = True
                        mastered_now = True
                        print(f"✓ S{stage} mastered! (SSR={ssr:.1%})")
                    elif steps_in_stage >= MAX_STAGE_STEPS:
                        advance = True
                        print(f"⚠ Safety valve S{stage} after {steps_in_stage:,} steps")

                    if advance and stage < len(STAGE_NAMES) - 1:
                        if mastered_now and stage not in mastered:
                            mastered.append(stage)
                            mastered = sorted(set(mastered))
                            print(f"🧠 Mastered: {mastered}")

                        # Stage mixing
                        print(f"🔄 Mixing S{stage} ↔ S{stage+1}...")
                        for i in range(40):
                            env.set_curriculum_level(stage if i % 2 == 0 else stage + 1)
                            data = do_training_step(env, policy, device, 16)
                            with torch.no_grad():
                                a, r = compute_gae(data[2], data[3], data[5],
                                                  HYPERPARAMS["gamma"], HYPERPARAMS["gae_lambda"])
                            ppo_update(policy, optimizer, data[0], data[1], data[4], a, r,
                                      epochs=args.epochs, batch_size=args.batch_size,
                                      entropy_coef=get_entropy_coef(env.curriculum_level))

                        env.set_curriculum_level(stage + 1)
                        obs_dict, _ = env.reset()
                        obs = obs_dict["rgb"].to(device)
                        cur_reward.zero_()
                        cur_length.zero_()
                        stage_successes.clear()
                        steps_in_stage = 0
                        print(f"➡️ Advanced to S{stage + 1}")

                # Rehearsal
                if (REHEARSAL_ENABLED and mastered and 
                    env.curriculum_level >= REHEARSAL_MIN_STAGE and
                    step - last_rehearsal >= REHEARSAL_INTERVAL_STEPS):
                    
                    stage = env.curriculum_level
                    candidates = [s for s in mastered 
                                 if max(0, stage - REHEARSAL_MAX_HISTORY) <= s < stage]
                    
                    if candidates:
                        r_stage = int(np.random.choice(candidates))
                        print(f"🧪 Rehearsal S{r_stage}")
                        env.set_curriculum_level(r_stage)
                        
                        for _ in range(REHEARSAL_UPDATES):
                            data = do_training_step(env, policy, device, REHEARSAL_ROLLOUT_LEN)
                            with torch.no_grad():
                                a, r = compute_gae(data[2], data[3], data[5],
                                                  HYPERPARAMS["gamma"], HYPERPARAMS["gae_lambda"])
                            ppo_update(policy, optimizer, data[0], data[1], data[4], a, r,
                                      epochs=args.epochs, batch_size=args.batch_size,
                                      entropy_coef=get_entropy_coef(r_stage))

                        env.set_curriculum_level(stage)
                        obs_dict, _ = env.reset()
                        obs = obs_dict["rgb"].to(device)
                        cur_reward.zero_()
                        cur_length.zero_()
                        last_rehearsal = step
                        print(f"✅ Back to S{stage}")

                # Checkpoint
                if step - last_ckpt >= CHECKPOINT_INTERVAL:
                    path = f"{log_dir}/ckpt_{step}.pt"
                    torch.save({
                        "policy": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "curriculum_level": env.curriculum_level,
                        "steps_in_stage": steps_in_stage,
                        "mastered_stages": mastered,
                    }, path)
                    last_ckpt = step
                    print(f"💾 {path}")

        except KeyboardInterrupt:
            path = f"{log_dir}/interrupt_{step}.pt"
            torch.save({
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "curriculum_level": env.curriculum_level,
                "steps_in_stage": steps_in_stage,
                "mastered_stages": mastered,
            }, path)
            print(f"\n💾 Saved: {path}")

    finally:
        if writer:
            writer.close()
        if env:
            env.close()
        sim.close()


if __name__ == "__main__":
    main()