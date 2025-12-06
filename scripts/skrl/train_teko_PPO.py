#!/usr/bin/env python3
"""
TEKO PPO TRAINING (v11.0 - 17-STAGE LOW-SAMPLE OPTIMIZED)
==========================================================

Optimized for:
- 64×64 grayscale input with SimpleCNN v9.7
- 17-stage streamlined curriculum
- 65-100 parallel environments (VRAM-limited)
- Mixed precision training (AMP)
- More PPO epochs for better sample efficiency
- Balanced quality thresholds

Key optimizations:
- 6 PPO epochs (vs 3) for more gradient updates
- Longer rollouts (256 vs 128) for more data
- Mixed precision for VRAM efficiency
- Lower minimum steps per stage
- Minimal replay for accurate SSR

Author: Alexandre Schleier Neves da Silva
"""

import argparse
import os
import math
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from isaaclab.app import AppLauncher


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

HYPERPARAMS = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.15,
    "value_clip": 0.2,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "min_stage_steps": 40_000,  # Reduced from 50k for faster progression
}

# Stage-specific max steps (safety valve)
MAX_STAGE_STEPS_DEFAULT = 800_000
MAX_STAGE_STEPS_TURN = 1_200_000      # S8-S10 (goal visible)
MAX_STAGE_STEPS_BLIND = 2_000_000     # S11+ (blind search needs more time)

CHECKPOINT_INTERVAL = 30_000

# Rehearsal settings
REHEARSAL_ENABLED = True
REHEARSAL_MIN_STAGE = 2
REHEARSAL_MAX_HISTORY = 3
REHEARSAL_INTERVAL_STEPS = 150_000
REHEARSAL_ROLLOUT_LEN = 32
REHEARSAL_UPDATES = 2

args = None


# =============================================================================
# CURRICULUM FUNCTIONS (STAGE-AWARE, BALANCED THRESHOLDS)
# =============================================================================

def get_stage_threshold(level: int, is_blind: bool = False) -> float:
    """
    Success rate threshold to advance.
    Balanced: not too easy, not impossible.
    """
    if level <= 0:
        return 0.75  # Baby steps
    elif level <= 2:
        return 0.70  # Forward stages
    elif level <= 5:
        return 0.65  # Small offsets
    elif level <= 7:
        return 0.60  # Medium/large offsets
    elif level <= 10:  # Turn stages (goal visible)
        return 0.55
    else:  # Blind stages S11+
        return 0.45  # Lower for blind search


def get_max_stage_steps(level: int, is_blind: bool = False) -> int:
    """Get max steps allowed per stage before safety valve."""
    if is_blind or level >= 11:
        return MAX_STAGE_STEPS_BLIND
    elif level >= 8:
        return MAX_STAGE_STEPS_TURN
    else:
        return MAX_STAGE_STEPS_DEFAULT


def get_entropy_coef(level: int, is_blind: bool = False) -> float:
    """
    Entropy coefficient per curriculum stage.
    Higher for blind stages to encourage exploration.
    """
    if level <= 5:
        return 0.05
    elif level <= 7:
        return 0.06
    elif level <= 10:  # Turn stages (goal visible)
        return 0.06
    else:  # Blind stages S11+
        return 0.08


# =============================================================================
# PPO CORE FUNCTIONS (WITH MIXED PRECISION)
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

    returns = advantages + values
    return advantages, returns


def ppo_update(policy, optimizer, obs, actions, logp_old,
               advantages, returns,
               epochs=6, batch_size=1024,
               clip_ratio=0.15, value_clip=0.2,
               entropy_coef=0.05, value_coef=0.5,
               max_grad_norm=0.5):

    device = next(policy.parameters()).device
    T, N, C, H, W = obs.shape
    total = T * N

    obs_flat = obs.view(total, C, H, W)
    actions_flat = actions.view(total, 2)
    logp_flat = logp_old.view(-1)
    adv_flat = advantages.view(-1)
    ret_flat = returns.view(-1)

    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    p_losses, v_losses, entropies, grad_norms = [], [], [], []
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    for _ in range(epochs):
        idx = torch.randperm(total)
        for start in range(0, total, batch_size):
            mb = idx[start:start + batch_size]

            mb_obs = obs_flat[mb].to(device)
            mb_act = actions_flat[mb].to(device)
            mb_logp = logp_flat[mb].to(device)
            mb_adv = adv_flat[mb].to(device)
            mb_ret = ret_flat[mb].to(device)

            optimizer.zero_grad()
            
            # Forward pass with automatic mixed precision
            with torch.cuda.amp.autocast():
                logp, value, entropy = policy.evaluate(mb_obs, mb_act)
                ratio = (logp - mb_logp).exp()

                p_loss = -torch.min(
                    ratio * mb_adv,
                    torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv
                ).mean()

                if value_clip:
                    value = torch.clamp(value, mb_ret - value_clip, mb_ret + value_clip)
                v_loss = F.mse_loss(value, mb_ret)

                loss = p_loss + value_coef * v_loss - entropy_coef * entropy.mean()

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm).item()
            scaler.step(optimizer)
            scaler.update()

            grad_norms.append(grad_norm)
            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(entropy.mean().item())

    return np.mean(p_losses), np.mean(v_losses), np.mean(entropies), np.mean(grad_norms)


def collect_rollout(env, policy, device, rollout_len):
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

    return (
        torch.stack(obs_buf),
        torch.stack(act_buf),
        torch.stack(rew_buf),
        torch.stack(val_buf),
        torch.stack(logp_buf),
        torch.stack(done_buf),
    )


# =============================================================================
# POLICY NETWORK
# =============================================================================

class Policy(nn.Module):
    LOG_STD_V_MIN, LOG_STD_V_MAX = -1.5, 0.2
    LOG_STD_W_MIN, LOG_STD_W_MAX = -1.0, 0.6

    def __init__(self, create_visual_encoder):
        super().__init__()

        self.encoder = create_visual_encoder(
            architecture="simple",
            feature_dim=256,
            pretrained=False,
            num_frame_stack=4,
            input_h=84,
            input_w=84,
        )

        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.log_std_v = nn.Parameter(torch.tensor(0.0))
        self.log_std_w = nn.Parameter(torch.tensor(0.0))

        self._init_heads()

    def _init_heads(self):
        for module in [self.actor, self.critic]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)

    def forward(self, obs):
        feat = self.encoder(obs)
        mean = self.actor(feat)
        value = self.critic(feat)
        log_std = torch.stack([
            self.log_std_v.clamp(self.LOG_STD_V_MIN, self.LOG_STD_V_MAX),
            self.log_std_w.clamp(self.LOG_STD_W_MIN, self.LOG_STD_W_MAX),
        ])
        return mean, value, log_std

    def act(self, obs):
        mean, value, log_std = self.forward(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), value.squeeze(-1)

    def evaluate(self, obs, actions):
        mean, value, log_std = self.forward(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        return (
            dist.log_prob(actions).sum(-1),
            value.squeeze(-1),
            dist.entropy().sum(-1),
        )


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def save_checkpoint(path, policy, optimizer, step, curriculum_level,
                    steps_in_stage, mastered_stages):
    torch.save({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "curriculum_level": curriculum_level,
        "steps_in_stage": steps_in_stage,
        "mastered_stages": mastered_stages,
    }, path)


def load_checkpoint(path, policy, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return (
        ckpt.get("step", 0),
        ckpt.get("curriculum_level", 0),
        ckpt.get("steps_in_stage", 0),
        ckpt.get("mastered_stages", []),
    )


def log_metrics(writer, step, metrics):
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)


def log_vram_usage(writer, step):
    """Log GPU VRAM usage to TensorBoard."""
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)    # GB
    max_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)  # GB
    
    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
    
    # Calculate percentages
    allocated_pct = (allocated / total_memory) * 100
    reserved_pct = (reserved / total_memory) * 100
    
    writer.add_scalar("system/vram_allocated_gb", allocated, step)
    writer.add_scalar("system/vram_reserved_gb", reserved, step)
    writer.add_scalar("system/vram_max_allocated_gb", max_allocated, step)
    writer.add_scalar("system/vram_allocated_percent", allocated_pct, step)
    writer.add_scalar("system/vram_reserved_percent", reserved_pct, step)
    writer.add_scalar("system/vram_total_gb", total_memory, step)


# =============================================================================
# CURRICULUM MANAGEMENT
# =============================================================================

def do_stage_mixing(env, policy, optimizer, device, current_stage,
                    num_iterations=40, rollout_len=16, epochs=6, batch_size=1024):
    print(f"🔄 Mixing S{current_stage} ↔ S{current_stage + 1}...")

    for i in range(num_iterations):
        stage = current_stage if i % 2 == 0 else current_stage + 1
        env.set_curriculum_level(stage)

        data = collect_rollout(env, policy, device, rollout_len)
        with torch.no_grad():
            adv, ret = compute_gae(
                data[2], data[3], data[5],
                HYPERPARAMS["gamma"], HYPERPARAMS["gae_lambda"]
            )

        ppo_update(
            policy, optimizer,
            data[0], data[1], data[4], adv, ret,
            epochs=epochs,
            batch_size=batch_size,
            entropy_coef=get_entropy_coef(stage, stage >= 11),
        )


def do_rehearsal(env, policy, optimizer, device, current_stage, mastered_stages,
                 num_updates=2, rollout_len=32, epochs=6, batch_size=1024):
    candidates = [
        s for s in mastered_stages
        if max(0, current_stage - REHEARSAL_MAX_HISTORY) <= s < current_stage
    ]

    if not candidates:
        return False

    r_stage = int(np.random.choice(candidates))
    print(f"🧪 Rehearsal S{r_stage}")
    env.set_curriculum_level(r_stage)

    for _ in range(num_updates):
        data = collect_rollout(env, policy, device, rollout_len)
        with torch.no_grad():
            adv, ret = compute_gae(
                data[2], data[3], data[5],
                HYPERPARAMS["gamma"], HYPERPARAMS["gae_lambda"]
            )

        ppo_update(
            policy, optimizer,
            data[0], data[1], data[4], adv, ret,
            epochs=epochs,
            batch_size=batch_size,
            entropy_coef=get_entropy_coef(r_stage, r_stage >= 11),
        )

    return True


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TEKO PPO Training (17-Stage Optimized)")

    parser.add_argument("--num_envs", type=int, default=65)
    parser.add_argument("--steps", type=int, default=200_000_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--rollout_len", type=int, default=256)  # Longer rollouts
    parser.add_argument("--epochs", type=int, default=6)  # More epochs
    parser.add_argument("--batch_size", type=int, default=1024)  # Smaller batches

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_lr_decay", action="store_true", help="Disable LR scheduler")

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    global args
    args = parse_args()
    args.enable_cameras = True

    print("\n" + "=" * 75)
    print("🎓 TEKO - 17-STAGE CURRICULUM (64×64 GRAYSCALE, PPO v11.0)")
    print("=" * 75)
    print(f"Envs: {args.num_envs} | Steps: {args.steps:,}")
    print(f"LR: {args.lr} → {args.lr_min} | Batch: {args.batch_size} | Rollout: {args.rollout_len}")
    print(f"Epochs: {args.epochs} | Mixed Precision: Enabled")
    print("=" * 75 + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    app = AppLauncher(args)
    sim = app.app

    from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
    from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder
    from teko.tasks.direct.teko.curriculum.curriculum_manager import (
        NUM_STAGES, STAGE_NAMES, is_turn_stage, is_blind_stage, get_stage_angle
    )

    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_curriculum = True

    env = None
    writer = None

    try:
        env = TekoEnv(cfg=cfg)
        policy = Policy(create_visual_encoder).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

        total_updates = args.steps // (args.num_envs * args.rollout_len)
        if not args.no_lr_decay:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_updates, eta_min=args.lr_min
            )
        else:
            scheduler = None

        start_step = 0
        steps_in_stage = 0
        last_ckpt = 0
        last_rehearsal = 0
        mastered = []

        if args.checkpoint:
            print(f"🔁 Loading checkpoint: {args.checkpoint}")
            start_step, curr_level, steps_in_stage, mastered = load_checkpoint(
                args.checkpoint, policy, optimizer, device
            )
            env.set_curriculum_level(curr_level)
            last_ckpt = start_step
            last_rehearsal = start_step

            if scheduler:
                completed_updates = start_step // (args.num_envs * args.rollout_len)
                for _ in range(completed_updates):
                    scheduler.step()

            print(f"Resumed from step {start_step}, stage S{curr_level}")

        log_dir = f"teko_curriculum/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"📊 Logs: {log_dir}\n")

        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"🧠 Model: {total_params:,} params ({trainable_params:,} trainable)\n")
        
        # Log initial VRAM usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"💾 VRAM: {allocated:.2f} / {total_memory:.2f} GB allocated ({allocated/total_memory*100:.1f}%)")
            print(f"💾 VRAM: {reserved:.2f} / {total_memory:.2f} GB reserved ({reserved/total_memory*100:.1f}%)\n")

        obs_dict, _ = env.reset()
        obs = obs_dict["rgb"].to(device)

        ep_rewards = deque(maxlen=100)
        ep_lengths = deque(maxlen=100)
        ep_successes = deque(maxlen=100)
        stage_successes = deque(maxlen=200)
        
        recent_successes = deque(maxlen=50)
        recent_timeouts = deque(maxlen=50)

        cur_reward = torch.zeros(args.num_envs, device=device)
        cur_length = torch.zeros(args.num_envs, dtype=torch.int32, device=device)

        step = start_step

        print("🚀 Starting training loop...\n")

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
                        timeout = trunc[i].item() and not term[i].item()
                        
                        ep_successes.append(1.0 if success else 0.0)
                        stage_successes.append(1.0 if success else 0.0)
                        recent_successes.append(1.0 if success else 0.0)
                        recent_timeouts.append(1.0 if timeout else 0.0)
                        
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

            mean_r = np.mean(ep_rewards) if ep_rewards else 0.0
            mean_len = np.mean(ep_lengths) if ep_lengths else 0.0
            ssr = np.mean(stage_successes) if stage_successes else 0.0
            
            success_count = sum(recent_successes) if recent_successes else 0
            timeout_count = sum(recent_timeouts) if recent_timeouts else 0

            stage = env.curriculum_level
            is_blind = is_blind_stage(stage)
            is_turn = is_turn_stage(stage)
            angle = get_stage_angle(stage)
            
            threshold = get_stage_threshold(stage, is_blind)
            ent_coef = get_entropy_coef(stage, is_blind)
            max_steps = get_max_stage_steps(stage, is_blind)
            current_lr = optimizer.param_groups[0]['lr']

            with torch.no_grad():
                adv, ret = compute_gae(
                    rew_t, val_t, done_t,
                    HYPERPARAMS["gamma"], HYPERPARAMS["gae_lambda"]
                )

            p_loss, v_loss, entropy, grad_norm = ppo_update(
                policy, optimizer,
                obs_t, act_t, logp_t, adv, ret,
                epochs=args.epochs,
                batch_size=args.batch_size,
                clip_ratio=HYPERPARAMS["clip_ratio"],
                value_clip=HYPERPARAMS["value_clip"],
                entropy_coef=ent_coef,
                value_coef=HYPERPARAMS["value_coef"],
                max_grad_norm=HYPERPARAMS["max_grad_norm"],
            )

            if scheduler:
                scheduler.step()

            log_metrics(writer, step, {
                "train/reward": mean_r,
                "train/episode_length": mean_len,
                "train/stage_success": ssr,
                "train/stage": stage,
                "train/stage_angle": angle,
                "train/is_blind_stage": float(is_blind),
                "train/entropy_coef": ent_coef,
                "train/learning_rate": current_lr,
                "train/grad_norm": grad_norm,
                "train/steps_in_stage": steps_in_stage,
                "train/max_stage_steps": max_steps,
                "train/success_count_50ep": success_count,
                "train/timeout_count_50ep": timeout_count,
                "loss/policy": p_loss,
                "loss/value": v_loss,
                "loss/entropy": entropy,
                "policy/log_std_v": policy.log_std_v.item(),
                "policy/log_std_w": policy.log_std_w.item(),
            })
            
            # Log VRAM usage every iteration
            log_vram_usage(writer, step)

            if hasattr(env, "reward_components"):
                for k, v in env.reward_components.items():
                    if v:
                        writer.add_scalar(f"rewards/{k}", np.mean(v), step)
                    env.reward_components[k].clear()

            stage_marker = "🔍" if is_blind else ("🔄" if is_turn else "  ")
            
            # Add VRAM info every 10 iterations
            vram_str = ""
            if step % (args.rollout_len * args.num_envs * 10) < (args.rollout_len * args.num_envs):
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    vram_str = f" | VRAM={allocated:.1f}/{total_memory:.1f}GB"
            
            print(
                f"[{step:9d}] S{stage:02d} ({angle:3.0f}°) {stage_marker} | "
                f"R={mean_r:6.1f} | Succ={success_count:2.0f}/50 T/O={timeout_count:2.0f} | "
                f"SSR={ssr*100:4.1f}% >{threshold*100:.0f}% | "
                f"Steps={steps_in_stage/1e6:.2f}M/{max_steps/1e6:.1f}M{vram_str}"
            )

            if steps_in_stage >= HYPERPARAMS["min_stage_steps"]:
                advance = False
                mastered_now = False

                if len(stage_successes) >= 50 and ssr >= threshold:
                    advance = True
                    mastered_now = True
                    print(f"✓ S{stage} ({angle}°) mastered! (SSR={ssr:.1%})")

                elif steps_in_stage >= max_steps:
                    advance = True
                    print(f"⚠ Safety valve at S{stage} ({angle}°) after {steps_in_stage/1e6:.1f}M steps")

                if advance and stage < NUM_STAGES - 1:
                    if mastered_now and stage not in mastered:
                        mastered.append(stage)
                        mastered = sorted(set(mastered))
                        print(f"🧠 Mastered stages: {mastered}")

                    do_stage_mixing(
                        env, policy, optimizer, device, stage,
                        num_iterations=40, rollout_len=16,
                        epochs=args.epochs, batch_size=args.batch_size
                    )

                    env.set_curriculum_level(stage + 1)
                    obs_dict, _ = env.reset()
                    obs = obs_dict["rgb"].to(device)
                    cur_reward.zero_()
                    cur_length.zero_()
                    stage_successes.clear()
                    recent_successes.clear()
                    recent_timeouts.clear()
                    steps_in_stage = 0
                    
                    next_angle = get_stage_angle(stage + 1)
                    print(f"➡️ Advanced to S{stage + 1} ({next_angle}°)")

            if (REHEARSAL_ENABLED and mastered and
                env.curriculum_level >= REHEARSAL_MIN_STAGE and
                step - last_rehearsal >= REHEARSAL_INTERVAL_STEPS):

                stage = env.curriculum_level

                if do_rehearsal(
                    env, policy, optimizer, device, stage, mastered,
                    num_updates=REHEARSAL_UPDATES,
                    rollout_len=REHEARSAL_ROLLOUT_LEN,
                    epochs=args.epochs, batch_size=args.batch_size
                ):
                    env.set_curriculum_level(stage)
                    obs_dict, _ = env.reset()
                    obs = obs_dict["rgb"].to(device)
                    cur_reward.zero_()
                    cur_length.zero_()
                    last_rehearsal = step
                    print(f"✅ Back to S{stage}")

            if step - last_ckpt >= CHECKPOINT_INTERVAL:
                path = f"{log_dir}/ckpt_{step}.pt"
                save_checkpoint(
                    path, policy, optimizer, step,
                    env.curriculum_level, steps_in_stage, mastered
                )
                last_ckpt = step
                print(f"💾 Saved: {path}")

    except KeyboardInterrupt:
        path = f"{log_dir}/interrupt_{step}.pt"
        save_checkpoint(
            path, policy, optimizer, step,
            env.curriculum_level, steps_in_stage, mastered
        )
        print(f"\n💾 Interrupted: saved to {path}")

    finally:
        if writer:
            writer.close()
        if env:
            env.close()
        sim.close()


if __name__ == "__main__":
    main()