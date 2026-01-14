#!/usr/bin/env python3
"""
TEKO Vision + Attention + YawAux - OPTIMAL CONFIG (Final)
=========================================================
Hyperparameters from Optuna Trial 80 (reached S41/180°)
With TensorBoard logging for thesis figures.

Author: Alexandre Schleier Neves da Silva
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import argparse
import sys
import math
import socket
import time
import csv
from collections import deque
from functools import partial
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from isaaclab.app import AppLauncher
print = partial(print, flush=True)

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("[WARN] TensorBoard not available")

# =============================================================================
# OPTIMAL HYPERPARAMETERS from Optuna Trial 80 (S41/180°)
# =============================================================================
CONFIG = {
    "max_steps": 200_000_000,
    "max_hours": 168,  # 7 days
    
    # OPTIMAL from Optuna Trial 80
    "learning_rate": 0.00016,
    "entropy_coef": 0.0062,
    "gae_lambda": 0.94,
    "gamma": 0.99,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "epochs": 5,
    "batch_size": 1024,
    
    "aux_yaw_coef": 0.31,
    
    "num_envs": 120,
    "rollout_len": 128,
    
    "advance_threshold": 0.75,
    "min_steps_before_advance": 200_000,
    "max_stage": 41,
    
    "log_interval": 50_000,
    "save_interval": 2_000_000,
}


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        y = x.view(b, c, -1).mean(-1)
        return x * self.fc(y).view(b, c, 1, 1)


class VisionEncoderAttentionYaw(nn.Module):
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.channel_attn = ChannelAttention(64)
        self.spatial_attn = SpatialAttention(64)
        self.gn1 = nn.GroupNorm(8, 32)
        self.gn2 = nn.GroupNorm(8, 64)
        self.gn3 = nn.GroupNorm(8, 64)
        
        self._init_weights()
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            flat_size = self._forward_conv(dummy).shape[1]
        
        self.fc = nn.Linear(flat_size, feature_dim)
        nn.init.orthogonal_(self.fc.weight, gain=1.0)
        nn.init.zeros_(self.fc.bias)
        
        self.yaw_head = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(True),
            nn.Linear(64, 32), nn.ReLU(True),
            nn.Linear(32, 1), nn.Tanh()
        )
        self.feature_dim = feature_dim
    
    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
    
    def _forward_conv(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x.flatten(1)
    
    def forward(self, x):
        return F.relu(self.fc(self._forward_conv(x)))
    
    def predict_yaw(self, features):
        return self.yaw_head(features) * math.pi


class VisionIMUAttentionYawPolicy(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
    
    def __init__(self, vis_dim=256, imu_dim=6, hidden=256, action_dim=2):
        super().__init__()
        self.vision_encoder = VisionEncoderAttentionYaw(in_channels=4, feature_dim=vis_dim)
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, 64), nn.ReLU(True),
            nn.Linear(64, 64), nn.ReLU(True),
        )
        fused_dim = vis_dim + 64
        self.actor_head = nn.Sequential(
            nn.Linear(fused_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        priv_dim = 7
        self.critic_head = nn.Sequential(
            nn.Linear(fused_dim + priv_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden // 2), nn.ReLU(True),
            nn.Linear(hidden // 2, 1),
        )
    
    def _std(self):
        return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
    
    def forward_features(self, rgb, imu):
        vis_feat = self.vision_encoder(rgb)
        imu_feat = self.imu_encoder(imu)
        return torch.cat([vis_feat, imu_feat], dim=-1), vis_feat
    
    def act(self, rgb, imu, privileged=None, deterministic=False):
        fused, vis_feat = self.forward_features(rgb, imu)
        mean = self.actor_head(fused)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        u = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        
        if privileged is not None:
            critic_in = torch.cat([fused, privileged], dim=-1)
        else:
            critic_in = torch.cat([fused, torch.zeros(fused.shape[0], 7, device=fused.device)], dim=-1)
        value = self.critic_head(critic_in).squeeze(-1)
        yaw_pred = self.vision_encoder.predict_yaw(vis_feat)
        return action, log_prob, value, yaw_pred
    
    def evaluate(self, rgb, imu, actions, privileged=None):
        fused, vis_feat = self.forward_features(rgb, imu)
        mean = self.actor_head(fused)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        u = torch.clamp(actions, -0.999, 0.999)
        u = 0.5 * (torch.log1p(u) - torch.log1p(-u))
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - actions.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        if privileged is not None:
            critic_in = torch.cat([fused, privileged], dim=-1)
        else:
            critic_in = torch.cat([fused, torch.zeros(fused.shape[0], 7, device=fused.device)], dim=-1)
        value = self.critic_head(critic_in).squeeze(-1)
        yaw_pred = self.vision_encoder.predict_yaw(vis_feat)
        return log_prob, value, entropy, yaw_pred


def compute_gae(rewards, values, dones, gamma, lam, last_value):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def ppo_update_with_yaw(policy, optimizer, rgb, imu, actions, old_logp, advantages, returns, yaw_targets, privileged, cfg):
    device = next(policy.parameters()).device
    T, N = rgb.shape[:2]
    total = T * N
    
    rgb_flat = rgb.view(total, *rgb.shape[2:])
    imu_flat = imu.view(total, -1)
    actions_flat = actions.view(total, -1)
    old_logp_flat = old_logp.view(total)
    adv_flat = (advantages.view(total) - advantages.mean()) / (advantages.std() + 1e-8)
    ret_flat = returns.view(total)
    yaw_flat = yaw_targets.view(total, 1)
    priv_flat = privileged.view(total, -1) if privileged is not None else None
    
    metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "yaw_loss": 0, "grad_norm": 0}
    n_updates = 0
    
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(total, device=device)
        for start in range(0, total, cfg["batch_size"]):
            mb = idx[start:start + cfg["batch_size"]]
            priv_mb = priv_flat[mb] if priv_flat is not None else None
            logp, val, ent, yaw_pred = policy.evaluate(rgb_flat[mb], imu_flat[mb], actions_flat[mb], priv_mb)
            
            ratio = torch.exp(logp - old_logp_flat[mb])
            surr1 = ratio * adv_flat[mb]
            surr2 = torch.clamp(ratio, 1 - cfg["clip_ratio"], 1 + cfg["clip_ratio"]) * adv_flat[mb]
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * F.mse_loss(val, ret_flat[mb])
            yaw_loss = F.mse_loss(yaw_pred, yaw_flat[mb])
            
            loss = p_loss + cfg["value_coef"] * v_loss - cfg["entropy_coef"] * ent.mean() + cfg["aux_yaw_coef"] * yaw_loss
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"]).item()
            optimizer.step()
            
            metrics["policy_loss"] += p_loss.item()
            metrics["value_loss"] += v_loss.item()
            metrics["entropy"] += ent.mean().item()
            metrics["yaw_loss"] += yaw_loss.item()
            metrics["grad_norm"] += grad_norm
            n_updates += 1
    
    return {k: v / max(n_updates, 1) for k, v in metrics.items()}


def train(args):
    # Fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    
    app = AppLauncher(args)
    sim = app.app
    
    sys.path.insert(0, "/workspace/teko/source/teko")
    from teko.tasks.direct.teko.teko_env_tiled_imu import TekoEnvTiledIMU
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
    
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = CONFIG["num_envs"]
    cfg.enable_curriculum = True
    cfg.asymmetric_critic = True
    
    env = TekoEnvTiledIMU(cfg=cfg)
    
    policy = VisionIMUAttentionYawPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=CONFIG["learning_rate"])
    
    # TensorBoard setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/home/schux00/tensorboard/vision_optimal_{timestamp}"
    csv_path = f"/home/schux00/logs/vision_optimal_{timestamp}.csv"
    
    writer = None
    if HAS_TENSORBOARD:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"[TB] Logging to {log_dir}")
    
    # CSV logging
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'stage', 'ssr', 'reward', 'entropy', 'yaw_loss', 'policy_loss', 'value_loss', 'hours'])
    
    num_envs = CONFIG["num_envs"]
    rollout_len = CONFIG["rollout_len"]
    
    # Buffers
    rgb_buf = torch.zeros((rollout_len, num_envs, 4, 128, 128), device=device)
    imu_buf = torch.zeros((rollout_len, num_envs, 6), device=device)
    actions_buf = torch.zeros((rollout_len, num_envs, 2), device=device)
    rewards_buf = torch.zeros((rollout_len, num_envs), device=device)
    values_buf = torch.zeros((rollout_len, num_envs), device=device)
    logprobs_buf = torch.zeros((rollout_len, num_envs), device=device)
    dones_buf = torch.zeros((rollout_len, num_envs), device=device)
    yaw_targets_buf = torch.zeros((rollout_len, num_envs, 1), device=device)
    priv_buf = torch.zeros((rollout_len, num_envs, 7), device=device)
    
    ep_rewards = deque(maxlen=300)
    stage_successes = deque(maxlen=300)
    cur_reward = torch.zeros(num_envs, device=device)
    
    current_stage = 0
    max_stage_reached = 0
    last_advance_step = 0
    
    obs_dict, _ = env.reset()
    step = 0
    t0 = time.time()
    next_log = CONFIG["log_interval"]
    next_save = CONFIG["save_interval"]
    
    print("=" * 70)
    print("TEKO Vision + Attention + YawAux - OPTIMAL (Final)")
    print("=" * 70)
    print(f"Host: {socket.gethostname()}")
    print(f"Envs: {num_envs} | Max Steps: {CONFIG['max_steps']:,}")
    print(f"LR: {CONFIG['learning_rate']} | Entropy: {CONFIG['entropy_coef']}")
    print(f"GAE Lambda: {CONFIG['gae_lambda']} | Batch: {CONFIG['batch_size']}")
    print(f"YawAux Coef: {CONFIG['aux_yaw_coef']}")
    print(f"TensorBoard: {log_dir}")
    print(f"CSV: {csv_path}")
    print("=" * 70)
    
    has_privileged = "privileged" in obs_dict and obs_dict["privileged"] is not None
    
    try:
        while step < CONFIG["max_steps"]:
            elapsed_h = (time.time() - t0) / 3600
            if elapsed_h > CONFIG["max_hours"]:
                print(f"[TIME] Reached {CONFIG['max_hours']}h limit")
                break
            
            for t in range(rollout_len):
                rgb = obs_dict["rgb"].to(device)
                imu = obs_dict["imu"].to(device)
                priv = obs_dict.get("privileged")
                if priv is not None:
                    priv = priv.to(device)
                    yaw_target = priv[:, 3:4]
                else:
                    yaw_target = torch.zeros(num_envs, 1, device=device)
                
                with torch.no_grad():
                    action, logp, value, _ = policy.act(rgb, imu, priv)
                
                rgb_buf[t] = rgb
                imu_buf[t] = imu
                actions_buf[t] = action
                logprobs_buf[t] = logp
                values_buf[t] = value
                yaw_targets_buf[t] = yaw_target
                if priv is not None:
                    priv_buf[t] = priv
                
                obs_dict, reward, term, trunc, info = env.step(action)
                done = term | trunc
                
                rewards_buf[t] = reward
                dones_buf[t] = done.float()
                cur_reward += reward
                
                if done.any():
                    done_idx = done.nonzero(as_tuple=False).squeeze(-1)
                    if hasattr(env, "_last_success"):
                        succ = env._last_success.float()
                    else:
                        _, _, sxy, _ = env.get_sphere_distances_from_physics()
                        succ = (sxy < 0.03).float()
                    ep_rewards.extend(cur_reward[done_idx].cpu().tolist())
                    stage_successes.extend(succ[done_idx].cpu().tolist())
                    cur_reward[done_idx] = 0
                
                step += num_envs
            
            with torch.no_grad():
                last_rgb = obs_dict["rgb"].to(device)
                last_imu = obs_dict["imu"].to(device)
                last_priv = obs_dict.get("privileged")
                if last_priv is not None:
                    last_priv = last_priv.to(device)
                _, _, last_value, _ = policy.act(last_rgb, last_imu, last_priv)
            
            advantages, returns = compute_gae(
                rewards_buf, values_buf, dones_buf,
                CONFIG["gamma"], CONFIG["gae_lambda"], last_value
            )
            
            metrics = ppo_update_with_yaw(
                policy, optimizer, rgb_buf, imu_buf, actions_buf, logprobs_buf,
                advantages, returns, yaw_targets_buf,
                priv_buf if has_privileged else None, CONFIG
            )
            
            ssr = float(np.mean(stage_successes)) if stage_successes else 0.0
            mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            
            # Curriculum advancement
            if (len(stage_successes) >= 100 and
                ssr >= CONFIG["advance_threshold"] and
                step - last_advance_step >= CONFIG["min_steps_before_advance"] and
                current_stage < CONFIG["max_stage"]):
                
                print(f"[ADVANCE] Stage {current_stage} -> {current_stage + 1} (SSR={ssr:.1%})")
                current_stage += 1
                max_stage_reached = max(max_stage_reached, current_stage)
                env.set_curriculum_level(current_stage)
                stage_successes.clear()
                last_advance_step = step
                
                if writer:
                    writer.add_scalar("curriculum/stage", current_stage, step)
            
            # Logging
            if step >= next_log:
                print(f"[{step:,}] S{current_stage:02d} | SSR: {ssr:.1%} | R: {mean_r:.1f} | "
                      f"YawL: {metrics['yaw_loss']:.3f} | Ent: {metrics['entropy']:.3f} | "
                      f"MaxS: {max_stage_reached} | {elapsed_h:.1f}h")
                
                # TensorBoard
                if writer:
                    writer.add_scalar("train/ssr", ssr, step)
                    writer.add_scalar("train/reward", mean_r, step)
                    writer.add_scalar("train/entropy", metrics["entropy"], step)
                    writer.add_scalar("train/yaw_loss", metrics["yaw_loss"], step)
                    writer.add_scalar("train/policy_loss", metrics["policy_loss"], step)
                    writer.add_scalar("train/value_loss", metrics["value_loss"], step)
                    writer.add_scalar("train/grad_norm", metrics["grad_norm"], step)
                    writer.add_scalar("curriculum/stage", current_stage, step)
                    writer.add_scalar("curriculum/max_stage", max_stage_reached, step)
                
                # CSV
                csv_writer.writerow([step, current_stage, f"{ssr:.4f}", f"{mean_r:.2f}",
                                    f"{metrics['entropy']:.4f}", f"{metrics['yaw_loss']:.4f}",
                                    f"{metrics['policy_loss']:.4f}", f"{metrics['value_loss']:.4f}",
                                    f"{elapsed_h:.2f}"])
                csv_file.flush()
                
                next_log += CONFIG["log_interval"]
            
            # Save checkpoint
            if step >= next_save:
                ckpt_path = f"/home/schux00/checkpoints/vision_optimal_S{current_stage}_{step//1000}k.pt"
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    "step": step,
                    "stage": current_stage,
                    "max_stage": max_stage_reached,
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": CONFIG,
                }, ckpt_path)
                print(f"[SAVE] {ckpt_path}")
                next_save += CONFIG["save_interval"]
            
            # Early success
            if current_stage >= CONFIG["max_stage"] and ssr >= 0.70:
                print("=" * 70)
                print(f"[SUCCESS] Reached Stage {CONFIG['max_stage']} with SSR={ssr:.1%}!")
                print("=" * 70)
                break
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final save
        final_path = f"/home/schux00/checkpoints/vision_optimal_FINAL_S{max_stage_reached}.pt"
        torch.save({
            "step": step,
            "stage": current_stage,
            "max_stage": max_stage_reached,
            "policy": policy.state_dict(),
            "config": CONFIG,
        }, final_path)
        print(f"[FINAL] Saved to {final_path}")
        print(f"[DONE] MaxStage={max_stage_reached}, Steps={step:,}, Time={(time.time()-t0)/3600:.1f}h")
        
        if writer:
            writer.close()
        csv_file.close()
        env.close()
        sim.close()


def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    train(args)


if __name__ == "__main__":
    main()