#!/usr/bin/env python3
"""
TEKO Vision + Spatial Attention + YawAux Head - Debug Training
Best of both worlds: Attention helps CNN focus, YawAux supervises orientation
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import argparse
import sys
import math
import socket
import time
from collections import deque
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.app import AppLauncher
print = partial(print, flush=True)

CONFIG = {
    "max_steps": 200_000_000,
    "max_hours": 72,
    
    "learning_rate": 1e-4,
    "entropy_coef": 0.015,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "epochs": 5,
    "batch_size": 2048,
    
    "aux_yaw_coef": 0.3,  # YawAux loss weight
    
    "num_envs": 120,  # Slightly less due to attention overhead
    "rollout_len": 128,
    
    "advance_threshold": 0.75,
    "min_steps_before_advance": 200_000,
    "max_stage": 32,
    
    "log_interval": 50_000,
    "save_interval": 2_000_000,
}


class SpatialAttention(nn.Module):
    """Spatial attention module - highlights important regions."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x * attn


class ChannelAttention(nn.Module):
    """Channel attention - weighs feature channels."""
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
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class VisionEncoderAttentionYaw(nn.Module):
    """CNN with Spatial+Channel Attention and YawAux head."""
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        # Attention after conv3
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
        
        # Yaw auxiliary head
        self.yaw_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
            nn.Tanh()  # Output in [-1, 1], scaled to [-pi, pi]
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
        
        # Apply attention
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        
        return x.flatten(1)
    
    def forward(self, x):
        x = self._forward_conv(x)
        features = F.relu(self.fc(x))
        return features
    
    def predict_yaw(self, features):
        """Predict yaw error from features. Output in [-pi, pi]."""
        return self.yaw_head(features) * math.pi


class VisionIMUAttentionYawPolicy(nn.Module):
    """Policy with Vision+Attention+IMU fusion and YawAux head."""
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
    
    def __init__(self, vis_dim=256, imu_dim=6, hidden=256, action_dim=2):
        super().__init__()
        
        self.vision_encoder = VisionEncoderAttentionYaw(in_channels=4, feature_dim=vis_dim)
        
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
        )
        
        fused_dim = vis_dim + 64
        
        self.actor_head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        
        # Asymmetric critic with privileged info
        priv_dim = 7  # dx, dy, dz, yaw_err, vx, vy, omega
        self.critic_head = nn.Sequential(
            nn.Linear(fused_dim + priv_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(True),
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
        
        # Yaw prediction
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
    
    metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "yaw_loss": 0}
    n_updates = 0
    
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(total, device=device)
        for start in range(0, total, cfg["batch_size"]):
            mb = idx[start:start + cfg["batch_size"]]
            
            priv_mb = priv_flat[mb] if priv_flat is not None else None
            logp, val, ent, yaw_pred = policy.evaluate(
                rgb_flat[mb], imu_flat[mb], actions_flat[mb], priv_mb
            )
            
            ratio = torch.exp(logp - old_logp_flat[mb])
            surr1 = ratio * adv_flat[mb]
            surr2 = torch.clamp(ratio, 1 - cfg["clip_ratio"], 1 + cfg["clip_ratio"]) * adv_flat[mb]
            
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * F.mse_loss(val, ret_flat[mb])
            
            # Yaw auxiliary loss
            yaw_loss = F.mse_loss(yaw_pred, yaw_flat[mb])
            
            loss = p_loss + cfg["value_coef"] * v_loss - cfg["entropy_coef"] * ent.mean() + cfg["aux_yaw_coef"] * yaw_loss
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            
            metrics["policy_loss"] += p_loss.item()
            metrics["value_loss"] += v_loss.item()
            metrics["entropy"] += ent.mean().item()
            metrics["yaw_loss"] += yaw_loss.item()
            n_updates += 1
    
    return {k: v / max(n_updates, 1) for k, v in metrics.items()}


def train(args):
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
    
    num_envs = CONFIG["num_envs"]
    rollout_len = CONFIG["rollout_len"]
    
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
    print("TEKO Vision + Attention + YawAux - Debug Training")
    print("=" * 70)
    print(f"Host: {socket.gethostname()}")
    print(f"Envs: {num_envs} | Max Steps: {CONFIG['max_steps']:,}")
    print(f"Entropy: {CONFIG['entropy_coef']} | YawAux Coef: {CONFIG['aux_yaw_coef']}")
    print("=" * 70)
    
    has_privileged = "privileged" in obs_dict and obs_dict["privileged"] is not None
    if has_privileged:
        print("[OK] Privileged observations available - YawAux + Attention ENABLED")
    else:
        print("[WARN] No privileged observations - YawAux supervision limited")
    
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
                    yaw_target = priv[:, 3:4]  # yaw_error at index 3
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
                policy, optimizer,
                rgb_buf, imu_buf, actions_buf, logprobs_buf,
                advantages, returns, yaw_targets_buf,
                priv_buf if has_privileged else None,
                CONFIG
            )
            
            ssr = float(np.mean(stage_successes)) if stage_successes else 0.0
            
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
            
            if step >= next_log:
                mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
                print(
                    f"[{step:,}] S{current_stage:02d} | SSR: {ssr:.1%} | "
                    f"R: {mean_r:.1f} | YawL: {metrics['yaw_loss']:.4f} | "
                    f"Ent: {metrics['entropy']:.3f} | MaxS: {max_stage_reached} | {elapsed_h:.1f}h"
                )
                next_log += CONFIG["log_interval"]
            
            if step >= next_save:
                ckpt_path = f"/home/schux00/checkpoints/vision_attn_yaw_S{current_stage}_{step//1000}k.pt"
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    "step": step,
                    "stage": current_stage,
                    "max_stage": max_stage_reached,
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, ckpt_path)
                print(f"[SAVE] {ckpt_path}")
                next_save += CONFIG["save_interval"]
            
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
        print("\n[INTERRUPTED]")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        print("\n[INTERRUPTED]")
    
    finally:
        final_path = f"/home/schux00/checkpoints/vision_attn_yaw_FINAL_S{max_stage_reached}.pt"
        torch.save({
            "step": step,
            "stage": current_stage,
            "max_stage": max_stage_reached,
            "policy": policy.state_dict(),
        }, final_path)
        print(f"[FINAL] Saved to {final_path}")
        print(f"[DONE] MaxStage={max_stage_reached}, Steps={step:,}, Time={(time.time()-t0)/3600:.1f}h")
        
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
