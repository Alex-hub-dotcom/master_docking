#!/usr/bin/env python3
"""
TEKO Vision+IMU+Attention Debug - NO OPTUNA
============================================
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
    "max_steps": 150_000_000,
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
    
    "num_envs": 120,  # Slightly less due to attention overhead
    "rollout_len": 128,
    
    "advance_threshold": 0.75,
    "min_steps_before_advance": 200_000,
    "max_stage": 32,
    
    "log_interval": 50_000,
    "save_interval": 2_000_000,
}

IMG_SIZE = 128
NUM_FRAMES = 4
IMU_DIM = 6
PRIVILEGED_DIM = 7


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        hidden = max(1, in_channels // 4)
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_channels, hidden), nn.ReLU(True),
            nn.Linear(hidden, in_channels), nn.Sigmoid(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ch_att = self.channel_fc(x).view(B, C, 1, 1)
        x = x * ch_att
        max_pool = x.max(dim=1, keepdim=True)[0]
        avg_pool = x.mean(dim=1, keepdim=True)
        sp_att = self.spatial_conv(torch.cat([max_pool, avg_pool], dim=1))
        return x * sp_att


class VisionEncoderWithAttention(nn.Module):
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, 8, 3, 1), nn.GroupNorm(8, 32), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.ReLU(True))
        self.attn = SpatialAttention(128)
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.GroupNorm(16, 256), nn.ReLU(True))
        
        with torch.no_grad():
            x = torch.zeros(1, in_channels, IMG_SIZE, IMG_SIZE)
            x = self.conv4(self.attn(self.conv3(self.conv2(self.conv1(x)))))
            flat_dim = x.view(1, -1).shape[1]
        
        self.fc = nn.Sequential(nn.Linear(flat_dim, 512), nn.ReLU(True), nn.Linear(512, feature_dim), nn.ReLU(True))
        self.feature_dim = feature_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attn(x)
        x = self.conv4(x)
        return self.fc(x.flatten(1))


class VisionIMUAttentionPolicy(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -1.0, 0.5
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.vision_encoder = VisionEncoderWithAttention(NUM_FRAMES, hidden_dim)
        
        self.imu_encoder = nn.Sequential(
            nn.Linear(IMU_DIM, 64), nn.ReLU(True),
            nn.Linear(64, 64), nn.ReLU(True),
        )
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 2),
        )
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        
        self.state_encoder = nn.Sequential(
            nn.Linear(PRIVILEGED_DIM, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 128, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 1),
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.actor_head, self.imu_encoder, self.state_encoder, self.critic_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
    
    def _std(self):
        return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
    
    def act(self, obs, deterministic=False):
        vision_feat = self.vision_encoder(obs["rgb"])
        imu_feat = self.imu_encoder(obs["imu"])
        actor_in = torch.cat([vision_feat, imu_feat], dim=-1)
        
        mean = self.actor_head(actor_in)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        
        u = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        
        if "privileged" in obs:
            state_feat = self.state_encoder(obs["privileged"])
            value = self.critic_head(torch.cat([vision_feat, imu_feat, state_feat], dim=-1)).squeeze(-1)
        else:
            value = torch.zeros(action.shape[0], device=action.device)
        
        return action, log_prob, value
    
    def evaluate(self, obs, actions):
        vision_feat = self.vision_encoder(obs["rgb"])
        imu_feat = self.imu_encoder(obs["imu"])
        actor_in = torch.cat([vision_feat, imu_feat], dim=-1)
        
        mean = self.actor_head(actor_in)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        
        u = torch.clamp(actions, -0.999, 0.999)
        u = 0.5 * (torch.log1p(u) - torch.log1p(-u))
        
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - actions.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        if "privileged" in obs:
            state_feat = self.state_encoder(obs["privileged"])
            value = self.critic_head(torch.cat([vision_feat, imu_feat, state_feat], dim=-1)).squeeze(-1)
        else:
            value = torch.zeros(actions.shape[0], device=actions.device)
        
        return log_prob, value, entropy


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


def ppo_update(policy, optimizer, rgb, imu, priv, actions, old_logp, adv, ret, cfg):
    device = next(policy.parameters()).device
    T, N = rgb.shape[:2]
    total = T * N
    
    rgb_flat = rgb.view(total, NUM_FRAMES, IMG_SIZE, IMG_SIZE)
    imu_flat = imu.view(total, IMU_DIM)
    priv_flat = priv.view(total, PRIVILEGED_DIM) if priv is not None else None
    act_flat = actions.view(total, 2)
    old_logp_flat = old_logp.view(total)
    adv_flat = (adv.view(total) - adv.mean()) / (adv.std() + 1e-8)
    ret_flat = ret.view(total)
    
    metrics = {"entropy": 0}
    n_updates = 0
    
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(total, device=device)
        for start in range(0, total, cfg["batch_size"]):
            mb = idx[start:start + cfg["batch_size"]]
            
            mb_obs = {"rgb": rgb_flat[mb].float() / 255.0, "imu": imu_flat[mb]}
            if priv_flat is not None:
                mb_obs["privileged"] = priv_flat[mb]
            
            logp, val, ent = policy.evaluate(mb_obs, act_flat[mb])
            
            ratio = torch.exp(logp - old_logp_flat[mb])
            surr1 = ratio * adv_flat[mb]
            surr2 = torch.clamp(ratio, 1 - cfg["clip_ratio"], 1 + cfg["clip_ratio"]) * adv_flat[mb]
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * F.mse_loss(val, ret_flat[mb])
            
            loss = p_loss + cfg["value_coef"] * v_loss - cfg["entropy_coef"] * ent.mean()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            
            metrics["entropy"] += ent.mean().item()
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
    cfg.tiled_camera.width = IMG_SIZE
    cfg.tiled_camera.height = IMG_SIZE
    cfg.enable_curriculum = True
    cfg.asymmetric_critic = True
    
    env = TekoEnvTiledIMU(cfg=cfg)
    
    policy = VisionIMUAttentionPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=CONFIG["learning_rate"])
    
    num_envs = CONFIG["num_envs"]
    rollout_len = CONFIG["rollout_len"]
    
    rgb_buf = torch.zeros((rollout_len, num_envs, NUM_FRAMES, IMG_SIZE, IMG_SIZE), device=device, dtype=torch.uint8)
    imu_buf = torch.zeros((rollout_len, num_envs, IMU_DIM), device=device)
    priv_buf = torch.zeros((rollout_len, num_envs, PRIVILEGED_DIM), device=device)
    actions_buf = torch.zeros((rollout_len, num_envs, 2), device=device)
    rewards_buf = torch.zeros((rollout_len, num_envs), device=device)
    values_buf = torch.zeros((rollout_len, num_envs), device=device)
    logprobs_buf = torch.zeros((rollout_len, num_envs), device=device)
    dones_buf = torch.zeros((rollout_len, num_envs), device=device)
    
    ep_rewards = deque(maxlen=300)
    stage_successes = deque(maxlen=300)
    cur_reward = torch.zeros(num_envs, device=device)
    
    current_stage = 0
    max_stage_reached = 0
    last_advance_step = 0
    env.set_curriculum_level(0)
    
    obs_dict, _ = env.reset()
    step = 0
    t0 = time.time()
    next_log = CONFIG["log_interval"]
    next_save = CONFIG["save_interval"]
    
    print("=" * 70)
    print("TEKO Vision+IMU+Attention Debug - NO OPTUNA")
    print("=" * 70)
    print(f"Host: {socket.gethostname()} | Envs: {num_envs}")
    print(f"CNN: Spatial Attention after conv3")
    print(f"Target: Stage {CONFIG['max_stage']}")
    print("=" * 70)
    
    has_priv = "privileged" in obs_dict
    
    try:
        while step < CONFIG["max_steps"]:
            elapsed_h = (time.time() - t0) / 3600
            if elapsed_h > CONFIG["max_hours"]:
                break
            
            for t in range(rollout_len):
                rgb = obs_dict["rgb"].to(device, dtype=torch.float32)
                imu = obs_dict["imu"].to(device, dtype=torch.float32)
                obs = {"rgb": rgb, "imu": imu}
                if has_priv:
                    obs["privileged"] = obs_dict["privileged"].to(device, dtype=torch.float32)
                
                with torch.no_grad():
                    action, logp, value = policy.act(obs)
                
                rgb_buf[t].copy_((rgb.clamp(0, 1) * 255).to(torch.uint8))
                imu_buf[t].copy_(imu)
                if has_priv:
                    priv_buf[t].copy_(obs["privileged"])
                
                actions_buf[t].copy_(action)
                logprobs_buf[t].copy_(logp)
                values_buf[t].copy_(value)
                
                obs_dict, reward, term, trunc, _ = env.step(action)
                done = term | trunc
                
                rewards_buf[t].copy_(reward)
                dones_buf[t].copy_(done.float())
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
                last_obs = {"rgb": obs_dict["rgb"].to(device, dtype=torch.float32),
                           "imu": obs_dict["imu"].to(device, dtype=torch.float32)}
                if has_priv:
                    last_obs["privileged"] = obs_dict["privileged"].to(device, dtype=torch.float32)
                _, _, last_value = policy.act(last_obs)
            
            adv, ret = compute_gae(rewards_buf, values_buf, dones_buf, CONFIG["gamma"], CONFIG["gae_lambda"], last_value)
            
            metrics = ppo_update(policy, optimizer, rgb_buf, imu_buf, priv_buf if has_priv else None,
                                actions_buf, logprobs_buf, adv, ret, CONFIG)
            
            ssr = float(np.mean(stage_successes)) if stage_successes else 0.0
            
            if (len(stage_successes) >= 100 and ssr >= CONFIG["advance_threshold"] and
                step - last_advance_step >= CONFIG["min_steps_before_advance"] and
                current_stage < CONFIG["max_stage"]):
                print(f"[ADVANCE] S{current_stage} -> S{current_stage + 1} (SSR={ssr:.1%})")
                current_stage += 1
                max_stage_reached = max(max_stage_reached, current_stage)
                env.set_curriculum_level(current_stage)
                stage_successes.clear()
                last_advance_step = step
            
            if step >= next_log:
                mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
                print(f"[{step:,}] S{current_stage:02d} | SSR: {ssr:.1%} | R: {mean_r:.1f} | "
                      f"Ent: {metrics['entropy']:.3f} | MaxS: {max_stage_reached} | {elapsed_h:.1f}h")
                next_log += CONFIG["log_interval"]
            
            if step >= next_save:
                ckpt = f"/home/schux00/checkpoints/vision_attn_S{current_stage}_{step//1000}k.pt"
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                torch.save({"step": step, "stage": current_stage, "policy": policy.state_dict()}, ckpt)
                print(f"[SAVE] {ckpt}")
                next_save += CONFIG["save_interval"]
            
            if current_stage >= CONFIG["max_stage"] and ssr >= 0.70:
                print(f"[SUCCESS] Reached S{CONFIG['max_stage']}!")
                break
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    finally:
        final = f"/home/schux00/checkpoints/vision_attn_FINAL_S{max_stage_reached}.pt"
        torch.save({"step": step, "stage": current_stage, "max_stage": max_stage_reached, "policy": policy.state_dict()}, final)
        print(f"[DONE] MaxStage={max_stage_reached}, Steps={step:,}")
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
