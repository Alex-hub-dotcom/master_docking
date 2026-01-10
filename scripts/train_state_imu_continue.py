#!/usr/bin/env python3
"""
TEKO State+IMU - Continue from S28 checkpoint
Increased entropy to break plateau
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
    "max_steps": 150_000_000,       # 100M steps
    "max_hours": 12,                 # 12h max
    
    "learning_rate": 5e-5,           # Lower LR for fine-tuning
    "entropy_coef": 0.02,            # INCREASED from 0.005 to force exploration
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "epochs": 5,
    "batch_size": 2048,
    
    "num_envs": 120,
    "rollout_len": 128,
    
    "advance_threshold": 0.75,       # MANTIDO 75%!
    "min_steps_before_advance": 200_000,
    "max_stage": 32,
    
    "log_interval": 50_000,
    "save_interval": 1_000_000,
    
    # Checkpoint to load
    "checkpoint": "/home/schux00/checkpoints/state_imu_debug_FINAL_S28.pt",
    "start_stage": 28,
}


class MLPPolicy(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
    
    def __init__(self, state_dim=10, action_dim=2, hidden=(256, 256, 128)):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(True)])
            in_dim = h
        self.features = nn.Sequential(*layers)
        
        self.actor = nn.Sequential(
            nn.Linear(hidden[-1], 64), nn.ReLU(True), nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        
        self.critic = nn.Sequential(
            nn.Linear(hidden[-1], 64), nn.ReLU(True), nn.Linear(64, 1)
        )
    
    def _std(self):
        return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
    
    def act(self, state, deterministic=False):
        feat = self.features(state)
        mean = self.actor(feat)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        
        u = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        value = self.critic(feat).squeeze(-1)
        
        return action, log_prob, value
    
    def evaluate(self, state, actions):
        feat = self.features(state)
        mean = self.actor(feat)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        
        u = torch.clamp(actions, -0.999, 0.999)
        u = 0.5 * (torch.log1p(u) - torch.log1p(-u))
        
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - actions.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(feat).squeeze(-1)
        
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


def ppo_update(policy, optimizer, states, actions, old_logp, advantages, returns, cfg):
    device = next(policy.parameters()).device
    T, N = states.shape[:2]
    total = T * N
    
    states_flat = states.view(total, -1)
    actions_flat = actions.view(total, -1)
    old_logp_flat = old_logp.view(total)
    adv_flat = (advantages.view(total) - advantages.mean()) / (advantages.std() + 1e-8)
    ret_flat = returns.view(total)
    
    metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
    n_updates = 0
    
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(total, device=device)
        for start in range(0, total, cfg["batch_size"]):
            mb = idx[start:start + cfg["batch_size"]]
            
            logp, val, ent = policy.evaluate(states_flat[mb], actions_flat[mb])
            
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
            
            metrics["policy_loss"] += p_loss.item()
            metrics["value_loss"] += v_loss.item()
            metrics["entropy"] += ent.mean().item()
            n_updates += 1
    
    return {k: v / max(n_updates, 1) for k, v in metrics.items()}


def train(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    
    app = AppLauncher(args)
    sim = app.app
    
    sys.path.insert(0, "/workspace/teko/source/teko")
    from teko.tasks.direct.teko.teko_env_state_imu import TekoEnvStateIMU
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
    
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = CONFIG["num_envs"]
    cfg.enable_curriculum = True
    
    env = TekoEnvStateIMU(cfg=cfg)
    
    # Load checkpoint
    policy = MLPPolicy(state_dim=10, action_dim=2).to(device)
    
    print(f"[LOAD] Loading checkpoint: {CONFIG['checkpoint']}")
    ckpt = torch.load(CONFIG["checkpoint"], map_location=device)
    policy.load_state_dict(ckpt["policy"])
    print(f"[LOAD] Loaded from Stage {ckpt.get('stage', '?')}, Step {ckpt.get('step', '?')}")
    
    # Reset log_std to allow more exploration
    with torch.no_grad():
        policy.log_std.fill_(-0.3)  # Higher than trained value for more exploration
    print(f"[INFO] Reset log_std to -0.3 for more exploration")
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=CONFIG["learning_rate"])
    
    num_envs = CONFIG["num_envs"]
    rollout_len = CONFIG["rollout_len"]
    
    states_buf = torch.zeros((rollout_len, num_envs, 10), device=device)
    actions_buf = torch.zeros((rollout_len, num_envs, 2), device=device)
    rewards_buf = torch.zeros((rollout_len, num_envs), device=device)
    values_buf = torch.zeros((rollout_len, num_envs), device=device)
    logprobs_buf = torch.zeros((rollout_len, num_envs), device=device)
    dones_buf = torch.zeros((rollout_len, num_envs), device=device)
    
    ep_rewards = deque(maxlen=300)
    stage_successes = deque(maxlen=300)
    cur_reward = torch.zeros(num_envs, device=device)
    
    # Start from checkpoint stage
    current_stage = CONFIG["start_stage"]
    max_stage_reached = current_stage
    last_advance_step = 0
    env.set_curriculum_level(current_stage)
    
    obs_dict, _ = env.reset()
    step = 0
    t0 = time.time()
    next_log = CONFIG["log_interval"]
    next_save = CONFIG["save_interval"]
    
    print("=" * 70)
    print("TEKO State+IMU CONTINUE - From S28 to S32")
    print("=" * 70)
    print(f"Host: {socket.gethostname()}")
    print(f"Start Stage: {current_stage} | Target: S32 (180°)")
    print(f"Entropy Coef: {CONFIG['entropy_coef']} (INCREASED)")
    print(f"Learning Rate: {CONFIG['learning_rate']} (reduced for fine-tuning)")
    print(f"Threshold: {CONFIG['advance_threshold']} (UNCHANGED)")
    print("=" * 70)
    
    try:
        while step < CONFIG["max_steps"]:
            elapsed_h = (time.time() - t0) / 3600
            if elapsed_h > CONFIG["max_hours"]:
                print(f"[TIME] Reached {CONFIG['max_hours']}h limit")
                break
            
            for t in range(rollout_len):
                state = obs_dict["policy"].to(device)
                
                with torch.no_grad():
                    action, logp, value = policy.act(state)
                
                states_buf[t] = state
                actions_buf[t] = action
                logprobs_buf[t] = logp
                values_buf[t] = value
                
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
                last_state = obs_dict["policy"].to(device)
                _, _, last_value = policy.act(last_state)
            
            advantages, returns = compute_gae(
                rewards_buf, values_buf, dones_buf,
                CONFIG["gamma"], CONFIG["gae_lambda"], last_value
            )
            
            metrics = ppo_update(
                policy, optimizer,
                states_buf, actions_buf, logprobs_buf,
                advantages, returns, CONFIG
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
                    f"R: {mean_r:.1f} | Ent: {metrics['entropy']:.3f} | "
                    f"MaxS: {max_stage_reached} | {elapsed_h:.1f}h"
                )
                next_log += CONFIG["log_interval"]
            
            if step >= next_save:
                ckpt_path = f"/home/schux00/checkpoints/state_imu_cont_S{current_stage}_{step//1000}k.pt"
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
            
            # Victory!
            if current_stage >= CONFIG["max_stage"] and ssr >= 0.70:
                print("=" * 70)
                print(f"[SUCCESS] Reached Stage {CONFIG['max_stage']} (180°) with SSR={ssr:.1%}!")
                print(f"Total steps: {step:,} | Time: {elapsed_h:.1f}h")
                print("=" * 70)
                break
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    
    finally:
        final_path = f"/home/schux00/checkpoints/state_imu_cont_FINAL_S{max_stage_reached}.pt"
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
    args.enable_cameras = False
    
    train(args)


if __name__ == "__main__":
    main()
