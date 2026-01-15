#!/usr/bin/env python3
"""
TEKO State Full v4 - Fair Comparison with Vision
=================================================
Same hyperparameters as Vision Optimal (Trial 80)
Only difference: uses state vector instead of camera.

Input: 10D [dx, dy, dz, yaw_err, vx, vy, vz, wx, wy, wz]
Output: 2D [v_cmd, w_cmd]

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

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# =============================================================================
# CONFIGURATION - IDENTICAL to Vision Optimal (Trial 80)
# =============================================================================
CONFIG = {
    "max_steps": 200_000_000,
    "max_hours": 168,  # 7 days
    
    # IDENTICAL to Vision Optimal (Trial 80)
    "learning_rate": 0.00016,
    "entropy_coef": 0.0062,
    "gae_lambda": 0.94,
    "gamma": 0.99,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "epochs": 5,
    "batch_size": 1024,
    
    # Same env settings
    "num_envs": 120,
    "rollout_len": 128,
    
    "advance_threshold": 0.75,
    "min_steps_before_advance": 200_000,
    "max_stage": 41,
    
    "log_interval": 50_000,
    "save_interval": 2_000_000,
}


def atanh(x):
    x = torch.clamp(x, -0.999, 0.999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class StatePolicy(nn.Module):
    """
    MLP policy with state information.
    Same LOG_STD bounds as Vision for fair comparison.
    
    Input: 10D state [dx, dy, dz, yaw_err, vx, vy, vz, wx, wy, wz]
    Output: 2D actions [v_cmd, w_cmd]
    """
    # SAME as Vision Optimal
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
    
    def __init__(self, state_dim=10, action_dim=2, hidden_dim=256):
        super().__init__()
        
        # Encoder (similar capacity to Vision encoder output)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, hidden_dim),
            nn.ReLU(True),
        )
        
        # Actor head (same structure as Vision)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        
        # Critic head with privileged info (asymmetric, same as Vision)
        priv_dim = 7
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim + priv_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init for actor output
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
    
    def _std(self):
        return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
    
    def act(self, state, privileged=None, deterministic=False):
        features = self.encoder(state)
        mean = self.actor_head(features)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        
        u = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        
        # Asymmetric critic (same as Vision)
        if privileged is not None:
            critic_in = torch.cat([features, privileged], dim=-1)
        else:
            critic_in = torch.cat([features, torch.zeros(features.shape[0], 7, device=features.device)], dim=-1)
        value = self.critic_head(critic_in).squeeze(-1)
        
        return action, log_prob, value
    
    def evaluate(self, state, actions, privileged=None):
        features = self.encoder(state)
        mean = self.actor_head(features)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        
        u = atanh(actions)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - actions.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        # Asymmetric critic
        if privileged is not None:
            critic_in = torch.cat([features, privileged], dim=-1)
        else:
            critic_in = torch.cat([features, torch.zeros(features.shape[0], 7, device=features.device)], dim=-1)
        value = self.critic_head(critic_in).squeeze(-1)
        
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


def ppo_update(policy, optimizer, states, actions, old_logp, advantages, returns, privileged, cfg):
    device = next(policy.parameters()).device
    T, N = states.shape[:2]
    total = T * N
    
    state_flat = states.view(total, -1)
    act_flat = actions.view(total, -1)
    old_logp_flat = old_logp.view(total)
    adv_flat = (advantages.view(total) - advantages.mean()) / (advantages.std() + 1e-8)
    ret_flat = returns.view(total)
    priv_flat = privileged.view(total, -1) if privileged is not None else None
    
    metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "grad_norm": 0}
    n_updates = 0
    
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(total, device=device)
        for start in range(0, total, cfg["batch_size"]):
            mb = idx[start:start + cfg["batch_size"]]
            priv_mb = priv_flat[mb] if priv_flat is not None else None
            
            logp, val, ent = policy.evaluate(state_flat[mb], act_flat[mb], priv_mb)
            
            ratio = torch.exp(logp - old_logp_flat[mb])
            surr1 = ratio * adv_flat[mb]
            surr2 = torch.clamp(ratio, 1 - cfg["clip_ratio"], 1 + cfg["clip_ratio"]) * adv_flat[mb]
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * F.mse_loss(val, ret_flat[mb])
            
            loss = p_loss + cfg["value_coef"] * v_loss - cfg["entropy_coef"] * ent.mean()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"]).item()
            optimizer.step()
            
            metrics["policy_loss"] += p_loss.item()
            metrics["value_loss"] += v_loss.item()
            metrics["entropy"] += ent.mean().item()
            metrics["grad_norm"] += grad_norm
            n_updates += 1
    
    return {k: v / max(n_updates, 1) for k, v in metrics.items()}


def train(args):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
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
    cfg.asymmetric_critic = True  # Same as Vision
    
    env = TekoEnvStateIMU(cfg=cfg)
    
    policy = StatePolicy(state_dim=10, action_dim=2).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=CONFIG["learning_rate"])
    
    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/home/schux00/tensorboard/state_full_{timestamp}"
    csv_path = f"/home/schux00/logs/state_full_{timestamp}.csv"
    
    writer = None
    if HAS_TENSORBOARD:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'stage', 'ssr', 'reward', 'entropy', 'policy_loss', 'value_loss', 'hours'])
    
    num_envs = CONFIG["num_envs"]
    rollout_len = CONFIG["rollout_len"]
    
    # Buffers
    states_buf = torch.zeros((rollout_len, num_envs, 10), device=device)
    actions_buf = torch.zeros((rollout_len, num_envs, 2), device=device)
    rewards_buf = torch.zeros((rollout_len, num_envs), device=device)
    values_buf = torch.zeros((rollout_len, num_envs), device=device)
    logprobs_buf = torch.zeros((rollout_len, num_envs), device=device)
    dones_buf = torch.zeros((rollout_len, num_envs), device=device)
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
    
    has_privileged = "privileged" in obs_dict and obs_dict["privileged"] is not None
    
    print("=" * 70)
    print("TEKO State Full v4 - Fair Comparison with Vision")
    print("=" * 70)
    print(f"Host: {socket.gethostname()}")
    print(f"Envs: {num_envs} | Max Steps: {CONFIG['max_steps']:,}")
    print(f"IDENTICAL HPs to Vision Optimal (Trial 80):")
    print(f"  LR: {CONFIG['learning_rate']} | Entropy: {CONFIG['entropy_coef']}")
    print(f"  GAE: {CONFIG['gae_lambda']} | Batch: {CONFIG['batch_size']}")
    print(f"  LOG_STD_MIN: -2.0 | Asymmetric Critic: Yes")
    print(f"TensorBoard: {log_dir}")
    print(f"CSV: {csv_path}")
    print("=" * 70)
    
    try:
        while step < CONFIG["max_steps"]:
            elapsed_h = (time.time() - t0) / 3600
            if elapsed_h > CONFIG["max_hours"]:
                print(f"[TIME] Reached {CONFIG['max_hours']}h limit")
                break
            
            for t in range(rollout_len):
                state = obs_dict["policy"].to(device)
                priv = obs_dict.get("privileged")
                if priv is not None:
                    priv = priv.to(device)
                
                with torch.no_grad():
                    action, logp, value = policy.act(state, priv)
                
                states_buf[t] = state
                actions_buf[t] = action
                logprobs_buf[t] = logp
                values_buf[t] = value
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
                last_state = obs_dict["policy"].to(device)
                last_priv = obs_dict.get("privileged")
                if last_priv is not None:
                    last_priv = last_priv.to(device)
                _, _, last_value = policy.act(last_state, last_priv)
            
            advantages, returns = compute_gae(
                rewards_buf, values_buf, dones_buf,
                CONFIG["gamma"], CONFIG["gae_lambda"], last_value
            )
            
            metrics = ppo_update(
                policy, optimizer, states_buf, actions_buf, logprobs_buf,
                advantages, returns,
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
                      f"Ent: {metrics['entropy']:.3f} | MaxS: {max_stage_reached} | {elapsed_h:.1f}h")
                
                if writer:
                    writer.add_scalar("train/ssr", ssr, step)
                    writer.add_scalar("train/reward", mean_r, step)
                    writer.add_scalar("train/entropy", metrics["entropy"], step)
                    writer.add_scalar("train/policy_loss", metrics["policy_loss"], step)
                    writer.add_scalar("train/value_loss", metrics["value_loss"], step)
                    writer.add_scalar("train/grad_norm", metrics["grad_norm"], step)
                    writer.add_scalar("curriculum/stage", current_stage, step)
                    writer.add_scalar("curriculum/max_stage", max_stage_reached, step)
                
                csv_writer.writerow([step, current_stage, f"{ssr:.4f}", f"{mean_r:.2f}",
                                    f"{metrics['entropy']:.4f}", f"{metrics['policy_loss']:.4f}",
                                    f"{metrics['value_loss']:.4f}", f"{elapsed_h:.2f}"])
                csv_file.flush()
                
                next_log += CONFIG["log_interval"]
            
            # Save checkpoints
            if step >= next_save:
                ckpt_path = f"/home/schux00/checkpoints/state_full_S{current_stage}_{step//1000}k.pt"
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
            
            # Success condition
            if current_stage >= CONFIG["max_stage"] and ssr >= 0.70:
                print("=" * 70)
                print(f"[SUCCESS] State Full reached S{CONFIG['max_stage']} with SSR={ssr:.1%}!")
                print("=" * 70)
                break
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        final_path = f"/home/schux00/checkpoints/state_full_FINAL_S{max_stage_reached}.pt"
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
    args.enable_cameras = False  # No cameras needed for state-based
    train(args)


if __name__ == "__main__":
    main()