#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO PPO OPTUNA HYPERPARAMETER OPTIMIZATION
============================================

Distributed hyperparameter optimization using Optuna with SQLite storage.
Supports multiple workers running in parallel on SLURM cluster.

Features:
- Distributed optimization with shared SQLite database
- Median pruning for early stopping of bad trials
- Optimizes: entropy_coef, gae_lambda, clip_ratio, epochs
- Reports stage success rate (SSR) as optimization metric

Usage:
    # First, create the study (run once):
    python train_optuna_ppo.py --create-study
    
    # Then, launch workers (can run multiple in parallel):
    sbatch run_optuna_worker.sh

Author: Alexandre Schleier Neves da Silva
Date: December 2024
"""

import argparse
import os
import sys
import math
import time
from datetime import datetime
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

from isaaclab.app import AppLauncher


# =============================================================================
# OPTUNA CONFIGURATION
# =============================================================================

OPTUNA_CONFIG = {
    # Study settings
    "study_name": "teko_ppo_nsgaii_v1",
    "storage_path": "/home/schux00/optuna/teko_nsgaii.db",
    
    # Optimization target
    "direction": "maximize",  # Maximize SSR
    "n_trials": 100,          # Total trials across all workers
    
    # Pruning settings
    "pruning_enabled": True,
    "pruning_warmup_steps": 100_000,      # Don't prune before this
    "pruning_check_interval": 50_000,     # Check for pruning every N steps
    "min_ssr_stage0": 0.30,               # Min SSR at stage 0 to continue
    "min_ssr_stage3": 0.40,               # Min SSR at stage 3 to continue
    "min_ssr_stage5": 0.50,               # Min SSR at stage 5 to continue
    
    # Training budget per trial
    "max_steps_per_trial": 5_000_000,     # 5M steps per trial
    "eval_interval": 25_000,              # Report metrics every N steps
}

# Hyperparameter search space
SEARCH_SPACE = {
    "entropy_coef": {
        "type": "float",
        "low": 0.0,
        "high": 0.01,
        "log": False,
    },
    "gae_lambda": {
        "type": "float",
        "low": 0.9,
        "high": 1.0,
        "log": False,
    },
    "clip_ratio": {
        "type": "categorical",
        "choices": [0.1, 0.2, 0.3],
    },
    "epochs": {
        "type": "int",
        "low": 3,
        "high": 30,
    },
    # Additional hyperparameters to explore
    "learning_rate": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-3,
        "log": True,
    },
    "batch_size": {
        "type": "categorical",
        "choices": [1024, 2048, 4096],
    },
}

# Fixed hyperparameters (not optimized)
FIXED_PARAMS = {
    "gamma": 0.99,
    "value_clip": 0.2,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "rollout_len": 128,
    "num_envs": 120,
}


# =============================================================================
# CNN ENCODER (SimpleCNN with LayerNorm)
# =============================================================================

class SimpleCNN(nn.Module):
    """CNN optimized for robot shape recognition at 84x84."""
    
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=6, stride=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.ln1 = nn.LayerNorm([32, 27, 27])
        self.ln2 = nn.LayerNorm([64, 13, 13])
        self.ln3 = nn.LayerNorm([128, 7, 7])
        
        self.fc = nn.Sequential(
            nn.Linear(6272, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )
        
        self.feature_dim = feature_dim
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = torch.relu(self.ln1(self.conv1(x)))
        x = torch.relu(self.ln2(self.conv2(x)))
        x = torch.relu(self.ln3(self.conv3(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


# =============================================================================
# ASYMMETRIC ACTOR-CRITIC POLICY
# =============================================================================

class AsymmetricPolicy(nn.Module):
    """Asymmetric actor-critic with vision encoder."""
    
    LOG_STD_MIN = -2.0
    LOG_STD_MAX = 0.5
    
    def __init__(self, vision_channels=4, privileged_dim=7, action_dim=2, hidden_dim=256):
        super().__init__()
        
        self.vision_encoder = SimpleCNN(in_channels=vision_channels, feature_dim=hidden_dim)
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )
        
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        
        self.state_encoder = nn.Sequential(
            nn.Linear(privileged_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        
        self._init_heads()
    
    def _init_heads(self):
        for module in [self.actor_head, self.state_encoder, self.critic_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-2].weight, gain=0.01)
    
    def _get_std(self):
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return torch.exp(log_std)
    
    def forward_actor(self, vision):
        features = self.vision_encoder(vision)
        mean = self.actor_head(features)
        std = self._get_std().unsqueeze(0).expand(mean.shape[0], -1)
        return mean, std
    
    def forward_critic(self, vision, privileged):
        vision_features = self.vision_encoder(vision)
        state_features = self.state_encoder(privileged)
        fused = torch.cat([vision_features, state_features], dim=-1)
        return self.critic_head(fused).squeeze(-1)
    
    def act(self, obs, deterministic=False):
        vision = obs["rgb"]
        privileged = obs.get("privileged", None)
        
        mean, std = self.forward_actor(vision)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        
        if privileged is not None:
            value = self.forward_critic(vision, privileged)
        else:
            value = torch.zeros(mean.shape[0], device=mean.device)
        
        return action, log_prob, value
    
    def evaluate(self, obs, actions):
        vision = obs["rgb"]
        privileged = obs.get("privileged", None)
        
        mean, std = self.forward_actor(vision)
        dist = torch.distributions.Normal(mean, std)
        
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        if privileged is not None:
            value = self.forward_critic(vision, privileged)
        else:
            value = torch.zeros(mean.shape[0], device=mean.device)
        
        return log_prob, value, entropy


# =============================================================================
# PPO FUNCTIONS
# =============================================================================

def compute_gae(rewards, values, dones, gamma, lam):
    """Compute Generalized Advantage Estimation."""
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


def ppo_update(policy, optimizer, obs_rgb, obs_privileged, actions, old_log_probs,
               advantages, returns, epochs, batch_size, clip_ratio, value_clip,
               entropy_coef, value_coef, max_grad_norm):
    """PPO policy update."""
    device = next(policy.parameters()).device
    T, N = obs_rgb.shape[:2]
    total = T * N
    
    obs_rgb_flat = obs_rgb.view(total, 4, 84, 84)
    obs_priv_flat = obs_privileged.view(total, -1) if obs_privileged is not None else None
    actions_flat = actions.view(total, 2)
    old_logp_flat = old_log_probs.view(-1)
    adv_flat = advantages.view(-1)
    ret_flat = returns.view(-1)
    
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    
    p_losses, v_losses, entropies = [], [], []
    
    for _ in range(epochs):
        indices = torch.randperm(total) 
        
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            mb_idx = indices[start:end]
            
            mb_obs = {"rgb": obs_rgb_flat[mb_idx].to(device)}
            if obs_priv_flat is not None:
                mb_obs["privileged"] = obs_priv_flat[mb_idx].to(device)
            
            mb_actions = actions_flat[mb_idx].to(device)
            mb_old_logp = old_logp_flat[mb_idx].to(device)
            mb_adv = adv_flat[mb_idx].to(device)
            mb_ret = ret_flat[mb_idx].to(device)
            
            log_prob, value, entropy = policy.evaluate(mb_obs, mb_actions)
            
            ratio = torch.exp(log_prob - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
            p_loss = -torch.min(surr1, surr2).mean()
            
            if value_clip:
                value_clipped = torch.clamp(value, mb_ret - value_clip, mb_ret + value_clip)
                v_loss = 0.5 * torch.max((value - mb_ret) ** 2, (value_clipped - mb_ret) ** 2).mean()
            else:
                v_loss = 0.5 * F.mse_loss(value, mb_ret)
            
            loss = p_loss + value_coef * v_loss - entropy_coef * entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            
            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(entropy.mean().item())
    
    return {
        "policy_loss": np.mean(p_losses),
        "value_loss": np.mean(v_losses),
        "entropy": np.mean(entropies),
    }


# =============================================================================
# OPTUNA OBJECTIVE FUNCTION
# =============================================================================

def objective_with_env(trial: optuna.Trial, env) -> float:
    """
    Optuna objective with shared environment - avoids Isaac Sim restart issues.
    """
    
    # Sample hyperparameters
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.01)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    clip_ratio = trial.suggest_categorical("clip_ratio", [0.1, 0.2, 0.3])
    epochs = trial.suggest_int("epochs", 3, 30)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    
    print(f"\n{'='*70}")
    print(f"TRIAL {trial.number}")
    print(f"{'='*70}")
    print(f"entropy_coef: {entropy_coef:.6f}")
    print(f"gae_lambda:   {gae_lambda:.4f}")
    print(f"clip_ratio:   {clip_ratio}")
    print(f"epochs:       {epochs}")
    print(f"learning_rate: {learning_rate:.2e}")
    print(f"batch_size:   {batch_size}")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda:0")
    
    # Reset environment to stage 0
    env.set_curriculum_level(0)
    
    # Create fresh policy for this trial
    policy = AsymmetricPolicy(
        vision_channels=4,
        privileged_dim=7,
        action_dim=2,
        hidden_dim=256,
    ).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Training loop
    step = 0
    max_steps = OPTUNA_CONFIG["max_steps_per_trial"]
    eval_interval = OPTUNA_CONFIG["eval_interval"]
    rollout_len = FIXED_PARAMS["rollout_len"]
    num_envs = FIXED_PARAMS["num_envs"]
    
    ep_rewards = deque(maxlen=100)
    stage_successes = deque(maxlen=200)
    
    cur_reward = torch.zeros(num_envs, device=device)
    cur_length = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    obs_dict, _ = env.reset()
    best_ssr = 0.0
    current_stage = 0
    next_eval = eval_interval
    
    try:
        while step < max_steps:
            # Collect rollout
            obs_rgb_buf, obs_priv_buf = [], []
            act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], []
            
            for _ in range(rollout_len):
                obs = {"rgb": obs_dict["rgb"].to(device)}
                if "privileged" in obs_dict:
                    obs["privileged"] = obs_dict["privileged"].to(device)
                
                with torch.no_grad():
                    action, log_prob, value = policy.act(obs)
                
                obs_dict, reward, term, trunc, _ = env.step(action)
                done = term | trunc
                
                cur_reward += reward
                cur_length += 1
                
                for i in range(num_envs):
                    if done[i]:
                        ep_rewards.append(cur_reward[i].item())
                        success = reward[i].item() > 50.0
                        stage_successes.append(1.0 if success else 0.0)
                        cur_reward[i] = 0.0
                        cur_length[i] = 0
                
                obs_rgb_buf.append(obs["rgb"].cpu())
                if "privileged" in obs:
                    obs_priv_buf.append(obs["privileged"].cpu())
                act_buf.append(action.cpu())
                rew_buf.append(reward.cpu())
                val_buf.append(value.cpu())
                logp_buf.append(log_prob.cpu())
                done_buf.append(done.float().cpu())
                
                step += num_envs
            
            # Stack buffers
            obs_rgb = torch.stack(obs_rgb_buf)
            obs_priv = torch.stack(obs_priv_buf) if obs_priv_buf else None
            actions = torch.stack(act_buf)
            rewards = torch.stack(rew_buf)
            values = torch.stack(val_buf)
            log_probs = torch.stack(logp_buf)
            dones = torch.stack(done_buf)
            
            # Compute GAE
            with torch.no_grad():
                adv, ret = compute_gae(
                    rewards, values, dones,
                    FIXED_PARAMS["gamma"], gae_lambda
                )
            
            # PPO update
            ppo_update(
                policy, optimizer,
                obs_rgb, obs_priv,
                actions, log_probs,
                adv, ret,
                epochs=epochs,
                batch_size=batch_size,
                clip_ratio=clip_ratio,
                value_clip=FIXED_PARAMS["value_clip"],
                entropy_coef=entropy_coef,
                value_coef=FIXED_PARAMS["value_coef"],
                max_grad_norm=FIXED_PARAMS["max_grad_norm"],
            )
            
            # Compute metrics
            ssr = np.mean(stage_successes) if stage_successes else 0.0
            mean_reward = np.mean(ep_rewards) if ep_rewards else 0.0
            current_stage = env.curriculum_level
            
            if ssr > best_ssr:
                best_ssr = ssr
            
            # Report to Optuna (fixed interval)
            if step >= next_eval:
                trial.report(ssr, step)
                next_eval += eval_interval
                
                print(f"[{step:,}] Stage {current_stage} | SSR: {ssr:.1%} | "
                      f"Reward: {mean_reward:.1f} | Best: {best_ssr:.1%}")
                
                # Check for pruning
                if OPTUNA_CONFIG["pruning_enabled"] and trial.should_prune():
                    print(f"Trial {trial.number} pruned at step {step:,}")
                    raise optuna.TrialPruned()
                
                # Manual pruning based on stage progress
                if step >= OPTUNA_CONFIG["pruning_warmup_steps"]:
                    if current_stage == 0 and ssr < OPTUNA_CONFIG["min_ssr_stage0"]:
                        print(f"Trial {trial.number} pruned: SSR {ssr:.1%} < {OPTUNA_CONFIG['min_ssr_stage0']:.0%} at S0")
                        raise optuna.TrialPruned()
                    
                    if step >= 200_000:
                        if current_stage <= 3 and ssr < OPTUNA_CONFIG["min_ssr_stage3"]:
                            print(f"Trial {trial.number} pruned: stuck at S{current_stage}")
                            raise optuna.TrialPruned()
                    
                    if step >= 500_000:
                        if current_stage <= 5 and ssr < OPTUNA_CONFIG["min_ssr_stage5"]:
                            print(f"Trial {trial.number} pruned: stuck at S{current_stage}")
                            raise optuna.TrialPruned()
            
            # Curriculum advancement
            if len(stage_successes) >= 50 and ssr >= 0.70:
                if current_stage < 27:
                    env.set_curriculum_level(current_stage + 1)
                    obs_dict, _ = env.reset()
                    cur_reward.zero_()
                    cur_length.zero_()
                    stage_successes.clear()
                    print(f"➡️ Advanced to S{current_stage + 1}")
        
        # Training complete
        print(f"\nTRIAL {trial.number} COMPLETE")
        print(f"Final SSR: {best_ssr:.1%}")
        print(f"Final Stage: S{current_stage}")
        
    except optuna.TrialPruned:
        # Reset env to clean state for next trial
        env.set_curriculum_level(0)
        env.reset()
        raise
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        env.set_curriculum_level(0)
        env.reset()
        raise optuna.TrialPruned()
    
    # Reset for next trial
    env.set_curriculum_level(0)
    
    # Return combined metric
    score = best_ssr + (current_stage * 0.01)
    return score


def objective(trial: optuna.Trial, args) -> float:
    """
    Optuna objective function - trains one trial and returns final SSR.
    """
    
    # =================================================================
    # SAMPLE HYPERPARAMETERS
    # =================================================================
    
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.01)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    clip_ratio = trial.suggest_categorical("clip_ratio", [0.1, 0.2, 0.3])
    epochs = trial.suggest_int("epochs", 3, 30)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    
    print(f"\n{'='*70}")
    print(f"TRIAL {trial.number}")
    print(f"{'='*70}")
    print(f"entropy_coef: {entropy_coef:.6f}")
    print(f"gae_lambda:   {gae_lambda:.4f}")
    print(f"clip_ratio:   {clip_ratio}")
    print(f"epochs:       {epochs}")
    print(f"learning_rate: {learning_rate:.2e}")
    print(f"batch_size:   {batch_size}")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda:0")
    
    # =================================================================
    # CREATE ENVIRONMENT
    # =================================================================
    
    sys.path.insert(0, "/workspace/teko/source/teko")
    
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
    
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = FIXED_PARAMS["num_envs"]
    cfg.enable_curriculum = True
    cfg.asymmetric_critic = True
    
    env = TekoEnv(cfg=cfg)
    
    # =================================================================
    # CREATE POLICY
    # =================================================================
    
    policy = AsymmetricPolicy(
        vision_channels=4,
        privileged_dim=7,
        action_dim=2,
        hidden_dim=256,
    ).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # =================================================================
    # TRAINING LOOP
    # =================================================================
    
    step = 0
    max_steps = OPTUNA_CONFIG["max_steps_per_trial"]
    eval_interval = OPTUNA_CONFIG["eval_interval"]
    rollout_len = FIXED_PARAMS["rollout_len"]
    num_envs = FIXED_PARAMS["num_envs"]
    
    ep_rewards = deque(maxlen=100)
    stage_successes = deque(maxlen=200)
    
    cur_reward = torch.zeros(num_envs, device=device)
    cur_length = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    obs_dict, _ = env.reset()
    best_ssr = 0.0
    current_stage = 0
    
    try:
        while step < max_steps:
            # Collect rollout
            obs_rgb_buf, obs_priv_buf = [], []
            act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], []
            
            for _ in range(rollout_len):
                obs = {"rgb": obs_dict["rgb"].to(device)}
                if "privileged" in obs_dict:
                    obs["privileged"] = obs_dict["privileged"].to(device)
                
                with torch.no_grad():
                    action, log_prob, value = policy.act(obs)
                
                obs_dict, reward, term, trunc, _ = env.step(action)
                done = term | trunc
                
                cur_reward += reward
                cur_length += 1
                
                for i in range(num_envs):
                    if done[i]:
                        ep_rewards.append(cur_reward[i].item())
                        success = reward[i].item() > 50.0
                        stage_successes.append(1.0 if success else 0.0)
                        cur_reward[i] = 0.0
                        cur_length[i] = 0
                
                obs_rgb_buf.append(obs["rgb"].cpu())
                if "privileged" in obs:
                    obs_priv_buf.append(obs["privileged"].cpu())
                act_buf.append(action.cpu())
                rew_buf.append(reward.cpu())
                val_buf.append(value.cpu())
                logp_buf.append(log_prob.cpu())
                done_buf.append(done.float().cpu())
                
                step += num_envs
            
            # Stack buffers
            obs_rgb = torch.stack(obs_rgb_buf)
            obs_priv = torch.stack(obs_priv_buf) if obs_priv_buf else None
            actions = torch.stack(act_buf)
            rewards = torch.stack(rew_buf)
            values = torch.stack(val_buf)
            log_probs = torch.stack(logp_buf)
            dones = torch.stack(done_buf)
            
            # Compute GAE
            with torch.no_grad():
                adv, ret = compute_gae(
                    rewards, values, dones,
                    FIXED_PARAMS["gamma"], gae_lambda
                )
            
            # PPO update
            ppo_update(
                policy, optimizer,
                obs_rgb, obs_priv,
                actions, log_probs,
                adv, ret,
                epochs=epochs,
                batch_size=batch_size,
                clip_ratio=clip_ratio,
                value_clip=FIXED_PARAMS["value_clip"],
                entropy_coef=entropy_coef,
                value_coef=FIXED_PARAMS["value_coef"],
                max_grad_norm=FIXED_PARAMS["max_grad_norm"],
            )
            
            # Compute metrics
            ssr = np.mean(stage_successes) if stage_successes else 0.0
            mean_reward = np.mean(ep_rewards) if ep_rewards else 0.0
            current_stage = env.curriculum_level
            
            if ssr > best_ssr:
                best_ssr = ssr
            
            # Report to Optuna (for pruning)
            if step % eval_interval < num_envs * rollout_len:
                trial.report(ssr, step)
                
                print(f"[{step:,}] Stage {current_stage} | SSR: {ssr:.1%} | "
                      f"Reward: {mean_reward:.1f} | Best: {best_ssr:.1%}")
                
                # Check for pruning
                if OPTUNA_CONFIG["pruning_enabled"] and trial.should_prune():
                    print(f"Trial {trial.number} pruned at step {step:,}")
                    raise optuna.TrialPruned()
                
                # Manual pruning based on stage progress
                if step >= OPTUNA_CONFIG["pruning_warmup_steps"]:
                    if current_stage == 0 and ssr < OPTUNA_CONFIG["min_ssr_stage0"]:
                        print(f"Trial {trial.number} pruned: SSR {ssr:.1%} < {OPTUNA_CONFIG['min_ssr_stage0']:.0%} at S0")
                        raise optuna.TrialPruned()
                    
                    if step >= 200_000:
                        if current_stage <= 3 and ssr < OPTUNA_CONFIG["min_ssr_stage3"]:
                            print(f"Trial {trial.number} pruned: stuck at S{current_stage}")
                            raise optuna.TrialPruned()
                    
                    if step >= 500_000:
                        if current_stage <= 5 and ssr < OPTUNA_CONFIG["min_ssr_stage5"]:
                            print(f"Trial {trial.number} pruned: stuck at S{current_stage}")
                            raise optuna.TrialPruned()
            
            # Curriculum advancement
            if len(stage_successes) >= 50 and ssr >= 0.70:
                if current_stage < 27:
                    env.set_curriculum_level(current_stage + 1)
                    obs_dict, _ = env.reset()
                    cur_reward.zero_()
                    cur_length.zero_()
                    stage_successes.clear()
                    print(f"➡️ Advanced to S{current_stage + 1}")
        
        # Training complete
        final_ssr = best_ssr
        final_stage = current_stage
        
        print(f"\n{'='*70}")
        print(f"TRIAL {trial.number} COMPLETE")
        print(f"Final SSR: {final_ssr:.1%}")
        print(f"Final Stage: S{final_stage}")
        print(f"{'='*70}\n")
        
    except optuna.TrialPruned:
        raise
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.TrialPruned()
    
    finally:
        env.close()
    
    # Return combined metric: SSR + stage bonus
    # This encourages reaching higher stages
    score = best_ssr + (current_stage * 0.01)
    return score


# =============================================================================
# MAIN
# =============================================================================

def create_study():
    """Create Optuna study with SQLite storage."""
    storage_path = OPTUNA_CONFIG["storage_path"]
    storage_dir = os.path.dirname(storage_path)
    os.makedirs(storage_dir, exist_ok=True)
    
    storage = f"sqlite:///{storage_path}"
    
    study = optuna.create_study(
        study_name=OPTUNA_CONFIG["study_name"],
        storage=storage,
        direction=OPTUNA_CONFIG["direction"],
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=100_000,
            interval_steps=25_000,
        ),
    )
    
    print(f"✅ Study created/loaded: {OPTUNA_CONFIG['study_name']}")
    print(f"   Storage: {storage_path}")
    print(f"   Trials so far: {len(study.trials)}")
    
    return study


def run_worker(args):
    """Run Optuna worker with shared environment."""
    # Initialize Isaac Sim
    app = AppLauncher(args)
    sim = app.app
    
    # Create environment ONCE
    import sys
    sys.path.insert(0, "/workspace/teko/source/teko")
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
    
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = FIXED_PARAMS["num_envs"]
    cfg.enable_curriculum = True
    cfg.asymmetric_critic = True
    
    print("Creating shared environment...")
    env = TekoEnv(cfg=cfg)
    print(f"✅ Shared environment created with {FIXED_PARAMS['num_envs']} envs")
    
    try:
        storage = f"sqlite:///{OPTUNA_CONFIG['storage_path']}"
        
        study = optuna.load_study(
            study_name=OPTUNA_CONFIG["study_name"],
            storage=storage,
        )
        
        print(f"🚀 Worker started for study: {OPTUNA_CONFIG['study_name']}")
        print(f"   Completed trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
        print(f"   Running trials: {len([t for t in study.trials if t.state == TrialState.RUNNING])}")
        
        # Run optimization with shared env
        study.optimize(
            lambda trial: objective_with_env(trial, env),
            n_trials=OPTUNA_CONFIG["n_trials"],
            timeout=None,
            catch=(Exception,),
        )
        
        # Print best results
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.4f}")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        
    finally:
        env.close()
        sim.close()


def main():
    parser = argparse.ArgumentParser(description="TEKO Optuna HPO")
    parser.add_argument("--create-study", action="store_true", help="Create study and exit")
    parser.add_argument("--num-trials", type=int, default=None, help="Override n_trials")
    
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    
    if args.create_study:
        # Just create study and exit
        create_study()
        return
    
    if args.num_trials:
        OPTUNA_CONFIG["n_trials"] = args.num_trials
    
    run_worker(args)


if __name__ == "__main__":
    main()