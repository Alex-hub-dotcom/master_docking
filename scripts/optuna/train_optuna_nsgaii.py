#!/usr/bin/env python3
"""
TEKO PPO NSGA-II HYPERPARAMETER OPTIMIZATION
=============================================

Multi-objective optimization using NSGA-II genetic algorithm.
Objectives: Maximize SSR + Maximize Stage Reached

Author: Alexandre Schleier Neves da Silva
Date: December 2024
"""

import argparse
import os
import sys
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("ERROR: optuna not installed")
    sys.exit(1)

from isaaclab.app import AppLauncher

# =============================================================================
# CONFIGURATION
# =============================================================================

OPTUNA_CONFIG = {
    "study_name": "teko_nsgaii_v1",
    "storage_path": "/home/schux00/optuna/teko_nsgaii.db",
    "n_trials": 150,
    "max_steps_per_trial": 5_000_000,  # 3M para mais trials
    "eval_interval": 25_000,
    
    # Pruning
    "pruning_enabled": True,
    "pruning_warmup_steps": 100_000,
    "min_ssr_stage0": 0.30,
    "min_ssr_stage3": 0.40,
    "min_ssr_stage5": 0.50,
}

FIXED_PARAMS = {
    "gamma": 0.99,
    "value_clip": 0.2,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "rollout_len": 128,
    "num_envs": 120,
}

# =============================================================================
# NEURAL NETWORK
# =============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=6, stride=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.ln1 = nn.LayerNorm([32, 27, 27])
        self.ln2 = nn.LayerNorm([64, 13, 13])
        self.ln3 = nn.LayerNorm([128, 7, 7])
        self.fc = nn.Sequential(
            nn.Linear(6272, 512), nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim), nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = torch.relu(self.ln1(self.conv1(x)))
        x = torch.relu(self.ln2(self.conv2(x)))
        x = torch.relu(self.ln3(self.conv3(x)))
        return self.fc(torch.flatten(x, 1))


class AsymmetricPolicy(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
    
    def __init__(self, vision_channels=4, privileged_dim=7, action_dim=2, hidden_dim=256):
        super().__init__()
        self.vision_encoder = SimpleCNN(in_channels=vision_channels, feature_dim=hidden_dim)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, action_dim), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.state_encoder = nn.Sequential(
            nn.Linear(privileged_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self._init_heads()
    
    def _init_heads(self):
        for module in [self.actor_head, self.state_encoder, self.critic_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-2].weight, gain=0.01)
    
    def _get_std(self):
        return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
    
    def forward_actor(self, vision):
        features = self.vision_encoder(vision)
        mean = self.actor_head(features)
        return mean, self._get_std().unsqueeze(0).expand(mean.shape[0], -1)
    
    def forward_critic(self, vision, privileged):
        vision_features = self.vision_encoder(vision)
        state_features = self.state_encoder(privileged)
        return self.critic_head(torch.cat([vision_features, state_features], dim=-1)).squeeze(-1)
    
    def act(self, obs, deterministic=False):
        mean, std = self.forward_actor(obs["rgb"])
        if deterministic:
            action, log_prob = mean, torch.zeros(mean.shape[0], device=mean.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        value = self.forward_critic(obs["rgb"], obs["privileged"]) if "privileged" in obs else torch.zeros(mean.shape[0], device=mean.device)
        return action, log_prob, value
    
    def evaluate(self, obs, actions):
        mean, std = self.forward_actor(obs["rgb"])
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.forward_critic(obs["rgb"], obs["privileged"]) if "privileged" in obs else torch.zeros(mean.shape[0], device=mean.device)
        return log_prob, value, entropy


# =============================================================================
# PPO FUNCTIONS
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


def ppo_update(policy, optimizer, obs_rgb, obs_priv, actions, old_log_probs,
               advantages, returns, epochs, batch_size, clip_ratio,
               entropy_coef, value_coef, max_grad_norm):
    device = next(policy.parameters()).device
    T, N = obs_rgb.shape[:2]
    total = T * N
    
    obs_rgb_flat = obs_rgb.view(total, 4, 84, 84)
    obs_priv_flat = obs_priv.view(total, -1) if obs_priv is not None else None
    actions_flat = actions.view(total, 2)
    old_logp_flat = old_log_probs.view(-1)
    adv_flat = (advantages.view(-1))
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    ret_flat = returns.view(-1)
    
    for _ in range(epochs):
        indices = torch.randperm(total)
        for start in range(0, total, batch_size):
            mb_idx = indices[start:start + batch_size]
            mb_obs = {"rgb": obs_rgb_flat[mb_idx].to(device)}
            if obs_priv_flat is not None:
                mb_obs["privileged"] = obs_priv_flat[mb_idx].to(device)
            
            log_prob, value, entropy = policy.evaluate(mb_obs, actions_flat[mb_idx].to(device))
            ratio = torch.exp(log_prob - old_logp_flat[mb_idx].to(device))
            mb_adv = adv_flat[mb_idx].to(device)
            
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * F.mse_loss(value, ret_flat[mb_idx].to(device))
            loss = p_loss + value_coef * v_loss - entropy_coef * entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()


# =============================================================================
# OBJECTIVE FUNCTION (MULTI-OBJECTIVE)
# =============================================================================

def objective(trial: optuna.Trial, env) -> tuple:
    """
    NSGA-II multi-objective function.
    Returns: (SSR, Stage) - both to maximize
    """
    # Sample hyperparameters
    entropy_coef = trial.suggest_float("entropy_coef", 0.005, 0.015)  # Narrowed based on TPE results
    gae_lambda = trial.suggest_float("gae_lambda", 0.93, 0.99)
    clip_ratio = trial.suggest_categorical("clip_ratio", [0.2, 0.3])  # 0.3 was best
    epochs = trial.suggest_int("epochs", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 2e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    
    print(f"\n{'='*70}")
    print(f"TRIAL {trial.number} [NSGA-II]")
    print(f"{'='*70}")
    print(f"entropy: {entropy_coef:.4f} | gae: {gae_lambda:.4f} | clip: {clip_ratio}")
    print(f"epochs: {epochs} | lr: {learning_rate:.2e} | batch: {batch_size}")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda:0")
    env.set_curriculum_level(0)
    
    policy = AsymmetricPolicy(vision_channels=4, privileged_dim=7, action_dim=2, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
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
    max_stage = 0
    next_eval = eval_interval
    
    try:
        while step < max_steps:
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
            
            obs_rgb = torch.stack(obs_rgb_buf)
            obs_priv = torch.stack(obs_priv_buf) if obs_priv_buf else None
            actions = torch.stack(act_buf)
            rewards = torch.stack(rew_buf)
            values = torch.stack(val_buf)
            log_probs = torch.stack(logp_buf)
            dones = torch.stack(done_buf)
            
            with torch.no_grad():
                adv, ret = compute_gae(rewards, values, dones, FIXED_PARAMS["gamma"], gae_lambda)
            
            ppo_update(policy, optimizer, obs_rgb, obs_priv, actions, log_probs,
                      adv, ret, epochs, batch_size, clip_ratio,
                      entropy_coef, FIXED_PARAMS["value_coef"], FIXED_PARAMS["max_grad_norm"])
            
            ssr = np.mean(stage_successes) if stage_successes else 0.0
            current_stage = env.curriculum_level
            if current_stage > max_stage:
                max_stage = current_stage
            if ssr > best_ssr:
                best_ssr = ssr
            
            if step >= next_eval:
                mean_reward = np.mean(ep_rewards) if ep_rewards else 0.0
                print(f"[{step:,}] S{current_stage} | SSR: {ssr:.1%} | R: {mean_reward:.1f} | MaxS: {max_stage}")
                next_eval += eval_interval
                
                # Pruning
                if OPTUNA_CONFIG["pruning_enabled"]:
                    if step >= OPTUNA_CONFIG["pruning_warmup_steps"]:
                        if current_stage == 0 and ssr < OPTUNA_CONFIG["min_ssr_stage0"]:
                            print(f"Trial {trial.number} pruned: S0 SSR {ssr:.1%}")
                            raise optuna.TrialPruned()
                        if step >= 200_000 and current_stage <= 3 and ssr < OPTUNA_CONFIG["min_ssr_stage3"]:
                            print(f"Trial {trial.number} pruned: stuck at S{current_stage}")
                            raise optuna.TrialPruned()
                        if step >= 500_000 and current_stage <= 5 and ssr < OPTUNA_CONFIG["min_ssr_stage5"]:
                            print(f"Trial {trial.number} pruned: stuck at S{current_stage}")
                            raise optuna.TrialPruned()
            
            # Curriculum advancement
            if len(stage_successes) >= 50 and ssr >= 0.70 and current_stage < 27:
                env.set_curriculum_level(current_stage + 1)
                obs_dict, _ = env.reset()
                cur_reward.zero_()
                cur_length.zero_()
                stage_successes.clear()
                print(f"➡️ Advanced to S{current_stage + 1}")
        
        print(f"\n✅ TRIAL {trial.number} COMPLETE | SSR: {best_ssr:.1%} | Stage: S{max_stage}")
        
    except optuna.TrialPruned:
        env.set_curriculum_level(0)
        env.reset()
        raise
    except Exception as e:
        print(f"Trial {trial.number} error: {e}")
        env.set_curriculum_level(0)
        env.reset()
        raise optuna.TrialPruned()
    
    env.set_curriculum_level(0)
    
    # Return both objectives (NSGA-II will maximize both)
    return best_ssr, max_stage


# =============================================================================
# MAIN
# =============================================================================

def create_study():
    storage_path = OPTUNA_CONFIG["storage_path"]
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    storage = f"sqlite:///{storage_path}"
    
    study = optuna.create_study(
        study_name=OPTUNA_CONFIG["study_name"],
        storage=storage,
        directions=["maximize", "maximize"],  # SSR and Stage
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(
            population_size=20,
            mutation_prob=0.1,
            crossover_prob=0.9,
            seed=42,
        ),
    )
    
    print(f"✅ NSGA-II Study created: {OPTUNA_CONFIG['study_name']}")
    print(f"   Objectives: maximize SSR, maximize Stage")
    print(f"   Population: 20, Mutation: 0.1, Crossover: 0.9")
    print(f"   Trials so far: {len(study.trials)}")
    return study


def run_worker(args):
    app = AppLauncher(args)
    sim = app.app
    
    sys.path.insert(0, "/workspace/teko/source/teko")
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
    
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = FIXED_PARAMS["num_envs"]
    cfg.enable_curriculum = True
    cfg.asymmetric_critic = True
    
    print("Creating shared environment...")
    env = TekoEnv(cfg=cfg)
    print(f"✅ Shared environment: {FIXED_PARAMS['num_envs']} envs")
    
    try:
        storage = f"sqlite:///{OPTUNA_CONFIG['storage_path']}"
        study = optuna.load_study(
            study_name=OPTUNA_CONFIG["study_name"],
            storage=storage,
        )
        
        print(f"🚀 NSGA-II Worker started")
        print(f"   Completed: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
        
        study.optimize(
            lambda trial: objective(trial, env),
            n_trials=OPTUNA_CONFIG["n_trials"],
            timeout=None,
            catch=(Exception,),
        )
        
        # Print Pareto front
        print(f"\n{'='*70}")
        print("PARETO FRONT (Best trials)")
        print(f"{'='*70}")
        for trial in study.best_trials:
            print(f"Trial {trial.number}: SSR={trial.values[0]:.3f}, Stage={trial.values[1]:.0f}")
            print(f"  Params: {trial.params}")
        
    finally:
        env.close()
        sim.close()


def main():
    parser = argparse.ArgumentParser(description="TEKO NSGA-II HPO")
    parser.add_argument("--create-study", action="store_true")
    parser.add_argument("--num-trials", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    
    if args.create_study:
        create_study()
        return
    
    if args.num_trials:
        OPTUNA_CONFIG["n_trials"] = args.num_trials
    
    run_worker(args)


if __name__ == "__main__":
    main()
