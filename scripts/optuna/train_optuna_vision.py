#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Optuna Vision Training (NSGA-II) v6
========================================

Multi-objective optimization for vision-based autonomous docking.
Objectives: maximize success_rate, maximize max_stage_reached

Hyperparameters tuned by Optuna:
- entropy_coef: exploration vs exploitation
- gae_lambda: bias-variance tradeoff in advantage estimation
- learning_rate: optimizer step size
- epochs: SGD passes per rollout
- batch_size: minibatch size for PPO updates

Fixed hyperparameters (based on literature):
- gamma: 0.99 (standard for continuous control)
- clip_ratio: 0.2 (PPO paper recommendation)
- rollout_len: 128 (horizon)
- num_envs: 150 (parallelization)

Author: Alexandre Schleier Neves da Silva
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import argparse
import sys
import math
import socket
import sqlite3
import time
import random
from collections import deque
from typing import Dict, Tuple, Optional
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("ERROR: optuna not installed", flush=True)
    sys.exit(1)

from isaaclab.app import AppLauncher
NullPool = None
print = partial(print, flush=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

OPTUNA_CONFIG = {
    "study_name": "teko_vision_v8",
    "storage_path": "/home/schux00/optuna/teko_vision_v8.db",

    "target_total_trials": 200,

    "max_steps_per_trial": 15_000_000,
    "max_walltime_s_per_trial": 7 * 24 * 3600,  # 7 days

    "eval_interval": 50_000,

    "pruning_enabled": True,
    "pruning_warmup_steps": 2_000_000,
    "bad_eval_streak_to_prune": 8,

    "min_ssr_thresholds": {
        0: 0.60,
        4: 0.50,
        8: 0.40,
        12: 0.35,
        16: 0.30,
    },

    "success_surface_xy": 0.03,
}

FIXED_PARAMS = {
    "gamma": 0.99,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_envs": 150,
    "rollout_len": 128,
}

# Curriculum advancement thresholds
ADVANCE_THRESHOLD_EARLY = 0.80   # S0-S6
ADVANCE_THRESHOLD_MID = 0.75    # S7-S12
ADVANCE_THRESHOLD_LATE = 0.70   # S13+

MIN_STEPS_BEFORE_ADVANCE = 200_000


# =============================================================================
# UTILITIES
# =============================================================================

def get_min_ssr_for_stage(stage: int) -> float:
    """Get minimum SSR threshold for pruning at given stage."""
    thresholds = OPTUNA_CONFIG["min_ssr_thresholds"]
    applicable_key = 0
    for key in sorted(thresholds.keys()):
        if key <= stage:
            applicable_key = key
    return thresholds[applicable_key]


def get_advance_threshold(stage: int) -> float:
    """Get SSR threshold needed to advance from given stage."""
    if stage <= 6:
        return ADVANCE_THRESHOLD_EARLY
    elif stage <= 12:
        return ADVANCE_THRESHOLD_MID
    else:
        return ADVANCE_THRESHOLD_LATE


def atanh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic tangent with clamping for numerical stability."""
    x = torch.clamp(x, -0.999, 0.999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def get_success_flags(env, device: torch.device) -> torch.Tensor:
    """Extract success flags from environment."""
    if hasattr(env, "_last_success"):
        s = getattr(env, "_last_success")
        if isinstance(s, torch.Tensor):
            return s.to(device=device, dtype=torch.bool)
    
    # Fallback: check connector distance
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    if not isinstance(surface_xy, torch.Tensor):
        surface_xy = torch.as_tensor(surface_xy, device=device)
    return surface_xy < OPTUNA_CONFIG["success_surface_xy"]


# =============================================================================
# SQLITE STORAGE
# =============================================================================

def _init_sqlite(path: str) -> None:
    """Initialize SQLite database with optimal settings."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=120)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=120000;")
        conn.commit()
    finally:
        conn.close()


def make_storage(db_path: str):
    """Create Optuna storage - simple URL to avoid SQLAlchemy conflicts."""
    _init_sqlite(db_path)
    return f"sqlite:///{db_path}"


def create_study(study_name: str, storage) -> optuna.Study:
    """Create or load NSGA-II study."""
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize", "maximize"],  # SSR, max_stage
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(
            population_size=20,
            mutation_prob=0.1,
            crossover_prob=0.9,
            seed=42,
        ),
    )
    print(f"[STUDY] {study_name} ready | Target: {OPTUNA_CONFIG['target_total_trials']} trials")
    return study


def _is_retryable_error(e: Exception) -> bool:
    """Check if error is a transient storage error."""
    if isinstance(e, optuna.exceptions.StorageInternalError):
        return True
    msg = str(e).lower()
    return any(x in msg for x in ["database is locked", "locked", "busy"])


# =============================================================================
# VISION ENCODER (CNN)
# =============================================================================

class VisionEncoder(nn.Module):
    """
    CNN encoder for 84x84 grayscale images with 4-frame stack.
    
    Architecture optimized for low-resolution visual features:
    - 3 conv layers with GroupNorm
    - Output: 256-dim feature vector
    """
    
    def __init__(self, in_channels: int = 4, feature_dim: int = 256):
        super().__init__()
        
        self.conv = nn.Sequential(
            # Layer 1: 84x84 -> 28x28
            nn.Conv2d(in_channels, 64, kernel_size=6, stride=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            
            # Layer 2: 28x28 -> 14x14
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            
            # Layer 3: 14x14 -> 7x7
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )
        
        # Compute flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            flat_dim = int(self.conv(dummy).view(1, -1).shape[1])
        
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )
        
        self.feature_dim = feature_dim
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


# =============================================================================
# POLICY NETWORK
# =============================================================================

class VisionPolicy(nn.Module):
    """
    Actor-Critic policy with asymmetric architecture.
    
    Actor: vision only (what the real robot would have)
    Critic: vision + privileged state (for better value estimation during training)
    """
    
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
    
    def __init__(
        self,
        vision_channels: int = 4,
        privileged_dim: int = 7,
        action_dim: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Vision encoder (shared concept, separate instances)
        self.vision_encoder = VisionEncoder(
            in_channels=vision_channels,
            feature_dim=hidden_dim
        )
        
        # Actor head (vision only)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )
        
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        
        # Critic: vision + privileged state
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
        """Initialize actor/critic heads with orthogonal init."""
        for module in [self.actor_head, self.state_encoder, self.critic_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Small init for final actor layer (helps exploration)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
    
    def _std(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
    
    def act(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: tanh-squashed action
            log_prob: log probability of action
            value: critic value estimate
        """
        vision = obs["rgb"]
        privileged = obs.get("privileged", None)
        
        # Encode vision
        vision_feat = self.vision_encoder(vision)
        
        # Actor distribution
        mean = self.actor_head(vision_feat)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        
        # Sample action
        u = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(u)
        
        # Log probability with tanh correction
        log_prob_u = dist.log_prob(u).sum(-1)
        log_det = torch.log(torch.clamp(1.0 - action * action, min=1e-6)).sum(-1)
        log_prob = log_prob_u - log_det
        
        # Critic value
        if privileged is not None:
            state_feat = self.state_encoder(privileged)
            value = self.critic_head(torch.cat([vision_feat, state_feat], dim=-1)).squeeze(-1)
        else:
            value = torch.zeros(action.shape[0], device=action.device)
        
        return action, log_prob, value
    
    def evaluate(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_prob: log probability of actions
            value: critic value estimate
            entropy: policy entropy
        """
        vision = obs["rgb"]
        privileged = obs.get("privileged", None)
        
        vision_feat = self.vision_encoder(vision)
        
        mean = self.actor_head(vision_feat)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        
        # Inverse tanh to get pre-squash action
        u = atanh(actions)
        log_prob_u = dist.log_prob(u).sum(-1)
        log_det = torch.log(torch.clamp(1.0 - actions * actions, min=1e-6)).sum(-1)
        log_prob = log_prob_u - log_det
        
        entropy = dist.entropy().sum(-1)
        
        if privileged is not None:
            state_feat = self.state_encoder(privileged)
            value = self.critic_head(torch.cat([vision_feat, state_feat], dim=-1)).squeeze(-1)
        else:
            value = torch.zeros(actions.shape[0], device=actions.device)
        
        return log_prob, value, entropy


# =============================================================================
# PPO ALGORITHM
# =============================================================================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    last_value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: [T, N] reward tensor
        values: [T, N] value estimates
        dones: [T, N] episode done flags
        gamma: discount factor
        lam: GAE lambda parameter
        last_value: [N] bootstrap value for final step
    
    Returns:
        advantages: [T, N] advantage estimates
        returns: [T, N] return targets for value function
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: VisionPolicy,
    optimizer: torch.optim.Optimizer,
    obs_rgb: torch.Tensor,
    obs_priv: Optional[torch.Tensor],
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    epochs: int,
    batch_size: int,
    clip_ratio: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    """
    Perform PPO policy update.
    
    Returns dict with loss metrics for logging.
    """
    device = next(policy.parameters()).device
    T, N = obs_rgb.shape[:2]
    total = T * N
    
    # Flatten tensors
    rgb_flat = obs_rgb.view(total, 4, 128, 128)
    act_flat = actions.view(total, 2)
    old_logp_flat = old_log_probs.view(total)
    adv_flat = advantages.view(total)
    ret_flat = returns.view(total)
    priv_flat = obs_priv.view(total, -1) if obs_priv is not None else None
    
    # Normalize advantages
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    
    metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "grad_norm": []}
    
    for _ in range(epochs):
        indices = torch.randperm(total, device=device)
        
        for start in range(0, total, batch_size):
            mb_idx = indices[start:start + batch_size]
            
            mb_obs = {"rgb": rgb_flat[mb_idx]}
            if priv_flat is not None:
                mb_obs["privileged"] = priv_flat[mb_idx]
            
            mb_actions = act_flat[mb_idx]
            mb_old_logp = old_logp_flat[mb_idx]
            mb_adv = adv_flat[mb_idx]
            mb_ret = ret_flat[mb_idx]
            
            # Evaluate current policy
            log_prob, value, entropy = policy.evaluate(mb_obs, mb_actions)
            
            # Policy loss (clipped surrogate)
            ratio = torch.exp(log_prob - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = 0.5 * F.mse_loss(value, mb_ret)
            
            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            
            # Optimize
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm).item()
            optimizer.step()
            
            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy.mean().item())
            metrics["grad_norm"].append(grad_norm)
    
    return {k: np.mean(v) for k, v in metrics.items()}


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def objective(trial: optuna.Trial, env) -> Tuple[float, float]:
    """
    Optuna objective: train PPO and return (best_ssr, max_stage).
    
    Tuned hyperparameters:
    - entropy_coef: [0.001, 0.02] - exploration bonus
    - gae_lambda: [0.90, 0.98] - advantage estimation bias-variance
    - learning_rate: [3e-5, 3e-4] - optimizer step size
    - epochs: [3, 8] - PPO update passes
    - batch_size: [1024, 4096] - minibatch size
    """
    
    # Sample hyperparameters
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.02, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.98)
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True)
    epochs = trial.suggest_int("epochs", 3, 8)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    
    device = torch.device("cuda:0")
    env.set_curriculum_level(0)
    
    # Initialize policy
    policy = VisionPolicy(
        vision_channels=4,
        privileged_dim=7,
        action_dim=2,
        hidden_dim=256
    ).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Training state
    step = 0
    max_steps = OPTUNA_CONFIG["max_steps_per_trial"]
    max_wall_s = OPTUNA_CONFIG["max_walltime_s_per_trial"]
    t0 = time.time()
    
    eval_interval = OPTUNA_CONFIG["eval_interval"]
    rollout_len = FIXED_PARAMS["rollout_len"]
    num_envs = FIXED_PARAMS["num_envs"]
    
    # Episode tracking
    ep_rewards = deque(maxlen=200)
    stage_successes = deque(maxlen=300)
    
    cur_reward = torch.zeros(num_envs, device=device)
    cur_length = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    obs_dict, _ = env.reset()
    has_privileged = "privileged" in obs_dict
    
    best_ssr = 0.0
    max_stage = 0
    next_eval = eval_interval
    bad_eval_streak = 0
    last_stage_change_step = 0
    
    # Rollout buffers (store as uint8 to save memory)
    obs_rgb_u8 = torch.empty((rollout_len, num_envs, 4, 128, 128), device=device, dtype=torch.uint8)
    obs_priv = torch.empty((rollout_len, num_envs, 7), device=device, dtype=torch.float32) if has_privileged else None
    actions = torch.empty((rollout_len, num_envs, 2), device=device, dtype=torch.float32)
    rewards = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    values = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    log_probs = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    dones = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    
    try:
        while step < max_steps and (time.time() - t0) < max_wall_s:
            # Collect rollout
            for t in range(rollout_len):
                vision_f32 = obs_dict["rgb"].to(device=device, dtype=torch.float32)
                obs = {"rgb": vision_f32}
                if has_privileged:
                    obs["privileged"] = obs_dict["privileged"].to(device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    action, log_prob, value = policy.act(obs)
                
                # Store in buffers
                obs_rgb_u8[t].copy_((vision_f32.clamp(0.0, 1.0) * 255.0).to(torch.uint8))
                if has_privileged:
                    obs_priv[t].copy_(obs["privileged"])
                actions[t].copy_(action)
                log_probs[t].copy_(log_prob)
                values[t].copy_(value)
                
                # Environment step
                obs_dict, reward, term, trunc, _ = env.step(action)
                done = term | trunc
                
                rewards[t].copy_(reward)
                dones[t].copy_(done.float())
                
                # Track episode statistics
                cur_reward += reward
                cur_length += 1
                
                if done.any():
                    done_idx = done.nonzero(as_tuple=False).squeeze(-1)
                    with torch.no_grad():
                        succ = get_success_flags(env, device=device).float()
                    
                    ep_rewards.extend(cur_reward[done_idx].cpu().tolist())
                    stage_successes.extend(succ[done_idx].cpu().tolist())
                    
                    cur_reward[done_idx] = 0.0
                    cur_length[done_idx] = 0
                
                step += num_envs
            
            # Compute advantages
            with torch.no_grad():
                last_obs = {"rgb": obs_dict["rgb"].to(device=device, dtype=torch.float32)}
                if has_privileged:
                    last_obs["privileged"] = obs_dict["privileged"].to(device=device, dtype=torch.float32)
                
                _, _, last_value = policy.act(last_obs)
            
            # Convert obs to float for update
            obs_rgb_f32 = obs_rgb_u8.to(dtype=torch.float32) / 255.0
            
            adv, ret = compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                gamma=FIXED_PARAMS["gamma"],
                lam=gae_lambda,
                last_value=last_value,
            )
            
            # PPO update
            metrics = ppo_update(
                policy=policy,
                optimizer=optimizer,
                obs_rgb=obs_rgb_f32,
                obs_priv=obs_priv,
                actions=actions,
                old_log_probs=log_probs,
                advantages=adv,
                returns=ret,
                epochs=epochs,
                batch_size=batch_size,
                clip_ratio=FIXED_PARAMS["clip_ratio"],
                entropy_coef=entropy_coef,
                value_coef=FIXED_PARAMS["value_coef"],
                max_grad_norm=FIXED_PARAMS["max_grad_norm"],
            )
            
            # Evaluation
            ssr = float(np.mean(stage_successes)) if len(stage_successes) > 0 else 0.0
            current_stage = int(env.curriculum_level)
            max_stage = max(max_stage, current_stage)
            best_ssr = max(best_ssr, ssr)
            
            if step >= next_eval:
                mean_reward = float(np.mean(ep_rewards)) if len(ep_rewards) > 0 else 0.0
                elapsed = time.time() - t0
                
                print(f"[{step:,}] S{current_stage:02d} | SSR: {ssr:.1%} | R: {mean_reward:.1f} | "
                      f"MaxS: {max_stage} | Ent: {metrics['entropy']:.3f} | {elapsed/3600:.1f}h")
                
                next_eval += eval_interval
                
                # Pruning check
                if OPTUNA_CONFIG["pruning_enabled"] and step >= OPTUNA_CONFIG["pruning_warmup_steps"]:
                    min_ssr = get_min_ssr_for_stage(current_stage)
                    
                    if ssr < min_ssr:
                        bad_eval_streak += 1
                        if bad_eval_streak >= OPTUNA_CONFIG["bad_eval_streak_to_prune"]:
                            print(f"[PRUNE] SSR {ssr:.1%} < {min_ssr:.0%} for {bad_eval_streak} evals")
                            raise optuna.TrialPruned()
                    else:
                        bad_eval_streak = 0
            
            # Curriculum advancement
            advance_threshold = get_advance_threshold(current_stage)
            steps_in_stage = step - last_stage_change_step
            
            if (len(stage_successes) >= 100 and
                ssr >= advance_threshold and
                steps_in_stage >= MIN_STEPS_BEFORE_ADVANCE and
                current_stage < 27):
                
                env.set_curriculum_level(current_stage + 1)
                obs_dict, _ = env.reset()
                cur_reward.zero_()
                cur_length.zero_()
                stage_successes.clear()
                last_stage_change_step = step
                bad_eval_streak = 0
                
                print(f"[ADVANCE] S{current_stage} -> S{current_stage + 1} (SSR={ssr:.1%})")
    
    except torch.cuda.OutOfMemoryError:
        print(f"[OOM] Trial {trial.number}")
        torch.cuda.empty_cache()
        raise optuna.TrialPruned()
    
    except optuna.TrialPruned:
        env.set_curriculum_level(0)
        env.reset()
        raise
    
    except Exception as e:
        print(f"[ERROR] Trial {trial.number}: {repr(e)}")
        torch.cuda.empty_cache()
        env.set_curriculum_level(0)
        env.reset()
        raise optuna.TrialPruned()
    
    env.set_curriculum_level(0)
    elapsed = time.time() - t0
    print(f"[DONE] Trial {trial.number}: SSR={best_ssr:.1%}, MaxStage={max_stage}, Time={elapsed/3600:.1f}h")
    
    return best_ssr, float(max_stage)


# =============================================================================
# WORKER
# =============================================================================

def run_worker(args):
    """Main worker loop: run trials until target reached."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    hostname = socket.gethostname()
    slurm_job = os.environ.get("SLURM_JOB_ID", "local")
    slurm_array = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    # Launch Isaac Sim
    app = AppLauncher(args)
    sim = app.app
    
    # Import after sim launch
    sys.path.insert(0, "/workspace/teko/source/teko")
    from teko.tasks.direct.teko.teko_env_tiled import TekoEnvTiled as TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
    
    # Create environment
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = FIXED_PARAMS["num_envs"]
    cfg.enable_curriculum = True
    
    env = TekoEnv(cfg=cfg)
    
    # Setup Optuna
    db_path = OPTUNA_CONFIG["storage_path"]
    storage = make_storage(db_path)
    study = create_study(OPTUNA_CONFIG["study_name"], storage)
    
    print("=" * 70)
    print(f"TEKO Vision Training v6 - Worker {slurm_array}")
    print("=" * 70)
    print(f"Host: {hostname} | Job: {slurm_job}")
    print(f"Envs: {FIXED_PARAMS['num_envs']} | Rollout: {FIXED_PARAMS['rollout_len']}")
    print(f"Max steps/trial: {OPTUNA_CONFIG['max_steps_per_trial']:,}")
    print("=" * 70)
    
    local_trials = 0
    max_retries = 10
    retry_count = 0
    
    try:
        while len(study.get_trials(deepcopy=False)) < OPTUNA_CONFIG["target_total_trials"]:
            try:
                study.optimize(lambda tr: objective(tr, env), n_trials=1)
                local_trials += 1
                retry_count = 0
                
                # Refresh study periodically
                if local_trials % 3 == 0:
                    study = optuna.load_study(
                        study_name=OPTUNA_CONFIG["study_name"],
                        storage=storage
                    )
                    
            except Exception as e:
                if _is_retryable_error(e):
                    retry_count += 1
                    if retry_count > max_retries:
                        raise
                    time.sleep(2.0 + random.random() * 3.0)
                    storage = make_storage(db_path)
                    study = optuna.load_study(
                        study_name=OPTUNA_CONFIG["study_name"],
                        storage=storage
                    )
                else:
                    raise
    
    finally:
        env.close()
        sim.close()
    
    print(f"[WORKER] Completed {local_trials} trials")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TEKO Vision Optuna Training")
    parser.add_argument("--create-study", action="store_true", help="Create study and exit")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    args.headless = True
    args.enable_cameras = True
    
    if args.create_study:
        storage = make_storage(OPTUNA_CONFIG["storage_path"])
        create_study(OPTUNA_CONFIG["study_name"], storage)
        return
    
    run_worker(args)


if __name__ == "__main__":
    main()