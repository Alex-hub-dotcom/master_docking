#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Optuna PPO (NSGA-II) v5 - 84x84 Pure Vision + High Replay
==============================================================

Based on v4 with higher replay probabilities to combat catastrophic forgetting.

Changes from v4:
- Increased replay probs (0.40/0.45/0.50/0.55)
- Removed anti-stuck (never drop stages)
- New study name

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

try:
    from sqlalchemy.pool import NullPool
except Exception:
    NullPool = None

from isaaclab.app import AppLauncher

print = partial(print, flush=True)


# =============================================================================
# CONFIG - PURE VISION + HIGH REPLAY
# =============================================================================

OPTUNA_CONFIG = {
    "study_name": "teko_nsgaii_v5_highrep",
    "storage_path": "/home/schux00/optuna/teko_nsgaii_v5_highrep.db",

    "target_total_trials": 150,

    "max_steps_per_trial": 12_000_000,
    "max_walltime_s_per_trial": 13_500,

    "eval_interval": 25_000,

    "pruning_enabled": True,
    "pruning_warmup_steps": 1_000_000,
    "bad_eval_streak_to_prune": 10,

    "min_ssr_thresholds": {
        0: 0.65,
        2: 0.55,
        4: 0.45,
        7: 0.35,
        10: 0.30,
    },

    "success_surface_xy": 0.03,
}

FIXED_PARAMS = {
    "gamma": 0.99,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,

    "num_envs": 120,
    "rollout_len": 96,
}

REWARD_OVERRIDES = {
    "alignment_scale": 0.40,
    "turning_bonus": 0.60,
    "facing_threshold_deg": 15.0,
    "progress_scale": 10.0,
}

# HIGH REPLAY to combat catastrophic forgetting
REPLAY_PROBS = {
    "early": 0.40,
    "micro": 0.45,
    "ultra": 0.50,
    "turn": 0.55,
}


# =============================================================================
# UTILS
# =============================================================================

def get_min_ssr_for_stage(stage: int) -> float:
    thresholds = OPTUNA_CONFIG["min_ssr_thresholds"]
    applicable_key = 0
    for key in sorted(thresholds.keys()):
        if key <= stage:
            applicable_key = key
        else:
            break
    return thresholds[applicable_key]


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999, 0.999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _as_bool_tensor(x, device: torch.device) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.bool)
    try:
        return torch.as_tensor(x, device=device, dtype=torch.bool)
    except Exception:
        return None


def get_success_flags(env, device: torch.device) -> torch.Tensor:
    if hasattr(env, "get_last_success") and callable(getattr(env, "get_last_success")):
        s = env.get_last_success()
        s_t = _as_bool_tensor(s, device)
        if s_t is not None:
            return s_t

    if hasattr(env, "_last_success"):
        s = getattr(env, "_last_success")
        s_t = _as_bool_tensor(s, device)
        if s_t is not None:
            return s_t

    print("[WARN] Fallback success detection - may be inaccurate!")
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    if not isinstance(surface_xy, torch.Tensor):
        surface_xy = torch.as_tensor(surface_xy, device=device)
    return surface_xy.to(device=device) < OPTUNA_CONFIG["success_surface_xy"]


# =============================================================================
# SQLITE SAFETY
# =============================================================================

def _init_sqlite(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=120)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=120000;")
        conn.commit()
    finally:
        conn.close()


def _make_storage_url(path: str) -> str:
    _init_sqlite(path)
    return f"sqlite:///{path}"


def _sqlite_state_counts(db_path: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    try:
        con = sqlite3.connect(db_path, timeout=60)
        cur = con.cursor()
        rows = cur.execute("SELECT state, COUNT(*) FROM trials GROUP BY state;").fetchall()
        con.close()
        for state_int, c in rows:
            try:
                name = TrialState(state_int).name
            except Exception:
                name = str(state_int)
            counts[name] = int(c)
    except Exception:
        pass
    return counts


def make_storage(db_path: str):
    url = _make_storage_url(db_path)
    if NullPool is None:
        return url
    return optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={
            "connect_args": {"timeout": 120, "check_same_thread": False},
            "poolclass": NullPool,
        },
    )


def create_study(study_name: str, storage) -> optuna.Study:
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize", "maximize"],
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(
            population_size=20,
            mutation_prob=0.1,
            crossover_prob=0.9,
            seed=42,
        ),
    )
    print(f"✅ Study ready: {study_name} | Target: {OPTUNA_CONFIG['target_total_trials']} trials")
    return study


def load_or_create_study(study_name: str, storage) -> optuna.Study:
    try:
        return optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"[WARN] load_study failed ({repr(e)}). Creating...")
        return create_study(study_name, storage)


def _total_trials(study: optuna.Study) -> int:
    return len(study.get_trials(deepcopy=False))


def _is_retryable_storage_error(e: Exception) -> bool:
    if isinstance(e, optuna.exceptions.StorageInternalError):
        return True
    if isinstance(e, UnboundLocalError) and "updated_state" in str(e):
        return True
    msg = str(e).lower()
    return any(x in msg for x in ["database is locked", "database is busy", "locked", "operationalerror"])


# =============================================================================
# VISION ENCODER
# =============================================================================

class VisionEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, feature_dim: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=6, stride=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
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
# POLICY: Pure Vision (No IMU)
# =============================================================================

class VisionPolicy(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5

    def __init__(
        self,
        vision_channels: int = 4,
        privileged_dim: int = 7,
        action_dim: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.vision_encoder = VisionEncoder(in_channels=vision_channels, feature_dim=hidden_dim)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
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
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)

    def _std(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))

    def _vision_features(self, vision: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(vision)

    def _actor_dist(self, vision_feat: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor_head(vision_feat)
        std = self._std().unsqueeze(0).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def _critic_value(self, vision_feat: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        s = self.state_encoder(privileged)
        v = self.critic_head(torch.cat([vision_feat, s], dim=-1)).squeeze(-1)
        return v

    def act(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vision = obs["rgb"]
        privileged = obs.get("privileged", None)

        vision_feat = self._vision_features(vision)
        dist = self._actor_dist(vision_feat)

        u = dist.mean if deterministic else dist.rsample()
        a = torch.tanh(u)

        log_prob_u = dist.log_prob(u).sum(-1)
        log_det = torch.log(torch.clamp(1.0 - a * a, min=1e-6)).sum(-1)
        log_prob = log_prob_u - log_det

        if privileged is not None:
            value = self._critic_value(vision_feat, privileged)
        else:
            value = torch.zeros(a.shape[0], device=a.device)

        return a, log_prob, value

    def evaluate(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vision = obs["rgb"]
        privileged = obs.get("privileged", None)

        vision_feat = self._vision_features(vision)
        dist = self._actor_dist(vision_feat)

        u = atanh(actions)
        log_prob_u = dist.log_prob(u).sum(-1)
        log_det = torch.log(torch.clamp(1.0 - actions * actions, min=1e-6)).sum(-1)
        log_prob = log_prob_u - log_det

        entropy = dist.entropy().sum(-1)

        if privileged is not None:
            value = self._critic_value(vision_feat, privileged)
        else:
            value = torch.zeros(actions.shape[0], device=actions.device)

        return log_prob, value, entropy


# =============================================================================
# PPO / GAE
# =============================================================================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    last_value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    obs_rgb_u8: torch.Tensor,
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
):
    device = next(policy.parameters()).device
    T, N = obs_rgb_u8.shape[:2]
    total = T * N

    rgb_flat_u8 = obs_rgb_u8.view(total, 4, 84, 84)
    act_flat = actions.view(total, 2)
    old_logp_flat = old_log_probs.view(total)
    adv_flat = advantages.view(total)
    ret_flat = returns.view(total)
    priv_flat = obs_priv.view(total, -1) if obs_priv is not None else None

    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    indices = torch.randperm(total, device=device)

    for _ in range(epochs):
        for start in range(0, total, batch_size):
            mb_idx = indices[start:start + batch_size]

            mb_rgb = rgb_flat_u8[mb_idx].to(dtype=torch.float32) / 255.0
            mb_obs = {"rgb": mb_rgb}
            if priv_flat is not None:
                mb_obs["privileged"] = priv_flat[mb_idx]

            mb_actions = act_flat[mb_idx]
            mb_old_logp = old_logp_flat[mb_idx]
            mb_adv = adv_flat[mb_idx]
            mb_ret = ret_flat[mb_idx]

            log_prob, value, entropy = policy.evaluate(mb_obs, mb_actions)

            ratio = torch.exp(log_prob - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
            p_loss = -torch.min(surr1, surr2).mean()

            v_loss = 0.5 * F.mse_loss(value, mb_ret)
            loss = p_loss + value_coef * v_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()


# =============================================================================
# OBJECTIVE
# =============================================================================

def objective(trial: optuna.Trial, env) -> Tuple[float, float]:
    """Train pure vision PPO and return (best_ssr, max_stage)."""

    entropy_coef = trial.suggest_float("entropy_coef", 0.004, 0.012)
    gae_lambda = trial.suggest_float("gae_lambda", 0.93, 0.97)
    clip_ratio = trial.suggest_categorical("clip_ratio", [0.25, 0.30])
    epochs = trial.suggest_int("epochs", 5, 8)
    learning_rate = trial.suggest_float("learning_rate", 8e-5, 2e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2048, 4096])

    device = torch.device("cuda:0")
    env.set_curriculum_level(0)

    policy = VisionPolicy(
        vision_channels=4,
        privileged_dim=7,
        action_dim=2,
        hidden_dim=256
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    step = 0
    max_steps = OPTUNA_CONFIG["max_steps_per_trial"]
    max_wall_s = OPTUNA_CONFIG["max_walltime_s_per_trial"]
    t0 = time.time()

    eval_interval = OPTUNA_CONFIG["eval_interval"]
    rollout_len = FIXED_PARAMS["rollout_len"]
    num_envs = FIXED_PARAMS["num_envs"]

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
    min_steps_before_advance = 150_000

    # Rollout buffers
    obs_rgb_u8 = torch.empty((rollout_len, num_envs, 4, 84, 84), device=device, dtype=torch.uint8)
    obs_priv = torch.empty((rollout_len, num_envs, 7), device=device, dtype=torch.float32) if has_privileged else None

    actions = torch.empty((rollout_len, num_envs, 2), device=device, dtype=torch.float32)
    rewards = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    values = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    log_probs = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    dones = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)

    try:
        while step < max_steps and (time.time() - t0) < max_wall_s:
            for t in range(rollout_len):
                vision_f32 = obs_dict["rgb"].to(device=device, dtype=torch.float32)
                obs = {"rgb": vision_f32}
                if has_privileged:
                    obs["privileged"] = obs_dict["privileged"].to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    action, log_prob, value = policy.act(obs)

                obs_rgb_u8[t].copy_((vision_f32.clamp(0.0, 1.0) * 255.0).to(torch.uint8))
                if has_privileged:
                    obs_priv[t].copy_(obs["privileged"])

                actions[t].copy_(action)
                log_probs[t].copy_(log_prob)
                values[t].copy_(value)

                obs_dict, reward, term, trunc, _ = env.step(action)
                done = (term | trunc)

                rewards[t].copy_(reward)
                dones[t].copy_(done.float())

                cur_reward += reward
                cur_length += 1

                if done.any():
                    done_idx = done.nonzero(as_tuple=False).squeeze(-1)
                    with torch.no_grad():
                        succ = get_success_flags(env, device=device).float()

                    ep_rewards.extend(cur_reward[done_idx].detach().cpu().tolist())
                    stage_successes.extend(succ[done_idx].detach().cpu().tolist())

                    cur_reward[done_idx] = 0.0
                    cur_length[done_idx] = 0

                step += num_envs

            # Compute advantages
            with torch.no_grad():
                last_obs = {"rgb": obs_dict["rgb"].to(device=device, dtype=torch.float32)}
                if has_privileged:
                    last_obs["privileged"] = obs_dict["privileged"].to(device=device, dtype=torch.float32)

                vf = policy._vision_features(last_obs["rgb"])
                last_value = policy._critic_value(vf, last_obs["privileged"]) if has_privileged else torch.zeros(num_envs, device=device)

            adv, ret = compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                gamma=FIXED_PARAMS["gamma"],
                lam=gae_lambda,
                last_value=last_value,
            )

            # PPO update
            ppo_update(
                policy=policy,
                optimizer=optimizer,
                obs_rgb_u8=obs_rgb_u8,
                obs_priv=obs_priv,
                actions=actions,
                old_log_probs=log_probs,
                advantages=adv,
                returns=ret,
                epochs=epochs,
                batch_size=batch_size,
                clip_ratio=clip_ratio,
                entropy_coef=entropy_coef,
                value_coef=FIXED_PARAMS["value_coef"],
                max_grad_norm=FIXED_PARAMS["max_grad_norm"],
            )

            ssr = float(np.mean(stage_successes)) if len(stage_successes) > 0 else 0.0
            current_stage = int(env.curriculum_level)
            max_stage = max(max_stage, current_stage)
            best_ssr = max(best_ssr, ssr)

            if step >= next_eval:
                mean_reward = float(np.mean(ep_rewards)) if len(ep_rewards) > 0 else 0.0
                elapsed = time.time() - t0
                print(f"[{step:,}] S{current_stage:02d} | SSR: {ssr:.1%} | R: {mean_reward:.1f} | MaxS: {max_stage} | {elapsed/60:.1f}min")
                next_eval += eval_interval

                # Pruning
                if OPTUNA_CONFIG["pruning_enabled"] and step >= OPTUNA_CONFIG["pruning_warmup_steps"]:
                    min_ssr = get_min_ssr_for_stage(current_stage)
                    bad = ssr < min_ssr

                    if bad:
                        bad_eval_streak += 1
                        print(f"[PRUNE-CHECK] streak={bad_eval_streak} | SSR={ssr:.1%} < {min_ssr:.0%} | S{current_stage:02d}")
                    else:
                        bad_eval_streak = 0

                    if bad_eval_streak >= OPTUNA_CONFIG["bad_eval_streak_to_prune"]:
                        print(f"[PRUNE] Triggered: {bad_eval_streak} consecutive bad evals")
                        raise optuna.TrialPruned()

            # Curriculum advancement (NO anti-stuck - never drop stages)
            advance_thr = 0.75 if current_stage < 10 else 0.70

            if (
                len(stage_successes) >= 80
                and ssr >= advance_thr
                and (step - last_stage_change_step) >= min_steps_before_advance
                and current_stage < 27
            ):
                env.set_curriculum_level(current_stage + 1)
                obs_dict, _ = env.reset()
                cur_reward.zero_()
                cur_length.zero_()
                stage_successes.clear()
                last_stage_change_step = step
                bad_eval_streak = 0
                print(f"➡️ Advanced to S{current_stage + 1:02d} (SSR={ssr:.1%} >= {advance_thr:.0%})")

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
    print(f"✅ Trial {trial.number} complete: SSR={best_ssr:.1%}, MaxStage={max_stage}, Time={elapsed/60:.1f}min")

    return best_ssr, float(max_stage)


# =============================================================================
# WORKER
# =============================================================================

def run_worker(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    hostname = socket.gethostname()
    slurm_job = os.environ.get("SLURM_JOB_ID", "NA")
    slurm_array = os.environ.get("SLURM_ARRAY_TASK_ID", "NA")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    app = AppLauncher(args)
    sim = app.app

    sys.path.insert(0, "/workspace/teko/source/teko")
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    from teko.tasks.direct.teko.rewards.reward_functions import REWARD_CONFIG
    REWARD_CONFIG.update(REWARD_OVERRIDES)
    print(f"📊 Reward overrides: {REWARD_OVERRIDES}")

    cfg = TekoEnvCfg()
    cfg.scene.num_envs = FIXED_PARAMS["num_envs"]
    cfg.enable_curriculum = True
    cfg.asymmetric_critic = True

    env = TekoEnv(cfg=cfg)

    if not hasattr(env, "_last_success") and not hasattr(env, "get_last_success"):
        print("⚠️  WARNING: env missing _last_success! SSR will be wrong!")

    import teko.tasks.direct.teko.curriculum.curriculum_manager as cm
    cm.REPLAY_PROBS = REPLAY_PROBS
    print(f"📊 Replay probs (HIGH): {REPLAY_PROBS}")

    db_path = OPTUNA_CONFIG["storage_path"]
    storage = make_storage(db_path)
    study_name = OPTUNA_CONFIG["study_name"]

    storage_err_streak = 0

    try:
        print("=" * 60)
        print("🚀 TEKO Optuna v5 - PURE VISION + HIGH REPLAY")
        print("=" * 60)
        print(f"Host: {hostname} | SLURM: job={slurm_job}, array={slurm_array}")
        print(f"Study: {study_name}")
        print(f"Target: {OPTUNA_CONFIG['target_total_trials']} trials")
        print(f"Budget: {OPTUNA_CONFIG['max_steps_per_trial']/1e6:.1f}M steps/trial")
        print(f"Replay: {REPLAY_PROBS}")
        print("=" * 60)

        local_trials = 0
        study = load_or_create_study(study_name=study_name, storage=storage)

        print(f"[WORKER] Initial total_trials={_total_trials(study)}")

        while True:
            total_trials_now = _total_trials(study)
            states = _sqlite_state_counts(db_path)

            print(f"[WORKER] TOTAL={total_trials_now}/{OPTUNA_CONFIG['target_total_trials']} | {states} | local={local_trials}")

            if total_trials_now >= OPTUNA_CONFIG["target_total_trials"]:
                print(f"✅ Target reached: {total_trials_now} trials")
                break

            if args.worker_max_trials is not None and local_trials >= args.worker_max_trials:
                print(f"✅ Worker limit reached: {local_trials} trials")
                break

            try:
                print("[WORKER] Starting trial...")
                study.optimize(lambda tr: objective(tr, env), n_trials=1)
                local_trials += 1
                storage_err_streak = 0

            except Exception as e:
                print(f"[WORKER][ERROR] {repr(e)}")

                if _is_retryable_storage_error(e):
                    storage_err_streak += 1
                    sleep_s = min(20.0, 1.0 + random.random() * (2.0 + storage_err_streak))
                    print(f"[WORKER] Retryable - sleeping {sleep_s:.1f}s (streak={storage_err_streak})")
                    time.sleep(sleep_s)

                    storage = make_storage(db_path)
                    try:
                        study = optuna.load_study(study_name=study_name, storage=storage)
                    except Exception:
                        study = load_or_create_study(study_name=study_name, storage=storage)

                    if storage_err_streak >= 30:
                        print("[WORKER][FATAL] Too many storage errors")
                        raise
                    continue

                raise

            if local_trials % 5 == 0:
                try:
                    study = optuna.load_study(study_name=study_name, storage=storage)
                    print(f"[WORKER] Refreshed | total={_total_trials(study)}")
                except Exception as e:
                    print(f"[WORKER][WARN] Refresh failed: {repr(e)}")

    finally:
        env.close()
        sim.close()


def main():
    parser = argparse.ArgumentParser(description="TEKO v5 - Pure Vision + High Replay")
    parser.add_argument("--create-study", action="store_true")
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--worker-max-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.headless = True
    args.enable_cameras = True

    if args.num_trials is not None:
        OPTUNA_CONFIG["target_total_trials"] = int(args.num_trials)

    db_path = OPTUNA_CONFIG["storage_path"]
    storage = make_storage(db_path)

    if args.create_study:
        create_study(OPTUNA_CONFIG["study_name"], storage)
        return

    run_worker(args)


if __name__ == "__main__":
    main()