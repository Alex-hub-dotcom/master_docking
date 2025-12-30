#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Optuna PPO v5 State-Based - Debug/Validation
==================================================

Structurally identical to v5 vision but uses ground-truth state.
Purpose: Validate curriculum + rewards without vision bottleneck.

If this reaches S20+ easily, the problem is vision.
If this also gets stuck at S5-6, the problem is curriculum/rewards.

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
# CONFIG - STATE-BASED (HIGH ENV COUNT)
# =============================================================================

OPTUNA_CONFIG = {
    "study_name": "teko_nsgaii_v5_state_v5",
    "storage_path": "/home/schux00/optuna/teko_nsgaii_v5_state_v5.db",

    "target_total_trials": 100,

    "max_steps_per_trial": 15_000_000,
    "max_walltime_s_per_trial": 10_800,  # 3h (state is faster)

    "eval_interval": 50_000,

    "pruning_enabled": True,
    "pruning_warmup_steps": 500_000,
    "bad_eval_streak_to_prune": 8,

    "min_ssr_thresholds": {
        0: 0.80,   # State should be much better
        2: 0.75,
        4: 0.70,
        7: 0.65,
        10: 0.60,
        15: 0.55,
        20: 0.50,
    },

    "success_surface_xy": 0.03,
}

FIXED_PARAMS = {
    "gamma": 0.99,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,

    "num_envs": 256,  # Much more envs for state-based
    "rollout_len": 64,
}

REWARD_OVERRIDES = {
    "alignment_scale": 0.40,
    "turning_bonus": 0.60,
    "facing_threshold_deg": 15.0,
    "progress_scale": 10.0,
}

REPLAY_PROBS = {
    "forward": 0.0,
    "yaw_only": 0.30,
    "combined": 0.40,
    "extreme": 0.50,
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

    print("[WARN] Fallback success detection!")
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
    print(f"✅ Study ready: {study_name}")
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
# STATE-BASED MLP POLICY
# =============================================================================

class StatePolicy(nn.Module):
    """
    Simple MLP for state-based control.
    Input: privileged state [dx, dy, dz, yaw_err, vx, vy, w] (7D)
    Output: actions [v, omega] (2D)
    """
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5

    def __init__(self, state_dim: int = 7, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def _std(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))

    def act(self, state: torch.Tensor, deterministic: bool = False):
        mean = self.actor(state)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        u = dist.mean if deterministic else dist.rsample()
        a = torch.tanh(u)

        log_prob_u = dist.log_prob(u).sum(-1)
        log_det = torch.log(torch.clamp(1.0 - a * a, min=1e-6)).sum(-1)
        log_prob = log_prob_u - log_det

        value = self.critic(state).squeeze(-1)

        return a, log_prob, value

    def evaluate(self, state: torch.Tensor, actions: torch.Tensor):
        mean = self.actor(state)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        # Inverse tanh
        actions_clamped = torch.clamp(actions, -0.999, 0.999)
        u = 0.5 * (torch.log1p(actions_clamped) - torch.log1p(-actions_clamped))

        log_prob_u = dist.log_prob(u).sum(-1)
        log_det = torch.log(torch.clamp(1.0 - actions * actions, min=1e-6)).sum(-1)
        log_prob = log_prob_u - log_det

        entropy = dist.entropy().sum(-1)
        value = self.critic(state).squeeze(-1)

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
    policy: StatePolicy,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
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
    T, N = states.shape[:2]
    total = T * N
    state_dim = states.shape[-1]

    states_flat = states.view(total, state_dim)
    act_flat = actions.view(total, 2)
    old_logp_flat = old_log_probs.view(total)
    adv_flat = advantages.view(total)
    ret_flat = returns.view(total)

    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    indices = torch.randperm(total, device=device)

    for _ in range(epochs):
        for start in range(0, total, batch_size):
            mb_idx = indices[start:start + batch_size]

            mb_states = states_flat[mb_idx]
            mb_actions = act_flat[mb_idx]
            mb_old_logp = old_logp_flat[mb_idx]
            mb_adv = adv_flat[mb_idx]
            mb_ret = ret_flat[mb_idx]

            log_prob, value, entropy = policy.evaluate(mb_states, mb_actions)

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
    """Train state-based PPO and return (best_ssr, max_stage)."""

    entropy_coef = trial.suggest_float("entropy_coef", 0.004, 0.015)
    gae_lambda = trial.suggest_float("gae_lambda", 0.92, 0.98)
    clip_ratio = trial.suggest_categorical("clip_ratio", [0.20, 0.25, 0.30])
    epochs = trial.suggest_int("epochs", 4, 8)
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4096, 8192])

    device = torch.device("cuda:0")
    env.set_curriculum_level(0)

    policy = StatePolicy(state_dim=7, action_dim=2, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    step = 0
    max_steps = OPTUNA_CONFIG["max_steps_per_trial"]
    max_wall_s = OPTUNA_CONFIG["max_walltime_s_per_trial"]
    t0 = time.time()

    eval_interval = OPTUNA_CONFIG["eval_interval"]
    rollout_len = FIXED_PARAMS["rollout_len"]
    num_envs = FIXED_PARAMS["num_envs"]

    ep_rewards = deque(maxlen=500)
    stage_successes = deque(maxlen=500)

    cur_reward = torch.zeros(num_envs, device=device)
    cur_length = torch.zeros(num_envs, dtype=torch.int32, device=device)

    obs_dict, _ = env.reset()

    best_ssr = 0.0
    max_stage = 0
    next_eval = eval_interval
    bad_eval_streak = 0

    last_stage_change_step = 0
    min_steps_before_advance = 500_000  # ~2-3 min per stage minimum  # Faster for state-based

    # Rollout buffers
    states = torch.empty((rollout_len, num_envs, 7), device=device, dtype=torch.float32)
    actions = torch.empty((rollout_len, num_envs, 2), device=device, dtype=torch.float32)
    rewards = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    values = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    log_probs = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    dones = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)

    try:
        while step < max_steps and (time.time() - t0) < max_wall_s:
            for t in range(rollout_len):
                state = obs_dict["privileged"].to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    action, log_prob, value = policy.act(state)

                states[t].copy_(state)
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
                last_state = obs_dict["privileged"].to(device=device, dtype=torch.float32)
                last_value = policy.critic(last_state).squeeze(-1)

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
                states=states,
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

            # Curriculum advancement - require CONSISTENT performance
            advance_thr = 0.80 if current_stage < 10 else 0.75

            # Track consecutive good evaluations
            if not hasattr(objective, '_consec_good_evals'):
                objective._consec_good_evals = 0
            
            if ssr >= advance_thr:
                objective._consec_good_evals += 1
            else:
                objective._consec_good_evals = 0
            
            required_consecutive = 5  # Need 5 consecutive good evals
            
            if (
                len(stage_successes) >= 100
                and ssr >= advance_thr
                and objective._consec_good_evals >= required_consecutive
                and (step - last_stage_change_step) >= min_steps_before_advance
                and current_stage < 8
            ):
                objective._consec_good_evals = 0  # Reset for next stage
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
        import traceback
        traceback.print_exc()
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
    
    # Use the VISION environment but only use privileged observations
    from teko.tasks.direct.teko.teko_env_state_debug import TekoEnvStateDebug
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    from teko.tasks.direct.teko.rewards.reward_functions import REWARD_CONFIG
    REWARD_CONFIG.update(REWARD_OVERRIDES)
    print(f"📊 Reward overrides: {REWARD_OVERRIDES}")

    cfg = TekoEnvCfg()
    cfg.scene.num_envs = FIXED_PARAMS["num_envs"]
    cfg.enable_curriculum = True
    cfg.asymmetric_critic = True  # This gives us privileged obs

    env = TekoEnvStateDebug(cfg=cfg)

    if not hasattr(env, "_last_success") and not hasattr(env, "get_last_success"):
        print("⚠️  WARNING: env missing _last_success!")

    import teko.tasks.direct.teko.curriculum.curriculum_manager as cm
    cm.REPLAY_PROBS = REPLAY_PROBS
    print(f"📊 Replay probs: {REPLAY_PROBS}")

    db_path = OPTUNA_CONFIG["storage_path"]
    storage = make_storage(db_path)
    study_name = OPTUNA_CONFIG["study_name"]

    storage_err_streak = 0

    try:
        print("=" * 60)
        print("🚀 TEKO Optuna v5 - STATE-BASED (Debug)")
        print("=" * 60)
        print(f"Host: {hostname} | SLURM: job={slurm_job}, array={slurm_array}")
        print(f"Study: {study_name}")
        print(f"Envs: {FIXED_PARAMS['num_envs']} (state-based)")
        print(f"Target: {OPTUNA_CONFIG['target_total_trials']} trials")
        print(f"Budget: {OPTUNA_CONFIG['max_steps_per_trial']/1e6:.1f}M steps/trial")
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
                    print(f"[WORKER] Retryable - sleeping {sleep_s:.1f}s")
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
    parser = argparse.ArgumentParser(description="TEKO v5 State-Based Debug")
    parser.add_argument("--create-study", action="store_true")
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--worker-max-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.headless = True
    args.enable_cameras = True  # Still needed for env init

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
