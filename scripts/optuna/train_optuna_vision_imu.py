#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Optuna Vision+IMU Training (NSGA-II) v6
Actor: vision (84x84x4) + IMU (6D)
Critic: vision + IMU + privileged (7D)
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import argparse, sys, math, socket, sqlite3, time, random
from collections import deque
from typing import Dict, Tuple, Optional
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed", flush=True)
    sys.exit(1)

from isaaclab.app import AppLauncher
print = partial(print, flush=True)

OPTUNA_CONFIG = {
    "study_name": "teko_vision_imu_v9",
    "storage_path": "/home/schux00/optuna/teko_vision_imu_v9.db",
    "target_total_trials": 200,
    "max_steps_per_trial": 15_000_000,
    "max_walltime_s_per_trial": 7 * 24 * 3600,
    "eval_interval": 50_000,
    "pruning_enabled": True,
    "pruning_warmup_steps": 2_000_000,
    "bad_eval_streak_to_prune": 8,
    "min_ssr_thresholds": {0: 0.60, 4: 0.50, 8: 0.40, 12: 0.35, 16: 0.30},
    "success_surface_xy": 0.03,
}

FIXED_PARAMS = {"gamma": 0.99, "clip_ratio": 0.2, "value_coef": 0.5, "max_grad_norm": 0.5, "num_envs": 150, "rollout_len": 128}
ADVANCE_THRESHOLD_EARLY, ADVANCE_THRESHOLD_MID, ADVANCE_THRESHOLD_LATE = 0.80, 0.75, 0.70
MIN_STEPS_BEFORE_ADVANCE = 200_000

def get_min_ssr_for_stage(stage):
    thresholds = OPTUNA_CONFIG["min_ssr_thresholds"]
    applicable_key = 0
    for key in sorted(thresholds.keys()):
        if key <= stage: applicable_key = key
    return thresholds[applicable_key]

def get_advance_threshold(stage):
    if stage <= 6: return ADVANCE_THRESHOLD_EARLY
    elif stage <= 12: return ADVANCE_THRESHOLD_MID
    return ADVANCE_THRESHOLD_LATE

def atanh(x):
    x = torch.clamp(x, -0.999, 0.999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def get_success_flags(env, device):
    if hasattr(env, "_last_success"):
        s = getattr(env, "_last_success")
        if isinstance(s, torch.Tensor): return s.to(device=device, dtype=torch.bool)
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    if not isinstance(surface_xy, torch.Tensor): surface_xy = torch.as_tensor(surface_xy, device=device)
    return surface_xy < OPTUNA_CONFIG["success_surface_xy"]

def _init_sqlite(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=120000;")
    conn.commit()
    conn.close()

def make_storage(db_path):
    _init_sqlite(db_path)
    return f"sqlite:///{db_path}"

def create_study(study_name, storage):
    study = optuna.create_study(study_name=study_name, storage=storage, directions=["maximize", "maximize"],
        load_if_exists=True, sampler=optuna.samplers.NSGAIISampler(population_size=20, mutation_prob=0.1, crossover_prob=0.9, seed=42))
    print(f"[STUDY] {study_name} ready | Target: {OPTUNA_CONFIG['target_total_trials']} trials")
    return study

def _is_retryable_error(e):
    if isinstance(e, optuna.exceptions.StorageInternalError): return True
    return any(x in str(e).lower() for x in ["database is locked", "locked", "busy"])

class VisionEncoder(nn.Module):
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 6, 3, 1), nn.GroupNorm(8, 64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.GroupNorm(16, 256), nn.ReLU(True))
        with torch.no_grad():
            flat_dim = int(self.conv(torch.zeros(1, in_channels, 128, 128)).view(1, -1).shape[1])
        self.fc = nn.Sequential(nn.Linear(flat_dim, 512), nn.ReLU(True), nn.Linear(512, feature_dim), nn.ReLU(True))
        self.feature_dim = feature_dim
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))

class VisionIMUPolicy(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
    
    def __init__(self, vision_channels=4, imu_dim=6, privileged_dim=7, action_dim=2, hidden_dim=256):
        super().__init__()
        self.vision_encoder = VisionEncoder(vision_channels, hidden_dim)
        self.imu_encoder = nn.Sequential(nn.Linear(imu_dim, 64), nn.ReLU(True), nn.Linear(64, 64), nn.ReLU(True))
        self.actor_head = nn.Sequential(nn.Linear(hidden_dim + 64, 128), nn.ReLU(True), nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.state_encoder = nn.Sequential(nn.Linear(privileged_dim, 128), nn.ReLU(True), nn.Linear(128, 128), nn.ReLU(True))
        self.critic_head = nn.Sequential(nn.Linear(hidden_dim + 64 + 128, 128), nn.ReLU(True), nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, 1))
        for module in [self.actor_head, self.imu_encoder, self.state_encoder, self.critic_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
    
    def _std(self): return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
    
    def act(self, obs, deterministic=False):
        vision_feat = self.vision_encoder(obs["rgb"])
        imu_feat = self.imu_encoder(obs["imu"])
        actor_input = torch.cat([vision_feat, imu_feat], dim=-1)
        mean = self.actor_head(actor_input)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        u = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(torch.clamp(1.0 - action * action, min=1e-6)).sum(-1)
        if "privileged" in obs:
            state_feat = self.state_encoder(obs["privileged"])
            value = self.critic_head(torch.cat([vision_feat, imu_feat, state_feat], dim=-1)).squeeze(-1)
        else:
            value = torch.zeros(action.shape[0], device=action.device)
        return action, log_prob, value
    
    def evaluate(self, obs, actions):
        vision_feat = self.vision_encoder(obs["rgb"])
        imu_feat = self.imu_encoder(obs["imu"])
        actor_input = torch.cat([vision_feat, imu_feat], dim=-1)
        mean = self.actor_head(actor_input)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        u = atanh(actions)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(torch.clamp(1.0 - actions * actions, min=1e-6)).sum(-1)
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
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values

def ppo_update(policy, optimizer, obs_rgb, obs_imu, obs_priv, actions, old_log_probs, advantages, returns, epochs, batch_size, clip_ratio, entropy_coef, value_coef, max_grad_norm):
    device = next(policy.parameters()).device
    T, N = obs_rgb.shape[:2]
    total = T * N
    rgb_flat, imu_flat = obs_rgb.view(total, 4, 128, 128), obs_imu.view(total, 6)
    act_flat, old_logp_flat = actions.view(total, 2), old_log_probs.view(total)
    adv_flat = (advantages.view(total) - advantages.mean()) / (advantages.std() + 1e-8)
    ret_flat = returns.view(total)
    priv_flat = obs_priv.view(total, -1) if obs_priv is not None else None
    metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "grad_norm": []}
    for _ in range(epochs):
        for start in range(0, total, batch_size):
            mb_idx = torch.randperm(total, device=device)[start:start + batch_size]
            mb_obs = {"rgb": rgb_flat[mb_idx], "imu": imu_flat[mb_idx]}
            if priv_flat is not None: mb_obs["privileged"] = priv_flat[mb_idx]
            log_prob, value, entropy = policy.evaluate(mb_obs, act_flat[mb_idx])
            ratio = torch.exp(log_prob - old_logp_flat[mb_idx])
            surr1, surr2 = ratio * adv_flat[mb_idx], torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_flat[mb_idx]
            loss = -torch.min(surr1, surr2).mean() + value_coef * 0.5 * F.mse_loss(value, ret_flat[mb_idx]) - entropy_coef * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            metrics["grad_norm"].append(nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm).item())
            optimizer.step()
            metrics["policy_loss"].append(-torch.min(surr1, surr2).mean().item())
            metrics["value_loss"].append(F.mse_loss(value, ret_flat[mb_idx]).item())
            metrics["entropy"].append(entropy.mean().item())
    return {k: np.mean(v) for k, v in metrics.items()}

def objective(trial, env):
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.02, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.98)
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True)
    epochs = trial.suggest_int("epochs", 3, 8)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    device = torch.device("cuda:0")
    env.set_curriculum_level(0)
    policy = VisionIMUPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    step, t0, num_envs, rollout_len = 0, time.time(), FIXED_PARAMS["num_envs"], FIXED_PARAMS["rollout_len"]
    ep_rewards, stage_successes = deque(maxlen=200), deque(maxlen=300)
    cur_reward, cur_length = torch.zeros(num_envs, device=device), torch.zeros(num_envs, dtype=torch.int32, device=device)
    obs_dict, _ = env.reset()
    has_privileged = "privileged" in obs_dict
    best_ssr, max_stage, next_eval, bad_eval_streak, last_stage_change_step = 0.0, 0, OPTUNA_CONFIG["eval_interval"], 0, 0
    obs_rgb_u8 = torch.empty((rollout_len, num_envs, 4, 128, 128), device=device, dtype=torch.uint8)
    obs_imu = torch.empty((rollout_len, num_envs, 6), device=device, dtype=torch.float32)
    obs_priv = torch.empty((rollout_len, num_envs, 7), device=device, dtype=torch.float32) if has_privileged else None
    actions = torch.empty((rollout_len, num_envs, 2), device=device, dtype=torch.float32)
    rewards, values, log_probs, dones = [torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32) for _ in range(4)]
    try:
        while step < OPTUNA_CONFIG["max_steps_per_trial"] and (time.time() - t0) < OPTUNA_CONFIG["max_walltime_s_per_trial"]:
            for t in range(rollout_len):
                vision_f32, imu_f32 = obs_dict["rgb"].to(device=device, dtype=torch.float32), obs_dict["imu"].to(device=device, dtype=torch.float32)
                obs = {"rgb": vision_f32, "imu": imu_f32}
                if has_privileged: obs["privileged"] = obs_dict["privileged"].to(device=device, dtype=torch.float32)
                with torch.no_grad(): action, log_prob, value = policy.act(obs)
                obs_rgb_u8[t].copy_((vision_f32.clamp(0.0, 1.0) * 255.0).to(torch.uint8))
                obs_imu[t].copy_(imu_f32)
                if has_privileged: obs_priv[t].copy_(obs["privileged"])
                actions[t].copy_(action); log_probs[t].copy_(log_prob); values[t].copy_(value)
                obs_dict, reward, term, trunc, _ = env.step(action)
                rewards[t].copy_(reward); dones[t].copy_((term | trunc).float())
                cur_reward += reward; cur_length += 1
                if (term | trunc).any():
                    done_idx = (term | trunc).nonzero(as_tuple=False).squeeze(-1)
                    with torch.no_grad(): succ = get_success_flags(env, device=device).float()
                    ep_rewards.extend(cur_reward[done_idx].cpu().tolist())
                    stage_successes.extend(succ[done_idx].cpu().tolist())
                    cur_reward[done_idx] = 0.0; cur_length[done_idx] = 0
                step += num_envs
            with torch.no_grad():
                last_obs = {"rgb": obs_dict["rgb"].to(device=device, dtype=torch.float32), "imu": obs_dict["imu"].to(device=device, dtype=torch.float32)}
                if has_privileged: last_obs["privileged"] = obs_dict["privileged"].to(device=device, dtype=torch.float32)
                _, _, last_value = policy.act(last_obs)
            adv, ret = compute_gae(rewards, values, dones, FIXED_PARAMS["gamma"], gae_lambda, last_value)
            metrics = ppo_update(policy, optimizer, obs_rgb_u8.to(dtype=torch.float32) / 255.0, obs_imu, obs_priv, actions, log_probs, adv, ret, epochs, batch_size, FIXED_PARAMS["clip_ratio"], entropy_coef, FIXED_PARAMS["value_coef"], FIXED_PARAMS["max_grad_norm"])
            ssr = float(np.mean(stage_successes)) if stage_successes else 0.0
            current_stage = int(env.curriculum_level)
            max_stage, best_ssr = max(max_stage, current_stage), max(best_ssr, ssr)
            if step >= next_eval:
                print(f"[{step:,}] S{current_stage:02d} | SSR: {ssr:.1%} | R: {np.mean(ep_rewards) if ep_rewards else 0:.1f} | MaxS: {max_stage} | Ent: {metrics['entropy']:.3f} | {(time.time()-t0)/3600:.1f}h")
                next_eval += OPTUNA_CONFIG["eval_interval"]
                if OPTUNA_CONFIG["pruning_enabled"] and step >= OPTUNA_CONFIG["pruning_warmup_steps"]:
                    if ssr < get_min_ssr_for_stage(current_stage):
                        bad_eval_streak += 1
                        if bad_eval_streak >= OPTUNA_CONFIG["bad_eval_streak_to_prune"]:
                            print(f"[PRUNE] SSR {ssr:.1%} < threshold for {bad_eval_streak} evals"); raise optuna.TrialPruned()
                    else: bad_eval_streak = 0
            if len(stage_successes) >= 100 and ssr >= get_advance_threshold(current_stage) and step - last_stage_change_step >= MIN_STEPS_BEFORE_ADVANCE and current_stage < 27:
                env.set_curriculum_level(current_stage + 1); obs_dict, _ = env.reset()
                cur_reward.zero_(); cur_length.zero_(); stage_successes.clear(); last_stage_change_step = step; bad_eval_streak = 0
                print(f"[ADVANCE] S{current_stage} -> S{current_stage + 1} (SSR={ssr:.1%})")
    except (torch.cuda.OutOfMemoryError, optuna.TrialPruned): torch.cuda.empty_cache(); env.set_curriculum_level(0); env.reset(); raise optuna.TrialPruned()
    except Exception as e: print(f"[ERROR] {repr(e)}"); torch.cuda.empty_cache(); env.set_curriculum_level(0); env.reset(); raise optuna.TrialPruned()
    env.set_curriculum_level(0)
    print(f"[DONE] Trial {trial.number}: SSR={best_ssr:.1%}, MaxStage={max_stage}, Time={(time.time()-t0)/3600:.1f}h")
    return best_ssr, float(max_stage)

def run_worker(args):
    torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True
    if args.seed: torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    app = AppLauncher(args); sim = app.app
    sys.path.insert(0, "/workspace/teko/source/teko")
    from teko.tasks.direct.teko.teko_env_tiled_imu import TekoEnvTiledIMU
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
    cfg = TekoEnvCfg(); cfg.scene.num_envs = FIXED_PARAMS["num_envs"]; cfg.enable_curriculum = True
    env = TekoEnvTiledIMU(cfg=cfg)
    storage = make_storage(OPTUNA_CONFIG["storage_path"]); study = create_study(OPTUNA_CONFIG["study_name"], storage)
    print("=" * 70 + f"\nTEKO Vision+IMU Training v6\nHost: {socket.gethostname()} | Envs: {FIXED_PARAMS['num_envs']}\nActor: Vision + IMU (6D) | Critic: + Privileged (7D)\n" + "=" * 70)
    try:
        while len(study.get_trials(deepcopy=False)) < OPTUNA_CONFIG["target_total_trials"]:
            try: study.optimize(lambda tr: objective(tr, env), n_trials=1)
            except Exception as e:
                if _is_retryable_error(e): time.sleep(2 + random.random() * 3); storage = make_storage(OPTUNA_CONFIG["storage_path"]); study = optuna.load_study(study_name=OPTUNA_CONFIG["study_name"], storage=storage)
                else: raise
    finally: env.close(); sim.close()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--create-study", action="store_true"); parser.add_argument("--seed", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser); args = parser.parse_args(); args.headless = True; args.enable_cameras = True
    if args.create_study: create_study(OPTUNA_CONFIG["study_name"], make_storage(OPTUNA_CONFIG["storage_path"])); return
    run_worker(args)

if __name__ == "__main__": main()
