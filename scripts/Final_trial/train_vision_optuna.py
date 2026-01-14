#!/usr/bin/env python3
"""
TEKO Vision + Attention + YawAux - Optuna NSGA-II (Final v2)
============================================================
Fresh study for thesis comparison graphs.
With TensorBoard logging per trial.

Author: Alexandre Schleier Neves da Silva
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import argparse
import sys
import math
import socket
import time
import random
import csv
from collections import deque
from functools import partial
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed", flush=True)
    sys.exit(1)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from isaaclab.app import AppLauncher
print = partial(print, flush=True)

# =============================================================================
# CONFIGURATION - New study v2 for fresh experiments
# =============================================================================
OPTUNA_CONFIG = {
    "study_name": "teko_vision_final_v2",
    "storage_path": "/home/schux00/optuna/teko_vision_final_v2.log",
    "target_total_trials": 100,
    "max_steps_per_trial": 200_000_000,
    "max_walltime_s_per_trial": 24 * 3600,  # 24h per trial
    "eval_interval": 50_000,
    "pruning_enabled": True,
    "pruning_warmup_steps": 3_000_000,
    "bad_eval_streak_to_prune": 8,
    "min_ssr_thresholds": {0: 0.60, 3: 0.50, 6: 0.40, 10: 0.30},
    "success_surface_xy": 0.03,
}

FIXED_PARAMS = {
    "gamma": 0.99,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "clip_ratio": 0.2,
    "num_envs": 120,
    "rollout_len": 128,
}

IMG_SIZE = 128
ADVANCE_THRESHOLD = 0.75
MIN_STEPS_BEFORE_ADVANCE = 200_000
MAX_STAGE = 41


def get_min_ssr_for_stage(stage):
    thresholds = OPTUNA_CONFIG["min_ssr_thresholds"]
    applicable = 0
    for k in sorted(thresholds.keys()):
        if k <= stage:
            applicable = k
    return thresholds[applicable]


def atanh(x):
    x = torch.clamp(x, -0.999, 0.999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def get_success_flags(env, device):
    if hasattr(env, "_last_success"):
        s = env._last_success
        if isinstance(s, torch.Tensor):
            return s.to(device=device, dtype=torch.bool)
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    return surface_xy.to(device) < OPTUNA_CONFIG["success_surface_xy"]


def make_storage(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        from optuna.storages.journal import JournalFileBackend
        backend = JournalFileBackend(file_path)
        return optuna.storages.JournalStorage(backend)
    except:
        pass
    try:
        from optuna.storages.journal import JournalFileStorage
        backend = JournalFileStorage(file_path)
        return optuna.storages.JournalStorage(backend)
    except:
        pass
    from optuna.storages import JournalFileStorage
    backend = JournalFileStorage(file_path)
    return optuna.storages.JournalStorage(backend)


def create_study(study_name, storage):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize", "maximize"],
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(population_size=20, seed=42),
    )
    print(f"[STUDY] {study_name} ready")
    return study


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))


class ChannelAttention(nn.Module):
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
        return x * self.fc(y).view(b, c, 1, 1)


class VisionEncoderAttentionYaw(nn.Module):
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.channel_attn = ChannelAttention(64)
        self.spatial_attn = SpatialAttention(64)
        self.gn1 = nn.GroupNorm(8, 32)
        self.gn2 = nn.GroupNorm(8, 64)
        self.gn3 = nn.GroupNorm(8, 64)
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, IMG_SIZE, IMG_SIZE)
            flat_size = self._forward_conv(dummy).shape[1]
        
        self.fc = nn.Linear(flat_size, feature_dim)
        self.yaw_head = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(True),
            nn.Linear(64, 32), nn.ReLU(True),
            nn.Linear(32, 1), nn.Tanh()
        )
        self.feature_dim = feature_dim
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
    
    def _forward_conv(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x.flatten(1)
    
    def forward(self, x):
        return F.relu(self.fc(self._forward_conv(x)))
    
    def predict_yaw(self, features):
        return self.yaw_head(features) * math.pi


class VisionIMUAttentionYawPolicy(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
    
    def __init__(self, vis_dim=256, imu_dim=6, hidden=256, action_dim=2):
        super().__init__()
        self.vision_encoder = VisionEncoderAttentionYaw(in_channels=4, feature_dim=vis_dim)
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, 64), nn.ReLU(True),
            nn.Linear(64, 64), nn.ReLU(True),
        )
        fused_dim = vis_dim + 64
        self.actor_head = nn.Sequential(
            nn.Linear(fused_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        priv_dim = 7
        self.critic_head = nn.Sequential(
            nn.Linear(fused_dim + priv_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden // 2), nn.ReLU(True),
            nn.Linear(hidden // 2, 1),
        )
    
    def _std(self):
        return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
    
    def forward_features(self, rgb, imu):
        vis_feat = self.vision_encoder(rgb)
        imu_feat = self.imu_encoder(imu)
        return torch.cat([vis_feat, imu_feat], dim=-1), vis_feat
    
    def act(self, rgb, imu, privileged=None):
        fused, vis_feat = self.forward_features(rgb, imu)
        mean = self.actor_head(fused)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        
        if privileged is not None:
            critic_in = torch.cat([fused, privileged], dim=-1)
        else:
            critic_in = torch.cat([fused, torch.zeros(fused.shape[0], 7, device=fused.device)], dim=-1)
        value = self.critic_head(critic_in).squeeze(-1)
        yaw_pred = self.vision_encoder.predict_yaw(vis_feat)
        return action, log_prob, value, yaw_pred
    
    def evaluate(self, rgb, imu, actions, privileged=None):
        fused, vis_feat = self.forward_features(rgb, imu)
        mean = self.actor_head(fused)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        u = atanh(actions)
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


def ppo_update_with_yaw(policy, optimizer, rgb, imu, actions, old_logp, advantages, returns, 
                         yaw_targets, privileged, epochs, batch_size, clip_ratio, 
                         entropy_coef, value_coef, aux_yaw_coef, max_grad_norm):
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
    
    metrics = {"entropy": 0, "yaw_loss": 0, "policy_loss": 0, "value_loss": 0}
    n_updates = 0
    
    for _ in range(epochs):
        idx = torch.randperm(total, device=device)
        for start in range(0, total, batch_size):
            mb = idx[start:start + batch_size]
            priv_mb = priv_flat[mb] if priv_flat is not None else None
            logp, val, ent, yaw_pred = policy.evaluate(rgb_flat[mb], imu_flat[mb], actions_flat[mb], priv_mb)
            
            ratio = torch.exp(logp - old_logp_flat[mb])
            surr1 = ratio * adv_flat[mb]
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_flat[mb]
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * F.mse_loss(val, ret_flat[mb])
            yaw_loss = F.mse_loss(yaw_pred, yaw_flat[mb])
            
            loss = p_loss + value_coef * v_loss - entropy_coef * ent.mean() + aux_yaw_coef * yaw_loss
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            
            metrics["entropy"] += ent.mean().item()
            metrics["yaw_loss"] += yaw_loss.item()
            metrics["policy_loss"] += p_loss.item()
            metrics["value_loss"] += v_loss.item()
            n_updates += 1
    
    return {k: v / max(n_updates, 1) for k, v in metrics.items()}


def objective(trial, env, base_log_dir):
    # Hyperparameters to optimize
    entropy_coef = trial.suggest_float("entropy_coef", 0.005, 0.03, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.98)
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True)
    epochs = trial.suggest_int("epochs", 3, 8)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    aux_yaw_coef = trial.suggest_float("aux_yaw_coef", 0.1, 0.5)
    
    device = torch.device("cuda:0")
    env.set_curriculum_level(0)
    
    policy = VisionIMUAttentionYawPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # TensorBoard for this trial
    writer = None
    if HAS_TENSORBOARD:
        trial_log_dir = f"{base_log_dir}/trial_{trial.number}"
        os.makedirs(trial_log_dir, exist_ok=True)
        writer = SummaryWriter(trial_log_dir)
    
    num_envs = FIXED_PARAMS["num_envs"]
    rollout_len = FIXED_PARAMS["rollout_len"]
    
    obs_dict, _ = env.reset()
    has_priv = "privileged" in obs_dict
    
    # Buffers
    rgb_buf = torch.zeros((rollout_len, num_envs, 4, IMG_SIZE, IMG_SIZE), device=device)
    imu_buf = torch.zeros((rollout_len, num_envs, 6), device=device)
    actions_buf = torch.zeros((rollout_len, num_envs, 2), device=device)
    rewards_buf = torch.zeros((rollout_len, num_envs), device=device)
    values_buf = torch.zeros((rollout_len, num_envs), device=device)
    logprobs_buf = torch.zeros((rollout_len, num_envs), device=device)
    dones_buf = torch.zeros((rollout_len, num_envs), device=device)
    yaw_targets_buf = torch.zeros((rollout_len, num_envs, 1), device=device)
    priv_buf = torch.zeros((rollout_len, num_envs, 7), device=device) if has_priv else None
    
    ep_rewards = deque(maxlen=300)
    stage_successes = deque(maxlen=300)
    cur_reward = torch.zeros(num_envs, device=device)
    
    step = 0
    current_stage = 0
    max_stage = 0
    best_ssr = 0.0
    last_advance_step = 0
    bad_streak = 0
    t0 = time.time()
    next_eval = OPTUNA_CONFIG["eval_interval"]
    
    try:
        while step < OPTUNA_CONFIG["max_steps_per_trial"] and (time.time() - t0) < OPTUNA_CONFIG["max_walltime_s_per_trial"]:
            for t in range(rollout_len):
                rgb = obs_dict["rgb"].to(device)
                imu = obs_dict["imu"].to(device)
                priv = obs_dict.get("privileged")
                if priv is not None:
                    priv = priv.to(device)
                    yaw_target = priv[:, 3:4]
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
                
                obs_dict, reward, term, trunc, _ = env.step(action)
                done = term | trunc
                
                rewards_buf[t] = reward
                dones_buf[t] = done.float()
                cur_reward += reward
                
                if done.any():
                    done_idx = done.nonzero(as_tuple=False).squeeze(-1)
                    succ = get_success_flags(env, device).float()
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
            
            advantages, returns = compute_gae(rewards_buf, values_buf, dones_buf,
                                              FIXED_PARAMS["gamma"], gae_lambda, last_value)
            
            metrics = ppo_update_with_yaw(
                policy, optimizer, rgb_buf, imu_buf, actions_buf, logprobs_buf,
                advantages, returns, yaw_targets_buf,
                priv_buf if has_priv else None,
                epochs, batch_size, FIXED_PARAMS["clip_ratio"],
                entropy_coef, FIXED_PARAMS["value_coef"], aux_yaw_coef,
                FIXED_PARAMS["max_grad_norm"]
            )
            
            ssr = float(np.mean(stage_successes)) if stage_successes else 0.0
            current_stage = int(env.curriculum_level)
            max_stage = max(max_stage, current_stage)
            best_ssr = max(best_ssr, ssr)
            
            if step >= next_eval:
                elapsed = (time.time() - t0) / 3600
                mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
                
                print(f"[T{trial.number}][{step:,}] S{current_stage:02d} | SSR: {ssr:.1%} | "
                      f"YawL: {metrics['yaw_loss']:.3f} | Ent: {metrics['entropy']:.3f} | "
                      f"MaxS: {max_stage} | {elapsed:.1f}h")
                
                # TensorBoard
                if writer:
                    writer.add_scalar("train/ssr", ssr, step)
                    writer.add_scalar("train/reward", mean_r, step)
                    writer.add_scalar("train/entropy", metrics["entropy"], step)
                    writer.add_scalar("train/yaw_loss", metrics["yaw_loss"], step)
                    writer.add_scalar("curriculum/stage", current_stage, step)
                    writer.add_scalar("curriculum/max_stage", max_stage, step)
                
                next_eval += OPTUNA_CONFIG["eval_interval"]
                
                # Pruning
                if OPTUNA_CONFIG["pruning_enabled"] and step >= OPTUNA_CONFIG["pruning_warmup_steps"]:
                    if ssr < get_min_ssr_for_stage(current_stage):
                        bad_streak += 1
                        if bad_streak >= OPTUNA_CONFIG["bad_eval_streak_to_prune"]:
                            print(f"[PRUNE] Bad streak {bad_streak}")
                            raise optuna.TrialPruned()
                    else:
                        bad_streak = 0
            
            # Curriculum advancement
            if (len(stage_successes) >= 100 and
                ssr >= ADVANCE_THRESHOLD and
                step - last_advance_step >= MIN_STEPS_BEFORE_ADVANCE and
                current_stage < MAX_STAGE):
                print(f"[ADVANCE] S{current_stage} -> S{current_stage + 1} (SSR={ssr:.1%})")
                current_stage += 1
                env.set_curriculum_level(current_stage)
                stage_successes.clear()
                last_advance_step = step
                bad_streak = 0
    
    except optuna.TrialPruned:
        if writer:
            writer.close()
        raise
    except Exception as e:
        print(f"[ERROR] {repr(e)}")
        if writer:
            writer.close()
        raise optuna.TrialPruned()
    finally:
        env.set_curriculum_level(0)
        if writer:
            writer.add_hparams(
                {"lr": learning_rate, "entropy": entropy_coef, "gae_lambda": gae_lambda,
                 "epochs": epochs, "batch_size": batch_size, "aux_yaw": aux_yaw_coef},
                {"hparam/best_ssr": best_ssr, "hparam/max_stage": max_stage}
            )
            writer.close()
    

    # Save checkpoint for good trials
    if max_stage >= 30:
        ckpt_path = f"/home/schux00/checkpoints/optuna_trial{trial.number}_S{max_stage}.pt"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            "trial": trial.number, "max_stage": max_stage, "best_ssr": best_ssr,
            "policy": policy.state_dict(),
            "params": {"lr": learning_rate, "entropy": entropy_coef, "gae_lambda": gae_lambda,
                      "epochs": epochs, "batch_size": batch_size, "aux_yaw": aux_yaw_coef}
        }, ckpt_path)
        print(f"[SAVE] {ckpt_path}")

        
    print(f"[DONE] Trial {trial.number}: SSR={best_ssr:.1%}, MaxStage={max_stage}")
    return best_ssr, float(max_stage)


def run_worker(args):
    torch.backends.cudnn.benchmark = True
    
    # Fixed seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    app = AppLauncher(args)
    sim = app.app
    
    sys.path.insert(0, "/workspace/teko/source/teko")
    from teko.tasks.direct.teko.teko_env_tiled_imu import TekoEnvTiledIMU
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
    
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = FIXED_PARAMS["num_envs"]
    cfg.enable_curriculum = True
    cfg.asymmetric_critic = True
    
    env = TekoEnvTiledIMU(cfg=cfg)
    
    storage = make_storage(OPTUNA_CONFIG["storage_path"])
    study = create_study(OPTUNA_CONFIG["study_name"], storage)
    
    # Base TensorBoard directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = f"/home/schux00/tensorboard/vision_optuna_{timestamp}"
    os.makedirs(base_log_dir, exist_ok=True)
    
    print("=" * 70)
    print("TEKO Vision + Attention + YawAux - Optuna NSGA-II (Final v2)")
    print("=" * 70)
    print(f"Host: {socket.gethostname()} | Envs: {FIXED_PARAMS['num_envs']}")
    print(f"Max Stage: {MAX_STAGE} (180°)")
    print(f"TensorBoard: {base_log_dir}")
    print("=" * 70)
    
    try:
        while len(study.get_trials(deepcopy=False)) < OPTUNA_CONFIG["target_total_trials"]:
            try:
                study.optimize(lambda tr: objective(tr, env, base_log_dir), n_trials=1)
            except Exception as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    time.sleep(2 + random.random() * 3)
                    storage = make_storage(OPTUNA_CONFIG["storage_path"])
                    study = optuna.load_study(study_name=OPTUNA_CONFIG["study_name"], storage=storage)
                else:
                    raise
    finally:
        env.close()
        sim.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-study", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    
    if args.create_study:
        create_study(OPTUNA_CONFIG["study_name"], make_storage(OPTUNA_CONFIG["storage_path"]))
        return
    
    run_worker(args)


if __name__ == "__main__":
    main()