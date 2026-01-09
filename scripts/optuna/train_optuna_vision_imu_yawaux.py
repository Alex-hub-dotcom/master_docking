#!/usr/bin/env python3
"""
TEKO Vision + IMU + Auxiliary Yaw Head (v10)
============================================
- Vision: 128x128 grayscale, 4-frame stack
- IMU: 6D (vx, vy, vz, wx, wy, wz)
- Aux Yaw Head: supervised yaw prediction from vision features (bias orientation learning)

IMPORTANT:
- Uses Optuna JournalStorage (file-based) to avoid SQLAlchemy / RDBStorage issues in Isaac Sim.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import argparse
import sys
import math
import socket
import time
import random
from collections import deque
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

# =============================================================================
# CONFIG
# =============================================================================
OPTUNA_CONFIG = {
    "study_name": "teko_vision_imu_yawaux_v10",
    # JournalStorage log file (NOT sqlite .db)
    "storage_path": "/home/schux00/optuna/teko_vision_imu_yawaux_v10.log",
    "target_total_trials": 200,

    "max_steps_per_trial": 15_000_000,
    "max_walltime_s_per_trial": 14400,   # 4h
    "eval_interval": 38_400,

    "pruning_enabled": True,
    "pruning_warmup_steps": 2_000_000,
    "bad_eval_streak_to_prune": 8,
    "min_ssr_thresholds": {0: 0.60, 3: 0.50, 6: 0.40, 10: 0.30},

    "success_surface_xy": 0.03,
}

FIXED_PARAMS = {
    "gamma": 0.99,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "clip_ratio": 0.2,
    "num_envs": 150,
    "rollout_len": 128,
}

IMG_SIZE = 128
NUM_FRAMES = 4
IMU_DIM = 6
PRIVILEGED_DIM = 7

# Which privileged index corresponds to yaw_error in YOUR env
# (you had [:, 3] before, keep it as default but make it explicit)
YAW_TARGET_INDEX = 3
# If your yaw_error is normalized (e.g., in [-1,1]), set this to math.pi
YAW_TARGET_SCALE = 1.0

# Debug print to verify yaw targets are non-zero (very useful to confirm aux head is active)
DEBUG_YAW_TARGETS = True

ADVANCE_THRESHOLD_EARLY = 0.80
ADVANCE_THRESHOLD_MID = 0.75
ADVANCE_THRESHOLD_LATE = 0.70
MIN_STEPS_BEFORE_ADVANCE = 200_000


def get_advance_threshold(stage: int) -> float:
    if stage <= 6:
        return ADVANCE_THRESHOLD_EARLY
    elif stage <= 12:
        return ADVANCE_THRESHOLD_MID
    return ADVANCE_THRESHOLD_LATE


def get_min_ssr_for_stage(stage: int) -> float:
    thresholds = OPTUNA_CONFIG["min_ssr_thresholds"]
    applicable = 0
    for k in sorted(thresholds.keys()):
        if k <= stage:
            applicable = k
    return thresholds[applicable]


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999, 0.999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _as_bool_tensor(x, device):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.bool)
    try:
        return torch.as_tensor(x, device=device, dtype=torch.bool)
    except Exception:
        return None


def get_success_flags(env, device):
    if hasattr(env, "_last_success"):
        s = _as_bool_tensor(env._last_success, device)
        if s is not None:
            return s
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    return surface_xy.to(device) < OPTUNA_CONFIG["success_surface_xy"]


# =============================================================================
# STORAGE (NO SQLAlchemy)
# =============================================================================
def make_storage(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Optuna v4: JournalFileBackend
    try:
        from optuna.storages.journal import JournalFileBackend
        backend = JournalFileBackend(file_path)
        return optuna.storages.JournalStorage(backend)
    except Exception:
        pass

    # Optuna v3.x: JournalFileStorage often lives in optuna.storages.journal
    try:
        from optuna.storages.journal import JournalFileStorage
        backend = JournalFileStorage(file_path)
        return optuna.storages.JournalStorage(backend)
    except Exception:
        pass

    # Older fallback: direct import
    from optuna.storages import JournalFileStorage
    backend = JournalFileStorage(file_path)
    return optuna.storages.JournalStorage(backend)


def create_study(study_name, storage):
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
    print(f"[STUDY] {study_name} ready | Target: {OPTUNA_CONFIG['target_total_trials']} trials")
    return study


def _is_retryable_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(x in msg for x in ["locked", "busy", "operational", "ioerror", "i/o error", "resource temporarily unavailable"])


# =============================================================================
# NETWORKS
# =============================================================================
class VisionEncoder(nn.Module):
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 3, 1),
            nn.GroupNorm(8, 32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 5, 2, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.GroupNorm(16, 256),
            nn.ReLU(True),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, in_channels, IMG_SIZE, IMG_SIZE)).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, feature_dim),
            nn.ReLU(True),
        )
        self.feature_dim = feature_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class VisionIMUYawAuxPolicy(nn.Module):
    """Vision + IMU + Auxiliary Yaw Head."""
    LOG_STD_MAX = 0.5

    def __init__(
        self,
        vision_channels=4,
        imu_dim=6,
        privileged_dim=7,
        action_dim=2,
        hidden_dim=256,
        log_std_min=-1.0,   # IMPORTANT: less entropy collapse than -2.0
    ):
        super().__init__()
        self.LOG_STD_MIN = float(log_std_min)

        self.vision_encoder = VisionEncoder(vision_channels, hidden_dim)

        # IMU encoder
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
        )

        # Actor: vision + IMU
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # Critic: vision + IMU + privileged
        self.state_encoder = nn.Sequential(
            nn.Linear(privileged_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 128, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )

        # AUX yaw head: from vision features only
        self.yaw_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        self._init_heads()

    def _init_heads(self):
        for module in [self.actor_head, self.imu_encoder, self.state_encoder, self.critic_head, self.yaw_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.yaw_head[-2].weight, gain=0.1)

    def _std(self):
        return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))

    def act(self, obs, deterministic=False):
        vision_feat = self.vision_encoder(obs["rgb"])
        imu_feat = self.imu_encoder(obs["imu"])
        actor_input = torch.cat([vision_feat, imu_feat], dim=-1)

        mean = self.actor_head(actor_input)
        std = self._std().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        u = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(torch.clamp(1 - action * action, min=1e-6)).sum(-1)

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
        log_prob = dist.log_prob(u).sum(-1) - torch.log(torch.clamp(1 - actions * actions, min=1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)

        if "privileged" in obs:
            state_feat = self.state_encoder(obs["privileged"])
            value = self.critic_head(torch.cat([vision_feat, imu_feat, state_feat], dim=-1)).squeeze(-1)
        else:
            value = torch.zeros(actions.shape[0], device=actions.device)

        yaw_pred = self.yaw_head(vision_feat).squeeze(-1) * math.pi
        return log_prob, value, entropy, yaw_pred


# =============================================================================
# GAE / PPO
# =============================================================================
def compute_gae(rewards, values, dones, gamma, lam, last_value):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def ppo_update_with_yaw_aux(
    policy,
    optimizer,
    obs_rgb_u8,
    obs_imu,
    obs_priv,
    actions,
    old_log_probs,
    advantages,
    returns,
    yaw_targets,          # can be None
    epochs,
    batch_size,
    clip_ratio,
    entropy_coef,
    value_coef,
    aux_yaw_coef,
    max_grad_norm,
):
    device = next(policy.parameters()).device
    T, N = obs_rgb_u8.shape[:2]
    total = T * N

    rgb_flat = obs_rgb_u8.view(total, NUM_FRAMES, IMG_SIZE, IMG_SIZE)
    imu_flat = obs_imu.view(total, IMU_DIM)
    priv_flat = obs_priv.view(total, PRIVILEGED_DIM) if obs_priv is not None else None
    act_flat = actions.view(total, 2)
    old_logp_flat = old_log_probs.view(total)
    adv_flat = advantages.view(total)
    ret_flat = returns.view(total)

    yaw_flat = yaw_targets.view(total) if yaw_targets is not None else None

    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    indices = torch.randperm(total, device=device)

    total_aux_loss = 0.0
    num_updates = 0
    last_entropy = 0.0

    use_aux = (yaw_flat is not None) and (aux_yaw_coef > 0.0)

    for _ in range(epochs):
        for start in range(0, total, batch_size):
            mb_idx = indices[start : start + batch_size]

            mb_rgb = rgb_flat[mb_idx].float() / 255.0
            mb_obs = {"rgb": mb_rgb, "imu": imu_flat[mb_idx]}
            if priv_flat is not None:
                mb_obs["privileged"] = priv_flat[mb_idx]

            log_prob, value, entropy, yaw_pred = policy.evaluate(mb_obs, act_flat[mb_idx])

            ratio = torch.exp(log_prob - old_logp_flat[mb_idx])
            surr1 = ratio * adv_flat[mb_idx]
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_flat[mb_idx]
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * F.mse_loss(value, ret_flat[mb_idx])

            if use_aux:
                yaw_aux_loss = F.mse_loss(yaw_pred, yaw_flat[mb_idx])
            else:
                yaw_aux_loss = torch.zeros((), device=device)

            loss = p_loss + value_coef * v_loss - entropy_coef * entropy.mean() + aux_yaw_coef * yaw_aux_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_aux_loss += float(yaw_aux_loss.item())
            last_entropy = float(entropy.mean().item())
            num_updates += 1

    return {"yaw_aux_loss": total_aux_loss / max(num_updates, 1), "entropy": last_entropy}


# =============================================================================
# OBJECTIVE
# =============================================================================
def objective(trial, env):
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.02, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.98)
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True)
    epochs = trial.suggest_int("epochs", 3, 8)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    aux_yaw_coef = trial.suggest_float("aux_yaw_coef", 0.1, 1.0)

    device = torch.device("cuda:0")
    env.set_curriculum_level(0)

    policy = VisionIMUYawAuxPolicy(
        NUM_FRAMES, IMU_DIM, PRIVILEGED_DIM, 2, 256,
        log_std_min=-1.0,  # <-- keep this
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    num_envs = FIXED_PARAMS["num_envs"]
    rollout_len = FIXED_PARAMS["rollout_len"]

    obs_dict, _ = env.reset()
    has_priv = "privileged" in obs_dict
    has_imu = "imu" in obs_dict

    if not has_imu:
        print("[ERROR] Environment doesn't provide IMU!")
        raise optuna.TrialPruned()

    if not has_priv:
        print("[WARN] No privileged observations found: yaw aux head will be DISABLED (aux_yaw_coef=0).")
        aux_yaw_coef = 0.0

    # Buffers
    obs_rgb_u8 = torch.empty((rollout_len, num_envs, NUM_FRAMES, IMG_SIZE, IMG_SIZE), device=device, dtype=torch.uint8)
    obs_imu = torch.empty((rollout_len, num_envs, IMU_DIM), device=device, dtype=torch.float32)
    obs_priv = torch.empty((rollout_len, num_envs, PRIVILEGED_DIM), device=device, dtype=torch.float32) if has_priv else None

    yaw_targets = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32) if has_priv else None

    actions = torch.empty((rollout_len, num_envs, 2), device=device, dtype=torch.float32)
    rewards = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    values = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    log_probs = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)
    dones = torch.empty((rollout_len, num_envs), device=device, dtype=torch.float32)

    cur_reward = torch.zeros(num_envs, device=device)
    ep_rewards = deque(maxlen=200)
    stage_successes = deque(maxlen=300)

    step = 0
    max_steps = OPTUNA_CONFIG["max_steps_per_trial"]
    t0 = time.time()
    max_wall = OPTUNA_CONFIG["max_walltime_s_per_trial"]

    best_ssr = 0.0
    max_stage = 0
    next_eval = OPTUNA_CONFIG["eval_interval"]
    bad_streak = 0
    last_stage_step = 0

    try:
        while step < max_steps and (time.time() - t0) < max_wall:
            for t in range(rollout_len):
                rgb_f32 = obs_dict["rgb"].to(device, dtype=torch.float32)
                imu_f32 = obs_dict["imu"].to(device, dtype=torch.float32)

                obs = {"rgb": rgb_f32, "imu": imu_f32}
                if has_priv:
                    obs["privileged"] = obs_dict["privileged"].to(device, dtype=torch.float32)

                with torch.no_grad():
                    action, log_prob, value = policy.act(obs)

                obs_rgb_u8[t].copy_((rgb_f32.clamp(0, 1) * 255).to(torch.uint8))
                obs_imu[t].copy_(imu_f32)

                if has_priv:
                    obs_priv[t].copy_(obs["privileged"])
                    # yaw target
                    yaw_targets[t].copy_(obs["privileged"][:, YAW_TARGET_INDEX] * float(YAW_TARGET_SCALE))

                actions[t].copy_(action)
                log_probs[t].copy_(log_prob)
                values[t].copy_(value)

                obs_dict, reward, term, trunc, _ = env.step(action)
                done = term | trunc

                rewards[t].copy_(reward)
                dones[t].copy_(done.float())
                cur_reward += reward

                if done.any():
                    done_idx = done.nonzero(as_tuple=False).squeeze(-1)
                    succ = get_success_flags(env, device).float()
                    ep_rewards.extend(cur_reward[done_idx].cpu().tolist())
                    stage_successes.extend(succ[done_idx].cpu().tolist())
                    cur_reward[done_idx] = 0

                step += num_envs

            # GAE
            with torch.no_grad():
                last_rgb = obs_dict["rgb"].to(device, dtype=torch.float32)
                last_imu = obs_dict["imu"].to(device, dtype=torch.float32)
                last_obs = {"rgb": last_rgb, "imu": last_imu}
                if has_priv:
                    last_obs["privileged"] = obs_dict["privileged"].to(device, dtype=torch.float32)
                _, _, last_value = policy.act(last_obs)

            adv, ret = compute_gae(rewards, values, dones, FIXED_PARAMS["gamma"], gae_lambda, last_value)

            stats = ppo_update_with_yaw_aux(
                policy,
                optimizer,
                obs_rgb_u8,
                obs_imu,
                obs_priv,
                actions,
                log_probs,
                adv,
                ret,
                yaw_targets,
                epochs,
                batch_size,
                FIXED_PARAMS["clip_ratio"],
                entropy_coef,
                FIXED_PARAMS["value_coef"],
                aux_yaw_coef,
                FIXED_PARAMS["max_grad_norm"],
            )

            ssr = float(np.mean(stage_successes)) if stage_successes else 0.0
            current_stage = int(env.curriculum_level)
            max_stage = max(max_stage, current_stage)
            best_ssr = max(best_ssr, ssr)

            if step >= next_eval:
                mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
                elapsed = (time.time() - t0) / 3600

                extra = ""
                if has_priv and DEBUG_YAW_TARGETS and yaw_targets is not None:
                    y = yaw_targets.reshape(-1)
                    extra = f" | yaw_tgt abs_mean={y.abs().mean().item():.4f} std={y.std().item():.4f} min={y.min().item():+.4f} max={y.max().item():+.4f}"

                print(
                    f"[{step:,}] S{current_stage:02d} | SSR: {ssr:.1%} | R: {mean_r:.1f} | "
                    f"YawL: {stats['yaw_aux_loss']:.3f} | Ent: {stats['entropy']:.3f} | "
                    f"MaxS: {max_stage} | {elapsed:.1f}h{extra}"
                )

                next_eval += OPTUNA_CONFIG["eval_interval"]

                # Pruning
                if OPTUNA_CONFIG["pruning_enabled"] and step >= OPTUNA_CONFIG["pruning_warmup_steps"]:
                    min_ssr = get_min_ssr_for_stage(current_stage)
                    if ssr < min_ssr:
                        bad_streak += 1
                        if bad_streak >= OPTUNA_CONFIG["bad_eval_streak_to_prune"]:
                            print(f"[PRUNE] Bad streak {bad_streak}")
                            raise optuna.TrialPruned()
                    else:
                        bad_streak = 0

            # Curriculum advancement
            thr = get_advance_threshold(current_stage)
            if (
                len(stage_successes) >= 80
                and ssr >= thr
                and (step - last_stage_step) >= MIN_STEPS_BEFORE_ADVANCE
                and current_stage < 27
            ):
                print(f"[ADVANCE] S{current_stage} -> S{current_stage + 1} (SSR={ssr:.1%})")
                env.set_curriculum_level(current_stage + 1)
                stage_successes.clear()
                last_stage_step = step
                bad_streak = 0

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"[ERROR] {repr(e)}")
        raise optuna.TrialPruned()
    finally:
        env.set_curriculum_level(0)

    print(f"[DONE] Trial {trial.number}: SSR={best_ssr:.1%}, MaxStage={max_stage}")
    return best_ssr, float(max_stage)


# =============================================================================
# WORKER
# =============================================================================
def run_worker(args):
    torch.backends.cudnn.benchmark = True
    app = AppLauncher(args)
    sim = app.app

    sys.path.insert(0, "/workspace/teko/source/teko")
    from teko.tasks.direct.teko.teko_env_tiled_imu import TekoEnvTiledIMU as TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    cfg = TekoEnvCfg()
    cfg.scene.num_envs = FIXED_PARAMS["num_envs"]
    cfg.tiled_camera.width = IMG_SIZE
    cfg.tiled_camera.height = IMG_SIZE
    cfg.enable_curriculum = True
    cfg.asymmetric_critic = True

    env = TekoEnv(cfg=cfg)

    storage = make_storage(OPTUNA_CONFIG["storage_path"])
    study = create_study(OPTUNA_CONFIG["study_name"], storage)

    print("=" * 70)
    print("TEKO Vision + IMU + Auxiliary Yaw Head (v10)")
    print("=" * 70)
    print(f"Host: {socket.gethostname()} | Envs: {FIXED_PARAMS['num_envs']}")
    print("Actor: Vision + IMU | Aux: Yaw Head (supervised)")
    print(f"Storage: JournalStorage -> {OPTUNA_CONFIG['storage_path']}")
    print("=" * 70)

    try:
        while len(study.get_trials(deepcopy=False)) < OPTUNA_CONFIG["target_total_trials"]:
            try:
                study.optimize(lambda tr: objective(tr, env), n_trials=1)
            except Exception as e:
                if _is_retryable_error(e):
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
    parser.add_argument("--seed", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.headless = True
    args.enable_cameras = True

    if args.create_study:
        create_study(OPTUNA_CONFIG["study_name"], make_storage(OPTUNA_CONFIG["storage_path"]))
        return

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    run_worker(args)


if __name__ == "__main__":
    main()
