# SPDX-License-Identifier: BSD-3-Clause
"""
9-STAGE CURRICULUM FOR TEKO (v3 – GRADUAL YAW)
==============================================

More gradual yaw progression to ensure learning.

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations

import numpy as np
import torch
from ..utils.geometry_utils import yaw_to_quat


STAGE_NAMES = [
    "Stage 0: Forward (5-20cm, straight back)",
    "Stage 1: Yaw Only (±15°, 0 lateral)",   # Mais fácil
    "Stage 2: Yaw Only (±30°, 0 lateral)",
    "Stage 3: Yaw Only (±60°, 0 lateral)",
    "Stage 4: Yaw Only (±90°, 0 lateral)",
    "Stage 5: Combined (±45°, ±5cm)",
    "Stage 6: Combined (±90°, ±8cm)",
    "Stage 7: Rear (±135°, ±5cm)",
    "Stage 8: Full Turn (±180°, ±3cm)",
]

STAGE_CONFIGS = {
    0: (0.0, 0.0, 0.05, 0.20),
    1: (15.0, 0.0, 0.15, 0.30),   # ±15° primeiro
    2: (30.0, 0.0, 0.20, 0.35),
    3: (60.0, 0.0, 0.25, 0.40),
    4: (90.0, 0.0, 0.30, 0.45),
    5: (45.0, 0.05, 0.25, 0.40),
    6: (90.0, 0.08, 0.30, 0.45),
    7: (135.0, 0.05, 0.35, 0.50),
    8: (180.0, 0.03, 0.40, 0.55),
}

REPLAY_PROBS = {
    "forward": 0.0,
    "yaw_only": 0.25,
    "combined": 0.35,
    "extreme": 0.45,
}


def reset_environment_curriculum(env, env_ids: torch.Tensor) -> None:
    current_stage = int(env.curriculum_level)
    num = int(env_ids.numel())

    if current_stage == 0:
        _reset_stage(env, env_ids, current_stage)
        return

    device = env.device
    mix_prob = _get_replay_probability(current_stage)

    replay_mask = torch.rand(num, device=device) < mix_prob
    replay_ids = env_ids[replay_mask]
    current_ids = env_ids[~replay_mask]

    if replay_ids.numel() > 0:
        replay_stages = torch.randint(0, current_stage, (replay_ids.numel(),), device=device)
        for stage in range(current_stage):
            stage_mask = replay_stages == stage
            stage_ids = replay_ids[stage_mask]
            if stage_ids.numel() > 0:
                _reset_stage(env, stage_ids, stage)
    
    if current_ids.numel() > 0:
        _reset_stage(env, current_ids, current_stage)


def _get_replay_probability(stage: int) -> float:
    if stage == 0:
        return REPLAY_PROBS["forward"]
    elif stage <= 4:
        return REPLAY_PROBS["yaw_only"]
    elif stage <= 6:
        return REPLAY_PROBS["combined"]
    else:
        return REPLAY_PROBS["extreme"]


def _reset_stage(env, env_ids: torch.Tensor, stage: int) -> None:
    if stage < 0 or stage >= len(STAGE_NAMES):
        raise ValueError(f"Invalid stage: {stage}")

    yaw_deg, lateral_m, min_dist, max_dist = STAGE_CONFIGS[stage]
    
    num = int(env_ids.numel())
    device = env.device

    dist = torch.rand(num, device=device) * (max_dist - min_dist) + min_dist

    if yaw_deg == 0.0:
        yaw = torch.ones(num, device=device) * np.pi
    else:
        max_yaw_rad = float(np.deg2rad(yaw_deg))
        yaw_offset = torch.rand(num, device=device) * (2.0 * max_yaw_rad) - max_yaw_rad
        yaw = np.pi + yaw_offset

    x = env.goal_positions[env_ids, 0] - dist
    
    if lateral_m == 0.0:
        y = env.goal_positions[env_ids, 1]
    else:
        lateral_offset = torch.rand(num, device=device) * (2.0 * lateral_m) - lateral_m
        y = env.goal_positions[env_ids, 1] + lateral_offset
    
    z = torch.full((num,), 0.40, device=device)

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)

    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids)
    _zero_root_velocity_if_available(env, env_ids)


def _zero_root_velocity_if_available(env, env_ids: torch.Tensor) -> None:
    num = int(env_ids.numel())
    device = env.device
    root_vel = torch.zeros((num, 6), device=device, dtype=torch.float32)

    if hasattr(env.robot, "write_root_velocity_to_sim"):
        env.robot.write_root_velocity_to_sim(root_vel, env_ids)
    elif hasattr(env.robot, "write_root_vel_to_sim"):
        env.robot.write_root_vel_to_sim(root_vel, env_ids)


def set_curriculum_level(env, level: int) -> None:
    max_level = len(STAGE_NAMES) - 1
    level = max(0, min(max_level, int(level)))
    env.curriculum_level = level
    print(f"\n{'=' * 70}")
    print(f"[CURRICULUM] {STAGE_NAMES[level]}")
    print(f"{'=' * 70}\n")


def should_advance_curriculum(success_rate: float, current_level: int) -> bool:
    if current_level >= len(STAGE_NAMES) - 1:
        return False
    return success_rate >= 0.80


def get_stage_info(stage: int) -> dict:
    if stage < 0 or stage >= len(STAGE_NAMES):
        raise ValueError(f"Invalid stage: {stage}")
    yaw_deg, lateral_m, min_dist, max_dist = STAGE_CONFIGS[stage]
    return {
        "name": STAGE_NAMES[stage],
        "stage": stage,
        "yaw_deg": yaw_deg,
        "lateral_m": lateral_m,
        "min_dist": min_dist,
        "max_dist": max_dist,
        "replay_prob": _get_replay_probability(stage),
    }


def get_all_stage_configs() -> list[dict]:
    return [get_stage_info(i) for i in range(len(STAGE_NAMES))]
