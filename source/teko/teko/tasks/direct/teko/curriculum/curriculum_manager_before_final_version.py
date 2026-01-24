# SPDX-License-Identifier: BSD-3-Clause
"""
42-STAGE CURRICULUM FOR TEKO (v10 – Gradual 180°)
=================================================

Key change from v9.1:
- Stages S23-S41 now have ~7° increments instead of 15°
- This allows smoother learning for large angle maneuvers

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations

import numpy as np
import torch
from ..utils.geometry_utils import yaw_to_quat


# =============================================================================
# CONNECTOR GEOMETRY (must match teko_env.py)
# =============================================================================

FEMALE_OFFSET_X = 0.24
MALE_OFFSET_X = -0.227
CONNECTOR_GAP = FEMALE_OFFSET_X - MALE_OFFSET_X


# =============================================================================
# STAGE DEFINITIONS (42 stages total)
# =============================================================================

STAGE_NAMES = [
    # Forward stages (S0-S3)
    "Stage 0:  Baby Steps (5–12 cm, forward)",
    "Stage 1:  Forward 1 (10–18 cm, forward)",
    "Stage 2:  Forward 2 (15–25 cm, forward)",
    "Stage 3:  Medium Forward (20–35 cm, forward)",
    
    # First offsets (S4-S6)
    "Stage 4:  Tiny Offset (±4°, ±2 cm)",
    "Stage 5:  Small Offset (±7°, ±3 cm)",
    "Stage 6:  Offset (±10°, ±3 cm)",
    
    # Micro-steps (S7-S12)
    "Stage 7:  Offset (±11°, ±3 cm)",
    "Stage 8:  Offset (±12°, ±3 cm)",
    "Stage 9:  Offset (±13°, ±3 cm)",
    "Stage 10: Offset (±15°, ±3 cm)",
    "Stage 11: Offset (±17°, ±3 cm)",
    "Stage 12: Offset (±19°, ±4 cm)",
    
    # Ultra micro-steps (S13-S22)
    "Stage 13: Offset (±20°, ±4 cm)",
    "Stage 14: Offset (±20°, ±5 cm)",
    "Stage 15: Offset (±20°, ±6 cm)",
    "Stage 16: Offset (±22°, ±6 cm)",
    "Stage 17: Offset (±24°, ±6 cm)",
    "Stage 18: Offset (±24°, ±7 cm)",
    "Stage 19: Offset (±24°, ±8 cm)",
    "Stage 20: Offset (±27°, ±8 cm)",
    "Stage 21: Offset (±30°, ±8 cm)",
    "Stage 22: Offset (±30°, ±10 cm)",
    
    # 180° turn stages - GRADUAL ~7° steps (S23-S41)
    "Stage 23: Large Angle (±45°, ±8 cm)",
    "Stage 24: Large Angle (±52°, ±7 cm)",
    "Stage 25: Large Angle (±60°, ±7 cm)",
    "Stage 26: Large Angle (±67°, ±6 cm)",
    "Stage 27: Large Angle (±75°, ±6 cm)",
    "Stage 28: Large Angle (±82°, ±5 cm)",
    "Stage 29: Perpendicular (±90°, ±5 cm)",
    "Stage 30: Past Perpendicular (±97°, ±5 cm)",
    "Stage 31: Past Perpendicular (±105°, ±5 cm)",
    "Stage 32: Large Turn (±112°, ±4 cm)",
    "Stage 33: Large Turn (±120°, ±4 cm)",
    "Stage 34: Large Turn (±127°, ±4 cm)",
    "Stage 35: Rear Angle (±135°, ±4 cm)",
    "Stage 36: Rear Angle (±142°, ±3 cm)",
    "Stage 37: Rear Angle (±150°, ±3 cm)",
    "Stage 38: Rear Angle (±157°, ±3 cm)",
    "Stage 39: Near Full (±165°, ±3 cm)",
    "Stage 40: Near Full (±172°, ±3 cm)",
    "Stage 41: Full Turn (±180°, ±3 cm)",
]


# =============================================================================
# STAGE CONFIGURATIONS
# =============================================================================

FORWARD_CONFIGS = {
    0: (0.05, 0.12),
    1: (0.10, 0.18),
    2: (0.15, 0.25),
    3: (0.20, 0.35),
}

OFFSET_CONFIGS = {
    # First offsets
    4:  (4.0,  0.02, 0.25, 0.36),
    5:  (7.0,  0.03, 0.25, 0.37),
    6:  (10.0, 0.03, 0.25, 0.38),
    
    # Micro-steps
    7:  (11.0, 0.03, 0.25, 0.38),
    8:  (12.0, 0.03, 0.25, 0.38),
    9:  (13.0, 0.03, 0.25, 0.38),
    10: (15.0, 0.03, 0.25, 0.38),
    11: (17.0, 0.03, 0.25, 0.38),
    12: (19.0, 0.04, 0.25, 0.38),
    
    # Ultra micro-steps
    13: (20.0, 0.04, 0.25, 0.38),
    14: (20.0, 0.05, 0.25, 0.38),
    15: (20.0, 0.06, 0.25, 0.38),
    16: (22.0, 0.06, 0.25, 0.40),
    17: (24.0, 0.06, 0.25, 0.40),
    18: (24.0, 0.07, 0.25, 0.40),
    19: (24.0, 0.08, 0.25, 0.40),
    20: (27.0, 0.08, 0.25, 0.40),
    21: (30.0, 0.08, 0.25, 0.40),
    22: (30.0, 0.10, 0.25, 0.40),
    
    # 180° stages - GRADUAL ~7° steps
    23: (45.0,  0.08, 0.28, 0.45),
    24: (52.0,  0.07, 0.28, 0.46),
    25: (60.0,  0.07, 0.29, 0.47),
    26: (67.0,  0.06, 0.29, 0.48),
    27: (75.0,  0.06, 0.30, 0.49),
    28: (82.0,  0.05, 0.31, 0.50),
    29: (90.0,  0.05, 0.32, 0.50),
    30: (97.0,  0.05, 0.32, 0.51),
    31: (105.0, 0.05, 0.33, 0.51),
    32: (112.0, 0.04, 0.33, 0.52),
    33: (120.0, 0.04, 0.34, 0.52),
    34: (127.0, 0.04, 0.34, 0.53),
    35: (135.0, 0.04, 0.35, 0.53),
    36: (142.0, 0.03, 0.35, 0.54),
    37: (150.0, 0.03, 0.36, 0.54),
    38: (157.0, 0.03, 0.36, 0.55),
    39: (165.0, 0.03, 0.37, 0.55),
    40: (172.0, 0.03, 0.37, 0.56),
    41: (180.0, 0.03, 0.38, 0.56),
}

REPLAY_PROBS = {
    "early": 0.15,
    "micro": 0.18,
    "ultra": 0.22,
    "turn": 0.25,
}


# =============================================================================
# CURRICULUM RESET FUNCTIONS
# =============================================================================

def reset_environment_curriculum(env, env_ids: torch.Tensor) -> None:
    current_stage = int(env.curriculum_level)
    num = len(env_ids)

    if current_stage == 0:
        _reset_stage_dispatch(env, env_ids, current_stage)
        return

    device = env.device
    mix_prob = _get_replay_probability(current_stage)
    mix_prev = torch.rand(num, device=device) < mix_prob
    prev_ids = env_ids[mix_prev]
    curr_ids = env_ids[~mix_prev]

    if len(prev_ids) > 0:
        _reset_stage_dispatch(env, prev_ids, current_stage - 1)
    if len(curr_ids) > 0:
        _reset_stage_dispatch(env, curr_ids, current_stage)


def _get_replay_probability(stage: int) -> float:
    if stage >= 23:
        return REPLAY_PROBS["turn"]
    elif stage >= 13:
        return REPLAY_PROBS["ultra"]
    elif stage >= 7:
        return REPLAY_PROBS["micro"]
    else:
        return REPLAY_PROBS["early"]


def _reset_stage_dispatch(env, env_ids: torch.Tensor, stage: int) -> None:
    if stage < 0 or stage >= len(STAGE_NAMES):
        raise ValueError(f"Invalid stage: {stage}")

    if stage <= 3:
        min_d, max_d = FORWARD_CONFIGS[stage]
        _forward_reset(env, env_ids, min_d, max_d)
    else:
        angle, lateral, min_d, max_d = OFFSET_CONFIGS[stage]
        _offset_reset(env, env_ids, angle, lateral, min_d, max_d)


def _forward_reset(env, env_ids: torch.Tensor, min_dist: float, max_dist: float) -> None:
    num = len(env_ids)
    device = env.device

    dist = torch.rand(num, device=device) * (max_dist - min_dist) + min_dist
    yaw = torch.ones(num, device=device) * np.pi

    x = env.goal_positions[env_ids, 0] - CONNECTOR_GAP - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.full((num,), 0.40, device=device)

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids)


def _offset_reset(
    env,
    env_ids: torch.Tensor,
    angle_deg: float,
    lateral_m: float,
    min_dist: float,
    max_dist: float,
) -> None:
    num = len(env_ids)
    device = env.device

    dist = torch.rand(num, device=device) * (max_dist - min_dist) + min_dist
    max_yaw = np.deg2rad(angle_deg)
    yaw = np.pi + (torch.rand(num, device=device) * 2 * max_yaw - max_yaw)

    x = env.goal_positions[env_ids, 0] - CONNECTOR_GAP - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=device) * 2 * lateral_m - lateral_m)
    z = torch.full((num,), 0.40, device=device)

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids)


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
    return success_rate >= 0.85


def get_stage_info(stage: int) -> dict:
    if stage < 0 or stage >= len(STAGE_NAMES):
        raise ValueError(f"Invalid stage: {stage}")

    info = {
        "name": STAGE_NAMES[stage],
        "stage": stage,
        "type": "forward" if stage <= 3 else "offset",
        "replay_prob": _get_replay_probability(stage),
    }

    if stage <= 3:
        min_d, max_d = FORWARD_CONFIGS[stage]
        info.update({"min_dist": min_d, "max_dist": max_d, "angle_deg": 0.0, "lateral_m": 0.0})
    else:
        angle, lateral, min_d, max_d = OFFSET_CONFIGS[stage]
        info.update({"min_dist": min_d, "max_dist": max_d, "angle_deg": angle, "lateral_m": lateral})

    return info


def get_all_stage_configs() -> list[dict]:
    return [get_stage_info(i) for i in range(len(STAGE_NAMES))]
