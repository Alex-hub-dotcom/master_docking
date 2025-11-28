# SPDX-License-Identifier: BSD-3-Clause
"""
28-STAGE CURRICULUM FOR TEKO (v8.0 - ULTRA MICRO-STEPS)
========================================================

GOLDEN RULE: Never increase YAW and LATERAL simultaneously!
EXTRA RULE: Maximum +1cm lateral or +2° yaw per stage!

v8.0 - Fixes S14 plateau by adding intermediate steps.

Key changes from v7:
- S14 now +1cm lateral (not +2cm)
- Added more intermediate stages
- Total 28 stages (S0-S27)
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils.geometry_utils import yaw_to_quat

STAGE_NAMES = [
    # Forward stages (S0-S3)
    "Stage 0:  Baby Steps (5–12 cm, forward)",
    "Stage 1:  Forward 1 (10–18 cm, forward)",
    "Stage 2:  Forward 2 (15–25 cm, forward)",
    "Stage 3:  Medium Forward (20–35 cm, forward)",
    # First offsets (S4-S6)
    "Stage 4:  Tiny Offset (±3°, ±2 cm)",
    "Stage 5:  Small Offset (±6°, ±3 cm)",
    "Stage 6:  Offset (±9°, ±3 cm)",
    # Micro-steps S7-S12 (proven to work)
    "Stage 7:  Offset (±10°, ±3 cm)",
    "Stage 8:  Offset (±11°, ±3 cm)",
    "Stage 9:  Offset (±12°, ±3 cm)",
    "Stage 10: Offset (±14°, ±3 cm)",
    "Stage 11: Offset (±16°, ±3 cm)",
    "Stage 12: Offset (±18°, ±4 cm)",
    # ULTRA micro-steps S13-S22 (max +2° yaw OR +1cm lateral per stage)
    "Stage 13: Offset (±20°, ±4 cm)",   # +2° yaw
    "Stage 14: Offset (±20°, ±5 cm)",   # +1cm lateral
    "Stage 15: Offset (±20°, ±6 cm)",   # +1cm lateral
    "Stage 16: Offset (±22°, ±6 cm)",   # +2° yaw
    "Stage 17: Offset (±24°, ±6 cm)",   # +2° yaw
    "Stage 18: Offset (±24°, ±7 cm)",   # +1cm lateral
    "Stage 19: Offset (±24°, ±8 cm)",   # +1cm lateral
    "Stage 20: Offset (±27°, ±8 cm)",   # +3° yaw
    "Stage 21: Offset (±30°, ±8 cm)",   # +3° yaw
    "Stage 22: Offset (±30°, ±10 cm)",  # +2cm lateral - OFFSET MASTERY
    # 180° turn stages S23-S27
    "Stage 23: Large Angle (±45°, ±8 cm)",
    "Stage 24: Large Angle (±60°, ±6 cm)",
    "Stage 25: Perpendicular (±90°, ±5 cm)",
    "Stage 26: Rear Angle (±135°, ±4 cm)",
    "Stage 27: Full Turn (±180°, ±3 cm)",
]


def reset_environment_curriculum(env, env_ids: torch.Tensor) -> None:
    """Reset environments according to current curriculum stage with replay."""
    current_stage = int(env.curriculum_level)
    num = len(env_ids)

    if current_stage == 0:
        _reset_stage_dispatch(env, env_ids, current_stage)
        return

    device = env.device

    # Conservative replay probability
    if current_stage >= 23:
        mix_prob = 0.30  # 180° stages
    elif current_stage >= 13:
        mix_prob = 0.25  # Advanced offset stages
    elif current_stage >= 7:
        mix_prob = 0.20  # Micro-step stages
    else:
        mix_prob = 0.15  # Early stages

    mix_prev = torch.rand(num, device=device) < mix_prob
    prev_ids = env_ids[mix_prev]
    curr_ids = env_ids[~mix_prev]

    if len(prev_ids) > 0:
        _reset_stage_dispatch(env, prev_ids, current_stage - 1)
    if len(curr_ids) > 0:
        _reset_stage_dispatch(env, curr_ids, current_stage)


def _reset_stage_dispatch(env, env_ids: torch.Tensor, stage: int) -> None:
    """Route to correct reset function."""
    if stage < 0 or stage >= len(STAGE_NAMES):
        raise ValueError(f"Invalid stage: {stage}")
    
    if stage <= 3:
        # Forward stages
        forward_configs = [
            (0.05, 0.12),  # S0
            (0.10, 0.18),  # S1
            (0.15, 0.25),  # S2
            (0.20, 0.35),  # S3
        ]
        min_d, max_d = forward_configs[stage]
        _forward_reset(env, env_ids, min_d, max_d)
    else:
        # Offset stages - (angle_deg, lateral_m, min_dist, max_dist)
        offset_configs = {
            # First offsets
            4:  (3.0,  0.02, 0.20, 0.30),
            5:  (6.0,  0.03, 0.20, 0.35),
            6:  (9.0,  0.03, 0.20, 0.35),
            # Proven micro-steps
            7:  (10.0, 0.03, 0.20, 0.35),
            8:  (11.0, 0.03, 0.20, 0.35),
            9:  (12.0, 0.03, 0.20, 0.35),
            10: (14.0, 0.03, 0.20, 0.35),
            11: (16.0, 0.03, 0.20, 0.35),
            12: (18.0, 0.04, 0.20, 0.35),
            # ULTRA micro-steps (max +2° yaw OR +1cm lateral)
            13: (20.0, 0.04, 0.20, 0.35),  # +2° yaw
            14: (20.0, 0.05, 0.20, 0.35),  # +1cm lateral
            15: (20.0, 0.06, 0.20, 0.35),  # +1cm lateral
            16: (22.0, 0.06, 0.20, 0.35),  # +2° yaw
            17: (24.0, 0.06, 0.20, 0.35),  # +2° yaw
            18: (24.0, 0.07, 0.20, 0.35),  # +1cm lateral
            19: (24.0, 0.08, 0.20, 0.35),  # +1cm lateral
            20: (27.0, 0.08, 0.20, 0.35),  # +3° yaw
            21: (30.0, 0.08, 0.20, 0.35),  # +3° yaw
            22: (30.0, 0.10, 0.20, 0.35),  # +2cm lateral - OFFSET MASTERY
            # 180° stages
            23: (45.0,  0.08, 0.20, 0.40),
            24: (60.0,  0.06, 0.20, 0.40),
            25: (90.0,  0.05, 0.20, 0.40),
            26: (135.0, 0.04, 0.25, 0.45),
            27: (180.0, 0.03, 0.30, 0.50),
        }
        angle, lateral, min_d, max_d = offset_configs[stage]
        _offset_reset(env, env_ids, angle, lateral, min_d, max_d)


def _forward_reset(env, env_ids: torch.Tensor, min_dist: float, max_dist: float) -> None:
    """Forward docking (yaw = π, no lateral offset)."""
    num = len(env_ids)
    device = env.device

    dist = torch.rand(num, device=device) * (max_dist - min_dist) + min_dist
    yaw = torch.ones(num, device=device) * np.pi

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.ones(num, device=device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


def _offset_reset(
    env,
    env_ids: torch.Tensor,
    angle_deg: float,
    lateral_m: float,
    min_dist: float,
    max_dist: float,
) -> None:
    """Offset docking with yaw and lateral variation."""
    num = len(env_ids)
    device = env.device

    dist = torch.rand(num, device=device) * (max_dist - min_dist) + min_dist
    max_yaw = np.deg2rad(angle_deg)
    yaw = np.pi + (torch.rand(num, device=device) * 2 * max_yaw - max_yaw)

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=device) * 2 * lateral_m - lateral_m)
    z = torch.ones(num, device=device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


def set_curriculum_level(env, level: int) -> None:
    """Set curriculum stage (0-27)."""
    max_level = len(STAGE_NAMES) - 1
    level = max(0, min(max_level, int(level)))
    env.curriculum_level = level
    print(f"\n{'=' * 70}")
    print(f"[CURRICULUM] {STAGE_NAMES[level]}")
    print(f"{'=' * 70}\n")


def should_advance_curriculum(success_rate: float, current_level: int) -> bool:
    """Check if should advance."""
    if current_level >= len(STAGE_NAMES) - 1:
        return False
    return success_rate >= 0.85