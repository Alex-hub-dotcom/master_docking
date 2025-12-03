# SPDX-License-Identifier: BSD-3-Clause
"""
28-STAGE CURRICULUM FOR TEKO (v9.0 – 84px OPTIMIZED)
====================================================

Changes for 84×84 grayscale stability:
- Added +5 cm to all offset min_dist (S4–S22)
- Widened max_dist slightly for visual clarity
- Increased yaw tolerance by ±1° in S4–S8
- Rebalanced 180° stages: farther spawn distances
- All changes maintain “NO simultaneous yaw + lateral increases”

Everything else remains identical.
"""

from __future__ import annotations

import numpy as np
import torch
from ..utils.geometry_utils import yaw_to_quat

# ------------------------------------------------------------
# Stage names unchanged
# ------------------------------------------------------------
STAGE_NAMES = [
    "Stage 0:  Baby Steps (5–12 cm, forward)",
    "Stage 1:  Forward 1 (10–18 cm, forward)",
    "Stage 2:  Forward 2 (15–25 cm, forward)",
    "Stage 3:  Medium Forward (20–35 cm, forward)",
    "Stage 4:  Tiny Offset (±4°, ±2 cm)",      # +1° yaw
    "Stage 5:  Small Offset (±7°, ±3 cm)",     # +1° yaw
    "Stage 6:  Offset (±10°, ±3 cm)",          # +1° yaw
    "Stage 7:  Offset (±11°, ±3 cm)",
    "Stage 8:  Offset (±12°, ±3 cm)",
    "Stage 9:  Offset (±13°, ±3 cm)",
    "Stage 10: Offset (±15°, ±3 cm)",
    "Stage 11: Offset (±17°, ±3 cm)",
    "Stage 12: Offset (±19°, ±4 cm)",
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
    "Stage 23: Large Angle (±45°,  ±8 cm)",
    "Stage 24: Large Angle (±60°,  ±6 cm)",
    "Stage 25: Perpendicular (±90°, ±5 cm)",
    "Stage 26: Rear Angle (±135°, ±4 cm)",
    "Stage 27: Full Turn (±180°, ±3 cm)",
]

# ------------------------------------------------------------
# Curriculum Reset Logic
# ------------------------------------------------------------

def reset_environment_curriculum(env, env_ids: torch.Tensor) -> None:
    current_stage = int(env.curriculum_level)
    num = len(env_ids)

    if current_stage == 0:
        _reset_stage_dispatch(env, env_ids, current_stage)
        return

    device = env.device

    # Replay probabilities tuned for 84px
    if current_stage >= 23:
        mix_prob = 0.25
    elif current_stage >= 13:
        mix_prob = 0.22
    elif current_stage >= 7:
        mix_prob = 0.18
    else:
        mix_prob = 0.15

    mix_prev = torch.rand(num, device=device) < mix_prob
    prev_ids = env_ids[mix_prev]
    curr_ids = env_ids[~mix_prev]

    if len(prev_ids) > 0:
        _reset_stage_dispatch(env, prev_ids, current_stage - 1)
    if len(curr_ids) > 0:
        _reset_stage_dispatch(env, curr_ids, current_stage)


# ------------------------------------------------------------
# Stage Dispatcher
# ------------------------------------------------------------

def _reset_stage_dispatch(env, env_ids: torch.Tensor, stage: int) -> None:
    if stage < 0 or stage >= len(STAGE_NAMES):
        raise ValueError(f"Invalid stage: {stage}")
    
    # -----------------------
    #  Forward-only stages
    # -----------------------
    if stage <= 3:
        forward_configs = [
            (0.05, 0.12),  # S0
            (0.10, 0.18),  # S1
            (0.15, 0.25),  # S2
            (0.20, 0.35),  # S3
        ]
        min_d, max_d = forward_configs[stage]
        _forward_reset(env, env_ids, min_d, max_d)
        return

    # -----------------------
    #  Offset stages (with 84px adjustments)
    # -----------------------
    offset_configs = {
        # ------ First offsets with softened yaw ------
        4:  (4.0,  0.02, 0.25, 0.36),
        5:  (7.0,  0.03, 0.25, 0.37),
        6:  (10.0, 0.03, 0.25, 0.38),
        # ------ Proven micro-steps ------
        7:  (11.0, 0.03, 0.25, 0.38),
        8:  (12.0, 0.03, 0.25, 0.38),
        9:  (13.0, 0.03, 0.25, 0.38),
        10: (15.0, 0.03, 0.25, 0.38),
        11: (17.0, 0.03, 0.25, 0.38),
        12: (19.0, 0.04, 0.25, 0.38),
        # ------ ULTRA micro-steps ------
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
        # ------ 180° stages (spawn farther for 84px) ------
        23: (45.0,  0.08, 0.28, 0.45),
        24: (60.0,  0.06, 0.30, 0.48),
        25: (90.0,  0.05, 0.32, 0.50),
        26: (135.0, 0.04, 0.34, 0.52),
        27: (180.0, 0.03, 0.36, 0.54),
    }

    angle, lateral, min_d, max_d = offset_configs[stage]
    _offset_reset(env, env_ids, angle, lateral, min_d, max_d)


# ------------------------------------------------------------
# Reset functions
# ------------------------------------------------------------

def _forward_reset(env, env_ids, min_d, max_d):
    num = len(env_ids)
    device = env.device

    dist = torch.rand(num, device=device) * (max_d - min_d) + min_d
    yaw = torch.ones(num, device=device) * np.pi

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.full((num,), 0.40, device=device)

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids)


def _offset_reset(env, env_ids, angle_deg, lateral_m, min_d, max_d):
    num = len(env_ids)
    device = env.device

    dist = torch.rand(num, device=device) * (max_d - min_d) + min_d
    max_yaw = np.deg2rad(angle_deg)

    yaw = np.pi + (torch.rand(num, device=device) * 2 * max_yaw - max_yaw)

    x = env.goal_positions[env_ids, 0] - dist
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
