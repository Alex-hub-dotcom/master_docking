# SPDX-License-Identifier: BSD-3-Clause
"""
23-STAGE CURRICULUM FOR TEKO (v6.0 - FIXED S13 PLATEAU)
========================================================

v6.0 Changes:
- Added S12.5 (now S13) to fix the S12→S13 jump problem
- S12: ±18°, ±4cm → S13: ±20°, ±4cm (only yaw) → S14: ±20°, ±6cm (only lateral)
- Total: 23 stages (S0-S22)
- Extended to 180° turn capability

Key insight: Never increase YAW and LATERAL simultaneously!
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils.geometry_utils import yaw_to_quat

# Descriptive names for console logging (23 stages)
STAGE_NAMES = [
    "Stage 0:  Baby Steps (5–12 cm, forward)",
    "Stage 1:  Forward 1 (10–18 cm, forward)",
    "Stage 2:  Forward 2 (15–25 cm, forward)",
    "Stage 3:  Medium Forward (20–35 cm, forward)",
    "Stage 4:  Tiny Offset (20–30 cm, ±3°, ±2 cm)",
    "Stage 5:  Small Offset (20–35 cm, ±6°, ±3 cm)",
    "Stage 6:  Offset (20–35 cm, ±9°, ±3 cm)",
    "Stage 7:  Offset (20–35 cm, ±10°, ±3 cm)",
    "Stage 8:  Offset (20–35 cm, ±11°, ±3 cm)",
    "Stage 9:  Offset (20–35 cm, ±12°, ±3 cm)",
    "Stage 10: Offset (20–35 cm, ±14°, ±3 cm)",
    "Stage 11: Offset (20–35 cm, ±16°, ±3 cm)",
    "Stage 12: Offset (20–35 cm, ±18°, ±4 cm)",
    "Stage 13: Offset (20–35 cm, ±20°, ±4 cm)",      # NEW: only yaw increase
    "Stage 14: Offset (20–35 cm, ±20°, ±6 cm)",      # NEW: only lateral increase
    "Stage 15: Offset (20–35 cm, ±24°, ±6 cm)",
    "Stage 16: Offset (20–35 cm, ±30°, ±9 cm)",
    # Extended stages for 180° turn
    "Stage 17: Large Angle (20–40 cm, ±45°, ±8 cm)",
    "Stage 18: Large Angle (20–40 cm, ±60°, ±6 cm)",
    "Stage 19: Perpendicular (20–40 cm, ±90°, ±5 cm)",
    "Stage 20: Rear Angle (25–45 cm, ±120°, ±4 cm)",
    "Stage 21: Rear Angle (25–45 cm, ±150°, ±3 cm)",
    "Stage 22: Full Turn (30–50 cm, ±180°, ±3 cm)",
]


# =============================================================================
# Dispatcher
# =============================================================================

def reset_environment_curriculum(env, env_ids: torch.Tensor) -> None:
    """Reset environments according to the current curriculum stage."""
    current_stage = int(env.curriculum_level)
    num = len(env_ids)

    if current_stage == 0:
        _reset_stage_dispatch(env, env_ids, current_stage)
        return

    device = env.device

    # Stage-dependent replay probability
    if current_stage >= 17:
        mix_prob = 0.40  # Higher replay for 180° stages
    elif current_stage >= 13:
        mix_prob = 0.35  # Higher replay for advanced offset stages
    elif current_stage >= 7:
        mix_prob = 0.25
    else:
        mix_prob = 0.20

    mix_prev = torch.rand(num, device=device) < mix_prob

    prev_ids = env_ids[mix_prev]
    curr_ids = env_ids[~mix_prev]

    if len(prev_ids) > 0:
        _reset_stage_dispatch(env, prev_ids, current_stage - 1)

    if len(curr_ids) > 0:
        _reset_stage_dispatch(env, curr_ids, current_stage)


def _reset_stage_dispatch(env, env_ids: torch.Tensor, stage: int) -> None:
    """Route to the correct reset function."""
    reset_functions = [
        _reset_stage0, _reset_stage1, _reset_stage2, _reset_stage3,
        _reset_stage4, _reset_stage5, _reset_stage6, _reset_stage7,
        _reset_stage8, _reset_stage9, _reset_stage10, _reset_stage11,
        _reset_stage12, _reset_stage13, _reset_stage14, _reset_stage15,
        _reset_stage16, _reset_stage17, _reset_stage18, _reset_stage19,
        _reset_stage20, _reset_stage21, _reset_stage22,
    ]
    
    if 0 <= stage < len(reset_functions):
        reset_functions[stage](env, env_ids)
    else:
        raise ValueError(f"Invalid curriculum stage: {stage}")


# =============================================================================
# Helpers
# =============================================================================

def _base_forward_reset(
    env,
    env_ids: torch.Tensor,
    min_dist: float,
    max_dist: float,
    yaw: torch.Tensor,
) -> None:
    """Helper for pure forward docking stages."""
    num = len(env_ids)
    device = env.device

    dist = torch.rand(num, device=device) * (max_dist - min_dist) + min_dist

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
    """Helper for offset stages."""
    num = len(env_ids)
    device = env.device

    dist = torch.rand(num, device=device) * (max_dist - min_dist) + min_dist
    max_yaw = np.deg2rad(angle_deg)

    yaw = np.pi + (torch.rand(num, device=device) * (2 * max_yaw) - max_yaw)

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (
        torch.rand(num, device=device) * (2 * lateral_m) - lateral_m
    )
    z = torch.ones(num, device=device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# S0-S3: Forward stages (unchanged)
# =============================================================================

def _reset_stage0(env, env_ids: torch.Tensor) -> None:
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.05, 0.12, yaw)


def _reset_stage1(env, env_ids: torch.Tensor) -> None:
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.10, 0.18, yaw)


def _reset_stage2(env, env_ids: torch.Tensor) -> None:
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.15, 0.25, yaw)


def _reset_stage3(env, env_ids: torch.Tensor) -> None:
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.20, 0.35, yaw)


# =============================================================================
# S4-S12: Offset stages (unchanged - these work!)
# =============================================================================

def _reset_stage4(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=3.0, lateral_m=0.02, min_dist=0.20, max_dist=0.30)


def _reset_stage5(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=6.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


def _reset_stage6(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=9.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


def _reset_stage7(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=10.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


def _reset_stage8(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=11.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


def _reset_stage9(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=12.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


def _reset_stage10(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=14.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


def _reset_stage11(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=16.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


def _reset_stage12(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=18.0, lateral_m=0.04, min_dist=0.20, max_dist=0.35)


# =============================================================================
# S13-S16: FIXED - Never increase yaw AND lateral simultaneously!
# =============================================================================

# S13: ±20°, ±4cm (only +2° yaw, lateral stays at 4cm)
def _reset_stage13(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=20.0, lateral_m=0.04, min_dist=0.20, max_dist=0.35)


# S14: ±20°, ±6cm (yaw stays at 20°, only +2cm lateral)
def _reset_stage14(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=20.0, lateral_m=0.06, min_dist=0.20, max_dist=0.35)


# S15: ±24°, ±6cm (only +4° yaw)
def _reset_stage15(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=24.0, lateral_m=0.06, min_dist=0.20, max_dist=0.35)


# S16: ±30°, ±9cm (offset mastery - can increase both here, robot is stronger)
def _reset_stage16(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=30.0, lateral_m=0.09, min_dist=0.20, max_dist=0.35)


# =============================================================================
# S17-S22: Extended stages for 180° turn
# =============================================================================

# S17: ±45°, ±8cm
def _reset_stage17(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=45.0, lateral_m=0.08, min_dist=0.20, max_dist=0.40)


# S18: ±60°, ±6cm
def _reset_stage18(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=60.0, lateral_m=0.06, min_dist=0.20, max_dist=0.40)


# S19: ±90°, ±5cm (perpendicular - robot starts sideways)
def _reset_stage19(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=90.0, lateral_m=0.05, min_dist=0.20, max_dist=0.40)


# S20: ±120°, ±4cm (robot starts facing partially away)
def _reset_stage20(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=120.0, lateral_m=0.04, min_dist=0.25, max_dist=0.45)


# S21: ±150°, ±3cm (robot starts almost backwards)
def _reset_stage21(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=150.0, lateral_m=0.03, min_dist=0.25, max_dist=0.45)


# S22: ±180°, ±3cm (full turn - robot starts facing completely away)
def _reset_stage22(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=180.0, lateral_m=0.03, min_dist=0.30, max_dist=0.50)


# =============================================================================
# Curriculum control
# =============================================================================

def set_curriculum_level(env, level: int) -> None:
    """Set the curriculum stage (0–22) on the environment."""
    max_level = len(STAGE_NAMES) - 1
    level = max(0, min(max_level, int(level)))
    env.curriculum_level = level

    print(f"\n{'=' * 70}")
    print(f"[CURRICULUM] {STAGE_NAMES[level]}")
    print(f"{'=' * 70}\n")


def should_advance_curriculum(success_rate: float, current_level: int) -> bool:
    """Decide whether to advance to the next curriculum stage."""
    max_level = len(STAGE_NAMES) - 1
    if current_level >= max_level:
        return False
    return success_rate >= 0.85