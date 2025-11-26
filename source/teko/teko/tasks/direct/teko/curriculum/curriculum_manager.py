# SPDX-License-Identifier: BSD-3-Clause
"""
16-STAGE MICRO-STEP CURRICULUM FOR TEKO (VISION-ONLY)
======================================================

v3.0 Changes:
- S0-S6: UNCHANGED (these work)
- S7-S15: MICRO-STEPS - only 1-2° yaw increase per stage
- Lateral stays at 3cm until S12, then gradually increases
- Goal: Prevent catastrophic policy shifts from large jumps

Offset progression (v3.0 - MICRO-STEPS):
- S4:  ±3°,  ±2cm  (unchanged)
- S5:  ±6°,  ±3cm  (unchanged)
- S6:  ±9°,  ±3cm  (unchanged - this works!)
- S7:  ±10°, ±3cm  (micro: +1°)
- S8:  ±11°, ±3cm  (micro: +1°)
- S9:  ±12°, ±3cm  (micro: +1°)
- S10: ±14°, ±3cm  (micro: +2°)
- S11: ±16°, ±3cm  (micro: +2°)
- S12: ±18°, ±4cm  (micro: +2°, lateral starts increasing)
- S13: ±21°, ±6cm  (micro: +3°, lateral↑)
- S14: ±24°, ±9cm  (micro: +3°, lateral↑)
- S15: ±30°, ±12cm (final offset stage)
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils.geometry_utils import yaw_to_quat

# Descriptive names for console logging
STAGE_NAMES = [
    "Stage 0:  Baby Steps (5–12 cm, forward)",
    "Stage 1:  Forward 1 (10–18 cm, forward)",
    "Stage 2:  Forward 2 (15–25 cm, forward)",
    "Stage 3:  Medium Forward (20–35 cm, forward)",
    "Stage 4:  Tiny Offset (20–30 cm, ±3°, ±2 cm)",
    "Stage 5:  Small Offset (20–35 cm, ±6°, ±3 cm)",
    "Stage 6:  Offset (20–35 cm, ±9°, ±3 cm)",
    "Stage 7:  Offset (20–35 cm, ±10°, ±3 cm)",    # +1°
    "Stage 8:  Offset (20–35 cm, ±11°, ±3 cm)",    # +1°
    "Stage 9:  Offset (20–35 cm, ±12°, ±3 cm)",    # +1°
    "Stage 10: Offset (20–35 cm, ±14°, ±3 cm)",    # +2°
    "Stage 11: Offset (20–35 cm, ±16°, ±3 cm)",    # +2°
    "Stage 12: Offset (20–35 cm, ±18°, ±4 cm)",    # +2°, lateral↑
    "Stage 13: Offset (20–35 cm, ±21°, ±6 cm)",    # +3°, lateral↑
    "Stage 14: Offset (20–35 cm, ±24°, ±9 cm)",    # +3°, lateral↑
    "Stage 15: Offset (20–35 cm, ±30°, ±12 cm)",   # final
]


# =============================================================================
# Dispatcher
# =============================================================================

def reset_environment_curriculum(env, env_ids: torch.Tensor) -> None:
    """
    Reset environments according to the current curriculum stage.
    Includes replay from previous stage to prevent forgetting.
    """
    current_stage = int(env.curriculum_level)
    num = len(env_ids)

    if current_stage == 0:
        _reset_stage_dispatch(env, env_ids, current_stage)
        return

    device = env.device

    # Stage-dependent replay probability
    mix_prob = 0.2
    if current_stage >= 12:
        mix_prob = 0.35
    elif current_stage >= 7:
        mix_prob = 0.25

    mix_prev = torch.rand(num, device=device) < mix_prob

    prev_ids = env_ids[mix_prev]
    curr_ids = env_ids[~mix_prev]

    if len(prev_ids) > 0:
        _reset_stage_dispatch(env, prev_ids, current_stage - 1)

    if len(curr_ids) > 0:
        _reset_stage_dispatch(env, curr_ids, current_stage)


def _reset_stage_dispatch(env, env_ids: torch.Tensor, stage: int) -> None:
    """Route to the correct reset function."""
    if stage == 0:
        _reset_stage0(env, env_ids)
    elif stage == 1:
        _reset_stage1(env, env_ids)
    elif stage == 2:
        _reset_stage2(env, env_ids)
    elif stage == 3:
        _reset_stage3(env, env_ids)
    elif stage == 4:
        _reset_stage4(env, env_ids)
    elif stage == 5:
        _reset_stage5(env, env_ids)
    elif stage == 6:
        _reset_stage6(env, env_ids)
    elif stage == 7:
        _reset_stage7(env, env_ids)
    elif stage == 8:
        _reset_stage8(env, env_ids)
    elif stage == 9:
        _reset_stage9(env, env_ids)
    elif stage == 10:
        _reset_stage10(env, env_ids)
    elif stage == 11:
        _reset_stage11(env, env_ids)
    elif stage == 12:
        _reset_stage12(env, env_ids)
    elif stage == 13:
        _reset_stage13(env, env_ids)
    elif stage == 14:
        _reset_stage14(env, env_ids)
    elif stage == 15:
        _reset_stage15(env, env_ids)
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
# S0-S6: UNCHANGED (these work!)
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


def _reset_stage4(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=3.0, lateral_m=0.02, min_dist=0.20, max_dist=0.30)


def _reset_stage5(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=6.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


def _reset_stage6(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=9.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


# =============================================================================
# S7-S15: MICRO-STEPS (new!)
# =============================================================================

# S7: ±10°, ±3cm (+1° from S6)
def _reset_stage7(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=10.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


# S8: ±11°, ±3cm (+1°)
def _reset_stage8(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=11.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


# S9: ±12°, ±3cm (+1°)
def _reset_stage9(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=12.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


# S10: ±14°, ±3cm (+2°)
def _reset_stage10(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=14.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


# S11: ±16°, ±3cm (+2°)
def _reset_stage11(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=16.0, lateral_m=0.03, min_dist=0.20, max_dist=0.35)


# S12: ±18°, ±4cm (+2°, lateral starts increasing)
def _reset_stage12(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=18.0, lateral_m=0.04, min_dist=0.20, max_dist=0.35)


# S13: ±21°, ±6cm (+3°, lateral↑)
def _reset_stage13(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=21.0, lateral_m=0.06, min_dist=0.20, max_dist=0.35)


# S14: ±24°, ±9cm (+3°, lateral↑)
def _reset_stage14(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=24.0, lateral_m=0.09, min_dist=0.20, max_dist=0.35)


# S15: ±30°, ±12cm (final offset mastery)
def _reset_stage15(env, env_ids: torch.Tensor) -> None:
    _offset_reset(env, env_ids, angle_deg=30.0, lateral_m=0.12, min_dist=0.20, max_dist=0.35)


# =============================================================================
# Curriculum control
# =============================================================================

def set_curriculum_level(env, level: int) -> None:
    """Set the curriculum stage (0–15) on the environment."""
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