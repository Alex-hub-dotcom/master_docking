# SPDX-License-Identifier: BSD-3-Clause
"""
35-STAGE CURRICULUM FOR TEKO (v10.0 – 180° MICRO-STEPS)
=======================================================

Optimized for 84×84 grayscale vision-based docking with
smooth progression into blind search regime.

Key changes from v9.1:
- Added micro-steps from 90° to 180° (+10° per stage)
- All turn stages use ±5cm lateral (proof-of-concept)
- Separate replay probs for "visible" vs "blind" turn stages
- Helper functions for stage classification

Design principles:
1. NEVER increase YAW and LATERAL simultaneously
2. Maximum +10° yaw in turn stages, +2° in early stages
3. Farther spawn distances for lower resolution
4. Lower replay for blind stages to track true SSR

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations

import numpy as np
import torch
from ..utils.geometry_utils import yaw_to_quat


# =============================================================================
# STAGE DEFINITIONS (35 STAGES)
# =============================================================================

STAGE_NAMES = [
    # Forward stages (S0-S3): Learn basic approach
    "Stage 0:  Baby Steps (5–12 cm, forward)",
    "Stage 1:  Forward 1 (10–18 cm, forward)",
    "Stage 2:  Forward 2 (15–25 cm, forward)",
    "Stage 3:  Medium Forward (20–35 cm, forward)",
    
    # First offsets (S4-S6): Introduce small corrections
    "Stage 4:  Tiny Offset (±4°, ±2 cm)",
    "Stage 5:  Small Offset (±7°, ±3 cm)",
    "Stage 6:  Offset (±10°, ±3 cm)",
    
    # Micro-steps (S7-S12): Gradual yaw increase
    "Stage 7:  Offset (±11°, ±3 cm)",
    "Stage 8:  Offset (±12°, ±3 cm)",
    "Stage 9:  Offset (±13°, ±3 cm)",
    "Stage 10: Offset (±15°, ±3 cm)",
    "Stage 11: Offset (±17°, ±3 cm)",
    "Stage 12: Offset (±19°, ±4 cm)",
    
    # Ultra micro-steps (S13-S22): Fine-grained progression
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
    
    # Large angle stages (S23-S25): Goal still visible
    "Stage 23: Large Angle (±45°, ±8 cm)",
    "Stage 24: Large Angle (±60°, ±6 cm)",
    "Stage 25: Perpendicular (±90°, ±5 cm)",
    
    # Blind search stages (S26-S34): Goal out of FOV, +10° per stage
    "Stage 26: Blind Search (±100°, ±5 cm)",
    "Stage 27: Blind Search (±110°, ±5 cm)",
    "Stage 28: Blind Search (±120°, ±5 cm)",
    "Stage 29: Blind Search (±130°, ±5 cm)",
    "Stage 30: Blind Search (±140°, ±5 cm)",
    "Stage 31: Blind Search (±150°, ±5 cm)",
    "Stage 32: Blind Search (±160°, ±5 cm)",
    "Stage 33: Blind Search (±170°, ±5 cm)",
    "Stage 34: Full Turn (±180°, ±5 cm)",
]

# Number of stages
NUM_STAGES = len(STAGE_NAMES)

# =============================================================================
# STAGE CONFIGURATIONS
# =============================================================================

# Forward stages: (min_dist, max_dist)
FORWARD_CONFIGS = {
    0: (0.05, 0.12),
    1: (0.10, 0.18),
    2: (0.15, 0.25),
    3: (0.20, 0.35),
}

# Offset stages: (angle_deg, lateral_m, min_dist, max_dist)
OFFSET_CONFIGS = {
    # First offsets (softened yaw for 84px)
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
    
    # Large angle stages (goal still visible/edge of FOV)
    23: (45.0,  0.08, 0.28, 0.45),
    24: (60.0,  0.06, 0.30, 0.48),
    25: (90.0,  0.05, 0.32, 0.50),
    
    # Blind search stages (goal out of FOV, +10° per stage)
    26: (100.0, 0.05, 0.34, 0.52),
    27: (110.0, 0.05, 0.36, 0.54),
    28: (120.0, 0.05, 0.38, 0.56),
    29: (130.0, 0.05, 0.40, 0.58),
    30: (140.0, 0.05, 0.42, 0.60),
    31: (150.0, 0.05, 0.44, 0.62),
    32: (160.0, 0.05, 0.46, 0.64),
    33: (170.0, 0.05, 0.48, 0.66),
    34: (180.0, 0.05, 0.50, 0.70),
}

# Replay probabilities per stage range
REPLAY_PROBS = {
    "early": 0.15,      # S1-S6
    "micro": 0.18,      # S7-S12
    "ultra": 0.22,      # S13-S22
    "turn_visible": 0.25,   # S23-S25 (goal visible)
    "turn_blind": 0.15,     # S26-S34 (goal out of FOV - lower to track true SSR)
}

# Stage category boundaries
TURN_STAGE_START = 23       # S23+ are turn stages
BLIND_STAGE_START = 26      # S26+ are blind search stages


# =============================================================================
# STAGE CLASSIFICATION HELPERS
# =============================================================================

def is_turn_stage(stage: int) -> bool:
    """Check if stage is a turn stage (>=45°)."""
    return stage >= TURN_STAGE_START


def is_blind_stage(stage: int) -> bool:
    """Check if stage is a blind search stage (>90°, goal out of FOV)."""
    return stage >= BLIND_STAGE_START


def get_stage_angle(stage: int) -> float:
    """Get the yaw angle (in degrees) for a stage."""
    if stage <= 3:
        return 0.0
    return OFFSET_CONFIGS[stage][0]


# =============================================================================
# CURRICULUM RESET FUNCTIONS
# =============================================================================

def reset_environment_curriculum(env, env_ids: torch.Tensor) -> None:
    """
    Reset environments according to current curriculum stage.
    
    Includes replay of previous stage for anti-forgetting.
    """
    current_stage = int(env.curriculum_level)
    num = len(env_ids)

    # Stage 0: no replay needed
    if current_stage == 0:
        _reset_stage_dispatch(env, env_ids, current_stage)
        return

    device = env.device

    # Determine replay probability based on stage
    mix_prob = _get_replay_probability(current_stage)

    # Split environments between current and previous stage
    mix_prev = torch.rand(num, device=device) < mix_prob
    prev_ids = env_ids[mix_prev]
    curr_ids = env_ids[~mix_prev]

    if len(prev_ids) > 0:
        _reset_stage_dispatch(env, prev_ids, current_stage - 1)
    if len(curr_ids) > 0:
        _reset_stage_dispatch(env, curr_ids, current_stage)


def _get_replay_probability(stage: int) -> float:
    """Get replay probability for anti-forgetting based on stage."""
    if stage >= BLIND_STAGE_START:
        return REPLAY_PROBS["turn_blind"]
    elif stage >= TURN_STAGE_START:
        return REPLAY_PROBS["turn_visible"]
    elif stage >= 13:
        return REPLAY_PROBS["ultra"]
    elif stage >= 7:
        return REPLAY_PROBS["micro"]
    else:
        return REPLAY_PROBS["early"]


def _reset_stage_dispatch(env, env_ids: torch.Tensor, stage: int) -> None:
    """Route to correct reset function based on stage type."""
    if stage < 0 or stage >= NUM_STAGES:
        raise ValueError(f"Invalid stage: {stage}")

    if stage <= 3:
        # Forward stages
        min_d, max_d = FORWARD_CONFIGS[stage]
        _forward_reset(env, env_ids, min_d, max_d)
    else:
        # Offset stages
        angle, lateral, min_d, max_d = OFFSET_CONFIGS[stage]
        _offset_reset(env, env_ids, angle, lateral, min_d, max_d)


# =============================================================================
# RESET IMPLEMENTATIONS
# =============================================================================

def _forward_reset(env, env_ids: torch.Tensor, min_dist: float, max_dist: float) -> None:
    """
    Forward docking reset (yaw = π, no lateral offset).
    
    Robot spawns directly behind the goal, facing it.
    """
    num = len(env_ids)
    device = env.device

    # Random distance within range
    dist = torch.rand(num, device=device) * (max_dist - min_dist) + min_dist
    
    # Facing the goal (yaw = π)
    yaw = torch.ones(num, device=device) * np.pi

    # Position: behind the goal
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.full((num,), 0.40, device=device)

    # Apply pose
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
    """
    Offset docking reset with yaw and lateral variation.
    
    Robot spawns with random yaw offset and lateral displacement.
    """
    num = len(env_ids)
    device = env.device

    # Random distance within range
    dist = torch.rand(num, device=device) * (max_dist - min_dist) + min_dist
    
    # Random yaw offset around π
    max_yaw = np.deg2rad(angle_deg)
    yaw = np.pi + (torch.rand(num, device=device) * 2 * max_yaw - max_yaw)

    # Position with lateral offset
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=device) * 2 * lateral_m - lateral_m)
    z = torch.full((num,), 0.40, device=device)

    # Apply pose
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids)


# =============================================================================
# CURRICULUM CONTROL
# =============================================================================

def set_curriculum_level(env, level: int) -> None:
    """Set curriculum stage (0-34) with bounds checking."""
    max_level = NUM_STAGES - 1
    level = max(0, min(max_level, int(level)))
    env.curriculum_level = level
    
    # Get stage info for display
    angle = get_stage_angle(level)
    blind_marker = " 🔍" if is_blind_stage(level) else ""
    turn_marker = " 🔄" if is_turn_stage(level) and not is_blind_stage(level) else ""
    
    print(f"\n{'=' * 70}")
    print(f"[CURRICULUM] {STAGE_NAMES[level]}{turn_marker}{blind_marker}")
    print(f"{'=' * 70}\n")


def get_stage_info(stage: int) -> dict:
    """
    Get detailed info about a stage.
    
    Returns:
        dict with stage parameters and metadata
    """
    if stage < 0 or stage >= NUM_STAGES:
        raise ValueError(f"Invalid stage: {stage}")

    info = {
        "name": STAGE_NAMES[stage],
        "stage": stage,
        "type": "forward" if stage <= 3 else "offset",
        "replay_prob": _get_replay_probability(stage),
        "is_turn_stage": is_turn_stage(stage),
        "is_blind_stage": is_blind_stage(stage),
    }

    if stage <= 3:
        min_d, max_d = FORWARD_CONFIGS[stage]
        info.update({
            "min_dist": min_d,
            "max_dist": max_d,
            "angle_deg": 0.0,
            "lateral_m": 0.0,
        })
    else:
        angle, lateral, min_d, max_d = OFFSET_CONFIGS[stage]
        info.update({
            "min_dist": min_d,
            "max_dist": max_d,
            "angle_deg": angle,
            "lateral_m": lateral,
        })

    return info


def get_all_stage_configs() -> list[dict]:
    """Get configurations for all stages."""
    return [get_stage_info(i) for i in range(NUM_STAGES)]
