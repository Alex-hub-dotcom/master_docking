# SPDX-License-Identifier: BSD-3-Clause
"""
17-STAGE CURRICULUM FOR TEKO (v11.0 – LOW-SAMPLE OPTIMIZED)
============================================================

Optimized for 64×64 grayscale vision-based docking with
limited parallel environments (65-100).

Key changes from 35-stage version:
- Consolidated to 17 stages (removed micro-steps)
- Minimal replay to ensure accurate SSR tracking
- Balanced thresholds (not too easy, not impossible)
- Larger yaw jumps are safe with fewer stages

Design principles:
1. Each stage is meaningfully different
2. Progression is smooth but not overly gradual
3. Lower replay = more accurate success metrics
4. Quality over quantity of stages

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations

import numpy as np
import torch
from ..utils.geometry_utils import yaw_to_quat


# =============================================================================
# STAGE DEFINITIONS (17 STAGES)
# =============================================================================

STAGE_NAMES = [
    "Stage 0:  Baby Steps (5–15 cm, forward)",
    "Stage 1:  Forward (15–30 cm, forward)",
    "Stage 2:  Long Forward (25–40 cm, forward)",
    "Stage 3:  Small Offset (±5°, ±2 cm)",
    "Stage 4:  Offset (±10°, ±3 cm)",
    "Stage 5:  Offset (±15°, ±4 cm)",
    "Stage 6:  Offset (±18°, ±5 cm)",      # updated
    "Stage 7:  Offset (±22°, ±6 cm)",      # updated
    "Stage 8:  Offset (±28°, ±7 cm)",      # updated
    "Stage 9:  Offset (±35°, ±8 cm)",      # updated
    "Stage 10: Turn (±45°, ±8 cm)",        # updated
    "Stage 11: Turn (±60°, ±6 cm)",        # updated
    "Stage 12: Turn (±75°, ±5 cm)",        # updated
    "Stage 13: Turn (±90°, ±5 cm)",        # updated
    "Stage 14: Blind (±120°, ±5 cm)",
    "Stage 15: Blind (±150°, ±5 cm)",
    "Stage 16: Full Circle (±180°, ±5 cm)",
]

# Number of stages
NUM_STAGES = len(STAGE_NAMES)

# =============================================================================
# STAGE CONFIGURATIONS
# =============================================================================

# Forward stages: (min_dist, max_dist)
# Forward stages: (min_dist, max_dist)
FORWARD_CONFIGS = {
    0: (0.05, 0.15),
    1: (0.15, 0.30),
    2: (0.25, 0.40),
}

# Offset stages: (angle_deg, lateral_m, min_dist, max_dist)
OFFSET_CONFIGS = {
    3:  (5.0,  0.02, 0.25, 0.36),
    4:  (10.0, 0.03, 0.25, 0.38),
    5:  (15.0, 0.04, 0.25, 0.40),
    6:  (15.0, 0.06, 0.25, 0.40),   # was 20°
    7:  (22.0, 0.06, 0.25, 0.42),   # was 30°
    8:  (28.0, 0.07, 0.25, 0.42),   # new
    9:  (35.0, 0.08, 0.28, 0.45),   # was 45°
    10: (45.0, 0.08, 0.28, 0.45),   # was 60°
    11: (60.0, 0.06, 0.30, 0.48),   # was 90°
    12: (75.0, 0.05, 0.30, 0.50),   # new
    13: (90.0, 0.05, 0.32, 0.50),   # was 105°
    14: (120.0, 0.05, 0.36, 0.54),
    15: (150.0, 0.05, 0.40, 0.58),
    16: (180.0, 0.05, 0.45, 0.65),
}
# =============================================================================
# REPLAY PROBABILITIES (MINIMAL FOR ACCURATE SSR)
# =============================================================================

REPLAY_PROBS = {
    "early": 0.05,          # S1-S5
    "medium": 0.05,         # S6-S7
    "turn_visible": 0.08,   # S8-S10
    "turn_blind": 0.03,     # S11-S16 (minimal replay)
}

# Stage category boundaries
TURN_STAGE_START = 8        # S8+ are turn stages
BLIND_STAGE_START = 11      # S11+ are blind search stages


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
    if stage <= 2:
        return 0.0
    return OFFSET_CONFIGS[stage][0]


# =============================================================================
# CURRICULUM RESET FUNCTIONS
# =============================================================================

def reset_environment_curriculum(env, env_ids: torch.Tensor) -> None:
    """
    Reset environments according to current curriculum stage.
    
    Includes minimal replay of previous stage for anti-forgetting.
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
    elif stage >= 6:
        return REPLAY_PROBS["medium"]
    else:
        return REPLAY_PROBS["early"]


def _reset_stage_dispatch(env, env_ids: torch.Tensor, stage: int) -> None:
    """Route to correct reset function based on stage type."""
    if stage < 0 or stage >= NUM_STAGES:
        raise ValueError(f"Invalid stage: {stage}")

    if stage <= 2:
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
    """Set curriculum stage (0-16) with bounds checking."""
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
        "type": "forward" if stage <= 2 else "offset",
        "replay_prob": _get_replay_probability(stage),
        "is_turn_stage": is_turn_stage(stage),
        "is_blind_stage": is_blind_stage(stage),
    }

    if stage <= 2:
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