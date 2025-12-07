# SPDX-License-Identifier: BSD-3-Clause
"""
17-STAGE CURRICULUM FOR TEKO (v11.2 – LOW-SAMPLE OPTIMIZED)
============================================================

Optimized for 64×64 grayscale vision-based docking with
limited parallel environments (≈65–100 envs).

Key design ideas:
- 17 consolidated stages (no micro-steps)
- Minimal replay for reliable SSR estimates
- Smooth but meaningful progression between stages
- Same logic works for vision and state-based (dx, dy, dz, yaw_error)

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
    "Stage 6:  Offset (±18°, ±5 cm)",
    "Stage 7:  Offset (±22°, ±6 cm)",
    "Stage 8:  Offset (±28°, ±7 cm)",
    "Stage 9:  Offset (±35°, ±8 cm)",
    "Stage 10: Turn (±45°, ±8 cm)",
    "Stage 11: Turn (±60°, ±6 cm)",
    "Stage 12: Turn (±75°, ±5 cm)",
    "Stage 13: Turn (±90°, ±5 cm)",
    "Stage 14: Blind (±120°, ±5 cm)",
    "Stage 15: Blind (±150°, ±5 cm)",
    "Stage 16: Full Circle (±180°, ±5 cm)",
]

# Total number of stages
NUM_STAGES = len(STAGE_NAMES)


# =============================================================================
# STAGE CONFIGURATIONS
# =============================================================================

# Forward stages: (min_dist, max_dist)
FORWARD_CONFIGS = {
    0: (0.05, 0.15),
    1: (0.15, 0.30),
    2: (0.25, 0.40),
}

# Offset / turn stages: (angle_deg, lateral_m, min_dist, max_dist)
OFFSET_CONFIGS = {
    3:  (5.0,   0.02, 0.25, 0.36),
    4:  (10.0,  0.03, 0.25, 0.38),
    5:  (15.0,  0.04, 0.25, 0.40),
    6:  (18.0,  0.05, 0.25, 0.40),  # slightly harder than S5
    7:  (22.0,  0.06, 0.25, 0.42),
    8:  (28.0,  0.07, 0.25, 0.42),
    9:  (35.0,  0.08, 0.28, 0.45),
    10: (45.0,  0.08, 0.28, 0.45),
    11: (60.0,  0.06, 0.30, 0.48),
    12: (75.0,  0.05, 0.30, 0.50),
    13: (90.0,  0.05, 0.32, 0.50),
    14: (120.0, 0.05, 0.36, 0.54),
    15: (150.0, 0.05, 0.40, 0.58),
    16: (180.0, 0.05, 0.45, 0.65),
}


# =============================================================================
# REPLAY PROBABILITIES (MINIMAL FOR CLEAN SSR)
# =============================================================================

REPLAY_PROBS = {
    "early": 0.05,          # S1–S5
    "medium": 0.05,         # S6–S7
    "turn_visible": 0.08,   # S8–S10
    "turn_blind": 0.03,     # S11–S16
}

# Category thresholds (aligned with OFFSET_CONFIGS above)
TURN_STAGE_START = 8        # S8+ = larger turns / offsets
BLIND_STAGE_START = 11      # S11+ = more “blind” search


# =============================================================================
# STAGE CLASSIFICATION HELPERS
# =============================================================================

def is_turn_stage(stage: int) -> bool:
    """Check if the stage is a turn/offset stage (>= TURN_STAGE_START)."""
    return stage >= TURN_STAGE_START


def is_blind_stage(stage: int) -> bool:
    """Check if the stage is a blind-search stage (>= BLIND_STAGE_START)."""
    return stage >= BLIND_STAGE_START


def get_stage_angle(stage: int) -> float:
    """Return nominal yaw angle (in degrees) for the given stage (0° for forward stages)."""
    if stage <= 2:
        return 0.0
    return OFFSET_CONFIGS[stage][0]


# =============================================================================
# CURRICULUM RESET FUNCTIONS
# =============================================================================

def reset_environment_curriculum(env, env_ids: torch.Tensor) -> None:
    """
    Reset environments according to the current curriculum stage.

    Includes minimal replay of the previous stage to prevent catastrophic
    forgetting, while keeping SSR estimates clean (low mixing).
    """
    current_stage = int(env.curriculum_level)
    num = len(env_ids)

    # Stage 0: no replay, only S0
    if current_stage == 0:
        _reset_stage_dispatch(env, env_ids, current_stage)
        return

    device = env.device

    # Replay probability based on stage category
    mix_prob = _get_replay_probability(current_stage)

    # Split environments between current and previous stages
    mix_prev = torch.rand(num, device=device) < mix_prob
    prev_ids = env_ids[mix_prev]
    curr_ids = env_ids[~mix_prev]

    if len(prev_ids) > 0:
        _reset_stage_dispatch(env, prev_ids, current_stage - 1)
    if len(curr_ids) > 0:
        _reset_stage_dispatch(env, curr_ids, current_stage)


def _get_replay_probability(stage: int) -> float:
    """Return replay probability for the previous stage."""
    if stage >= BLIND_STAGE_START:
        return REPLAY_PROBS["turn_blind"]
    elif stage >= TURN_STAGE_START:
        return REPLAY_PROBS["turn_visible"]
    elif stage >= 6:
        return REPLAY_PROBS["medium"]
    else:
        return REPLAY_PROBS["early"]


def _reset_stage_dispatch(env, env_ids: torch.Tensor, stage: int) -> None:
    """Route to the appropriate reset function based on stage type."""
    if stage < 0 or stage >= NUM_STAGES:
        raise ValueError(f"Invalid stage: {stage}")

    if stage <= 2:
        # Forward stages
        min_d, max_d = FORWARD_CONFIGS[stage]
        _forward_reset(env, env_ids, min_d, max_d)
    else:
        # Offset / turn stages
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

    # Random distance within [min_dist, max_dist]
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
    Offset reset: random yaw and lateral offset within the given limits.

    Robot spawns behind the goal, with yaw offset and lateral displacement.
    """
    num = len(env_ids)
    device = env.device

    # Random distance within [min_dist, max_dist]
    dist = torch.rand(num, device=device) * (max_dist - min_dist) + min_dist

    # Random yaw around π with amplitude given by angle_deg
    max_yaw = np.deg2rad(angle_deg)
    yaw = np.pi + (torch.rand(num, device=device) * 2 * max_yaw - max_yaw)

    # Random lateral offset
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=device) * 2 * lateral_m - lateral_m)
    z = torch.full((num,), 0.40, device=device)

    # Apply pose
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids)


# =============================================================================
# CURRICULUM CONTROL / INFO
# =============================================================================

def set_curriculum_level(env, level: int) -> None:
    """Set the curriculum stage (0–16), with bounds checking and friendly logging."""
    max_level = NUM_STAGES - 1
    level = max(0, min(max_level, int(level)))
    env.curriculum_level = level

    blind_marker = " 🔍" if is_blind_stage(level) else ""
    turn_marker = " 🔄" if is_turn_stage(level) and not is_blind_stage(level) else ""

    print(f"\n{'=' * 70}")
    print(f"[CURRICULUM] {STAGE_NAMES[level]}{turn_marker}{blind_marker}")
    print(f"{'=' * 70}\n")


def get_stage_info(stage: int) -> dict:
    """
    Return detailed info for a given stage.

    Returns:
        dict with parameters and metadata for the stage.
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
    """Return a list with configurations for all stages (useful for debug/logging)."""
    return [get_stage_info(i) for i in range(NUM_STAGES)]
