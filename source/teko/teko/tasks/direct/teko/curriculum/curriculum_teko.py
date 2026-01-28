# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO UNIFIED CURRICULUM - Final Version (50 Stages)
====================================================

Combines:
- S0-S41:  Precision docking (original 42 stages) - EXACT same logic
- S42-S49: Arena-wide search (8 stages) - EXACT same logic

Progressive tolerance for more precise docking:
- S0-S20:  3.0cm (learning basics)
- S21-S30: 2.0cm (refinement)  
- S31-S41: 1.5cm (precision)
- S42-S49: 1.0cm (search + high precision)

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations

import math
import numpy as np
import torch
from ..utils.geometry_utils import yaw_to_quat


# =============================================================================
# CONNECTOR GEOMETRY (from original curriculum_manager.py)
# =============================================================================

FEMALE_OFFSET_X = 0.24
MALE_OFFSET_X = -0.227
CONNECTOR_GAP = FEMALE_OFFSET_X - MALE_OFFSET_X


# =============================================================================
# ARENA AND DOCK CONFIGURATION (from curriculum_manager_search.py)
# =============================================================================

ARENA_HALF_X = 1.8   # Arena: X ∈ [-1.8, +1.8]
ARENA_HALF_Y = 2.4   # Arena: Y ∈ [-2.4, +2.4]
ARENA_MARGIN = 0.35  # Safety margin from walls

# Dock LOCAL position (for search stages)
DOCK_LOCAL_X = 1.5
DOCK_LOCAL_Y = 0.0
DOCK_LOCAL_Z = 0.40

SPAWN_Z = 0.40


# =============================================================================
# PROGRESSIVE TOLERANCE - Makes docking progressively more precise
# =============================================================================

def get_success_threshold(stage: int) -> float:
    """
    Get success threshold (meters) based on current stage.
    
    Returns:
        Threshold in meters for considering a docking successful.
    """
    if stage <= 20:
        return 0.030  # 3.0cm
    elif stage <= 30:
        return 0.020  # 2.0cm
    elif stage <= 46:
        return 0.015  # 1.5cm
    else:
        return 0.010  # 1.0cm (most demanding)


# =============================================================================
# STAGE NAMES (50 stages total)
# =============================================================================

STAGE_NAMES = [
    # =========================================================================
    # PHASE 1: PRECISION DOCKING (S0-S41) - From original curriculum_manager.py
    # =========================================================================
    
    # Forward stages (S0-S3)
    "S0:  Baby Steps (5–12 cm, forward)",
    "S1:  Forward 1 (10–18 cm, forward)",
    "S2:  Forward 2 (15–25 cm, forward)",
    "S3:  Medium Forward (20–35 cm, forward)",
    
    # First offsets (S4-S6)
    "S4:  Tiny Offset (±4°, ±2 cm)",
    "S5:  Small Offset (±7°, ±3 cm)",
    "S6:  Offset (±10°, ±3 cm)",
    
    # Micro-steps (S7-S12)
    "S7:  Offset (±11°, ±3 cm)",
    "S8:  Offset (±12°, ±3 cm)",
    "S9:  Offset (±13°, ±3 cm)",
    "S10: Offset (±15°, ±3 cm)",
    "S11: Offset (±17°, ±3 cm)",
    "S12: Offset (±19°, ±4 cm)",
    
    # Ultra micro-steps (S13-S22)
    "S13: Offset (±20°, ±4 cm)",
    "S14: Offset (±20°, ±5 cm)",
    "S15: Offset (±20°, ±6 cm)",
    "S16: Offset (±22°, ±6 cm)",
    "S17: Offset (±24°, ±6 cm)",
    "S18: Offset (±24°, ±7 cm)",
    "S19: Offset (±24°, ±8 cm)",
    "S20: Offset (±27°, ±8 cm)",
    "S21: Offset (±30°, ±8 cm)",
    "S22: Offset (±30°, ±10 cm)",
    
    # 180° turn stages - GRADUAL ~7° steps (S23-S41)
    "S23: Large Angle (±45°, ±8 cm)",
    "S24: Large Angle (±52°, ±7 cm)",
    "S25: Large Angle (±60°, ±7 cm)",
    "S26: Large Angle (±67°, ±6 cm)",
    "S27: Large Angle (±75°, ±6 cm)",
    "S28: Large Angle (±82°, ±5 cm)",
    "S29: Perpendicular (±90°, ±5 cm)",
    "S30: Past Perpendicular (±97°, ±5 cm)",
    "S31: Past Perpendicular (±105°, ±5 cm)",
    "S32: Large Turn (±112°, ±4 cm)",
    "S33: Large Turn (±120°, ±4 cm)",
    "S34: Large Turn (±127°, ±4 cm)",
    "S35: Rear Angle (±135°, ±4 cm)",
    "S36: Rear Angle (±142°, ±3 cm)",
    "S37: Rear Angle (±150°, ±3 cm)",
    "S38: Rear Angle (±157°, ±3 cm)",
    "S39: Near Full (±165°, ±3 cm)",
    "S40: Near Full (±172°, ±3 cm)",
    "S41: Full Turn (±180°, ±3 cm)",
    
    # =========================================================================
    # PHASE 2: ARENA SEARCH (S42-S49) - From curriculum_manager_search.py
    # =========================================================================
    
    "S42: Search Close + Small Angle (0.5-0.8m, ±30°)",
    "S43: Search Close + Medium Angle (0.6-1.0m, ±60°)",
    "S44: Search Medium + Large Angle (0.8-1.2m, ±90°)",
    "S45: Search Medium + Larger Angle (1.0-1.4m, ±120°)",
    "S46: Search Far + Near-Rear (1.2-1.6m, ±150°)",
    "S47: Search Far + Full Rotation (1.4-2.0m, ±180°)",
    "S48: Search Very Far + Full Rotation (1.8-2.4m, ±180°)",
    "S49: Search Arena-Wide + Full Rotation (2.2-3.0m, ±180°)",
]


# =============================================================================
# STAGE CONFIGURATIONS
# =============================================================================

# -------------------------------------------------------------------------
# PRECISION STAGES (S0-S41) - EXACT same as original curriculum_manager.py
# -------------------------------------------------------------------------

FORWARD_CONFIGS = {
    0: (0.05, 0.12),
    1: (0.10, 0.18),
    2: (0.15, 0.25),
    3: (0.20, 0.35),
}

# Format: (angle_deg, lateral_m, min_dist, max_dist)
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

# -------------------------------------------------------------------------
# SEARCH STAGES (S42-S49) - EXACT same as curriculum_manager_search.py
# -------------------------------------------------------------------------

# Format: (min_dist, max_dist, max_angle_deg)
SEARCH_CONFIGS = {
    42: (0.50, 0.80, 30.0),
    43: (0.60, 1.00, 60.0),
    44: (0.80, 1.20, 90.0),
    45: (1.00, 1.40, 120.0),
    46: (1.20, 1.60, 150.0),
    47: (1.40, 2.00, 180.0),
    48: (1.80, 2.40, 180.0),
    49: (2.20, 3.00, 180.0),
}


# =============================================================================
# REPLAY PROBABILITIES (anti-forgetting)
# =============================================================================

def _get_replay_probability(stage: int) -> float:
    """Get replay probability for mixing in previous stage."""
    if stage == 0:
        return 0.0
    elif stage <= 6:
        return 0.15  # early
    elif stage <= 12:
        return 0.18  # micro
    elif stage <= 22:
        return 0.22  # ultra
    elif stage <= 46:
        return 0.25  # turn
    elif stage <= 46:
        return 0.015  # 1.5cm
    else:
        # Search stages (from curriculum_manager_search.py)
        search_replay = {
            42: 0.15,
            43: 0.15,
            44: 0.18,
            45: 0.18,
            46: 0.20,
            47: 0.22,
            48: 0.22,
            49: 0.25,
        }
        return search_replay.get(stage, 0.20)


# =============================================================================
# SPAWN FUNCTIONS - PRECISION (S0-S41) - EXACT same logic as original
# =============================================================================

def _forward_reset(env, env_ids: torch.Tensor, min_dist: float, max_dist: float) -> None:
    """Forward reset - EXACT same as original curriculum_manager.py."""
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
    """Offset reset - EXACT same as original curriculum_manager.py."""
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


# =============================================================================
# SPAWN FUNCTIONS - SEARCH (S42-S49) - EXACT same logic as original
# =============================================================================

def _is_valid_local_position(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Check if LOCAL position is within arena bounds with margin."""
    valid_x = (x > -(ARENA_HALF_X - ARENA_MARGIN)) & (x < (ARENA_HALF_X - ARENA_MARGIN))
    valid_y = (y > -(ARENA_HALF_Y - ARENA_MARGIN)) & (y < (ARENA_HALF_Y - ARENA_MARGIN))
    return valid_x & valid_y


def _search_reset(
    env,
    env_ids: torch.Tensor,
    min_dist: float,
    max_dist: float,
    max_angle_deg: float,
) -> None:
    """Search reset - EXACT same logic as curriculum_manager_search.py."""
    num = len(env_ids)
    device = env.device
    
    # Get environment origins (for converting local -> global)
    env_origins = env.scene.env_origins[env_ids]
    
    # Dock position in LOCAL coordinates
    dock_local_x = DOCK_LOCAL_X
    dock_local_y = DOCK_LOCAL_Y
    
    max_attempts = 20
    
    # Initialize output tensors (LOCAL coordinates)
    final_local_x = torch.zeros(num, device=device)
    final_local_y = torch.zeros(num, device=device)
    valid_mask = torch.zeros(num, dtype=torch.bool, device=device)
    
    for attempt in range(max_attempts):
        invalid_mask = ~valid_mask
        n_invalid = invalid_mask.sum().item()
        
        if n_invalid == 0:
            break
        
        # Sample distance and angle around dock (full 360° for position)
        dist = torch.rand(n_invalid, device=device) * (max_dist - min_dist) + min_dist
        theta = torch.rand(n_invalid, device=device) * 2 * math.pi
        
        # Convert to cartesian offset from dock
        offset_x = dist * torch.cos(theta)
        offset_y = dist * torch.sin(theta)
        
        # LOCAL position
        candidate_local_x = dock_local_x + offset_x
        candidate_local_y = dock_local_y + offset_y
        
        # Validate in LOCAL coordinates
        candidate_valid = _is_valid_local_position(candidate_local_x, candidate_local_y)
        
        # Update valid positions
        invalid_indices = invalid_mask.nonzero(as_tuple=False).squeeze(-1)
        for i, idx in enumerate(invalid_indices):
            if candidate_valid[i]:
                final_local_x[idx] = candidate_local_x[i]
                final_local_y[idx] = candidate_local_y[i]
                valid_mask[idx] = True
    
    # Fallback for still-invalid positions
    still_invalid = ~valid_mask
    if still_invalid.any():
        n_fallback = still_invalid.sum().item()
        fallback_dist = (min_dist + max_dist) / 2
        fallback_angle = torch.rand(n_fallback, device=device) * math.pi + math.pi/2
        
        fallback_x = dock_local_x + fallback_dist * torch.cos(fallback_angle)
        fallback_y = dock_local_y + fallback_dist * torch.sin(fallback_angle)
        
        fallback_x = torch.clamp(fallback_x, -(ARENA_HALF_X - ARENA_MARGIN), ARENA_HALF_X - ARENA_MARGIN)
        fallback_y = torch.clamp(fallback_y, -(ARENA_HALF_Y - ARENA_MARGIN), ARENA_HALF_Y - ARENA_MARGIN)
        
        final_local_x[still_invalid] = fallback_x
        final_local_y[still_invalid] = fallback_y
        
        print(f"[CURRICULUM] {n_fallback} envs used fallback spawn position")
    
    # Robot yaw: random offset from "facing the dock"
    vec_to_dock_x = dock_local_x - final_local_x
    vec_to_dock_y = dock_local_y - final_local_y
    angle_to_dock = torch.atan2(vec_to_dock_y, vec_to_dock_x)
    
    max_yaw_offset = math.radians(max_angle_deg)
    yaw_offset = (torch.rand(num, device=device) * 2 - 1) * max_yaw_offset
    
    # Final yaw: pointing away from dock (rear toward dock) + random offset
    robot_yaw = angle_to_dock + math.pi + yaw_offset
    
    # Convert LOCAL -> GLOBAL coordinates
    global_x = final_local_x + env_origins[:, 0]
    global_y = final_local_y + env_origins[:, 1]
    global_z = torch.full((num,), SPAWN_Z, device=device)
    
    pos = torch.stack([global_x, global_y, global_z], dim=1)
    quat = yaw_to_quat(robot_yaw)
    
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids)


# =============================================================================
# MAIN DISPATCH FUNCTION
# =============================================================================

def _reset_stage_dispatch(env, env_ids: torch.Tensor, stage: int) -> None:
    """Route to correct spawn function based on stage."""
    if stage < 0 or stage >= len(STAGE_NAMES):
        raise ValueError(f"Invalid stage: {stage}")
    
    # Phase 1: Precision stages (S0-S41)
    if stage <= 3:
        # Forward reset
        min_d, max_d = FORWARD_CONFIGS[stage]
        _forward_reset(env, env_ids, min_d, max_d)
    elif stage <= 46:
        # Offset reset
        angle, lateral, min_d, max_d = OFFSET_CONFIGS[stage]
        _offset_reset(env, env_ids, angle, lateral, min_d, max_d)
    elif stage <= 46:
        return 0.015  # 1.5cm
    else:
        # Phase 2: Search stages (S42-S49)
        min_dist, max_dist, max_angle = SEARCH_CONFIGS[stage]
        _search_reset(env, env_ids, min_dist, max_dist, max_angle)


# =============================================================================
# PUBLIC API - Main curriculum functions
# =============================================================================

def reset_environment_curriculum(env, env_ids: torch.Tensor) -> None:
    """
    Reset environments using unified curriculum.
    
    Includes anti-forgetting mechanism that occasionally spawns
    at previous stage configurations.
    """
    current_stage = int(env.curriculum_level)
    num = len(env_ids)

    if current_stage == 0:
        _reset_stage_dispatch(env, env_ids, current_stage)
        return

    # Mix in previous stage for anti-forgetting
    device = env.device
    mix_prob = _get_replay_probability(current_stage)
    mix_prev = torch.rand(num, device=device) < mix_prob
    prev_ids = env_ids[mix_prev]
    curr_ids = env_ids[~mix_prev]

    if len(prev_ids) > 0:
        _reset_stage_dispatch(env, prev_ids, current_stage - 1)
    if len(curr_ids) > 0:
        _reset_stage_dispatch(env, curr_ids, current_stage)


def set_curriculum_level(env, level: int) -> None:
    """Set curriculum level with detailed logging."""
    max_level = len(STAGE_NAMES) - 1
    level = max(0, min(max_level, int(level)))
    env.curriculum_level = level
    
    threshold = get_success_threshold(level)
    replay_prob = _get_replay_probability(level)
    
    print(f"\n{'=' * 70}")
    print(f"[CURRICULUM] {STAGE_NAMES[level]}")
    print(f"  Success threshold: {threshold*100:.1f}cm")
    print(f"  Replay probability: {replay_prob:.0%}")
    print(f"{'=' * 70}\n")


def should_advance_curriculum(success_rate: float, current_level: int) -> bool:
    """Check if curriculum should advance based on success rate."""
    if current_level >= len(STAGE_NAMES) - 1:
        return False
    return success_rate >= 0.75  # 75% threshold to advance


def get_stage_info(stage: int) -> dict:
    """Get detailed info about a curriculum stage."""
    if stage < 0 or stage >= len(STAGE_NAMES):
        raise ValueError(f"Invalid stage: {stage}")
    
    info = {
        "name": STAGE_NAMES[stage],
        "stage": stage,
        "success_threshold": get_success_threshold(stage),
        "replay_prob": _get_replay_probability(stage),
    }
    
    if stage <= 3:
        # Forward stage
        min_d, max_d = FORWARD_CONFIGS[stage]
        info.update({
            "type": "forward",
            "min_dist": min_d,
            "max_dist": max_d,
            "angle_deg": 0.0,
            "lateral_m": 0.0,
        })
    elif stage <= 46:
        # Offset stage
        angle, lateral, min_d, max_d = OFFSET_CONFIGS[stage]
        info.update({
            "type": "offset",
            "min_dist": min_d,
            "max_dist": max_d,
            "angle_deg": angle,
            "lateral_m": lateral,
        })
    elif stage <= 46:
        return 0.015  # 1.5cm
    else:
        # Search stage
        min_dist, max_dist, max_angle = SEARCH_CONFIGS[stage]
        info.update({
            "type": "search",
            "min_dist": min_dist,
            "max_dist": max_dist,
            "max_angle_deg": max_angle,
        })
    
    return info


def get_all_stage_configs() -> list[dict]:
    """Get configuration for all stages."""
    return [get_stage_info(i) for i in range(len(STAGE_NAMES))]


# =============================================================================
# CONSTANTS FOR EXTERNAL USE
# =============================================================================

MAX_STAGE = 49
TOTAL_STAGES = 50
PRECISION_STAGES = 42  # S0-S41
SEARCH_STAGES = 8      # S42-S49