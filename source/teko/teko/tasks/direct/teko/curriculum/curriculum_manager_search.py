# SPDX-License-Identifier: BSD-3-Clause
"""
SEARCH CURRICULUM FOR TEKO - Arena-Wide Spawning (FIXED)
========================================================

Progressive curriculum for learning search + approach + dock behavior.
Robot spawns anywhere in the arena at increasing distances from dock.

FIX: Proper handling of global vs local coordinates.

Stages:
- S0-S4: Progressive distance and angle (learning to search)
- S5-S7: Full arena coverage with 180° rotation

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations

import math
import torch
import numpy as np
from ..utils.geometry_utils import yaw_to_quat


# =============================================================================
# ARENA AND DOCK CONFIGURATION
# =============================================================================

# Arena limits (must match teko_env_cfg.py)
ARENA_HALF_X = 1.8  # Arena: X ∈ [-1.8, +1.8]
ARENA_HALF_Y = 2.4  # Arena: Y ∈ [-2.4, +2.4]
ARENA_MARGIN = 0.35  # Safety margin from walls

# Dock LOCAL position (must match teko_static.py)
DOCK_LOCAL_X = 1.5
DOCK_LOCAL_Y = 0.0
DOCK_LOCAL_Z = 0.40

# Robot spawn height
SPAWN_Z = 0.40

# Minimum distance from dock (to avoid spawning on top of it)
MIN_DOCK_DISTANCE = 0.45


# =============================================================================
# STAGE DEFINITIONS (8 stages: S0-S7)
# =============================================================================

STAGE_NAMES = [
    "Stage 0: Close + Small Angle (0.5-0.8m, ±30°)",
    "Stage 1: Close + Medium Angle (0.6-1.0m, ±60°)",
    "Stage 2: Medium + Large Angle (0.8-1.2m, ±90°)",
    "Stage 3: Medium + Larger Angle (1.0-1.4m, ±120°)",
    "Stage 4: Far + Near-Rear (1.2-1.6m, ±150°)",
    "Stage 5: Far + Full Rotation (1.4-2.0m, ±180°)",
    "Stage 6: Very Far + Full Rotation (1.8-2.4m, ±180°)",
    "Stage 7: Arena-Wide + Full Rotation (2.2-3.0m, ±180°)",
]

# Stage config: (min_dist, max_dist, max_angle_deg)
STAGE_CONFIGS = {
    0: (0.50, 0.80, 30.0),
    1: (0.60, 1.00, 60.0),
    2: (0.80, 1.20, 90.0),
    3: (1.00, 1.40, 120.0),
    4: (1.20, 1.60, 150.0),
    5: (1.40, 2.00, 180.0),
    6: (1.80, 2.40, 180.0),
    7: (2.20, 3.00, 180.0),
}

# Replay probability (mix in easier stages)
REPLAY_PROBS = {
    0: 0.0,   # No replay for S0
    1: 0.15,
    2: 0.15,
    3: 0.18,
    4: 0.18,
    5: 0.20,
    6: 0.22,
    7: 0.25,
}


# =============================================================================
# SPAWN FUNCTIONS
# =============================================================================

def _is_valid_local_position(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Check if LOCAL position is within arena bounds with margin."""
    valid_x = (x > -(ARENA_HALF_X - ARENA_MARGIN)) & (x < (ARENA_HALF_X - ARENA_MARGIN))
    valid_y = (y > -(ARENA_HALF_Y - ARENA_MARGIN)) & (y < (ARENA_HALF_Y - ARENA_MARGIN))
    return valid_x & valid_y


def _spawn_at_distance(
    env,
    env_ids: torch.Tensor,
    min_dist: float,
    max_dist: float,
    max_angle_deg: float,
) -> None:
    """
    Spawn robots at random positions around the dock.
    
    Works in LOCAL coordinates, then converts to GLOBAL for simulation.
    
    Args:
        env: Environment instance
        env_ids: Indices of environments to reset
        min_dist: Minimum distance from dock
        max_dist: Maximum distance from dock  
        max_angle_deg: Maximum robot yaw offset from facing dock (±degrees)
    """
    num = len(env_ids)
    device = env.device
    
    # Get environment origins (for converting local -> global later)
    env_origins = env.scene.env_origins[env_ids]  # [num, 3]
    
    # Dock position in LOCAL coordinates (same for all envs)
    dock_local_x = DOCK_LOCAL_X
    dock_local_y = DOCK_LOCAL_Y
    
    # Sample positions in polar coordinates relative to dock (LOCAL)
    max_attempts = 20
    
    # Initialize output tensors (LOCAL coordinates)
    final_local_x = torch.zeros(num, device=device)
    final_local_y = torch.zeros(num, device=device)
    valid_mask = torch.zeros(num, dtype=torch.bool, device=device)
    
    for attempt in range(max_attempts):
        # Only sample for invalid positions
        invalid_mask = ~valid_mask
        n_invalid = invalid_mask.sum().item()
        
        if n_invalid == 0:
            break
        
        # Sample distance from dock
        dist = torch.rand(n_invalid, device=device) * (max_dist - min_dist) + min_dist
        
        # Sample angle around dock (full 360° for position)
        # Bias toward angles that keep robot inside arena (away from +X wall)
        theta = torch.rand(n_invalid, device=device) * 2 * math.pi
        
        # Convert to cartesian offset from dock
        offset_x = dist * torch.cos(theta)
        offset_y = dist * torch.sin(theta)
        
        # LOCAL position (relative to env origin)
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
    
    # Fallback for any still-invalid positions: spawn behind dock (safe zone)
    still_invalid = ~valid_mask
    if still_invalid.any():
        n_fallback = still_invalid.sum().item()
        # Spawn behind the dock (negative X direction from dock)
        fallback_dist = (min_dist + max_dist) / 2
        fallback_angle = torch.rand(n_fallback, device=device) * math.pi + math.pi/2  # 90° to 270° (behind/sides)
        
        fallback_x = dock_local_x + fallback_dist * torch.cos(fallback_angle)
        fallback_y = dock_local_y + fallback_dist * torch.sin(fallback_angle)
        
        # Clamp to safe area
        fallback_x = torch.clamp(fallback_x, -(ARENA_HALF_X - ARENA_MARGIN), ARENA_HALF_X - ARENA_MARGIN)
        fallback_y = torch.clamp(fallback_y, -(ARENA_HALF_Y - ARENA_MARGIN), ARENA_HALF_Y - ARENA_MARGIN)
        
        final_local_x[still_invalid] = fallback_x
        final_local_y[still_invalid] = fallback_y
        
        print(f"[SEARCH CURRICULUM] {n_fallback} envs used fallback spawn position")
    
    # Robot yaw: random offset from "facing the dock"
    # Compute angle TO dock from spawn position (in local coords)
    vec_to_dock_x = dock_local_x - final_local_x
    vec_to_dock_y = dock_local_y - final_local_y
    angle_to_dock = torch.atan2(vec_to_dock_y, vec_to_dock_x)
    
    # Robot's rear should face dock for docking, so robot yaw = angle_to_dock + π
    # Add random offset based on stage difficulty
    max_yaw_offset = math.radians(max_angle_deg)
    yaw_offset = (torch.rand(num, device=device) * 2 - 1) * max_yaw_offset
    
    # Final yaw: pointing away from dock (rear toward dock) + random offset
    robot_yaw = angle_to_dock + math.pi + yaw_offset
    
    # Convert LOCAL -> GLOBAL coordinates
    global_x = final_local_x + env_origins[:, 0]
    global_y = final_local_y + env_origins[:, 1]
    global_z = torch.full((num,), SPAWN_Z, device=device)
    
    # Build pose in GLOBAL coordinates
    pos = torch.stack([global_x, global_y, global_z], dim=1)
    quat = yaw_to_quat(robot_yaw)
    
    # Write to simulation (expects GLOBAL coordinates)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids)


# =============================================================================
# CURRICULUM RESET FUNCTIONS
# =============================================================================

def reset_environment_curriculum_search(env, env_ids: torch.Tensor) -> None:
    """Reset environments using search curriculum."""
    current_stage = int(env.curriculum_level)
    num = len(env_ids)
    
    if current_stage == 0:
        # No replay for stage 0
        min_dist, max_dist, max_angle = STAGE_CONFIGS[0]
        _spawn_at_distance(env, env_ids, min_dist, max_dist, max_angle)
        return
    
    # Mix in previous stage for anti-forgetting
    device = env.device
    mix_prob = REPLAY_PROBS.get(current_stage, 0.2)
    mix_prev = torch.rand(num, device=device) < mix_prob
    prev_ids = env_ids[mix_prev]
    curr_ids = env_ids[~mix_prev]
    
    if len(prev_ids) > 0:
        prev_stage = max(0, current_stage - 1)
        min_dist, max_dist, max_angle = STAGE_CONFIGS[prev_stage]
        _spawn_at_distance(env, prev_ids, min_dist, max_dist, max_angle)
    
    if len(curr_ids) > 0:
        min_dist, max_dist, max_angle = STAGE_CONFIGS[current_stage]
        _spawn_at_distance(env, curr_ids, min_dist, max_dist, max_angle)


def set_curriculum_level_search(env, level: int) -> None:
    """Set curriculum level for search curriculum."""
    max_level = len(STAGE_NAMES) - 1
    level = max(0, min(max_level, int(level)))
    env.curriculum_level = level
    
    config = STAGE_CONFIGS[level]
    print(f"\n{'=' * 70}")
    print(f"[SEARCH CURRICULUM] {STAGE_NAMES[level]}")
    print(f"  Distance: {config[0]:.1f} - {config[1]:.1f}m")
    print(f"  Max Angle: ±{config[2]:.0f}°")
    print(f"{'=' * 70}\n")


def get_search_stage_info(stage: int) -> dict:
    """Get info about a search curriculum stage."""
    if stage < 0 or stage >= len(STAGE_NAMES):
        raise ValueError(f"Invalid stage: {stage}")
    
    min_dist, max_dist, max_angle = STAGE_CONFIGS[stage]
    return {
        "name": STAGE_NAMES[stage],
        "stage": stage,
        "min_dist": min_dist,
        "max_dist": max_dist,
        "max_angle_deg": max_angle,
        "replay_prob": REPLAY_PROBS.get(stage, 0.2),
    }