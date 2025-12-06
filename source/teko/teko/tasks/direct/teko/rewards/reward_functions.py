# SPDX-License-Identifier: BSD-3-Clause
"""
Reward Functions for TEKO Docking (v9.3 – UNIFIED FINAL)
========================================================

Unified reward function for both state-based and vision-based training.
Optimized for 64×64 and 84×84 grayscale observations.

Key design principles:
- Gentle time penalty (encourages exploration)
- Strong progress/alignment shaping (guides learning)
- Nuclear terminal penalties (prevents bad behaviors)
- Curriculum-friendly (allows time to master stages)

Reward structure:
1. Distance reward     - Continuous penalty for being far
2. Progress reward     - Reward for getting closer
3. Alignment reward    - Reward for correct orientation
4. Facing bonus        - Bonus when well-aligned and close
5. Approach bonus      - Bonus for approaching while aligned
6. Turning bonus       - Bonus for correcting misalignment
7. Collision penalty   - Terminal penalty for crashes
8. Boundary penalty    - Terminal penalty for leaving arena
9. Success bonus       - Terminal reward for docking
10. Time penalty       - Gentle penalty for slow episodes

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations
import torch
import numpy as np


# =============================================================================
# REWARD HYPERPARAMETERS
# =============================================================================

REWARD_CONFIG = {
    # Shaping rewards
    "distance_scale": -2.0,
    "distance_min": -4.0,
    "distance_max": 0.0,
    
    "progress_scale": 8.0,
    "progress_min": -4.0,
    "progress_max": 4.0,
    
    "alignment_scale": 0.20,
    
    "facing_bonus": 1.0,
    "facing_threshold_deg": 20.0,
    "facing_distance": 0.25,
    
    "approach_scale": 2.0,
    
    "turning_bonus": 0.35,
    "turning_threshold_deg": 20.0,
    
    # Terminal rewards
    "collision_penalty": -100.0,
    "collision_speed_threshold": 0.4,
    "collision_min_steps": 10,
    
    "boundary_penalty": -500.0,
    
    "success_bonus": 400.0,
    "success_distance": 0.03,
    "success_min_steps": 5,
    
    # Time penalty (ENABLED - prevents reward exploitation)
    "time_base": -0.01,
    "time_exp_factor": 2.0,
    "time_scale": 25.0,
    
    # Clipping
    "reward_min": -500.0,
    "reward_max": 500.0,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π] range."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _extract_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion [x, y, z, w]."""
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def _compute_yaw_error(robot_yaw: torch.Tensor, robot_pos: torch.Tensor, 
                        goal_pos: torch.Tensor) -> torch.Tensor:
    """Compute yaw error between robot rear and goal direction."""
    vec_to_goal = goal_pos - robot_pos
    goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])
    rear_yaw = robot_yaw + torch.pi
    return _angle_wrap(rear_yaw - goal_yaw)


# =============================================================================
# REWARD COMPONENTS
# =============================================================================

def _compute_distance_reward(surface_xy: torch.Tensor) -> torch.Tensor:
    """Continuous penalty for distance to goal."""
    cfg = REWARD_CONFIG
    reward = cfg["distance_scale"] * surface_xy
    return torch.clamp(reward, min=cfg["distance_min"], max=cfg["distance_max"])


def _compute_progress_reward(env, surface_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward for making progress toward goal."""
    cfg = REWARD_CONFIG
    
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()
    
    progress = env.prev_distance - surface_xy
    reward = cfg["progress_scale"] * progress
    reward = torch.clamp(reward, min=cfg["progress_min"], max=cfg["progress_max"])
    
    env.prev_distance = surface_xy.clone()
    return reward, progress


def _compute_alignment_reward(yaw_error_abs: torch.Tensor) -> torch.Tensor:
    """Reward for correct orientation."""
    cfg = REWARD_CONFIG
    normalized_error = yaw_error_abs / torch.pi
    return cfg["alignment_scale"] * (1.0 - normalized_error)


def _compute_facing_bonus(yaw_error_abs: torch.Tensor, 
                          surface_xy: torch.Tensor) -> torch.Tensor:
    """Bonus when well-aligned and close."""
    cfg = REWARD_CONFIG
    threshold = np.deg2rad(cfg["facing_threshold_deg"])
    
    well_aligned = yaw_error_abs < threshold
    close_enough = surface_xy < cfg["facing_distance"]
    
    return torch.where(
        well_aligned & close_enough,
        torch.full_like(surface_xy, cfg["facing_bonus"]),
        torch.zeros_like(surface_xy),
    )


def _compute_approach_bonus(progress: torch.Tensor, yaw_error_abs: torch.Tensor,
                            surface_xy: torch.Tensor) -> torch.Tensor:
    """Bonus for approaching while aligned."""
    cfg = REWARD_CONFIG
    threshold = np.deg2rad(cfg["facing_threshold_deg"])
    
    approaching = progress > 0.0
    well_aligned = yaw_error_abs < threshold
    close_enough = surface_xy < cfg["facing_distance"]
    
    return torch.where(
        approaching & well_aligned & close_enough,
        cfg["approach_scale"] * progress,
        torch.zeros_like(progress),
    )


def _compute_turning_bonus(yaw_error: torch.Tensor, yaw_error_abs: torch.Tensor,
                           ang_vel: torch.Tensor, surface_xy: torch.Tensor) -> torch.Tensor:
    """Bonus for turning toward goal when misaligned."""
    cfg = REWARD_CONFIG
    threshold = np.deg2rad(cfg["turning_threshold_deg"])
    
    yaw_rate = ang_vel[:, 2]
    turning_correct = (yaw_error * yaw_rate) < 0
    is_misaligned = yaw_error_abs > threshold
    
    return torch.where(
        is_misaligned & turning_correct,
        torch.full_like(surface_xy, cfg["turning_bonus"]),
        torch.zeros_like(surface_xy),
    )


def _compute_collision_penalty(env, surface_xy: torch.Tensor) -> torch.Tensor:
    """Terminal penalty for collisions."""
    cfg = REWARD_CONFIG
    
    raw_success = surface_xy < cfg["success_distance"]
    
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions
    diff = robot_pos - goal_pos
    dx, dy = diff[:, 0], diff[:, 1]
    
    sL = 0.5 * env._static_body_length
    sW = 0.5 * env._static_body_width
    aL = 0.5 * env._active_body_length
    aW = 0.5 * env._active_body_width
    
    overlap = (dx.abs() < (sL + aL)) & (dy.abs() < (sW + aW))
    speed = torch.norm(env.robot.data.root_lin_vel_w[:, :2], dim=-1)
    ep_len = env.episode_length_buf
    
    collision = (
        overlap & 
        (speed > cfg["collision_speed_threshold"]) & 
        (~raw_success) & 
        (ep_len >= cfg["collision_min_steps"])
    )
    
    return torch.where(
        collision,
        torch.full_like(surface_xy, cfg["collision_penalty"]),
        torch.zeros_like(surface_xy),
    )


def _compute_boundary_penalty(env, surface_xy: torch.Tensor) -> torch.Tensor:
    """Terminal penalty for leaving arena."""
    cfg = REWARD_CONFIG
    
    robot_pos = env.robot.data.root_pos_w
    origins = env.scene.env_origins
    local_pos = robot_pos - origins
    
    hx = float(env._arena_half_x)
    hy = float(env._arena_half_y)
    
    out_of_bounds = (local_pos[:, 0].abs() > hx) | (local_pos[:, 1].abs() > hy)
    
    return torch.where(
        out_of_bounds,
        torch.full_like(surface_xy, cfg["boundary_penalty"]),
        torch.zeros_like(surface_xy),
    )


def _compute_success_bonus(env, surface_xy: torch.Tensor) -> torch.Tensor:
    """Terminal bonus for successful docking."""
    cfg = REWARD_CONFIG
    
    raw_success = surface_xy < cfg["success_distance"]
    ep_len = env.episode_length_buf
    success = raw_success & (ep_len >= cfg["success_min_steps"])
    
    return torch.where(
        success,
        torch.full_like(surface_xy, cfg["success_bonus"]),
        torch.zeros_like(surface_xy),
    )


def _compute_time_penalty(env, surface_xy: torch.Tensor) -> torch.Tensor:
    """Gentle time penalty to encourage efficiency."""
    cfg = REWARD_CONFIG
    
    max_ep_len = float(env.max_episode_length)
    ep_len = env.episode_length_buf
    length_ratio = ep_len.float() / max_ep_len
    
    exp_factor = torch.exp(cfg["time_exp_factor"] * length_ratio) - 1.0
    exp_factor = exp_factor / (np.exp(cfg["time_exp_factor"]) - 1.0)
    
    return cfg["time_base"] * (1.0 + cfg["time_scale"] * exp_factor)


# =============================================================================
# MAIN REWARD FUNCTION
# =============================================================================

def compute_total_reward(env) -> torch.Tensor:
    """
    Compute total reward for TEKO docking task.
    
    Works for both state-based and vision-based training.
    
    Returns:
        Total reward tensor [num_envs]
    """
    cfg = REWARD_CONFIG
    
    # Get state information
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    
    robot_quat = env.robot.data.root_quat_w
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions
    ang_vel = env.robot.data.root_ang_vel_w
    
    # Compute yaw error
    robot_yaw = _extract_yaw(robot_quat)
    yaw_error = _compute_yaw_error(robot_yaw, robot_pos, goal_pos)
    yaw_error_abs = torch.abs(yaw_error)
    
    # Compute reward components
    distance_reward = _compute_distance_reward(surface_xy)
    progress_reward, progress = _compute_progress_reward(env, surface_xy)
    alignment_reward = _compute_alignment_reward(yaw_error_abs)
    facing_bonus = _compute_facing_bonus(yaw_error_abs, surface_xy)
    approach_bonus = _compute_approach_bonus(progress, yaw_error_abs, surface_xy)
    turning_bonus = _compute_turning_bonus(yaw_error, yaw_error_abs, ang_vel, surface_xy)
    collision_penalty = _compute_collision_penalty(env, surface_xy)
    boundary_penalty = _compute_boundary_penalty(env, surface_xy)
    success_bonus = _compute_success_bonus(env, surface_xy)
    time_penalty = _compute_time_penalty(env, surface_xy)
    
    # Sum all components
    total_reward = (
        distance_reward
        + progress_reward
        + alignment_reward
        + facing_bonus
        + approach_bonus
        + turning_bonus
        + collision_penalty
        + boundary_penalty
        + success_bonus
        + time_penalty
    )
    
    total_reward = torch.clamp(total_reward, min=cfg["reward_min"], max=cfg["reward_max"])
    
    # Logging
    _log_rewards(env, {
        "distance": distance_reward,
        "progress": progress_reward,
        "alignment": alignment_reward,
        "facing_bonus": facing_bonus,
        "approach_bonus": approach_bonus,
        "turning_bonus": turning_bonus,
        "collision_penalty": collision_penalty,
        "boundary_penalty": boundary_penalty,
        "success_bonus": success_bonus,
        "time_penalty": time_penalty,
    })
    
    return total_reward


def _log_rewards(env, rewards: dict) -> None:
    """Log reward components for TensorBoard."""
    if not hasattr(env, 'reward_components'):
        env.reward_components = {}
    
    rc = env.reward_components
    for name, val in rewards.items():
        if name not in rc:
            rc[name] = []
        rc[name].append(val.mean().item())


# =============================================================================
# UTILITY FOR GA OPTIMIZATION
# =============================================================================

def get_reward_config() -> dict:
    """Get current reward configuration."""
    return REWARD_CONFIG.copy()


def set_reward_config(new_config: dict) -> None:
    """Update reward configuration."""
    global REWARD_CONFIG
    REWARD_CONFIG.update(new_config)