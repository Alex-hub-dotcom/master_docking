# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for TEKO (v8.9 – BACK TO BASICS + TURNING BONUS)
-----------------------------------------------------------------

v8.9: Return to the simple structure that WORKED for S0-S5.
Only addition: small turning bonus for offset stages.

This is essentially v8.3 with:
- Same distance/progress/alignment structure
- Same collision/boundary/success logic
- ADDED: Small turning bonus when misaligned

NO gating, NO threshold changes, NO aggressive modifications.
"""

from __future__ import annotations
import torch
import numpy as np


def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π] range."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_total_reward(env) -> torch.Tensor:
    """
    Compute total reward for the TEKO docking task (v8.9 - SIMPLE + TURNING).
    """
    device = env.device  # currently unused but kept for consistency

    # ------------------------------------------------------------------
    # 0. Get distances
    # ------------------------------------------------------------------
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()

    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 1. Distance reward (continuous shaping) - UNCHANGED FROM v8.3
    # ------------------------------------------------------------------
    distance_reward = -2.0 * surface_xy
    distance_reward = torch.clamp(distance_reward, min=-4.0, max=0.0)

    # ------------------------------------------------------------------
    # 2. Progress reward - UNCHANGED FROM v8.3
    # ------------------------------------------------------------------
    progress = env.prev_distance - surface_xy
    progress_reward = 10.0 * progress
    progress_reward = torch.clamp(progress_reward, min=-4.0, max=4.0)
    env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 3. Alignment reward (yaw-based) - UNCHANGED FROM v8.3
    # ------------------------------------------------------------------
    robot_quat = env.robot.data.root_quat_w
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions

    # Extract robot yaw
    qx, qy, qz, qw = (
        robot_quat[:, 0],
        robot_quat[:, 1],
        robot_quat[:, 2],
        robot_quat[:, 3],
    )
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    robot_yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Yaw to goal
    vec_to_goal = goal_pos - robot_pos
    goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])

    # We want REAR to face goal
    rear_yaw = robot_yaw + torch.pi
    yaw_error = _angle_wrap(rear_yaw - goal_yaw)
    yaw_error_abs = torch.abs(yaw_error)

    # Normalized error [0, 1]
    normalized_yaw_error = yaw_error_abs / torch.pi

    # Simple alignment reward (scale=0.3, like original v8.3)
    alignment_reward = 0.3 * (1.0 - normalized_yaw_error)

    # ------------------------------------------------------------------
    # 4. Facing bonus (when well-aligned and close) - UNCHANGED
    # ------------------------------------------------------------------
    well_aligned = yaw_error_abs < np.deg2rad(15.0)
    close_enough = surface_xy < 0.25

    facing_bonus = torch.where(
        well_aligned & close_enough,
        torch.full_like(surface_xy, 1.0),
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 5. Approach bonus - UNCHANGED
    # ------------------------------------------------------------------
    approaching = progress > 0.0
    approach_bonus = torch.where(
        approaching & well_aligned & close_enough,
        2.0 * progress,
        torch.zeros_like(progress),
    )

    # ------------------------------------------------------------------
    # 6. TURNING BONUS (NEW - only addition to v8.3)
    # ------------------------------------------------------------------
    # Small bonus when misaligned AND turning toward goal
    # This helps in offset stages without breaking forward stages

    ang_vel = env.robot.data.root_ang_vel_w
    yaw_rate = ang_vel[:, 2]

    # Turning in correct direction?
    turning_correct = (yaw_error * yaw_rate) < 0

    # Only give bonus when significantly misaligned (>15°)
    is_misaligned = yaw_error_abs > np.deg2rad(15.0)

    turning_bonus = torch.where(
        is_misaligned & turning_correct,
        torch.full_like(surface_xy, 0.2),  # Small bonus
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 7. Collision penalty (terminal)
    # ------------------------------------------------------------------
    raw_success = surface_xy < 0.03

    robot_pos_global = env.robot.data.root_pos_w
    static_root_pos = env.goal_positions
    diff = robot_pos_global - static_root_pos
    dx = diff[:, 0]
    dy = diff[:, 1]

    static_half_len = 0.5 * env._static_body_length
    static_half_wid = 0.5 * env._static_body_width
    active_half_len = 0.5 * env._active_body_length
    active_half_wid = 0.5 * env._active_body_width

    boxes_overlap = (
        (dx.abs() < (static_half_len + active_half_len))
        & (dy.abs() < (static_half_wid + active_half_wid))
    )

    lin_vel = env.robot.data.root_lin_vel_w
    speed = torch.norm(lin_vel[:, :2], dim=-1)

    min_collision_steps = 10
    ep_len = env.episode_length_buf

    collision = (
        boxes_overlap
        & (speed > 0.4)
        & (~raw_success)
        & (ep_len >= min_collision_steps)
    )

    collision_penalty = torch.where(
        collision,
        torch.full_like(surface_xy, -100.0),
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 8. Boundary penalty (terminal)
    # ------------------------------------------------------------------
    env_origins = env.scene.env_origins
    robot_pos_local = robot_pos_global - env_origins

    hx = float(env._arena_half_x)
    hy = float(env._arena_half_y)

    out_of_bounds = (
        (robot_pos_local[:, 0].abs() > hx)
        | (robot_pos_local[:, 1].abs() > hy)
    )

    boundary_penalty = torch.where(
        out_of_bounds,
        torch.full_like(surface_xy, -500.0),
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 9. Success bonus (terminal)
    # ------------------------------------------------------------------
    min_success_steps = 5
    terminal_success = raw_success & (ep_len >= min_success_steps)

    success_bonus = torch.where(
        terminal_success,
        torch.full_like(surface_xy, 400.0),
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 10. Time penalty (exponential)
    # ------------------------------------------------------------------
    max_ep_len = float(env.max_episode_length)
    length_ratio = ep_len.float() / max_ep_len

    exp_factor = torch.exp(4.0 * length_ratio) - 1.0
    exp_factor = exp_factor / 54.0

    base_time_penalty = -0.02
    time_penalty = base_time_penalty * (1.0 + 50.0 * exp_factor)

    # ------------------------------------------------------------------
    # Total reward
    # ------------------------------------------------------------------
    total_reward = (
        distance_reward
        + progress_reward
        + alignment_reward
        + facing_bonus
        + approach_bonus
        + turning_bonus      # NEW: small turning incentive
        + collision_penalty
        + boundary_penalty
        + success_bonus
        + time_penalty
    )

    total_reward = torch.clamp(total_reward, min=-500.0, max=500.0)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    rc = env.reward_components

    def _log(name: str, val: torch.Tensor):
        if name not in rc:
            rc[name] = []
        rc[name].append(val.mean().item())

    _log("distance", distance_reward)
    _log("progress", progress_reward)
    _log("alignment", alignment_reward)
    _log("facing_bonus", facing_bonus)
    _log("approach_bonus", approach_bonus)
    _log("turning_bonus", turning_bonus)
    _log("collision_penalty", collision_penalty)
    _log("boundary_penalty", boundary_penalty)
    _log("success_bonus", success_bonus)
    _log("time_penalty", time_penalty)

    return total_reward
