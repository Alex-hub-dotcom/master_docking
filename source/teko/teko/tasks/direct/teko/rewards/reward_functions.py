# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for TEKO (v9.0 – 84px OPTIMIZED + TURNING BONUS)
-----------------------------------------------------------------

Changes from v8.9:
- Alignment reward scale reduced (0.3 → 0.20)
- Progress reward softened (10 → 8)
- Turning bonus increased (0.2 → 0.35)
- Facing bonus threshold widened (15° → 20°)

Everything else identical to v8.9.

These small adjustments are ESSENTIAL for stability at 84×84 resolution.
"""

from __future__ import annotations
import torch
import numpy as np


def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_total_reward(env) -> torch.Tensor:

    # ------------------------------------------------------------------
    # 0. Distances
    # ------------------------------------------------------------------
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()

    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 1. Distance reward (same logic, unchanged scale)
    # ------------------------------------------------------------------
    distance_reward = -2.0 * surface_xy
    distance_reward = torch.clamp(distance_reward, min=-4.0, max=0.0)

    # ------------------------------------------------------------------
    # 2. Progress reward (10 → 8 for 84px stability)
    # ------------------------------------------------------------------
    progress = env.prev_distance - surface_xy
    progress_reward = 8.0 * progress           # CHANGED
    progress_reward = torch.clamp(progress_reward, min=-4.0, max=4.0)
    env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 3. Alignment reward (0.3 → 0.20)
    # ------------------------------------------------------------------
    quat = env.robot.data.root_quat_w
    pos = env.robot.data.root_pos_w
    goal = env.goal_positions

    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    vec_to_goal = goal - pos
    goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])

    rear_yaw = yaw + torch.pi
    yaw_error = _angle_wrap(rear_yaw - goal_yaw)
    yaw_error_abs = torch.abs(yaw_error)

    normalized_yaw_error = yaw_error_abs / torch.pi

    alignment_reward = 0.20 * (1.0 - normalized_yaw_error)   # CHANGED

    # ------------------------------------------------------------------
    # 4. Facing bonus (threshold widened: 15° → 20°)
    # ------------------------------------------------------------------
    well_aligned = yaw_error_abs < np.deg2rad(20.0)          # CHANGED
    close_enough = surface_xy < 0.25

    facing_bonus = torch.where(
        well_aligned & close_enough,
        torch.full_like(surface_xy, 1.0),
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 5. Approach bonus (unchanged)
    # ------------------------------------------------------------------
    approaching = progress > 0.0
    approach_bonus = torch.where(
        approaching & well_aligned & close_enough,
        2.0 * progress,
        torch.zeros_like(progress),
    )

    # ------------------------------------------------------------------
    # 6. Turning bonus (0.2 → 0.35)
    # ------------------------------------------------------------------
    ang_vel = env.robot.data.root_ang_vel_w
    yaw_rate = ang_vel[:, 2]

    turning_correct = (yaw_error * yaw_rate) < 0
    is_misaligned = yaw_error_abs > np.deg2rad(20.0)

    turning_bonus = torch.where(
        is_misaligned & turning_correct,
        torch.full_like(surface_xy, 0.35),      # CHANGED
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 7. Collision penalty (unchanged)
    # ------------------------------------------------------------------
    raw_success = surface_xy < 0.03

    robot_pos = pos
    static = goal
    diff = robot_pos - static
    dx, dy = diff[:, 0], diff[:, 1]

    sL = 0.5 * env._static_body_length
    sW = 0.5 * env._static_body_width
    aL = 0.5 * env._active_body_length
    aW = 0.5 * env._active_body_width

    overlap = (dx.abs() < (sL + aL)) & (dy.abs() < (sW + aW))
    speed = torch.norm(env.robot.data.root_lin_vel_w[:, :2], dim=-1)

    ep_len = env.episode_length_buf
    collision = overlap & (speed > 0.4) & (~raw_success) & (ep_len >= 10)

    collision_penalty = torch.where(
        collision,
        torch.full_like(surface_xy, -100.0),
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 8. Boundary penalty (unchanged)
    # ------------------------------------------------------------------
    origins = env.scene.env_origins
    local_pos = robot_pos - origins

    hx = float(env._arena_half_x)
    hy = float(env._arena_half_y)

    out_of_bounds = (local_pos[:, 0].abs() > hx) | (local_pos[:, 1].abs() > hy)

    boundary_penalty = torch.where(
        out_of_bounds,
        torch.full_like(surface_xy, -500.0),
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 9. Success bonus (unchanged)
    # ------------------------------------------------------------------
    success = raw_success & (ep_len >= 5)

    success_bonus = torch.where(
        success,
        torch.full_like(surface_xy, 400.0),
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 10. Time penalty (unchanged)
    # ------------------------------------------------------------------
    max_ep_len = float(env.max_episode_length)
    length_ratio = ep_len.float() / max_ep_len

    exp_factor = torch.exp(4.0 * length_ratio) - 1.0
    exp_factor = exp_factor / 54.0

    time_penalty = -0.02 * (1.0 + 50.0 * exp_factor)

    # ------------------------------------------------------------------
    # Total reward
    # ------------------------------------------------------------------
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

    total_reward = torch.clamp(total_reward, min=-500.0, max=500.0)

    # ------------------------------------------------------------------
    # Logging (unchanged)
    # ------------------------------------------------------------------
    rc = env.reward_components

    def _log(name, val):
        if name not in rc: rc[name] = []
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
