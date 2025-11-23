# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for TEKO (v7.3 – curriculum aligned, precision-focused)
------------------------------------------------------------------------

Main ideas:
- Progress reward is still the main dense signal, but less dominant.
- Alignment is much more important (robot should point the rear to the goal).
- Extra bonuses:
  - "Facing bonus" when close and well aligned.
  - "Approach bonus" when getting closer while roughly aligned.
- Collision penalty softened to -100 to allow more exploration on harder stages.
- Precision bonus kept for very tight docking (< 2 cm).
"""

from __future__ import annotations
import torch
import numpy as np


def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_total_reward(env) -> torch.Tensor:
    device = env.device
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()

    # Initialize previous distance on first call
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()

    # ---------------------------------------------------------------------
    # 1. Distance reward (moderate shaping: closer is better)
    # ---------------------------------------------------------------------
    distance_reward = -1.5 * surface_xy
    distance_reward = torch.clamp(distance_reward, min=-8.0, max=0.0)

    # ---------------------------------------------------------------------
    # 2. Progress reward (main dense signal, but not overwhelming)
    #    Positive when the robot gets closer to the goal.
    # ---------------------------------------------------------------------
    progress = env.prev_distance - surface_xy
    progress_reward = 8.0 * progress  # was 15.0
    progress_reward = torch.clamp(progress_reward, min=-3.0, max=3.0)
    env.prev_distance = surface_xy.clone()

    # ---------------------------------------------------------------------
    # 3. Alignment reward (rear must face the goal)
    # ---------------------------------------------------------------------
    robot_quat = env.robot.data.root_quat_w
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions

    qx, qy, qz, qw = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    robot_yaw = torch.atan2(siny_cosp, cosy_cosp)

    vec_to_goal = goal_pos - robot_pos
    goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])

    # rear_yaw = robot yaw + π (rear side / camera towards the goal)
    rear_yaw = robot_yaw + torch.pi
    yaw_error = _angle_wrap(rear_yaw - goal_yaw)

    # Stronger alignment weight: [-2, 2]
    alignment_reward = 2.0 * torch.cos(yaw_error)

    # 3a. Facing bonus: close and reasonably aligned
    close_and_aligned = (surface_xy < 0.15) & (torch.abs(yaw_error) < np.deg2rad(30.0))
    facing_bonus = torch.where(
        close_and_aligned,
        torch.tensor(3.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # 3b. Approach bonus: getting closer *while* roughly aligned
    approaching = (progress > 0.0) & (torch.abs(yaw_error) < np.deg2rad(45.0))
    approach_bonus = torch.where(
        approaching,
        2.0 * progress,  # extra reward only when progress > 0
        torch.tensor(0.0, device=device),
    )

    # ---------------------------------------------------------------------
    # 4. Velocity penalty (small – avoids rushing)
    # ---------------------------------------------------------------------
    lin_vel = env.robot.data.root_lin_vel_w
    speed = torch.norm(lin_vel[:, :2], dim=-1)
    velocity_penalty = -0.01 * speed

    # ---------------------------------------------------------------------
    # 5. Oscillation penalty (small – discourages twitching)
    # ---------------------------------------------------------------------
    if env.prev_actions is None:
        env.prev_actions = torch.zeros_like(env.actions)
    action_diff = torch.norm(env.actions - env.prev_actions, dim=-1)
    oscillation_penalty = -0.02 * action_diff
    env.prev_actions = env.actions.clone()

    # ---------------------------------------------------------------------
    # 6. Collision penalty (AABB overlap, softened to -100)
    # ---------------------------------------------------------------------
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
        (dx.abs() < (static_half_len + active_half_len)) &
        (dy.abs() < (static_half_wid + active_half_wid))
    )

    collision = boxes_overlap & (speed > 0.4) & (~raw_success)

    collision_penalty = torch.where(
        collision,
        torch.tensor(-100.0, device=device),  # was -300, then -200
        torch.tensor(0.0, device=device),
    )

    # ---------------------------------------------------------------------
    # 7. Boundary penalty (leaving the arena is always terrible)
    # ---------------------------------------------------------------------
    env_origins = env.scene.env_origins
    robot_pos_local = robot_pos_global - env_origins
    hx = float(env._arena_half_x)
    hy = float(env._arena_half_y)
    out_of_bounds = (
        (robot_pos_local[:, 0].abs() > hx) |
        (robot_pos_local[:, 1].abs() > hy)
    )
    boundary_penalty = torch.where(
        out_of_bounds,
        torch.tensor(-500.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ---------------------------------------------------------------------
    # 8. Success bonus (only on terminal success)
    # ---------------------------------------------------------------------
    min_success_steps = 5
    ep_len = env.episode_length_buf
    terminal_success = raw_success & (ep_len >= min_success_steps)

    success_bonus = torch.where(
        terminal_success,
        torch.tensor(250.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ---------------------------------------------------------------------
    # 9. Proximity + precision bonuses
    # ---------------------------------------------------------------------
    # Sweet spot: 3–10 cm (encourage staying close without crashing)
    close = (surface_xy < 0.10) & (surface_xy >= 0.03) & (~collision)
    proximity_bonus = torch.where(
        close,
        torch.tensor(4.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # Very precise docking: < 2 cm
    precise = (surface_xy < 0.02) & (~collision)
    precision_bonus = torch.where(
        precise,
        torch.tensor(20.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ---------------------------------------------------------------------
    # 10. Time penalty (very soft – avoids infinite wandering)
    # ---------------------------------------------------------------------
    time_penalty = torch.full_like(surface_xy, -0.02)

    # ---------------------------------------------------------------------
    # Total reward
    # ---------------------------------------------------------------------
    total_reward = (
        distance_reward +
        progress_reward +
        alignment_reward +
        facing_bonus +
        approach_bonus +
        velocity_penalty +
        oscillation_penalty +
        collision_penalty +
        boundary_penalty +
        success_bonus +
        proximity_bonus +
        precision_bonus +
        time_penalty
    )

    total_reward = torch.clamp(total_reward, min=-400.0, max=400.0)

    # ---------------------------------------------------------------------
    # Logging (for TensorBoard analysis)
    # ---------------------------------------------------------------------
    rc = env.reward_components

    def _log(name, val):
        if name not in rc:
            rc[name] = []
        rc[name].append(val.mean().item())

    _log("distance", distance_reward)
    _log("progress", progress_reward)
    _log("alignment", alignment_reward)
    _log("facing_bonus", facing_bonus)
    _log("approach_bonus", approach_bonus)
    _log("velocity_penalty", velocity_penalty)
    _log("oscillation_penalty", oscillation_penalty)
    _log("collision_penalty", collision_penalty)
    _log("wall_penalty", boundary_penalty)
    _log("success_bonus", success_bonus)
    _log("proximity_bonus", proximity_bonus)
    _log("precision_bonus", precision_bonus)
    _log("time_penalty", time_penalty)

    return total_reward
