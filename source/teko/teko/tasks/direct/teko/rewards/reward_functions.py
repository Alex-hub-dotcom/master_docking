# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for TEKO (v8.6 – OFFSET-FRIENDLY ALIGNMENT & APPROACH)
-----------------------------------------------------------------------

v8.6 Changes (FOCUSED ON STAGES 8–11):
- Alignment reward is now stronger when the robot is FAR from the dock
  and weaker when it is NEAR:
    * far (surface_xy > 0.15 m):  0.3 × (1 - error)
    * near (surface_xy ≤ 0.15 m): 0.05 × (1 - error)
- Approach bonus is slightly more permissive:
    * yaw error < 30°
    * distance < 0.40 m
    * scaled as 2.0 × progress

Goal:
- Encourage the robot to first rotate towards the docking target when
  spawned with large angular/lateral offsets (Stages 8–11),
  without reintroducing reward hacking via hovering.
"""

from __future__ import annotations
import torch
import numpy as np


def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π] range."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_total_reward(env) -> torch.Tensor:
    """
    Compute total reward for the TEKO docking task (v8.6).

    Reward structure (8 components):
    1. Distance shaping:        -2.0 × surface_xy
    2. Progress reward:        +10.0 × progress (main dense signal)
    3. Alignment shaping:   0.3 / 0.05 × (1 - normalized yaw error)
       - stronger when far from the dock, weaker near the dock
    4. Approach bonus:          +2.0 × progress
       - only when approaching, reasonably aligned, and within 0.40 m
    5. Collision penalty:      -100 (terminal)
    6. Boundary penalty:       -500 (terminal)
    7. Success bonus:          +400 (terminal, dominant positive event)
    8. Time penalty:    small negative term increasing with episode length
    """
    device = env.device
    num_envs = env.scene.cfg.num_envs  # kept for clarity, not strictly required

    # ------------------------------------------------------------------
    # 0. Distances between connector spheres
    # ------------------------------------------------------------------
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()

    # Initialize prev_distance on first call
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 1. Distance reward (shaping)
    # ------------------------------------------------------------------
    distance_reward = -2.0 * surface_xy
    distance_reward = torch.clamp(distance_reward, min=-10.0, max=0.0)

    # ------------------------------------------------------------------
    # 2. Progress reward (main dense signal)
    # ------------------------------------------------------------------
    progress = env.prev_distance - surface_xy
    progress_reward = 10.0 * progress
    progress_reward = torch.clamp(progress_reward, min=-4.0, max=4.0)

    # Update for next step
    env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 3. Alignment reward (offset-friendly)
    # ------------------------------------------------------------------
    # Extract robot yaw from quaternion
    robot_quat = env.robot.data.root_quat_w
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions

    qx, qy, qz, qw = (
        robot_quat[:, 0],
        robot_quat[:, 1],
        robot_quat[:, 2],
        robot_quat[:, 3],
    )
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    robot_yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Yaw required to point from robot to goal
    vec_to_goal = goal_pos - robot_pos
    goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])

    # We want the rear of the robot (camera side) to face the goal
    rear_yaw = robot_yaw + torch.pi
    yaw_error = _angle_wrap(rear_yaw - goal_yaw)

    # Normalized yaw error in [0, 1]
    normalized_error = torch.abs(yaw_error) / torch.pi

    # Stronger alignment shaping when far, weaker when near
    far_mask = surface_xy > 0.15
    near_mask = ~far_mask

    alignment_far = 0.3 * (1.0 - normalized_error)
    alignment_near = 0.05 * (1.0 - normalized_error)

    alignment_reward = torch.where(far_mask, alignment_far, alignment_near)

    # ------------------------------------------------------------------
    # 4. Approach bonus (slightly relaxed conditions)
    # ------------------------------------------------------------------
    well_aligned = torch.abs(yaw_error) < np.deg2rad(30.0)  # was 20°
    close_enough = surface_xy < 0.40                        # was 0.30 m

    approaching = (progress > 0.0) & well_aligned & close_enough

    approach_bonus = torch.where(
        approaching,
        2.0 * progress,  # was 3.0 × progress
        torch.tensor(0.0, device=device),
    )

    # ------------------------------------------------------------------
    # 5. Collision penalty (terminal)
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

    # Align collision condition with _get_dones (min_collision_steps)
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
        torch.tensor(-100.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ------------------------------------------------------------------
    # 6. Boundary penalty (terminal)
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
        torch.tensor(-500.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ------------------------------------------------------------------
    # 7. Success bonus (terminal)
    # ------------------------------------------------------------------
    min_success_steps = 5
    terminal_success = raw_success & (ep_len >= min_success_steps)

    success_bonus = torch.where(
        terminal_success,
        torch.tensor(400.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ------------------------------------------------------------------
    # 8. Time penalty (encourage efficient docking)
    # ------------------------------------------------------------------
    max_ep_len = float(env.max_episode_length)
    length_ratio = ep_len.float() / max_ep_len

    # Exponential growth with episode length, normalized to keep values small
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
        + approach_bonus
        + collision_penalty
        + boundary_penalty
        + success_bonus
        + time_penalty
    )

    total_reward = torch.clamp(total_reward, min=-400.0, max=400.0)

    # ------------------------------------------------------------------
    # Logging of reward components (for analysis)
    # ------------------------------------------------------------------
    rc = env.reward_components

    def _log(name: str, val: torch.Tensor):
        if name not in rc:
            rc[name] = []
        rc[name].append(val.mean().item())

    _log("distance", distance_reward)
    _log("progress", progress_reward)
    _log("alignment", alignment_reward)
    _log("approach_bonus", approach_bonus)
    _log("collision_penalty", collision_penalty)
    _log("boundary_penalty", boundary_penalty)
    _log("success_bonus", success_bonus)
    _log("time_penalty", time_penalty)

    return total_reward
