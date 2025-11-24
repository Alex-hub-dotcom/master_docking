# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for TEKO (v8.2 – ONE-TIME MILESTONE BONUSES)
-------------------------------------------------------------

v8.2 Changes (THE GOD VERSION):
- ONE-TIME milestone bonuses (completely prevents proximity farming)
- Bonuses only awarded FIRST TIME entering each distance zone
- Strong progressive time penalty (exponential growth)
- Approach requirement for all bonuses (must be moving toward goal)
- Episode length inflation is now IMPOSSIBLE

v8.0-8.1 fixes preserved:
- Survival bonus disabled
- Alignment reward always positive [0, 5]
- Progressive time penalty
"""

from __future__ import annotations
import torch
import numpy as np


def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_total_reward(env) -> torch.Tensor:
    device = env.device
    num_envs = env.scene.cfg.num_envs
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()

    # Initialize tracking buffers on first call
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()
    
    
    # ---------------------------------------------------------------------
    # 1. Distance reward (moderate shaping)
    # ---------------------------------------------------------------------
    distance_reward = -2.0 * surface_xy
    distance_reward = torch.clamp(distance_reward, min=-10.0, max=0.0)

    # ---------------------------------------------------------------------
    # 2. Progress reward (main signal)
    # ---------------------------------------------------------------------
    progress = env.prev_distance - surface_xy
    progress_reward = 10.0 * progress
    progress_reward = torch.clamp(progress_reward, min=-4.0, max=4.0)
    env.prev_distance = surface_xy.clone()

    # ---------------------------------------------------------------------
    # 3. Alignment reward (ALWAYS POSITIVE)
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

    rear_yaw = robot_yaw + torch.pi
    yaw_error = _angle_wrap(rear_yaw - goal_yaw)

    normalized_error = torch.abs(yaw_error) / torch.pi
    alignment_reward = 5.0 * (1.0 - normalized_error)

    # 3a. Facing bonus (close and aligned)
    close_and_aligned = (surface_xy < 0.15) & (torch.abs(yaw_error) < np.deg2rad(30.0))
    facing_bonus = torch.where(
        close_and_aligned,
        torch.tensor(5.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # 3b. Approach bonus (approaching while aligned)
    approaching = (progress > 0.0) & (torch.abs(yaw_error) < np.deg2rad(60.0))
    approach_bonus = torch.where(
        approaching,
        3.0 * progress,
        torch.tensor(0.0, device=device),
    )

    # ---------------------------------------------------------------------
    # 4. Velocity penalty (very small)
    # ---------------------------------------------------------------------
    lin_vel = env.robot.data.root_lin_vel_w
    speed = torch.norm(lin_vel[:, :2], dim=-1)
    velocity_penalty = -0.005 * speed

    # ---------------------------------------------------------------------
    # 5. Oscillation penalty
    # ---------------------------------------------------------------------
    if env.prev_actions is None:
        env.prev_actions = torch.zeros_like(env.actions)
    action_diff = torch.norm(env.actions - env.prev_actions, dim=-1)
    oscillation_penalty = -0.01 * action_diff
    env.prev_actions = env.actions.clone()

    # ---------------------------------------------------------------------
    # 6. Collision penalty
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
        torch.tensor(-100.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ---------------------------------------------------------------------
    # 7. Boundary penalty
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
    # 8. Success bonus (terminal only)
    # ---------------------------------------------------------------------
    min_success_steps = 5
    ep_len = env.episode_length_buf
    terminal_success = raw_success & (ep_len >= min_success_steps)

    success_bonus = torch.where(
        terminal_success,
        torch.tensor(300.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ---------------------------------------------------------------------
    # 9. ONE-TIME MILESTONE BONUSES (v8.2 - ANTI-FARMING GOD MODE)
    # ---------------------------------------------------------------------
    # CRITICAL: Each bonus awarded ONLY ONCE per episode when FIRST entering zone
    # This makes proximity farming IMPOSSIBLE
    
    # Detect first-time entries (not yet flagged AND currently in zone AND approaching)
    entering_20cm = (
        (surface_xy < 0.20) & 
        ~env.milestone_flags['entered_20cm'] & 
        (progress > 0.0)
    )
    entering_10cm = (
        (surface_xy < 0.10) & 
        ~env.milestone_flags['entered_10cm'] & 
        (progress > 0.0)
    )
    entering_5cm = (
        (surface_xy < 0.05) & 
        ~env.milestone_flags['entered_5cm'] & 
        (progress > 0.0)
    )
    entering_2cm = (
        (surface_xy < 0.02) & 
        ~env.milestone_flags['entered_2cm'] & 
        (progress > 0.0)
    )

    # Award bonuses (one-time only)
    milestone_bonus = torch.zeros_like(surface_xy)
    milestone_bonus += torch.where(entering_20cm, torch.tensor(5.0, device=device), torch.tensor(0.0, device=device))
    milestone_bonus += torch.where(entering_10cm, torch.tensor(10.0, device=device), torch.tensor(0.0, device=device))
    milestone_bonus += torch.where(entering_5cm, torch.tensor(20.0, device=device), torch.tensor(0.0, device=device))
    milestone_bonus += torch.where(entering_2cm, torch.tensor(50.0, device=device), torch.tensor(0.0, device=device))

    # Update flags (mark milestones as achieved)
    env.milestone_flags['entered_20cm'] |= (surface_xy < 0.20)
    env.milestone_flags['entered_10cm'] |= (surface_xy < 0.10)
    env.milestone_flags['entered_5cm'] |= (surface_xy < 0.05)
    env.milestone_flags['entered_2cm'] |= (surface_xy < 0.02)

    # ---------------------------------------------------------------------
    # 10. EXPONENTIAL time penalty (v8.2 - SUPER STRONG)
    # ---------------------------------------------------------------------
    # Exponential growth: gentle at first, BRUTAL at the end
    # ep_len=100:   penalty ≈ -0.01
    # ep_len=500:   penalty ≈ -0.03
    # ep_len=1000:  penalty ≈ -0.10
    # ep_len=1500:  penalty ≈ -0.30 (OUCH!)
    
    max_ep_len = float(env.max_episode_length)
    length_ratio = ep_len.float() / max_ep_len  # [0, 1]
    
    # Exponential growth: exp(4 * x) - 1 gives range [0, ~54]
    exp_factor = torch.exp(4.0 * length_ratio) - 1.0
    exp_factor = exp_factor / 54.0  # Normalize to [0, 1]
    
    base_time_penalty = -0.01
    time_penalty = base_time_penalty * (1.0 + 30.0 * exp_factor)

    # ---------------------------------------------------------------------
    # 11. Survival bonus (PERMANENTLY DISABLED)
    # ---------------------------------------------------------------------
    survival_bonus = torch.zeros_like(surface_xy)

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
        milestone_bonus +      # ONE-TIME bonuses (replaces proximity/precision)
        time_penalty +
        survival_bonus
    )

    total_reward = torch.clamp(total_reward, min=-400.0, max=400.0)

    # ---------------------------------------------------------------------
    # Logging
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
    _log("milestone_bonus", milestone_bonus)  # NEW: replaces proximity/precision
    _log("time_penalty", time_penalty)
    _log("survival_bonus", survival_bonus)

    return total_reward