# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for TEKO (v7.0 - BALANCED, NO SURVIVAL EXPLOIT)
================================================================

Key changes:
- REMOVED survival bonus (was encouraging passivity)
- KEPT collision penalty (-500) to prevent crash exploit
- INCREASED progress reward weight (main dense signal)
- Time penalty for long episodes (encourages efficiency)
"""

from __future__ import annotations
import torch


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

    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 1. Distance reward (stronger now that survival bonus is gone)
    # ------------------------------------------------------------------
    distance_reward = -2.0 * surface_xy      # was -1.0, now -2.0
    distance_reward = torch.clamp(distance_reward, min=-10.0, max=0.0)

    # ------------------------------------------------------------------
    # 2. Progress reward (MAIN DENSE SIGNAL - increased weight)
    # ------------------------------------------------------------------
    progress = env.prev_distance - surface_xy
    progress_reward = 20.0 * progress        # was 10.0, now 20.0
    progress_reward = torch.clamp(progress_reward, min=-5.0, max=5.0)
    env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 3. Alignment reward (rear facing goal)
    # ------------------------------------------------------------------
    robot_quat = env.robot.data.root_quat_w
    robot_yaw = _quat_to_yaw(robot_quat)
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions
    vec_to_goal = goal_pos - robot_pos
    goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])
    rear_yaw = robot_yaw + torch.pi
    yaw_error = _angle_wrap(rear_yaw - goal_yaw)
    alignment_reward = 0.5 * torch.cos(yaw_error)   # in [-0.5, 0.5]

    # ------------------------------------------------------------------
    # 4. Velocity penalty (small)
    # ------------------------------------------------------------------
    lin_vel = env.robot.data.root_lin_vel_w
    speed = torch.norm(lin_vel[:, :2], dim=-1)
    velocity_penalty = -0.01 * speed

    # ------------------------------------------------------------------
    # 5. Oscillation penalty (discourage action jitter)
    # ------------------------------------------------------------------
    if env.prev_actions is None:
        env.prev_actions = torch.zeros_like(env.actions)
    action_diff = torch.norm(env.actions - env.prev_actions, dim=-1)
    oscillation_penalty = -0.02 * action_diff
    env.prev_actions = env.actions.clone()

    # ------------------------------------------------------------------
    # 6. Collision penalty (CRITICAL - prevents crash exploit)
    # ------------------------------------------------------------------
    raw_success = surface_xy < 0.03
    collision = (surface_xy < 0.10) & (speed > 0.3) & (~raw_success)
    collision_penalty = torch.where(
        collision,
        torch.tensor(-500.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ------------------------------------------------------------------
    # 7. Boundary penalty
    # ------------------------------------------------------------------
    robot_pos_global = env.robot.data.root_pos_w
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

    # ------------------------------------------------------------------
    # 8. Success bonus (BIG reward for completion)
    # ------------------------------------------------------------------
    success = raw_success
    success_bonus = torch.where(
        success,
        torch.tensor(200.0, device=device),   # was 100, now 200
        torch.tensor(0.0, device=device),
    )

    # ------------------------------------------------------------------
    # 9. Proximity bonus
    # ------------------------------------------------------------------
    close = (surface_xy < 0.10) & (surface_xy >= 0.03) & (~collision)
    proximity_bonus = torch.where(
        close,
        torch.tensor(5.0, device=device),     # was 2.0, now 5.0
        torch.tensor(0.0, device=device),
    )

    # ------------------------------------------------------------------
    # 10. Time penalty (encourages efficiency, replaces survival bonus)
    # ------------------------------------------------------------------
    time_penalty = torch.full_like(surface_xy, -0.1)

    # ------------------------------------------------------------------
    # TOTAL REWARD
    # ------------------------------------------------------------------
    total_reward = (
        distance_reward +
        progress_reward +
        alignment_reward +
        velocity_penalty +
        oscillation_penalty +
        collision_penalty +
        boundary_penalty +
        success_bonus +
        proximity_bonus +
        time_penalty
    )

    total_reward = torch.clamp(total_reward, min=-500.0, max=400.0)

    # ------------------------------------------------------------------
    # Logging (robust: garante que as chaves existem)
    # ------------------------------------------------------------------
    rc = env.reward_components

    def _log(name: str, value: torch.Tensor):
        if name not in rc:
            rc[name] = []
        rc[name].append(value.mean().item())

    _log("distance", distance_reward)
    _log("progress", progress_reward)
    _log("alignment", alignment_reward)
    _log("velocity_penalty", velocity_penalty)
    _log("oscillation_penalty", oscillation_penalty)
    _log("collision_penalty", collision_penalty)
    _log("wall_penalty", boundary_penalty)
    _log("success_bonus", success_bonus)
    _log("proximity_bonus", proximity_bonus)
    _log("time_penalty", time_penalty)

    return total_reward
