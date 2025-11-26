# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for TEKO (v8.8 – TURN-FIRST CURRICULUM)
--------------------------------------------------------

v8.8 Changes (CRITICAL FIX FOR STAGES 4–11):

The core problem: In previous versions, the robot learned "back up straight"
in S0-S3 and this strategy fails in S4+ where lateral correction is needed.
The reward structure punished turning (temporary distance increase) more than
it rewarded alignment, so the robot refused to turn.

Key changes in v8.8:

1. NO DISTANCE PENALTY WHEN FAR (>0.20m):
   - When far, distance_reward = 0 (not negative)
   - Robot can freely maneuver without being punished

2. PROGRESS REWARD GATED BY ALIGNMENT:
   - Progress reward only activates when yaw error < 45°
   - When misaligned, progress = 0 (no reward or penalty for distance changes)
   - This forces the robot to align FIRST, then approach

3. STRONGER ALIGNMENT REWARD WHEN FAR:
   - Far (>0.20m): alignment_reward = 0.5 × (1 - error)
   - Near (≤0.20m): alignment_reward = 0.1 × (1 - error)

4. EXPLICIT TURNING BONUS:
   - When misaligned (yaw_error > 20°) AND turning toward goal: +0.3
   - Directly rewards the turning behavior we want

5. APPROACH BONUS UNCHANGED:
   - Still rewards approaching when aligned and close

Reward structure (9 components):
1. Distance shaping:     0 when far, -2.0×dist when near
2. Progress reward:      only when aligned (yaw < 45°)
3. Alignment shaping:    stronger when far
4. Turning bonus:        NEW - rewards turning toward goal when misaligned
5. Approach bonus:       extra reward when approaching + aligned + close
6. Collision penalty:    -100 (terminal)
7. Boundary penalty:     -500 (terminal)
8. Success bonus:        +400 (terminal)
9. Time penalty:         small increasing penalty
"""

from __future__ import annotations
import torch
import numpy as np


def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π] range."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_total_reward(env) -> torch.Tensor:
    """
    Compute total reward for the TEKO docking task (v8.8 - TURN-FIRST).
    """
    device = env.device

    # ------------------------------------------------------------------
    # 0. Distances and yaw error (used by multiple components)
    # ------------------------------------------------------------------
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()

    # Initialize prev_distance on first call
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()

    # Compute yaw error (how misaligned is the robot?)
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

    # We want the REAR of the robot (camera side) to face the goal
    rear_yaw = robot_yaw + torch.pi
    yaw_error = _angle_wrap(rear_yaw - goal_yaw)
    yaw_error_abs = torch.abs(yaw_error)

    # Normalized yaw error in [0, 1]
    normalized_yaw_error = yaw_error_abs / torch.pi

    # Distance thresholds
    FAR_THRESHOLD = 0.12  # meters - lowered so early stages aren't penalized
    is_far = surface_xy > FAR_THRESHOLD
    is_near = ~is_far

    # Alignment thresholds
    ALIGNED_THRESHOLD = np.deg2rad(45.0)  # 45 degrees
    MISALIGNED_THRESHOLD = np.deg2rad(20.0)  # 20 degrees
    is_aligned = yaw_error_abs < ALIGNED_THRESHOLD
    is_misaligned = yaw_error_abs > MISALIGNED_THRESHOLD

    # ------------------------------------------------------------------
    # 1. Distance reward – ZERO WHEN FAR, gentle shaping when near
    # ------------------------------------------------------------------
    # When far: no penalty, robot can maneuver freely
    # When near: gentle penalty (was -2.0, now -0.5)
    distance_reward = torch.where(
        is_far,
        torch.zeros_like(surface_xy),      # no penalty when far
        -0.5 * surface_xy,                  # GENTLE shaping when near
    )
    distance_reward = torch.clamp(distance_reward, min=-2.0, max=0.0)

    # ------------------------------------------------------------------
    # 2. Progress reward – ONLY WHEN ALIGNED
    # ------------------------------------------------------------------
    progress = env.prev_distance - surface_xy

    # Only give progress reward when reasonably aligned
    # This forces the robot to turn first, THEN approach
    progress_reward = torch.where(
        is_aligned,
        10.0 * progress,                    # full progress reward when aligned
        torch.zeros_like(progress),         # no progress signal when misaligned
    )
    progress_reward = torch.clamp(progress_reward, min=-4.0, max=4.0)

    # Update prev_distance AFTER using it
    env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 3. Alignment reward – STRONGER WHEN FAR
    # ------------------------------------------------------------------
    alignment_far = 0.5 * (1.0 - normalized_yaw_error)   # was 0.3
    alignment_near = 0.1 * (1.0 - normalized_yaw_error)  # was 0.05

    alignment_reward = torch.where(is_far, alignment_far, alignment_near)

    # ------------------------------------------------------------------
    # 4. TURNING BONUS – NEW: explicitly reward turning toward goal
    # ------------------------------------------------------------------
    # Get angular velocity (yaw rate)
    ang_vel = env.robot.data.root_ang_vel_w  # [N, 3]
    yaw_rate = ang_vel[:, 2]  # z-component is yaw rate

    # Check if turning in the correct direction
    # If yaw_error > 0, need to turn negative (and vice versa)
    turning_correct_direction = (yaw_error * yaw_rate) < 0

    # Turning bonus: reward when misaligned AND turning toward goal
    turning_bonus = torch.where(
        is_misaligned & turning_correct_direction & is_far,
        torch.full_like(surface_xy, 0.3),   # bonus for correct turning
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 5. Approach bonus (when approaching + aligned + close)
    # ------------------------------------------------------------------
    well_aligned = yaw_error_abs < np.deg2rad(30.0)
    close_enough = surface_xy < 0.40
    approaching = progress > 0.0

    approach_bonus = torch.where(
        approaching & well_aligned & close_enough,
        2.0 * progress,
        torch.zeros_like(progress),
    )

    # ------------------------------------------------------------------
    # 6. Collision penalty (terminal)
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
    # 7. Boundary penalty (terminal)
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
    # 8. Success bonus (terminal)
    # ------------------------------------------------------------------
    min_success_steps = 5
    terminal_success = raw_success & (ep_len >= min_success_steps)

    success_bonus = torch.where(
        terminal_success,
        torch.full_like(surface_xy, 400.0),
        torch.zeros_like(surface_xy),
    )

    # ------------------------------------------------------------------
    # 9. Time penalty (encourage efficient docking)
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
        + turning_bonus
        + approach_bonus
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
    _log("turning_bonus", turning_bonus)
    _log("approach_bonus", approach_bonus)
    _log("collision_penalty", collision_penalty)
    _log("boundary_penalty", boundary_penalty)
    _log("success_bonus", success_bonus)
    _log("time_penalty", time_penalty)

    return total_reward