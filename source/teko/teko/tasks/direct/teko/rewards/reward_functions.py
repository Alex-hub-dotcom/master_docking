# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for TEKO (v8.4 – FINAL SIMPLIFIED VERSION)
------------------------------------------------------------

v8.4 Changes (FINAL):
- REMOVED facing_bonus (never triggered, redundant with alignment_reward)
- Kept only essential, active reward components
- Maximum simplicity while maintaining effectiveness

Evolution:
- v8.0-8.2: Complex milestone system (caused issues)
- v8.3: Simplified, removed milestones, reduced alignment scale
- v8.4: FINAL - removed unused facing_bonus for clarity

Philosophy: 
- Every reward component must actively contribute to learning
- Success bonus (+300) is the dominant goal signal
- Alignment provides continuous rear-facing guidance
- Time penalty encourages efficiency
- No farmable continuous bonuses
- No unused conditional bonuses

This version prioritizes simplicity, clarity, and robustness.
"""

from __future__ import annotations
import torch
import numpy as np


def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π] range."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_total_reward(env) -> torch.Tensor:
    """
    Compute total reward for TEKO docking task (v8.4 - FINAL SIMPLIFIED).
    
    Reward structure (7 components):
    1. Distance shaping: Guides robot toward goal
    2. Progress reward: Main learning signal (rewards getting closer)
    3. Alignment shaping: Guides rear-facing approach (reduced scale: 1.0)
    4. Approach bonus: Extra reward for approaching while aligned
    5. Collision penalty: Terminal safety penalty (-100)
    6. Boundary penalty: Terminal out-of-bounds penalty (-500)
    7. Success bonus: Terminal goal reward (+300) - DOMINANT SIGNAL
    8. Time penalty: Encourages efficiency (moderate exponential)
    
    All components are active and contribute to learning.
    """
    device = env.device
    num_envs = env.scene.cfg.num_envs
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()

    # Initialize tracking buffers on first call
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()
    
    # ---------------------------------------------------------------------
    # 1. Distance reward (continuous shaping)
    # ---------------------------------------------------------------------
    # Guides robot toward goal: closer = less negative
    distance_reward = -2.0 * surface_xy
    distance_reward = torch.clamp(distance_reward, min=-10.0, max=0.0)

    # ---------------------------------------------------------------------
    # 2. Progress reward (main learning signal)
    # ---------------------------------------------------------------------
    # Rewards getting closer to goal
    progress = env.prev_distance - surface_xy
    progress_reward = 10.0 * progress
    progress_reward = torch.clamp(progress_reward, min=-4.0, max=4.0)
    env.prev_distance = surface_xy.clone()

    # ---------------------------------------------------------------------
    # 3. Alignment reward (REDUCED SCALE)
    # ---------------------------------------------------------------------
    # Guides rear-facing approach at all distances
    # Scale: 1.0 (reduced from 5.0 in v8.2 to prevent farming)
    # This provides continuous guidance without being exploitable
    
    robot_quat = env.robot.data.root_quat_w
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions

    # Extract yaw from quaternion
    qx, qy, qz, qw = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    robot_yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Compute desired yaw (rear facing goal)
    vec_to_goal = goal_pos - robot_pos
    goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])
    rear_yaw = robot_yaw + torch.pi
    yaw_error = _angle_wrap(rear_yaw - goal_yaw)

    # Normalized alignment reward: [0, 1.0]
    normalized_error = torch.abs(yaw_error) / torch.pi
    alignment_reward = 1.0 * (1.0 - normalized_error)

    # ---------------------------------------------------------------------
    # 4. Approach bonus (conditional)
    # ---------------------------------------------------------------------
    # Extra reward for approaching while reasonably aligned
    # This encourages forward progress when robot is roughly pointing correctly
    approaching = (progress > 0.0) & (torch.abs(yaw_error) < np.deg2rad(60.0))
    approach_bonus = torch.where(
        approaching,
        3.0 * progress,
        torch.tensor(0.0, device=device),
    )

    # ---------------------------------------------------------------------
    # 5. Collision penalty (terminal)
    # ---------------------------------------------------------------------
    # Penalize high-speed collisions (but not successful docking)
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

    # Calculate speed for collision detection
    lin_vel = env.robot.data.root_lin_vel_w
    speed = torch.norm(lin_vel[:, :2], dim=-1)

    # Collision = overlap + high speed + not successful docking
    collision = boxes_overlap & (speed > 0.4) & (~raw_success)

    collision_penalty = torch.where(
        collision,
        torch.tensor(-100.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ---------------------------------------------------------------------
    # 6. Boundary penalty (terminal)
    # ---------------------------------------------------------------------
    # Penalize leaving the arena boundaries
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
    # 7. Success bonus (terminal) - MAIN GOAL
    # ---------------------------------------------------------------------
    # Large bonus for successful docking
    # This is the dominant reward signal that drives learning
    min_success_steps = 5
    ep_len = env.episode_length_buf
    terminal_success = raw_success & (ep_len >= min_success_steps)

    success_bonus = torch.where(
        terminal_success,
        torch.tensor(300.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ---------------------------------------------------------------------
    # 8. Time penalty (moderate exponential)
    # ---------------------------------------------------------------------
    # Encourages efficiency without creating exploitable dynamics
    # Grows exponentially with episode length
    # 
    # Progression:
    # ep_len=100:   penalty ≈ -0.02/step
    # ep_len=300:   penalty ≈ -0.04/step
    # ep_len=500:   penalty ≈ -0.10/step
    # ep_len=700:   penalty ≈ -0.25/step
    # ep_len=1000:  penalty ≈ -0.60/step
    
    max_ep_len = float(env.max_episode_length)  # 1800
    length_ratio = ep_len.float() / max_ep_len  # [0, 1]
    
    # Exponential growth: exp(4 * x) - 1 gives range [0, ~54]
    exp_factor = torch.exp(4.0 * length_ratio) - 1.0
    exp_factor = exp_factor / 54.0  # Normalize to [0, 1]
    
    # Base penalty: -0.02, exponentially scaled up to 50×
    base_time_penalty = -0.02
    time_penalty = base_time_penalty * (1.0 + 50.0 * exp_factor)

    # ---------------------------------------------------------------------
    # Total reward (v8.4 - FINAL SIMPLIFIED)
    # ---------------------------------------------------------------------
    total_reward = (
        distance_reward +      # Continuous shaping
        progress_reward +      # Main learning signal
        alignment_reward +     # Rear-facing guidance (scale: 1.0)
        approach_bonus +       # Conditional bonus for aligned approach
        collision_penalty +    # Terminal safety penalty
        boundary_penalty +     # Terminal boundary penalty
        success_bonus +        # Terminal goal reward (DOMINANT)
        time_penalty           # Efficiency encouragement
    )

    # Clamp to prevent extreme values
    total_reward = torch.clamp(total_reward, min=-400.0, max=400.0)

    # ---------------------------------------------------------------------
    # Logging (all active components)
    # ---------------------------------------------------------------------
    rc = env.reward_components

    def _log(name, val):
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