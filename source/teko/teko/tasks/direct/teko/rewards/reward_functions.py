# SPDX-License-Identifier: BSD-3-Clause
"""
Reward Functions for TEKO Docking (v11 – PROXIMITY GATED)
=========================================================

Fix from v10: All yaw bonuses REQUIRE proximity.
No more reward for being aligned from far away.
"""

from __future__ import annotations

import numpy as np
import torch


REWARD_CONFIG = {
    # Distance (negative, pushes toward goal)
    "distance_scale": -2.0,
    "distance_min": -4.0,
    "distance_max": 0.0,

    # Progress (main driver)
    "progress_scale": 8.0,
    "progress_min": -3.0,
    "progress_max": 3.0,

    # Alignment (ONLY when close)
    "alignment_scale": 1.5,
    "alignment_gate_distance": 0.40,

    # Fine yaw (very close only)
    "fine_yaw_scale": 4.0,
    "fine_yaw_threshold_deg": 12.0,
    "fine_yaw_distance": 0.15,

    # Facing bonus
    "facing_bonus": 2.5,
    "facing_threshold_deg": 10.0,
    "facing_distance": 0.20,

    # Approach bonus
    "approach_scale": 2.0,

    # Turning bonus
    "turning_bonus": 0.8,
    "turning_threshold_deg": 15.0,
    "turning_distance": 0.50,
    "turning_rate_norm": 1.0,

    # Misaligned close penalty
    "misaligned_penalty": -3.0,
    "misaligned_threshold_deg": 20.0,
    "misaligned_distance": 0.15,

    # Terminal rewards
    "collision_penalty": -100.0,
    "collision_speed_threshold": 0.4,
    "collision_min_steps": 10,

    "boundary_penalty": -500.0,

    "success_bonus": 500.0,
    "success_distance": 0.03,
    "success_min_steps": 5,

    # Time penalty
    "time_base": -0.02,
    "time_exp_factor": 4.0,
    "time_scale": 50.0,

    # Clipping
    "reward_min": -500.0,
    "reward_max": 500.0,

    "log_components": False,
    "log_interval_steps": 50,
    "log_max_len": 2000,
}


def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _extract_yaw(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def _compute_yaw_error(robot_yaw: torch.Tensor, robot_pos: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
    vec_to_goal = goal_pos - robot_pos
    goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])
    rear_yaw = robot_yaw + torch.pi
    return _angle_wrap(rear_yaw - goal_yaw)


def _compute_distance_reward(surface_xy: torch.Tensor) -> torch.Tensor:
    cfg = REWARD_CONFIG
    r = cfg["distance_scale"] * surface_xy
    return torch.clamp(r, min=cfg["distance_min"], max=cfg["distance_max"])


def _compute_progress_reward(env, surface_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = REWARD_CONFIG
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()
    progress = env.prev_distance - surface_xy
    r = cfg["progress_scale"] * progress
    r = torch.clamp(r, min=cfg["progress_min"], max=cfg["progress_max"])
    env.prev_distance.copy_(surface_xy)
    return r, progress


def _compute_alignment_reward(yaw_error_abs: torch.Tensor, surface_xy: torch.Tensor) -> torch.Tensor:
    cfg = REWARD_CONFIG
    gate_dist = cfg["alignment_gate_distance"]
    close_enough = surface_xy < gate_dist
    normalized = yaw_error_abs / torch.pi
    alignment = cfg["alignment_scale"] * (1.0 - normalized)
    dist_scale = (1.0 - surface_xy / gate_dist).clamp(0.0, 1.0)
    return torch.where(close_enough, alignment * dist_scale, torch.zeros_like(alignment))


def _compute_fine_yaw_reward(yaw_error_abs: torch.Tensor, surface_xy: torch.Tensor) -> torch.Tensor:
    cfg = REWARD_CONFIG
    threshold = float(np.deg2rad(cfg["fine_yaw_threshold_deg"]))
    max_dist = cfg["fine_yaw_distance"]
    close = surface_xy < max_dist
    normalized_yaw = torch.clamp(yaw_error_abs / threshold, 0.0, 1.0)
    quadratic_bonus = (1.0 - normalized_yaw) ** 2
    dist_scale = (1.0 - surface_xy / max_dist).clamp(0.0, 1.0) ** 2
    reward = cfg["fine_yaw_scale"] * quadratic_bonus * dist_scale
    return torch.where(close, reward, torch.zeros_like(reward))


def _compute_facing_bonus(yaw_error_abs: torch.Tensor, surface_xy: torch.Tensor) -> torch.Tensor:
    cfg = REWARD_CONFIG
    threshold = float(np.deg2rad(cfg["facing_threshold_deg"]))
    aligned = yaw_error_abs < threshold
    close = surface_xy < cfg["facing_distance"]
    return torch.where(aligned & close, torch.full_like(surface_xy, cfg["facing_bonus"]), torch.zeros_like(surface_xy))


def _compute_approach_bonus(progress: torch.Tensor, yaw_error_abs: torch.Tensor, surface_xy: torch.Tensor) -> torch.Tensor:
    cfg = REWARD_CONFIG
    threshold = float(np.deg2rad(cfg["facing_threshold_deg"]))
    approaching = progress > 0.0
    aligned = yaw_error_abs < threshold
    close = surface_xy < cfg["facing_distance"]
    return torch.where(approaching & aligned & close, cfg["approach_scale"] * progress, torch.zeros_like(progress))


def _compute_turning_bonus(yaw_error: torch.Tensor, yaw_error_abs: torch.Tensor, ang_vel: torch.Tensor, surface_xy: torch.Tensor) -> torch.Tensor:
    cfg = REWARD_CONFIG
    threshold = float(np.deg2rad(cfg["turning_threshold_deg"]))
    yaw_rate = ang_vel[:, 2]
    turning_correct = (yaw_error * yaw_rate) < 0.0
    misaligned = yaw_error_abs > threshold
    close = surface_xy < cfg["turning_distance"]
    rate_scale = torch.clamp(yaw_rate.abs() / float(cfg["turning_rate_norm"]), 0.0, 1.0)
    bonus = cfg["turning_bonus"] * rate_scale
    return torch.where(misaligned & turning_correct & close, bonus, torch.zeros_like(surface_xy))


def _compute_misaligned_close_penalty(yaw_error_abs: torch.Tensor, surface_xy: torch.Tensor) -> torch.Tensor:
    cfg = REWARD_CONFIG
    threshold = float(np.deg2rad(cfg["misaligned_threshold_deg"]))
    close = surface_xy < cfg["misaligned_distance"]
    misaligned = yaw_error_abs > threshold
    return torch.where(close & misaligned, torch.full_like(surface_xy, cfg["misaligned_penalty"]), torch.zeros_like(surface_xy))


def _compute_collision_penalty(env, surface_xy: torch.Tensor) -> torch.Tensor:
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
    collision = overlap & (speed > cfg["collision_speed_threshold"]) & (~raw_success) & (ep_len >= cfg["collision_min_steps"])
    return torch.where(collision, torch.full_like(surface_xy, cfg["collision_penalty"]), torch.zeros_like(surface_xy))


def _compute_boundary_penalty(env, surface_xy: torch.Tensor) -> torch.Tensor:
    cfg = REWARD_CONFIG
    raw_success = surface_xy < cfg["success_distance"]
    robot_pos = env.robot.data.root_pos_w
    origins = env.scene.env_origins
    local_pos = robot_pos - origins
    hx = float(env._arena_half_x)
    hy = float(env._arena_half_y)
    out = (local_pos[:, 0].abs() > hx) | (local_pos[:, 1].abs() > hy)
    return torch.where(out & (~raw_success), torch.full_like(surface_xy, cfg["boundary_penalty"]), torch.zeros_like(surface_xy))


def _compute_success_bonus(env, surface_xy: torch.Tensor) -> torch.Tensor:
    cfg = REWARD_CONFIG
    raw_success = surface_xy < cfg["success_distance"]
    ep_len = env.episode_length_buf
    success = raw_success & (ep_len >= cfg["success_min_steps"])
    return torch.where(success, torch.full_like(surface_xy, cfg["success_bonus"]), torch.zeros_like(surface_xy))


def _compute_time_penalty(env) -> torch.Tensor:
    cfg = REWARD_CONFIG
    max_ep_len = float(env.max_episode_length)
    ep_len = env.episode_length_buf
    ratio = ep_len.float() / max_ep_len
    exp_factor = torch.exp(cfg["time_exp_factor"] * ratio) - 1.0
    exp_factor = exp_factor / 54.0
    return cfg["time_base"] * (1.0 + cfg["time_scale"] * exp_factor)


def compute_total_reward(env) -> torch.Tensor:
    cfg = REWARD_CONFIG
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    robot_quat = env.robot.data.root_quat_w
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions
    ang_vel = env.robot.data.root_ang_vel_w
    robot_yaw = _extract_yaw(robot_quat)
    yaw_error = _compute_yaw_error(robot_yaw, robot_pos, goal_pos)
    yaw_error_abs = torch.abs(yaw_error)

    total = (
        _compute_distance_reward(surface_xy)
        + _compute_progress_reward(env, surface_xy)[0]
        + _compute_alignment_reward(yaw_error_abs, surface_xy)
        + _compute_fine_yaw_reward(yaw_error_abs, surface_xy)
        + _compute_facing_bonus(yaw_error_abs, surface_xy)
        + _compute_approach_bonus(_compute_progress_reward(env, surface_xy)[1], yaw_error_abs, surface_xy)
        + _compute_turning_bonus(yaw_error, yaw_error_abs, ang_vel, surface_xy)
        + _compute_misaligned_close_penalty(yaw_error_abs, surface_xy)
        + _compute_collision_penalty(env, surface_xy)
        + _compute_boundary_penalty(env, surface_xy)
        + _compute_success_bonus(env, surface_xy)
        + _compute_time_penalty(env)
    )
    return torch.clamp(total, min=cfg["reward_min"], max=cfg["reward_max"])


def get_reward_config() -> dict:
    return REWARD_CONFIG.copy()


def set_reward_config(new_config: dict) -> None:
    global REWARD_CONFIG
    REWARD_CONFIG.update(new_config)
