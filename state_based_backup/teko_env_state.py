# SPDX-License-Identifier: BSD-3-Clause
"""
State-Based TEKO Environment (FINAL – Learnable)
================================================
Ground-truth state environment for PPO debugging and HPO.

Observation (normalized):
    [dx_norm, dy_norm, yaw_norm]

Reward:
    Dense shaping on distance + yaw
    Sparse bonus on success

This environment MUST learn fast (>90% SSR at Stage 0)
before enabling Optuna, curriculum, or vision-based policy.

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations
import torch
from .teko_env import TekoEnv


class TekoEnvState(TekoEnv):
    """State-based variant of TEKO environment."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    # --------------------------------------------------------------------- #
    # Camera handling (disabled)
    # --------------------------------------------------------------------- #
    def _setup_tiled_camera(self):
        self.tiled_camera = None
        print("[INFO] State-based mode: camera disabled")

    # --------------------------------------------------------------------- #
    # Observations (NORMALIZED)
    # --------------------------------------------------------------------- #
    def _get_observations(self) -> dict:
        """
        Returns normalized ground-truth state:
            [dx_norm, dy_norm, yaw_norm]
        """

        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w
        goal_pos = self.goal_positions

        # Relative position
        rel = goal_pos - robot_pos
        dx = rel[:, 0]
        dy = rel[:, 1]

        # Normalize position by arena size
        pos_scale = max(float(self._arena_half_x), float(self._arena_half_y))
        dx_norm = dx / pos_scale
        dy_norm = dy / pos_scale

        # Yaw error (front-facing for early learning)
        robot_yaw = self._extract_yaw(robot_quat)
        goal_yaw = torch.atan2(dy, dx)
        yaw_error = torch.atan2(
            torch.sin(robot_yaw - goal_yaw),
            torch.cos(robot_yaw - goal_yaw),
        )

        yaw_norm = yaw_error / torch.pi  # ∈ [-1, 1]

        state = torch.stack([dx_norm, dy_norm, yaw_norm], dim=-1)

        return {"policy": state}

    # --------------------------------------------------------------------- #
    # Reward (DENSE + SPARSE)
    # --------------------------------------------------------------------- #
    def compute_reward(self):
        """Dense shaping reward suitable for PPO."""

        robot_pos = self.robot.data.root_pos_w
        goal_pos = self.goal_positions
        rel = goal_pos - robot_pos

        # Distance (XY)
        dist_xy = torch.norm(rel[:, :2], dim=-1)

        # Yaw error
        robot_yaw = self._extract_yaw(self.robot.data.root_quat_w)
        goal_yaw = torch.atan2(rel[:, 1], rel[:, 0])
        yaw_error = torch.atan2(
            torch.sin(robot_yaw - goal_yaw),
            torch.cos(robot_yaw - goal_yaw),
        ).abs()

        # Dense shaping
        reward = (
            -1.0 * dist_xy
            -0.3 * yaw_error
        )

        reward = torch.clamp(reward, -5.0, 0.0)

        # Sparse events
        if hasattr(self, "_last_success"):
            reward += self._last_success.float() * 10.0
        if hasattr(self, "_last_collision"):
            reward -= self._last_collision.float() * 5.0
        if hasattr(self, "_last_out_of_bounds"):
            reward -= self._last_out_of_bounds.float() * 5.0

        return reward

    # --------------------------------------------------------------------- #
    # Termination
    # --------------------------------------------------------------------- #
    def _get_rewards(self):
    
        return self.compute_reward()

    def _get_dones(self):
        """Termination logic (relaxed for learning)."""

        _, _, surface_xy, _ = self.get_sphere_distances_from_physics()

        # SUCCESS (relaxed threshold)
        raw_success = surface_xy < 0.10  # 10 cm
        min_success_steps = 5
        success = raw_success & (self.episode_length_buf >= min_success_steps)

        # Out of bounds
        robot_pos = self.robot.data.root_pos_w
        env_origins = self.scene.env_origins
        local_pos = robot_pos - env_origins

        hx = float(self._arena_half_x)
        hy = float(self._arena_half_y)

        out_of_bounds = (
            (local_pos[:, 0].abs() > hx) |
            (local_pos[:, 1].abs() > hy)
        )

        # Collision DISABLED for state-based debugging
        collision = torch.zeros_like(success, dtype=torch.bool)

        # Store flags for reward
        self._last_success = success.clone()
        self._last_out_of_bounds = out_of_bounds.clone()
        self._last_collision = collision.clone()

        terminated = success | out_of_bounds | collision
        time_out = self.episode_length_buf >= self.max_episode_length

        return terminated, time_out

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
    def _extract_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw from quaternion [x, y, z, w]."""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return torch.atan2(siny_cosp, cosy_cosp)
