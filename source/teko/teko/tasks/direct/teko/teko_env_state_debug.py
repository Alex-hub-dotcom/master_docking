# SPDX-License-Identifier: BSD-3-Clause
"""
State-Based Debug Environment - Same rewards/curriculum as vision, no camera.

Purpose: Isolate whether S6 bottleneck is vision or curriculum/rewards.
"""

from __future__ import annotations
import torch
from .teko_env import TekoEnv


class TekoEnvStateDebug(TekoEnv):
    """
    TekoEnv with camera disabled.
    Uses SAME rewards, curriculum, and termination as vision version.
    Only difference: actor uses privileged state instead of RGB.
    """

    def _setup_tiled_camera(self):
        """Disable camera to save memory."""
        self.tiled_camera = None
        print("[INFO] StateDebug mode: TiledCamera DISABLED (using privileged state)")

    def _get_observations(self) -> dict:
        """
        Return privileged state for both policy and critic.
        This makes it a fair comparison - same info, just no vision bottleneck.
        """
        # Get privileged observations from parent
        priv = self._compute_privileged_obs()
        
        return {
            "policy": priv,      # Actor uses state directly
            "privileged": priv,  # Critic also uses state
        }

    def _compute_privileged_obs(self) -> torch.Tensor:
        """Compute 7D privileged state: [dx, dy, dz, yaw_err, vx, vy, omega]"""
        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w
        robot_vel = self.robot.data.root_lin_vel_w
        robot_ang_vel = self.robot.data.root_ang_vel_w
        goal_pos = self.goal_positions

        # Relative position
        rel_pos = goal_pos - robot_pos
        dx = rel_pos[:, 0]
        dy = rel_pos[:, 1]
        dz = rel_pos[:, 2]

        # Yaw error
        robot_yaw = self._quat_to_yaw(robot_quat)
        goal_yaw = torch.atan2(-dy, -dx)  # Point rear toward goal
        yaw_err = self._normalize_angle(goal_yaw - robot_yaw)

        # Velocities
        vx = robot_vel[:, 0]
        vy = robot_vel[:, 1]
        omega = robot_ang_vel[:, 2]

        # Normalize
        pos_scale = 2.0
        vel_scale = 1.0
        ang_scale = 3.14159

        state = torch.stack([
            dx / pos_scale,
            dy / pos_scale,
            dz / pos_scale,
            yaw_err / ang_scale,
            vx / vel_scale,
            vy / vel_scale,
            omega / ang_scale,
        ], dim=-1)

        return state

    def _quat_to_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw from quaternion [x, y, z, w]."""
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-pi, pi]."""
        return torch.atan2(torch.sin(angle), torch.cos(angle))
