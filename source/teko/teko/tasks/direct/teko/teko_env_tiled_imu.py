# SPDX-License-Identifier: BSD-3-Clause
"""TEKO Vision+IMU Environment with Privileged Observations
/home/schux00/teko/source/teko/teko/tasks/direct/teko/teko_env_tiled_imu.py

Author: Alexandre Schleier Neves da Silva

For questions or collaboration, contact:
    alexandre.schleiernevesdasilva@uni-hohenheim.de
"""
#/home/schux00/teko/source/teko/teko/tasks/direct/teko/teko_env_tiled_imu.py
from .teko_env_tiled import TekoEnvTiled
import torch


class TekoEnvTiledIMU(TekoEnvTiled):
    """Vision environment with IMU data and privileged state for asymmetric training."""
    
    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        
        # IMU: linear + angular velocity (6D)
        lin_vel = self.robot.data.root_lin_vel_w
        ang_vel = self.robot.data.root_ang_vel_w
        imu = torch.cat([lin_vel, ang_vel], dim=-1)
        obs["imu"] = imu
        
        # Privileged state for asymmetric critic and aux heads (7D)
        if getattr(self.cfg, "asymmetric_critic", False):
            obs["privileged"] = self._compute_privileged_obs()
        
        return obs

    def _compute_privileged_obs(self) -> torch.Tensor:
        """Compute 7D privileged state: [dx, dy, dz, yaw_err, vx, vy, omega]"""
        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w
        robot_vel = self.robot.data.root_lin_vel_w
        robot_ang_vel = self.robot.data.root_ang_vel_w
        goal_pos = self.goal_positions

        # Relative position (goal - robot)
        rel_pos = goal_pos - robot_pos
        dx = rel_pos[:, 0]
        dy = rel_pos[:, 1]
        dz = rel_pos[:, 2]

        # Yaw error: angle between robot's rear and goal direction
        robot_yaw = self._quat_to_yaw(robot_quat)
        goal_yaw = torch.atan2(-rel_pos[:, 1], -rel_pos[:, 0])  # Point rear toward goal
        yaw_err = self._normalize_angle(goal_yaw - robot_yaw)

        # Velocities
        vx = robot_vel[:, 0]
        vy = robot_vel[:, 1]
        omega = robot_ang_vel[:, 2]

        # Normalization scales
        pos_scale = 2.0      # Arena is ~4m wide
        vel_scale = 1.0      # Max velocity ~1 m/s
        ang_scale = 3.14159  # Max angle = pi

        state = torch.stack([
            dx / pos_scale,
            dy / pos_scale,
            dz / pos_scale,
            yaw_err ,  # INDEX 3 = yaw_error for aux head!
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
