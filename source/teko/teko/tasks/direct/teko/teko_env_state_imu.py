# SPDX-License-Identifier: BSD-3-Clause
"""TEKO State+IMU Environment - 10D observation"""

from .teko_env_state import TekoEnvState
import torch


class TekoEnvStateIMU(TekoEnvState):
    """State-based environment with full IMU data (10D)."""
    
    def _get_observations(self) -> dict:
        robot_pos = self.robot.data.root_pos_w
        goal_pos = self.goal_positions
        
        dx = goal_pos[:, 0] - robot_pos[:, 0]
        dy = goal_pos[:, 1] - robot_pos[:, 1]
        dz = goal_pos[:, 2] - robot_pos[:, 2]
        
        robot_quat = self.robot.data.root_quat_w
        robot_yaw = 2.0 * torch.atan2(robot_quat[:, 2], robot_quat[:, 3])
        goal_yaw = torch.zeros_like(robot_yaw)
        yaw_err = torch.atan2(torch.sin(goal_yaw - robot_yaw), torch.cos(goal_yaw - robot_yaw))
        
        lin_vel = self.robot.data.root_lin_vel_w
        ang_vel = self.robot.data.root_ang_vel_w
        
        obs = torch.stack([
            dx, dy, dz, yaw_err,
            lin_vel[:, 0], lin_vel[:, 1], lin_vel[:, 2],
            ang_vel[:, 0], ang_vel[:, 1], ang_vel[:, 2],
        ], dim=-1)
        
        return {"policy": obs}
