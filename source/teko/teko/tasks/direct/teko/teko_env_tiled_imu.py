# SPDX-License-Identifier: BSD-3-Clause
"""TEKO Vision+IMU Environment"""

from .teko_env_tiled import TekoEnvTiled
import torch


class TekoEnvTiledIMU(TekoEnvTiled):
    """Vision environment with IMU data for sensor fusion."""
    
    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        
        lin_vel = self.robot.data.root_lin_vel_w
        ang_vel = self.robot.data.root_ang_vel_w
        
        imu = torch.cat([lin_vel, ang_vel], dim=-1)
        obs["imu"] = imu
        
        return obs
