# SPDX-License-Identifier: BSD-3-Clause
"""
State-Based TEKO Environment (Debugging)
=========================================
Modified environment that returns ground truth state observations
instead of camera images. Used to validate curriculum + rewards.
Observation: [dx, dy, dz, yaw_error] with noise
Author: Alexandre Schleier Neves da Silva
"""
from __future__ import annotations
import torch
import numpy as np
from .teko_env import TekoEnv

class TekoEnvState(TekoEnv):
    """
    State-based variant of TEKO environment.
    Returns ground truth relative pose + noise instead of vision.
    """
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        """Override to skip camera initialization."""
        # MUST set this BEFORE calling super().__init__
        # Add dummy camera config
        from isaaclab.utils import configclass
        
        @configclass
        class DummyCameraCfg:
            width = 64
            height = 64
        
        if not hasattr(cfg, 'camera'):
            cfg.camera = DummyCameraCfg()
        
        # Now safe to call parent
        super().__init__(cfg, render_mode, **kwargs)
    
    def _setup_cameras(self):
        """Skip camera setup for state-based mode."""
        self.cameras = []
        print("[INFO] State-based mode: skipping camera setup")
    
    def _get_observations(self) -> dict:
        """
        Get state observations with noise.
        Returns:
            dict with "policy" key containing [dx, dy, dz, yaw_error]
        """
        # Get ground truth relative pose
        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w
        goal_pos = self.goal_positions
        
        # Relative position
        relative_pos = goal_pos - robot_pos  # [N, 3]
        
        # Yaw error
        robot_yaw = self._extract_yaw(robot_quat)
        goal_yaw = torch.atan2(relative_pos[:, 1], relative_pos[:, 0])
        rear_yaw = robot_yaw + torch.pi
        yaw_error = torch.atan2(
            torch.sin(rear_yaw - goal_yaw),
            torch.cos(rear_yaw - goal_yaw)
        )  # [N]
        
        # Combine into state: [dx, dy, dz, yaw_error]
        state = torch.cat([
            relative_pos,  # [N, 3]
            yaw_error.unsqueeze(-1),  # [N, 1]
        ], dim=-1)  # [N, 4]
        
        # Add noise for robustness
        if hasattr(self.cfg, "state_noise_pos"):
            pos_noise = torch.randn_like(relative_pos) * self.cfg.state_noise_pos
            state[:, :3] += pos_noise
        
        if hasattr(self.cfg, "state_noise_rot"):
            rot_noise = torch.randn_like(yaw_error) * self.cfg.state_noise_rot
            state[:, 3] += rot_noise
        
        return {"policy": state}
    
    def _extract_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw from quaternion [x, y, z, w]."""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return torch.atan2(siny_cosp, cosy_cosp)