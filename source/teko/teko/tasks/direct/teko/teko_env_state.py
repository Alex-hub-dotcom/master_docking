# SPDX-License-Identifier: BSD-3-Clause
"""
State-Based Environment for TEKO Docking
=========================================

Same rewards, curriculum, and termination as vision version.
Only difference: actor uses privileged state [7D] instead of RGB [4x84x84].

Purpose: Isolate whether learning bottleneck is vision or curriculum/rewards.
If state-based learns but vision doesn't -> problem is in CNN/visual features.
If both fail -> problem is in curriculum/rewards.

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations
import torch
import numpy as np
import gymnasium as gym

from .teko_env import TekoEnv
from .teko_env_cfg import TekoEnvCfg


class TekoEnvState(TekoEnv):
    """
    TekoEnv with camera disabled, using privileged state for policy.
    
    Observation space: 7D vector [dx, dy, dz, yaw_err, vx, vy, omega]
    Action space: 2D continuous [v_cmd, w_cmd]
    """

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        # Disable camera-related attributes
        self._cam_res = (1, 1)  # Dummy
        self.num_frame_stack = 1
        self.frame_stack = None
        self.frame_counts = None
        
        super().__init__(cfg, render_mode, **kwargs)

    def _init_observation_space(self):
        """Define observation space as 7D privileged state vector."""
        self.observation_space = gym.spaces.Dict({
            "policy": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7,),
                dtype=np.float32,
            ),
            "privileged": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7,),
                dtype=np.float32,
            ),
        })
        print("[INFO] State-based mode: observation space = 7D privileged state")

    def _setup_cameras(self):
        """Disable camera setup to save memory."""
        self.cameras = []
        print("[INFO] State-based mode: cameras DISABLED")

    def _get_observations(self) -> dict:
        """
        Return privileged state for both policy and critic.
        
        State vector [7D]:
            [0] dx: relative X position to goal (normalized)
            [1] dy: relative Y position to goal (normalized)
            [2] dz: relative Z position to goal (normalized)
            [3] yaw_err: angular error to goal direction (normalized)
            [4] vx: linear velocity X (normalized)
            [5] vy: linear velocity Y (normalized)
            [6] omega: angular velocity Z (normalized)
        """
        priv = self._compute_privileged_obs()
        
        return {
            "policy": priv,
            "privileged": priv,
        }

    def _compute_privileged_obs(self) -> torch.Tensor:
        """Compute 7D privileged state observation."""
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

        # Yaw error (same formula as reward function)
        robot_yaw = self._extract_yaw(robot_quat)
        goal_yaw = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
        rear_yaw = robot_yaw + torch.pi  # Robot's rear direction
        yaw_err = self._angle_wrap(rear_yaw - goal_yaw)

        # Velocities
        vx = robot_vel[:, 0]
        vy = robot_vel[:, 1]
        omega = robot_ang_vel[:, 2]

        # Normalization scales
        pos_scale = 2.0      # Arena is ~4m wide
        vel_scale = 1.0      # Max velocity ~1 m/s
        ang_scale = torch.pi # Max angle = pi

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

    def _extract_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Extract yaw from quaternion.
        
        Isaac Lab uses XYZW convention: quat = [x, y, z, w]
        """
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _angle_wrap(self, angle: torch.Tensor) -> torch.Tensor:
        """Wrap angle to [-pi, pi]."""
        return torch.atan2(torch.sin(angle), torch.cos(angle))