# SPDX-License-Identifier: BSD-3-Clause
"""
State-Based TEKO Environment (Debugging) - FIXED v1.1
======================================================
Modified environment that returns ground truth state observations
instead of camera images. Used to validate curriculum + rewards.

Observation: [dx, dy, dz, yaw_error] WITHOUT noise (for debugging)

Author: Alexandre Schleier Neves da Silva
"""
from __future__ import annotations
import torch
import numpy as np
from .teko_env import TekoEnv

class TekoEnvState(TekoEnv):
    """
    State-based variant of TEKO environment.
    Returns ground truth relative pose (NO NOISE) instead of vision.
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
        Get state observations WITHOUT noise (for debugging).
        
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
        
        # ❌ NOISE DISABLED FOR DEBUGGING
        # With ground truth state, the system should achieve 75%+ SSR
        # in Stage 0 within minutes. If not, the problem is elsewhere.
        
        return {"policy": state}
    
    def _extract_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw from quaternion [x, y, z, w]."""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return torch.atan2(siny_cosp, cosy_cosp)
    
    def _get_dones(self):
        """Episode termination logic (SILENT VERSION)."""
        _, _, surface_xy, _ = self.get_sphere_distances_from_physics()
        
        min_success_steps = 5
        min_collision_steps = 10
        
        raw_success = surface_xy < 0.03
        success = raw_success & (self.episode_length_buf >= min_success_steps)
        
        robot_pos_global = self.robot.data.root_pos_w
        env_origins = self.scene.env_origins
        robot_pos_local = robot_pos_global - env_origins
        
        hx = float(self._arena_half_x)
        hy = float(self._arena_half_y)
        
        out_of_bounds = (
            (robot_pos_local[:, 0].abs() > hx) |
            (robot_pos_local[:, 1].abs() > hy)
        )
        
        lin_vel = self.robot.data.root_lin_vel_w
        speed = torch.norm(lin_vel[:, :2], dim=-1)
        
        static_root_pos = self.goal_positions
        diff = robot_pos_global - static_root_pos
        dx = diff[:, 0]
        dy = diff[:, 1]
        
        static_half_len = 0.5 * self._static_body_length
        static_half_wid = 0.5 * self._static_body_width
        active_half_len = 0.5 * self._active_body_length
        active_half_wid = 0.5 * self._active_body_width
        
        boxes_overlap = (
            (dx.abs() < (static_half_len + active_half_len)) &
            (dy.abs() < (static_half_wid + active_half_wid))
        )
        
        collision = (
            boxes_overlap &
            (speed > 0.4) &
            ~raw_success &
            (self.episode_length_buf >= min_collision_steps)
        )

        # 🔴 IMPORTANTE: guardar flags antes do reset
        self._last_success = success.clone()
        self._last_out_of_bounds = out_of_bounds.clone()
        self._last_collision = collision.clone()
        
        terminated = success | out_of_bounds | collision
        time_out = self.episode_length_buf >= self.max_episode_length
        
        # Se quiseres debug visual, podes descomentar:
        # if success.any():
        #     print(f"[SUCCESS] {int(success.sum().item())} dockings!")
        
        return terminated, time_out
