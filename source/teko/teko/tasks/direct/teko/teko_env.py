# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal environment to visualize the TEKO robot moving forward."""

from __future__ import annotations
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from teko.robots.teko import TEKO_CONFIGURATION
from .teko_env_cfg import TekoEnvCfg


class TekoEnv(DirectRLEnv):
    """Simple single-robot environment."""

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action buffer (2 actions -> differential drive)
        self.actions = torch.zeros((1, 2), device=self.device)
        self._max_wheel_speed = float(cfg.max_wheel_speed)

        # Track joint indices
        self.dof_idx = None
        self.wheel_signs = None

    def _setup_scene(self):
        """Spawn ground plane and robot."""
        # Ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
        )

        # Spawn robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Finalize scene
        self.scene.clone_environments(copy_from_source=False)

    def _lazy_init_articulation(self):
        """Get joint indices once robot is ready."""
        if self.dof_idx is not None:
            return
        if getattr(self.robot, "root_physx_view", None) is None:
            return

        joint_names = list(self.robot.joint_names)
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        dof_idx_list = [name_to_idx[n] for n in self.cfg.dof_names if n in name_to_idx]
        self.dof_idx = torch.tensor(dof_idx_list, dtype=torch.long, device=self.device)
        self.wheel_signs = torch.ones(len(self.dof_idx), device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Update control actions."""
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        """Apply simple forward velocity to all wheels."""
        if self.dof_idx is None:
            return

        # Differential action: [left, right]
        left = self.actions[:, 0].unsqueeze(-1)
        right = self.actions[:, 1].unsqueeze(-1)

        # Aplica às 4 rodas (dianteiras + traseiras)
        targets = torch.hstack([left, right, left, right]) * self._max_wheel_speed

        # Corrige polaridade (por exemplo, inverter lado direito)
        polarity = torch.tensor(self.cfg.wheel_polarity, device=self.device)
        targets = targets * polarity

        # Envia velocidades para as juntas
        self.robot.set_joint_velocity_target(targets, joint_ids=self.dof_idx)

    def _get_observations(self):
        """Return empty observations (no RL here)."""
        return {}

    def _get_rewards(self):
        """No rewards (not RL)."""
        return torch.zeros(1, device=self.device)

    def _get_dones(self):
        """Never terminate."""
        return torch.zeros(1, dtype=torch.bool, device=self.device), torch.zeros(1, dtype=torch.bool, device=self.device)

    def _reset_idx(self, env_ids):
        """Reset robot pose."""
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()
