# Copyright (c) 2022-2025
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
import torch

# Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .teko_env_cfg import TekoEnvCfg


class TekoEnv(DirectRLEnv):
    """Tiny direct RL env for a 4-wheel TEKO robot (just ground + robot)."""

    cfg: TekoEnvCfg

    # ----------------------- init -----------------------

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        device = self.device
        nenv = self.cfg.scene.num_envs

        # Action buffer: 2 dims (left, right) unless independent_wheels=True
        a_dim = 4 if self.cfg.independent_wheels else 2
        self.actions = torch.zeros((nenv, a_dim), device=device)

        # Target smoothing
        self._targets_prev = torch.zeros((nenv, 4), device=device)
        self._max_wheel_speed = float(self.cfg.max_wheel_speed)
        self._max_wheel_accel = float(self.cfg.max_wheel_accel)
        self._env_dt = float(self.cfg.decimation) * float(self.cfg.sim.dt)

        # Will be set once PhysX views exist
        self.dof_idx: torch.Tensor | None = None
        self.wheel_signs: torch.Tensor | None = None  # (4,)

    # ----------------------- scene setup -----------------------

    def _setup_scene(self):
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground plane (moderate friction)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6,
                    dynamic_friction=0.5,
                    restitution=0.0,
                )
            ),
        )

        # Clone envs and register robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

    # ----------------------- internals -----------------------

    def _lazy_init_articulation(self):
        if self.dof_idx is not None and self.wheel_signs is not None:
            return
        if getattr(self.robot, "root_physx_view", None) is None:
            return

        # Map joint names -> indices
        joint_names = list(self.robot.joint_names)
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        missing = [n for n in self.cfg.dof_names if n not in name_to_idx]
        if missing:
            raise RuntimeError(
                "TEKO joint name(s) not found in asset.\n"
                f"Missing: {missing}\n"
                f"Available joints: {joint_names}"
            )

        dof_idx_list = [name_to_idx[n] for n in self.cfg.dof_names]
        self.dof_idx = torch.tensor(dof_idx_list, dtype=torch.long, device=self.device)

        # Fixed: identical spin sign for all wheels (L=R -> straight)
        self.wheel_signs = torch.ones(4, device=self.device, dtype=torch.float32)

        # Velocity-like drive: stiffness=0, damping>0
        n = len(self.dof_idx)
        stiffness = torch.zeros(n, device=self.device)
        damping = torch.full((n,), float(self.cfg.drive_damping), device=self.device)
        max_force = torch.full((n,), float(self.cfg.drive_max_force), device=self.device)
        if hasattr(self.robot, "set_joint_drive_property"):
            self.robot.set_joint_drive_property(
                stiffness=stiffness, damping=damping, max_force=max_force, joint_ids=self.dof_idx
            )

    # ----------------------- RL hooks -----------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions is None or actions.numel() == 0:
            a_dim = 4 if self.cfg.independent_wheels else 2
            actions = torch.zeros((self.cfg.scene.num_envs, a_dim), device=self.device)
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self) -> None:
        if self.dof_idx is None:
            return

        pol = float(self.cfg.wheel_polarity)
        if self.cfg.independent_wheels:
            # actions: [FL, FR, RL, RR]
            raw = float(self.cfg.action_scale) * self.actions
            targets = raw * self.wheel_signs.unsqueeze(0) * pol
        else:
            # actions: [left, right] -> [FL, FR, RL, RR]
            scale = float(self.cfg.action_scale)
            left = (scale * self.actions[:, 0]).unsqueeze(-1)
            right = (scale * self.actions[:, 1]).unsqueeze(-1)
            raw = torch.hstack([left, right, left, right])
            targets = raw * self.wheel_signs.unsqueeze(0) * pol

        # Slew-rate limit + clamp
        max_delta = self._max_wheel_accel * self._env_dt
        delta = torch.clamp(targets - self._targets_prev, -max_delta, +max_delta)
        targets = self._targets_prev + delta
        targets = torch.clamp(targets, -self._max_wheel_speed, +self._max_wheel_speed)
        self._targets_prev = targets.detach()

        # Send to sim
        self.robot.set_joint_velocity_target(targets, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].reshape(-1, 1)
        return {"policy": forward_speed}

    def _get_rewards(self) -> torch.Tensor:
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].reshape(-1, 1)
        return forward_speed

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)
        self._lazy_init_articulation()

        # Spawn at env origins, small lift, zero velocities
        if getattr(self.robot, "root_physx_view", None) is not None:
            root = self.robot.data.default_root_state[env_ids]
            root[:, :3] = self.scene.env_origins[env_ids]
            root[:, 2] += float(self.cfg.spawn_height)
            root[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0],
                                        device=self.device).repeat(len(env_ids), 1)
            root[:, 7:13] = 0.0
            self.robot.write_root_state_to_sim(root, env_ids)


#python ./scripts/skrl/train.py --task=Template-Teko-Direct-v0