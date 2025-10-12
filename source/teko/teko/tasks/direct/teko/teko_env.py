# /workspace/teko/source/teko/teko/tasks/direct/teko/teko_env.py
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

# USD stage access
from omni.usd import get_context
from pxr import Sdf

from .teko_env_cfg import TekoEnvCfg

# Arena parameters (per environment)
ARENA_SIDE = 8.0   # meters
WALL_THK   = 0.05  # meters
WALL_HGT   = 1.2   # meters


class TekoEnv(DirectRLEnv):
    """Tiny direct RL env for a 4-wheel TEKO robot (ground + robot + 8x8 m arena)."""

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

        # Will be resolved once PhysX views exist
        self.dof_idx: torch.Tensor | None = None
        self.wheel_signs: torch.Tensor | None = None  # (4,)

    # ----------------------- USD helpers -----------------------

    def _stage(self):
        return get_context().get_stage()

    def _prim_exists(self, path: str) -> bool:
        return self._stage().GetPrimAtPath(path).IsValid()

    def _remove_prim_if_exists(self, path: str):
        st = self._stage()
        if st.GetPrimAtPath(path).IsValid():
            st.RemovePrim(Sdf.Path(path))

    def _spawn_cuboid_unique(self, prim_path, cfg, translation, orientation):
        """Spawn a cuboid only if the prim doesn't already exist."""
        if self._prim_exists(prim_path):
            return
        sim_utils.spawn_cuboid(
            prim_path=prim_path, cfg=cfg, translation=translation, orientation=orientation
        )

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

        # Build an 8x8 m arena (four kinematic walls) around each env origin (in local coords)
        for i in range(self.cfg.scene.num_envs):
            self._spawn_arena_root(i)
            self._spawn_arena_walls(i)

    # ----------------------- arena builder -----------------------

    def _spawn_arena_root(self, env_index: int):
        """Ensure a clean '/arena' Xform exists under each env."""
        base = f"/World/envs/env_{env_index}/arena"
        stage = self._stage()
        # remove any existing arena subtree to avoid duplicate-prim errors
        if stage.GetPrimAtPath(base).IsValid():
            stage.RemovePrim(Sdf.Path(base))
        # define a fresh Xform prim
        stage.DefinePrim(Sdf.Path(base), "Xform")

    def _spawn_arena_walls(self, env_index: int):
        """Create 4 kinematic walls forming an 8x8 m box centered at env origin."""
        half = ARENA_SIDE * 0.5
        zc = WALL_HGT * 0.5
        base = f"/World/envs/env_{env_index}/arena"

        wall_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.28, 0.28, 0.28),
            roughness=1.0,
            metallic=0.0,
            opacity=1.0,
        )

        # N/S walls (long along X, thin along Y)
        wall_NS = sim_utils.CuboidCfg(
            size=(ARENA_SIDE, WALL_THK, WALL_HGT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=wall_mat,
        )
        # E/W walls (long along Y, thin along X)
        wall_EW = sim_utils.CuboidCfg(
            size=(WALL_THK, ARENA_SIDE, WALL_HGT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=wall_mat,
        )

        # All translations are RELATIVE to env_i frame (0,0,0 = env center)
        self._spawn_cuboid_unique(
            f"{base}/wall_n", wall_NS, translation=(0.0, +half, zc), orientation=(0, 0, 0, 1)
        )
        self._spawn_cuboid_unique(
            f"{base}/wall_s", wall_NS, translation=(0.0, -half, zc), orientation=(0, 0, 0, 1)
        )
        self._spawn_cuboid_unique(
            f"{base}/wall_e", wall_EW, translation=(+half, 0.0, zc), orientation=(0, 0, 0, 1)
        )
        self._spawn_cuboid_unique(
            f"{base}/wall_w", wall_EW, translation=(-half, 0.0, zc), orientation=(0, 0, 0, 1)
        )

    # ----------------------- internals -----------------------

    def _lazy_init_articulation(self):
        if self.dof_idx is not None and self.wheel_signs is not None:
            return
        if getattr(self.robot, "root_physx_view", None) is None:
            return

        # Joint names -> indices
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

        # Force identical spin sign for all wheels to avoid unintended yaw when L=R
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
            raw = float(self.cfg.action_scale) * self.actions  # [FL, FR, RL, RR]
            targets = raw * self.wheel_signs.unsqueeze(0) * pol
        else:
            scale = float(self.cfg.action_scale)
            left = (scale * self.actions[:, 0]).unsqueeze(-1)
            right = (scale * self.actions[:, 1]).unsqueeze(-1)
            raw = torch.hstack([left, right, left, right])  # [FL, FR, RL, RR]
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
