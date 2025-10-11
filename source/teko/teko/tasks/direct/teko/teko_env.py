# Copyright (c) 2022-2025
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence

# Isaac Lab / Omniverse
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # kept if your robot USD uses it
import isaaclab.utils.math as math_utils

# Omniverse USD stage access (no omni.isaac.core dependency)
from omni.usd import get_context
from pxr import Sdf

from .teko_env_cfg import TekoEnvCfg


# === ARENA CONFIG (per isolated environment) ===
ARENA_SIDE = 6.0     # square side in meters (exact)
WALL_THK   = 0.05    # wall thickness (m)
WALL_HGT   = 1.5     # wall height (m)
SPACING_MIN_MARGIN = 2.0  # extra gap between neighboring arenas (m)


class TekoEnv(DirectRLEnv):
    """Direct RL env for a 4-wheel TEKO robot with isolated 6x6 arenas (no markers)."""

    cfg: TekoEnvCfg

    # -------------- base setup --------------

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        device = self.device
        num_envs = self.cfg.scene.num_envs

        # Action buffer (4 when independent wheels, else 2 for left/right)
        a_dim = 4 if self.cfg.independent_wheels else 2
        self.actions = torch.zeros((num_envs, a_dim), device=device)

        # Smoothing buffers and limits
        self._targets_prev = torch.zeros((num_envs, 4), device=device)
        self._max_wheel_speed = float(getattr(self.cfg, "max_wheel_speed", 12.0))
        self._max_wheel_accel = float(getattr(self.cfg, "max_wheel_accel", 30.0))
        self._env_dt = float(self.cfg.decimation) * float(self.cfg.sim.dt)

        # Heading helpers (keep if your reward/obs uses them)
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
        self.commands = self._sample_planar_unit_vectors(num_envs, device=device)
        self.yaws = torch.zeros((num_envs, 1), device=device)
        self._update_command_yaws()

        # Will be filled lazily after PhysX views exist
        self.dof_idx = None
        self.wheel_signs = None  # (4,)

    # -------------- USD stage helpers (idempotent spawns) --------------

    def _get_stage(self):
        return get_context().get_stage()

    def _prim_exists(self, path: str) -> bool:
        stage = self._get_stage()
        return stage.GetPrimAtPath(path).IsValid()

    def _remove_prim_if_exists(self, path: str):
        stage = self._get_stage()
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(Sdf.Path(path))

    def _spawn_cuboid_unique(self, prim_path: str, cfg, translation, orientation):
        """Spawn a cuboid only if a prim does not already exist at prim_path."""
        if self._prim_exists(prim_path):
            return
        sim_utils.spawn_cuboid(
            prim_path=prim_path,
            cfg=cfg,
            translation=translation,
            orientation=orientation,
        )

    # -------------- arena helpers (walls) --------------

    def _reset_arena_group(self, env_index: int):
        """Delete the arena group to ensure a clean re-spawn (useful on relaunch)."""
        self._remove_prim_if_exists(f"/World/envs/env_{env_index}/arena")

    def _spawn_arena_walls(self, env_index: int, env_origin):
        """Create 4 opaque walls (6x6 m, 1.5 m tall) around the environment origin."""
        half = ARENA_SIDE / 2.0
        zc = env_origin[2] + WALL_HGT / 2.0

        # Consistent, non-reflective visual material
        wall_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.28, 0.28, 0.28), roughness=1.0, metallic=0.0, opacity=1.0
        )

        # N/S walls (aligned along Y)
        wall_NS = sim_utils.CuboidCfg(
            size=(ARENA_SIDE, WALL_THK, WALL_HGT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=wall_mat,
        )
        # E/W walls (aligned along X)
        wall_EW = sim_utils.CuboidCfg(
            size=(WALL_THK, ARENA_SIDE, WALL_HGT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=wall_mat,
        )

        # North (Y+)
        self._spawn_cuboid_unique(
            f"/World/envs/env_{env_index}/arena/wall_n",
            wall_NS,
            translation=(env_origin[0], env_origin[1] + half, zc),
            orientation=(0, 0, 0, 1),
        )
        # South (Y-)
        self._spawn_cuboid_unique(
            f"/World/envs/env_{env_index}/arena/wall_s",
            wall_NS,
            translation=(env_origin[0], env_origin[1] - half, zc),
            orientation=(0, 0, 0, 1),
        )
        # East (X+)
        self._spawn_cuboid_unique(
            f"/World/envs/env_{env_index}/arena/wall_e",
            wall_EW,
            translation=(env_origin[0] + half, env_origin[1], zc),
            orientation=(0, 0, 0, 1),
        )
        # West (X-)
        self._spawn_cuboid_unique(
            f"/World/envs/env_{env_index}/arena/wall_w",
            wall_EW,
            translation=(env_origin[0] - half, env_origin[1], zc),
            orientation=(0, 0, 0, 1),
        )

    # -------------- scene setup --------------

    def _setup_scene(self):
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground: high friction, zero restitution
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.5,
                    dynamic_friction=1.2,
                    restitution=0.0,
                )
            ),
        )

        # Clone envs and register robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        # Global lighting (uniform across all arenas)
        dome_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        dome_cfg.func("/World/lighting/dome", dome_cfg)

        # Optional: add a soft "sun" for consistent shadows (ignore if not available)
        try:
            sun_cfg = sim_utils.DistantLightCfg(intensity=3000.0, angle=0.5, color=(1.0, 1.0, 1.0))
            sun_cfg.func("/World/lighting/sun", sun_cfg, translation=(0, 0, 10), orientation=(0, 0, 0, 1))
        except AttributeError:
            pass

        # Check spacing to avoid visual overlaps (keeps all arenas same size)
        required_spacing = ARENA_SIDE + SPACING_MIN_MARGIN
        if getattr(self.cfg.scene, "env_spacing", required_spacing) < required_spacing:
            print(f"[warn] scene.env_spacing={self.cfg.scene.env_spacing} is small; "
                  f"set >= {required_spacing:.1f} to keep arenas cleanly separated.")

        # Spawn identical arenas (walls) for each environment
        env_origins = self.scene.env_origins  # (num_envs, 3)
        for i in range(self.cfg.scene.num_envs):
            self._reset_arena_group(i)  # clean group to avoid "prim already exists"
            self._spawn_arena_walls(env_index=i, env_origin=env_origins[i])

    # -------------- helpers --------------

    def _sample_planar_unit_vectors(self, n: int, device) -> torch.Tensor:
        """Sample random unit vectors in the XY plane."""
        v = torch.randn((n, 3), device=device)
        v[:, -1] = 0.0
        norm = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
        return v / norm

    def _update_command_yaws(self):
        """Compute yaw angles from command unit vectors."""
        self.yaws = torch.atan2(self.commands[:, 1], self.commands[:, 0]).reshape(-1, 1)

    def _build_wheel_signs(self):
        """Infer per-wheel spin signs from joint axes; fallback to [+1,-1,+1,-1]."""
        signs = torch.ones((len(self.dof_idx),), device=self.device, dtype=torch.float32)
        try:
            axes = self.robot.data.joint_axis[:, :3]
            raw = torch.sign(axes[self.dof_idx, 0]).to(torch.float32)
            raw[raw == 0] = 1.0
            raw[2] = raw[0]  # RL matches FL
            raw[3] = raw[1]  # RR matches FR
            signs = raw
        except Exception:
            signs = torch.tensor([+1.0, -1.0, +1.0, -1.0], device=self.device)
        self.wheel_signs = signs

    def _lazy_init_articulation(self):
        """Resolve joints and optionally set drive gains once PhysX views exist."""
        if self.dof_idx is not None and self.wheel_signs is not None:
            return
        if getattr(self.robot, "root_physx_view", None) is None:
            return

        # Name → index
        joint_names = list(self.robot.joint_names)
        name_to_idx = {n: i for i, n in enumerate(joint_names)}

        # Check mapping from config
        missing = [n for n in self.cfg.dof_names if n not in name_to_idx]
        if missing:
            raise RuntimeError(
                "Joint name(s) not found in TEKO asset.\n"
                f"Missing: {missing}\n"
                f"Available joints: {joint_names}"
            )

        dof_idx_list = [name_to_idx[n] for n in self.cfg.dof_names]
        self.dof_idx = torch.tensor(dof_idx_list, dtype=torch.long, device=self.device)

        # Build wheel signs
        self._build_wheel_signs()

        # Velocity-like drive (if supported in this build)
        n = len(self.dof_idx)
        stiffness = torch.zeros(n, device=self.device)
        damping = torch.full((n,), float(self.cfg.drive_damping), device=self.device)
        max_force = torch.full((n,), float(self.cfg.drive_max_force), device=self.device)
        if hasattr(self.robot, "set_joint_drive_property"):
            self.robot.set_joint_drive_property(
                stiffness=stiffness, damping=damping, max_force=max_force, joint_ids=self.dof_idx
            )

        print(f"[dbg] dof order [FL, FR, RL, RR] -> {self.cfg.dof_names}")
        print(f"[dbg] dof indices -> {self.dof_idx.tolist()}")
        print(f"[dbg] wheel_signs -> {self.wheel_signs.tolist()}")

    # -------------- RL API --------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Receive/shape actions and ensure articulation is initialized."""
        if actions is None or actions.numel() == 0:
            a_dim = 4 if self.cfg.independent_wheels else 2
            actions = 0.3 * torch.ones((self.cfg.scene.num_envs, a_dim), device=self.device)
        self.actions = actions.clone()

        self._lazy_init_articulation()

    def _apply_action(self) -> None:
        """Map actions to wheel angular velocity targets with slew-rate limiting."""
        if self.dof_idx is None:
            return

        if self.cfg.independent_wheels:
            # 4-dim actions -> FL, FR, RL, RR (later signed)
            assert self.actions.shape[1] == 4
            raw = float(self.cfg.action_scale) * self.actions  # (N,4)
            targets = raw * self.wheel_signs.unsqueeze(0) * float(self.cfg.wheel_polarity)
        else:
            # 2-dim actions -> left/right -> reorder into [R, L, R, L]
            assert self.actions.shape[1] == 2
            scale = float(self.cfg.action_scale)
            pol = float(self.cfg.wheel_polarity)
            left = (scale * self.actions[:, 0]).unsqueeze(-1)   # (N,1)
            right = (scale * self.actions[:, 1]).unsqueeze(-1)  # (N,1)
            raw = torch.hstack([right, left, right, left])      # (N,4)
            targets = raw * self.wheel_signs.unsqueeze(0) * pol

        # Slew-rate limit and saturation
        max_delta = self._max_wheel_accel * self._env_dt
        delta = torch.clamp(targets - self._targets_prev, -max_delta, +max_delta)
        targets = self._targets_prev + delta
        targets = torch.clamp(targets, -self._max_wheel_speed, +self._max_wheel_speed)
        self._targets_prev = targets.detach()

        # Send to sim
        self.robot.set_joint_velocity_target(targets, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        """Cosine alignment, signed cross-z, and forward speed."""
        forwards = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B)
        dot = torch.sum(forwards * self.commands, dim=-1, keepdim=True)
        cross_z = torch.cross(forwards, self.commands, dim=-1)[:, -1].reshape(-1, 1)
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].reshape(-1, 1)
        obs = torch.hstack((dot, cross_z, forward_speed))
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Reward forward progress and heading alignment."""
        forwards = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B)
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].reshape(-1, 1)
        alignment = torch.sum(forwards * self.commands, dim=-1, keepdim=True)
        return forward_speed + alignment

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Finite horizon termination (no failure termination here)."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset a subset of envs; resample commands; reset root state at arena center."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        self._lazy_init_articulation()

        # Resample commands if needed (kept for your reward/obs)
        if (self.commands is None) or (self.commands.shape[0] != self.cfg.scene.num_envs):
            self.commands = self._sample_planar_unit_vectors(self.cfg.scene.num_envs, device=self.device)
        self.commands[env_ids] = self._sample_planar_unit_vectors(len(env_ids), device=self.device)
        self._update_command_yaws()

        # Place robot at the exact center of each arena (env_origin), with spawn height
        if getattr(self.robot, "root_physx_view", None) is not None:
            default_root_state = self.robot.data.default_root_state[env_ids]
            default_root_state[:, :3] = self.scene.env_origins[env_ids]  # XY at arena center
            default_root_state[:, 2] += float(self.cfg.spawn_height)     # Z spawn height
            default_root_state[:, 7:10] = 0.0
            default_root_state[:, 10:13] = 0.0
            self.robot.write_root_state_to_sim(default_root_state, env_ids)


# --- HOW TO LAUNCH (examples) ---
# Simple (defaults):
# python ./scripts/skrl/train.py --task=Template-Teko-Direct-v0
#
# Recommended spacing for 6x6 arenas and UI:
# python ./scripts/skrl/train.py --task=Template-Teko-Direct-v0 scene.env_spacing=8.0 headless=false
#
# Full trace on errors:
# HYDRA_FULL_ERROR=1 python ./scripts/skrl/train.py --task=Template-Teko-Direct-v0


#python ./scripts/skrl/train.py --task=Template-Teko-Direct-v0