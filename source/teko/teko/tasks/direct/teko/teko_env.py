# Copyright (c) 2022-2025
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from collections.abc import Sequence
import math
import torch

# Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# USD / Omniverse
from omni.usd import get_context
from pxr import Sdf, Usd, UsdGeom, Gf

# Project cfg
from .teko_env_cfg import TekoEnvCfg


# ----- arena constants (visual walls only) -----
ARENA_SIDE = 8.0       # 8x8 m square
WALL_THK   = 0.05
WALL_HGT   = 1.2


class TekoEnv(DirectRLEnv):
    """
    Simple world: two robots per env + arena walls + brown visual floor.
    Injects a USD Camera under Robot's "teko_camera" Xform, rotated 180° (rear view).
    """

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        device = self.device
        nenv = self.cfg.scene.num_envs
        a_dim = 4 if self.cfg.independent_wheels else 2

        # actions / control buffers
        self.actions = torch.zeros((nenv, a_dim), device=device)
        self._targets_prev = torch.zeros((nenv, 4), device=device)     # robot A smoothing
        self._targets_prev_b = torch.zeros((nenv, 4), device=device)   # robot B smoothing
        self._max_wheel_speed = float(self.cfg.max_wheel_speed)
        self._max_wheel_accel = float(self.cfg.max_wheel_accel)
        self._env_dt = float(self.cfg.decimation) * float(self.cfg.sim.dt)

        # articulation handles
        self.dof_idx: torch.Tensor | None = None
        self.wheel_signs: torch.Tensor | None = None
        self.dof_idx_b: torch.Tensor | None = None
        self.wheel_signs_b: torch.Tensor | None = None

    # USD helpers
    def _stage(self):
        return get_context().get_stage()

    def _prim_exists(self, path: str) -> bool:
        return self._stage().GetPrimAtPath(path).IsValid()

    def _remove_prim_if_exists(self, path: str):
        st = self._stage()
        if st.GetPrimAtPath(path).IsValid():
            st.RemovePrim(Sdf.Path(path))

    def _spawn_cuboid_unique(self, prim_path, cfg, translation, orientation):
        if self._prim_exists(prim_path):
            return
        sim_utils.spawn_cuboid(
            prim_path=prim_path,
            cfg=cfg,
            translation=translation,
            orientation=orientation,
        )

    # Scene building
    def _setup_scene(self):
        """Build the simulation scene: two robots per environment, ground, arena, and camera."""
        # 1) Robots
        self.robot = Articulation(self.cfg.robot_cfg)
        self.robot_passive = Articulation(
            self.cfg.robot_cfg.replace(prim_path="/World/envs/env_.*/PassiveRobot")
        )

        # 2) Physical ground (infinite flat surface with friction)
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

        # 3) Register and clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["robot_passive"] = self.robot_passive

        # 4) Per-environment content
        for i in range(self.cfg.scene.num_envs):
            self._spawn_arena_root(i)
            self._spawn_arena_walls(i)
            self._spawn_visual_floor_for_env(i)  # one brown floor per env
            self._make_rpi_v2_camera(
                env_index=i,
                xform_name="teko_camera",
                cam_name="cam_rpi_v2",
                resolution=(1280, 960),
            )

    # arena root
    def _spawn_arena_root(self, env_index: int):
        base = f"/World/envs/env_{env_index}/arena"
        stage = self._stage()
        if stage.GetPrimAtPath(base).IsValid():
            stage.RemovePrim(Sdf.Path(base))
        stage.DefinePrim(Sdf.Path(base), "Xform")

    # arena walls (visual, kinematic)
    def _spawn_arena_walls(self, env_index: int):
        half = ARENA_SIDE * 0.5
        zc = WALL_HGT * 0.5
        base = f"/World/envs/env_{env_index}/arena"

        wall_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.28, 0.28, 0.28),
            roughness=1.0,
            metallic=0.0,
            opacity=1.0,
        )
        wall_NS = sim_utils.CuboidCfg(
            size=(ARENA_SIDE, WALL_THK, WALL_HGT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=wall_mat,
        )
        wall_EW = sim_utils.CuboidCfg(
            size=(WALL_THK, ARENA_SIDE, WALL_HGT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=wall_mat,
        )

        self._spawn_cuboid_unique(f"{base}/wall_n", wall_NS, (0.0, +half, zc), (0, 0, 0, 1))
        self._spawn_cuboid_unique(f"{base}/wall_s", wall_NS, (0.0, -half, zc), (0, 0, 0, 1))
        self._spawn_cuboid_unique(f"{base}/wall_e", wall_EW, (+half, 0.0, zc), (0, 0, 0, 1))
        self._spawn_cuboid_unique(f"{base}/wall_w", wall_EW, (-half, 0.0, zc), (0, 0, 0, 1))

    # per-env visual floor (brown, no collision)
    def _spawn_visual_floor_for_env(self, env_index: int):
        base = f"/World/envs/env_{env_index}"
        floor_path = f"{base}/visual_floor"
        if self._prim_exists(floor_path):
            return

        brown = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.32, 0.22, 0.15),  # earthy brown
            roughness=1.0,
            metallic=0.0,
            opacity=1.0,
        )

        # Make it a bit larger than the arena to avoid seeing edges.
        plate_cfg = sim_utils.CuboidCfg(
            size=(ARENA_SIDE + 2.0, ARENA_SIDE + 2.0, 0.01),
            visual_material=brown,
            rigid_props=None,          # visual-only
            collision_props=None,      # no collisions
        )

        # Place slightly above z=0 to avoid z-fighting with the physical ground.
        sim_utils.spawn_cuboid(
            prim_path=floor_path,
            cfg=plate_cfg,
            translation=(0.0, 0.0, 0.005),
            orientation=(0, 0, 0, 1),
        )

    # camera injection (rear view, 180° yaw, slight -X offset)
    def _set_resolution_meta_int2(self, prim, res_xy: tuple[int, int]):
        """
        Ensure prim has 'user:resolution' as Int2. If an attribute with another type
        exists (e.g., Vec2d), remove and recreate as Int2.
        """
        name = "user:resolution"
        attr = prim.GetAttribute(name)
        if attr and attr.GetTypeName() != Sdf.ValueTypeNames.Int2:
            prim.RemoveProperty(name)
            attr = None
        if not attr:
            attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Int2, False)
        attr.Set(Gf.Vec2i(int(res_xy[0]), int(res_xy[1])))

    def _make_rpi_v2_camera(
        self,
        env_index: int,
        xform_name: str = "teko_camera",
        cam_name: str = "cam_rpi_v2",
        resolution=(1280, 960),
    ):
        stage = self._stage()

        # 1) find robot in this env
        robot_root = f"/World/envs/env_{env_index}/Robot"
        robot_prim = stage.GetPrimAtPath(robot_root)
        if not robot_prim.IsValid():
            print(f"[cam] robot prim not found at {robot_root}")
            return

        # 2) find the Xform named xform_name under the robot
        xform_path = None
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() == xform_name:
                xform_path = prim.GetPath().pathString
                break
        if xform_path is None:
            print(f"[cam] '{xform_name}' not found under {robot_root}")
            return

        cam_path = f"{xform_path}/{cam_name}"

        # 3) define camera prim and schema
        cam_prim = stage.DefinePrim(Sdf.Path(cam_path), "Camera")
        cam = UsdGeom.Camera(cam_prim)

        # 4) transform
        xapi = UsdGeom.XformCommonAPI(cam_prim)
        xapi.SetRotate(Gf.Vec3f(0.0, 180.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)  # yaw 180°
        xapi.SetTranslate(Gf.Vec3d(-0.06, 0.0, 0.0))  # 6 cm outward along -X

        # 5) intrinsics (IMX219-like)
        cam.GetFocalLengthAttr().Set(3.04)
        cam.GetHorizontalApertureAttr().Set(3.68)
        cam.GetVerticalApertureAttr().Set(2.76)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
        cam.GetFocusDistanceAttr().Set(1.0)
        cam.GetFStopAttr().Set(2.0)

        # 6) resolution meta as Int2
        self._set_resolution_meta_int2(cam_prim, resolution)

        print(
            f"[cam] OK {cam_path}  res={int(resolution[0])}x{int(resolution[1])}  "
            f"primType={cam_prim.GetTypeName()}  rot=(0,180,0)  off=(-0.06,0,0)"
        )

    # RL plumbing (init drives for BOTH robots)
    def _lazy_init_articulation(self):
        # Robot A
        if self.dof_idx is None and getattr(self.robot, "root_physx_view", None) is not None:
            joint_names = list(self.robot.joint_names)
            name_to_idx = {n: i for i, n in enumerate(joint_names)}
            missing = [n for n in self.cfg.dof_names if n not in name_to_idx]
            if missing:
                raise RuntimeError(f"Missing joints {missing}. Available: {joint_names}")
            dof_idx_list = [name_to_idx[n] for n in self.cfg.dof_names]
            self.dof_idx = torch.tensor(dof_idx_list, dtype=torch.long, device=self.device)
            self.wheel_signs = torch.ones(4, device=self.device, dtype=torch.float32)

            n = len(self.dof_idx)
            stiffness = torch.zeros(n, device=self.device)
            damping = torch.full((n,), float(self.cfg.drive_damping), device=self.device)
            max_force = torch.full((n,), float(self.cfg.drive_max_force), device=self.device)
            if hasattr(self.robot, "set_joint_drive_property"):
                self.robot.set_joint_drive_property(
                    stiffness=stiffness, damping=damping, max_force=max_force, joint_ids=self.dof_idx
                )

        # Robot B (same drive setup)
        if self.dof_idx_b is None and getattr(self.robot_passive, "root_physx_view", None) is not None:
            joint_names_b = list(self.robot_passive.joint_names)
            name_to_idx_b = {n: i for i, n in enumerate(joint_names_b)}
            missing_b = [n for n in self.cfg.dof_names if n not in name_to_idx_b]
            if missing_b:
                raise RuntimeError(f"Missing joints on second robot {missing_b}. Available: {joint_names_b}")
            dof_idx_list_b = [name_to_idx_b[n] for n in self.cfg.dof_names]
            self.dof_idx_b = torch.tensor(dof_idx_list_b, dtype=torch.long, device=self.device)
            self.wheel_signs_b = torch.ones(4, device=self.device, dtype=torch.float32)

            n = len(self.dof_idx_b)
            stiffness = torch.zeros(n, device=self.device)
            damping = torch.full((n,), float(self.cfg.drive_damping), device=self.device)
            max_force = torch.full((n,), float(self.cfg.drive_max_force), device=self.device)
            if hasattr(self.robot_passive, "set_joint_drive_property"):
                self.robot_passive.set_joint_drive_property(
                    stiffness=stiffness, damping=damping, max_force=max_force, joint_ids=self.dof_idx_b
                )

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
            raw = float(self.cfg.action_scale) * self.actions
            targets = raw * self.wheel_signs.unsqueeze(0) * pol
        else:
            scale = float(self.cfg.action_scale)
            left = (scale * self.actions[:, 0]).unsqueeze(-1)
            right = (scale * self.actions[:, 1]).unsqueeze(-1)
            raw = torch.hstack([left, right, left, right])
            targets = raw * self.wheel_signs.unsqueeze(0) * pol

        # Smooth & clamp for Robot A
        max_delta = self._max_wheel_accel * self._env_dt
        delta = torch.clamp(targets - self._targets_prev, -max_delta, +max_delta)
        targets_a = self._targets_prev + delta
        targets_a = torch.clamp(targets_a, -self._max_wheel_speed, +self._max_wheel_speed)
        self._targets_prev = targets_a.detach()
        self.robot.set_joint_velocity_target(targets_a, joint_ids=self.dof_idx)

        # Same commands for Robot B (with its own smoothing buffer)
        if self.dof_idx_b is not None:
            delta_b = torch.clamp(targets - self._targets_prev_b, -max_delta, +max_delta)
            targets_b = self._targets_prev_b + delta_b
            targets_b = torch.clamp(targets_b, -self._max_wheel_speed, +self._max_wheel_speed)
            self._targets_prev_b = targets_b.detach()
            self.robot_passive.set_joint_velocity_target(targets_b, joint_ids=self.dof_idx_b)

    def _get_observations(self) -> dict:
        v_fwd = self.robot.data.root_com_lin_vel_b[:, 0].reshape(-1, 1)
        return {"policy": v_fwd}

    def _get_rewards(self) -> torch.Tensor:
        return self._get_observations()["policy"]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._lazy_init_articulation()

        # Force env_ids to a tensor of indices (avoids chained boolean indexing issues)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        origins_all = self.scene.env_origins[env_ids_t]

        # Spawn params: keep inside arena with wall margin and min separation
        half = ARENA_SIDE * 0.5
        margin = 0.7      # distance from walls
        min_sep = 1.2     # minimum A-B XY separation
        x_min = -half + margin
        x_max = +half - margin
        y_min = -half + margin
        y_max = +half - margin

        def _sample_xy(n: int):
            xs = x_min + (x_max - x_min) * torch.rand((n, 1), device=self.device)
            ys = y_min + (y_max - y_min) * torch.rand((n, 1), device=self.device)
            return torch.hstack([xs, ys])

        # ===== Robot A =====
        if getattr(self.robot, "root_physx_view", None) is not None:
            root_a = self.robot.data.default_root_state[env_ids_t]

            xy_a = _sample_xy(len(env_ids_t))
            z = torch.full((len(env_ids_t), 1), float(self.cfg.spawn_height), device=self.device)
            pos_a_local = torch.cat([xy_a, z], dim=-1)
            pos_a_world = origins_all + pos_a_local
            root_a[:, :3] = pos_a_world

            # Random yaw (keeping Fusion-import X-axis convention)
            yaw_a = 2 * math.pi * torch.rand(len(env_ids_t), device=self.device)
            quat_a = torch.stack([
                torch.sin(yaw_a / 2),            # x
                torch.zeros_like(yaw_a),         # y
                torch.zeros_like(yaw_a),         # z
                torch.cos(yaw_a / 2),            # w
            ], dim=1)
            root_a[:, 3:7] = quat_a

            root_a[:, 7:13] = 0.0
            self.robot.write_root_state_to_sim(root_a, env_ids_t)

        # ===== Robot B (rejection sampling with integer indices) =====
        if getattr(self.robot_passive, "root_physx_view", None) is not None:
            root_b = self.robot_passive.data.default_root_state[env_ids_t]

            remaining = env_ids_t.clone()                      # env indices still needing a pose
            pos_b_world = torch.empty((len(env_ids_t), 3), device=self.device)
            attempts = 0
            max_attempts = 10

            while remaining.numel() > 0 and attempts < max_attempts:
                origins_rem = self.scene.env_origins[remaining]
                xy_b = _sample_xy(remaining.numel())
                z = torch.full((remaining.numel(), 1), float(self.cfg.spawn_height), device=self.device)
                pos_b_local = torch.cat([xy_b, z], dim=-1)
                cand_world = origins_rem + pos_b_local

                # distance to A in the same envs
                pos_a_world_now = self.robot.data.root_state_w[remaining, :3]
                dist = torch.linalg.norm(cand_world[:, :2] - pos_a_world_now[:, :2], dim=1)
                ok_mask = dist >= min_sep

                if ok_mask.any():
                    approved_envs = remaining[ok_mask]          # absolute env indices
                    # map approved envs back to positions in the full env_ids_t array
                    map_mask = (env_ids_t.unsqueeze(1) == approved_envs).any(dim=1)
                    pos_b_world[map_mask] = cand_world[ok_mask]

                remaining = remaining[~ok_mask]
                attempts += 1

            # Fallback for any leftovers: mirror across origin and clamp to bounds
            if remaining.numel() > 0:
                origins_rem = self.scene.env_origins[remaining]
                pos_a_world_now = self.robot.data.root_state_w[remaining, :3]
                mirrored = origins_rem - (pos_a_world_now - origins_rem)
                rel = mirrored - origins_rem
                rel[:, 0] = torch.clamp(rel[:, 0], x_min, x_max)
                rel[:, 1] = torch.clamp(rel[:, 1], y_min, y_max)
                fixed = origins_rem + rel

                map_mask = (env_ids_t.unsqueeze(1) == remaining).any(dim=1)
                pos_b_world[map_mask] = fixed
                pos_b_world[map_mask, 2] = float(self.cfg.spawn_height)

            # Independent yaw for B (same X-axis convention)
            yaw_b = 2 * math.pi * torch.rand(len(env_ids_t), device=self.device)
            quat_b = torch.stack([
                torch.sin(yaw_b / 2),            # x
                torch.zeros_like(yaw_b),         # y
                torch.zeros_like(yaw_b),         # z
                torch.cos(yaw_b / 2),            # w
            ], dim=1)

            root_b[:, :3] = pos_b_world
            root_b[:, 3:7] = quat_b
            root_b[:, 7:13] = 0.0
            self.robot_passive.write_root_state_to_sim(root_b, env_ids_t)


# Optional local smoke test (handy to sanity check)
if __name__ == "__main__":
    cfg = TekoEnvCfg()
    env = TekoEnv(cfg, render_mode="window")  # use "headless" to run without a window
    env.reset()
    print("Env OK. Closing.")
    env.close()
