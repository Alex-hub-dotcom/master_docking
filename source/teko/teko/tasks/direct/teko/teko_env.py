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

# USD / Omniverse
from omni.usd import get_context
from pxr import Sdf, Usd, UsdGeom, Gf

# Project cfg
from .teko_env_cfg import TekoEnvCfg
import math


# ----- arena constants (visual walls only) -----
ARENA_SIDE = 8.0       # 8x8 m square
WALL_THK   = 0.05
WALL_HGT   = 1.2


class TekoEnv(DirectRLEnv):
    """
    Simple world: robot + per-env arena (walls) + per-env brown visual floor.
    Also injects a USD Camera under the robot's "teko_camera" Xform, rotated 180° (rear view).
    """

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        device = self.device
        nenv = self.cfg.scene.num_envs
        a_dim = 4 if self.cfg.independent_wheels else 2

        # action / control bookkeeping
        self.actions = torch.zeros((nenv, a_dim), device=device)
        self._targets_prev = torch.zeros((nenv, 4), device=device)
        self._max_wheel_speed = float(self.cfg.max_wheel_speed)
        self._max_wheel_accel = float(self.cfg.max_wheel_accel)
        self._env_dt = float(self.cfg.decimation) * float(self.cfg.sim.dt)

        # articulation handles
        self.dof_idx: torch.Tensor | None = None
        self.wheel_signs: torch.Tensor | None = None

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

    # scene building
    def _setup_scene(self):
        # 1) robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # 2) physical ground (flat infinite, with friction)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0
                )
            ),
        )

        # 3) register and clone envs
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        # 4) per-env content
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

        # 4) transform: USD cameras look along -Z. We want rear view (-X of robot),
        # so set yaw=180° to turn -Z -> +Z, then rely on the mounting Xform for alignment.
        # Use XformCommonAPI for broad compatibility.
        xapi = UsdGeom.XformCommonAPI(cam_prim)
        xapi.SetRotate(Gf.Vec3f(0.0, 180.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)  # yaw 180°
        xapi.SetTranslate(Gf.Vec3d(-0.06, 0.0, 0.0))  # 6 cm outward along -X

        # 5) intrinsics (IMX219-like)
        cam.GetFocalLengthAttr().Set(3.04)
        cam.GetHorizontalApertureAttr().Set(3.68)
        cam.GetVerticalApertureAttr().Set(2.76)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))  # near = 5 cm
        cam.GetFocusDistanceAttr().Set(1.0)
        cam.GetFStopAttr().Set(2.0)

        # 6) resolution meta as Int2
        self._set_resolution_meta_int2(cam_prim, resolution)

        print(
            f"[cam] OK {cam_path}  res={int(resolution[0])}x{int(resolution[1])}  "
            f"primType={cam_prim.GetTypeName()}  rot=(0,180,0)  off=(-0.06,0,0)"
        )

    # RL plumbing (minimal)
    def _lazy_init_articulation(self):
        if self.dof_idx is not None and self.wheel_signs is not None:
            return
        if getattr(self.robot, "root_physx_view", None) is None:
            return

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

        max_delta = self._max_wheel_accel * self._env_dt
        delta = torch.clamp(targets - self._targets_prev, -max_delta, +max_delta)
        targets = self._targets_prev + delta
        targets = torch.clamp(targets, -self._max_wheel_speed, +self._max_wheel_speed)
        self._targets_prev = targets.detach()

        self.robot.set_joint_velocity_target(targets, joint_ids=self.dof_idx)

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
        if getattr(self.robot, "root_physx_view", None) is not None:
            root = self.robot.data.default_root_state[env_ids]

            # ===== Random position in local arena (x, y in [-3.5, +3.5], z fixed) =====
            half_range = 3.5
            xy_random = 2 * half_range * (torch.rand((len(env_ids), 2), device=self.device) - 0.5)
            z = torch.full((len(env_ids), 1), float(self.cfg.spawn_height), device=self.device)
            xyz_local = torch.cat([xy_random, z], dim=-1)
            xyz_global = self.scene.env_origins[env_ids] + xyz_local
            root[:, :3] = xyz_global

            # ===== Random yaw (rotation around Z axis only, no tilt) =====
            import math
            yaw = 2 * math.pi * torch.rand(len(env_ids), device=self.device)
            quat_y = torch.stack([
                torch.sin(yaw / 2),            # x
                torch.zeros_like(yaw),         # y
                torch.zeros_like(yaw),         # z
                torch.cos(yaw / 2),            # w
            ], dim=1)
            root[:, 3:7] = quat_y

            # ===== Zero velocities =====
            root[:, 7:13] = 0.0

            self.robot.write_root_state_to_sim(root, env_ids)







# Optional local smoke test (handy to sanity check)
if __name__ == "__main__":
    cfg = TekoEnvCfg()
    env = TekoEnv(cfg, render_mode="window")  # use "headless" to run without a window
    env.reset()
    print("Env OK. Closing.")
    env.close()
