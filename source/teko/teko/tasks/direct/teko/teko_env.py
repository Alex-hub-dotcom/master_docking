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

from .teko_env_cfg import TekoEnvCfg

ARENA_SIDE = 8.0
WALL_THK   = 0.05
WALL_HGT   = 1.2

class TekoEnv(DirectRLEnv):
    """Ground + robot + 8x8 arena + (USD) camera marker."""

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        device = self.device
        nenv = self.cfg.scene.num_envs
        a_dim = 4 if self.cfg.independent_wheels else 2
        self.actions = torch.zeros((nenv, a_dim), device=device)
        self._targets_prev = torch.zeros((nenv, 4), device=device)
        self._max_wheel_speed = float(self.cfg.max_wheel_speed)
        self._max_wheel_accel = float(self.cfg.max_wheel_accel)
        self._env_dt = float(self.cfg.decimation) * float(self.cfg.sim.dt)
        self.dof_idx: torch.Tensor | None = None
        self.wheel_signs: torch.Tensor | None = None

    # ---------- USD helpers
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
        sim_utils.spawn_cuboid(prim_path=prim_path, cfg=cfg,
                               translation=translation, orientation=orientation)

    # ---------- scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0
                )
            ),
        )

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        for i in range(self.cfg.scene.num_envs):
            self._spawn_arena_root(i)
            self._spawn_arena_walls(i)
            self._make_rpi_v2_camera(
                env_index=i, xform_name="teko_camera", cam_name="cam_rpi_v2", resolution=(1280, 960)
            )

    def _spawn_arena_root(self, env_index: int):
        base = f"/World/envs/env_{env_index}/arena"
        stage = self._stage()
        if stage.GetPrimAtPath(base).IsValid():
            stage.RemovePrim(Sdf.Path(base))
        stage.DefinePrim(Sdf.Path(base), "Xform")

    def _spawn_arena_walls(self, env_index: int):
        half = ARENA_SIDE * 0.5
        zc = WALL_HGT * 0.5
        base = f"/World/envs/env_{env_index}/arena"

        wall_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.28, 0.28, 0.28), roughness=1.0, metallic=0.0, opacity=1.0
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

    # ---------- camera USD marker
    def _set_resolution_meta_int2(self, prim, res_xy: tuple[int, int]):
        """
        Ensure prim has 'user:resolution' as Int2. If an attribute with other type
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
        # USD imports locais
        from pxr import Usd, UsdGeom, Gf, Sdf

        stage = self._stage()

        # 1) Localiza o robô deste env
        robot_root = f"/World/envs/env_{env_index}/Robot"
        robot_prim = stage.GetPrimAtPath(robot_root)
        if not robot_prim.IsValid():
            print(f"[cam] robot prim not found at {robot_root}")
            return

        # 2) Acha o Xform alvo (teko_camera) dentro do robô
        xform_path = None
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() == xform_name:
                xform_path = prim.GetPath().pathString
                break
        if xform_path is None:
            print(f"[cam] '{xform_name}' not found under {robot_root}")
            return

        cam_path = f"{xform_path}/{cam_name}"

        # 3) Define o prim tipo Camera e cria o schema corretamente (NÃO transformar em bool)
        cam_prim = stage.DefinePrim(Sdf.Path(cam_path), "Camera")
        cam = UsdGeom.Camera(cam_prim)  # <- UsdGeom.Camera

        # 4) Define o transform local: olhar para a traseira (-X do robô)
        xf = UsdGeom.Xformable(cam_prim)
        for op in list(xf.GetOrderedXformOps()):
            xf.RemoveXformOp(op)
        # Rotaciona a câmera: no USD ela olha ao longo do -Z; Ry=+90° alinha -Z com -X do robô.
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 180.0, 0.0))
        # Empurra ~3 cm para fora no -X do robô pra não ficar dentro da carcaça
        xf.AddTranslateOp().Set(Gf.Vec3d(-0.03, 0.0, 0.0))

        # 5) Intrínsecas IMX219 (mm)
        cam.GetFocalLengthAttr().Set(3.04)
        cam.GetHorizontalApertureAttr().Set(3.68)
        cam.GetVerticalApertureAttr().Set(2.76)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))  # near=5 cm
        cam.GetFocusDistanceAttr().Set(1.0)
        cam.GetFStopAttr().Set(2.0)

        # 6) Meta 'user:resolution' com tipo correto (Int2)
        bad_attr = cam_prim.GetAttribute("user:resolution")
        if bad_attr and bad_attr.GetTypeName() != Sdf.ValueTypeNames.Int2:
            cam_prim.RemoveProperty("user:resolution")
        res_attr = cam_prim.GetAttribute("user:resolution")
        if not res_attr or res_attr.GetTypeName() != Sdf.ValueTypeNames.Int2:
            res_attr = cam_prim.CreateAttribute("user:resolution", Sdf.ValueTypeNames.Int2, False)
        res_attr.Set(Gf.Vec2i(int(resolution[0]), int(resolution[1])))

        print(
            f"[cam] OK {cam_path}  res={int(resolution[0])}x{int(resolution[1])}  "
            f"primType={cam_prim.GetTypeName()}  rot=(0,90,0)  off=(-0.03,0,0)"
        )

    # ---------- internals / RL hooks
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
            root[:, :3] = self.scene.env_origins[env_ids]
            root[:, 2] += float(self.cfg.spawn_height)
            root[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0],
                                        device=self.device).repeat(len(env_ids), 1)
            root[:, 7:13] = 0.0
            self.robot.write_root_state_to_sim(root, env_ids)
