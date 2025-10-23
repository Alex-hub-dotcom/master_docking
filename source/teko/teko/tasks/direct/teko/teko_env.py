# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Environment to visualize the TEKO robot moving inside the custom stage_arena.usd."""

from __future__ import annotations
import math
import torch
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from omni.usd import get_context
from pxr import Usd, Sdf, UsdGeom, Gf

from .teko_env_cfg import TekoEnvCfg
from .robots.teko import TEKO_CONFIGURATION
from .sensors.camera import ensure_teko_camera   # ✅ substitui Camera wrapper

class TekoEnv(DirectRLEnv):
    """Single-robot environment with random initial pose and modular sensors (camera, etc.)."""

    cfg: TekoEnvCfg

    # ------------------------------------------------------------------
    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((1, 2), device=self.device)
        self._max_wheel_speed = float(cfg.max_wheel_speed)
        self.dof_idx = None
        self.sensors = {}

    # ------------------------------------------------------------------
    def _setup_scene(self):
        """Load arena, spawn the TEKO robot and attach sensors."""
        try:
            stage = get_context().get_stage()
            if not stage:
                raise RuntimeError("Stage not initialized")

            arena_path = "/workspace/teko/documents/CAD/USD/stage_arena.usd"
            arena_prim_path = "/World/StageArena"

            stage.DefinePrim(Sdf.Path(arena_prim_path), "Xform")
            arena_prim = stage.GetPrimAtPath(arena_prim_path)
            arena_prim.GetReferences().AddReference(arena_path)

            UsdGeom.Xformable(arena_prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            print(f"✅ Arena carregada: {arena_path} → {arena_prim_path}")
        except Exception as e:
            print(f"⚠️ Erro ao carregar arena: {e}")
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # --- Spawn do robô TEKO
        self.robot = Articulation(TEKO_CONFIGURATION.replace(prim_path="/World/Robot"))
        self.scene.articulations["robot"] = self.robot

        # --- Sensores
        self._attach_sensors()

        # --- Pose inicial
        self._usd_randomize_robot_pose()

        # --- Multi-env clones
        self.scene.clone_environments(copy_from_source=False)

    # ------------------------------------------------------------------
    def _attach_sensors(self):
        """Cria e adiciona sensores (USD camera)."""
        # garante o dicionário
        if not hasattr(self, "sensors") or self.sensors is None:
            self.sensors = {}

        try:
            cam_path = ensure_teko_camera(resolution=(640, 480))
            self.sensors["front_camera"] = cam_path
            print(f"📷 Câmera TEKO garantida em {cam_path}")
        except Exception as e:
            print(f"⚠️ Falha ao criar câmera: {e}")

    # ------------------------------------------------------------------
    def _usd_randomize_robot_pose(self):
        """Define uma pose aleatória via USD (pré-física)."""
        try:
            stage = get_context().get_stage()
            robot_prim = stage.GetPrimAtPath("/World/Robot")
            if not robot_prim:
                print("⚠️ USD: prim do robô não encontrado.")
                return

            x = random.uniform(-1.4, 1.4)
            y = random.uniform(-1.9, 1.9)
            z = 0.43
            yaw_deg = random.uniform(-180.0, 180.0)

            xf = UsdGeom.Xformable(robot_prim)
            xf.ClearXformOpOrder()
            t_op = xf.AddTranslateOp()
            r_op = xf.AddRotateZOp()
            t_op.Set(Gf.Vec3d(x, y, z))
            r_op.Set(yaw_deg)

            print(f"🎯 Spawn inicial: x={x:.2f} y={y:.2f} z={z:.2f} yaw={yaw_deg:.1f}°")
        except Exception as e:
            print(f"⚠️ Falha na randomização USD: {e}")

    # ------------------------------------------------------------------
    def _lazy_init_articulation(self):
        """Inicializa índices de DOF quando a articulação estiver pronta."""
        if self.dof_idx is not None:
            return
        if getattr(self.robot, "root_physx_view", None) is None:
            return

        joint_names = list(self.robot.joint_names)
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        dof_list = [name_to_idx[n] for n in self.cfg.dof_names if n in name_to_idx]
        self.dof_idx = torch.tensor(dof_list, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        if self.dof_idx is None:
            return
        left = self.actions[:, 0].unsqueeze(-1)
        right = self.actions[:, 1].unsqueeze(-1)
        targets = torch.hstack([left, right, left, right]) * self._max_wheel_speed
        polarity = torch.tensor(self.cfg.wheel_polarity, device=self.device)
        self.robot.set_joint_velocity_target(targets * polarity, joint_ids=self.dof_idx)

    # ------------------------------------------------------------------
    def _get_observations(self):
        """Retorna observações (sem stream RGB ativo)."""
        obs = {
            "joint_pos": self.robot.data.joint_pos,
            "joint_vel": self.robot.data.joint_vel,
        }
        if "front_camera" in self.sensors:
            obs["camera_prim"] = self.sensors["front_camera"]
        return obs

    def _get_rewards(self):
        return torch.zeros(1, device=self.device)

    def _get_dones(self):
        return (
            torch.zeros(1, dtype=torch.bool, device=self.device),
            torch.zeros(1, dtype=torch.bool, device=self.device),
        )

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()

        num_envs = len(env_ids)
        x = torch.empty(num_envs, device=self.device).uniform_(-1.4, 1.4)
        y = torch.empty(num_envs, device=self.device).uniform_(-1.9, 1.9)
        z = torch.full((num_envs,), 0.43, device=self.device)
        yaw = torch.empty(num_envs, device=self.device).uniform_(-math.pi, math.pi)
        qw = torch.cos(yaw / 2)
        qz = torch.sin(yaw / 2)
        qx = torch.zeros_like(qw)
        qy = torch.zeros_like(qw)

        try:
            import omni.isaac.dynamic_control as dc
            dci = dc.acquire_dynamic_control_interface()
            art = dci.get_articulation("/World/Robot") or dci.get_articulation("/World/Robot/teko_urdf")
            if art:
                for i in range(num_envs):
                    pose = dc.Transform()
                    pose.p = dc.Vector3(float(x[i]), float(y[i]), float(z[i]))
                    pose.r = dc.Quaternion(float(qw[i]), float(qx[i]), float(qy[i]), float(qz[i]))
                    dci.set_articulation_root_pose(art, pose)
            else:
                print("⚠️ dynamic_control: articulação não encontrada.")
        except Exception as e:
            print(f"⚠️ dynamic_control indisponível: {e}")
