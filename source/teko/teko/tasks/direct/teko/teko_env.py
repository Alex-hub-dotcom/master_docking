# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Environment to visualize the TEKO robot moving inside a custom arena with random spawn."""

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


class TekoEnv(DirectRLEnv):
    """Single-robot environment with random initial pose."""

    cfg: TekoEnvCfg

    # ------------------------------------------------------------------
    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((1, 2), device=self.device)
        self._max_wheel_speed = float(cfg.max_wheel_speed)
        self.dof_idx = None

    # ------------------------------------------------------------------
    def _setup_scene(self):
        """Load arena, spawn the TEKO robot and place it at a random pose (USD-level, pre-physics)."""
        # Carrega arena via omni.usd (compatível com 0.47.1)
        try:
            stage = get_context().get_stage()
            if not stage:
                raise RuntimeError("Stage not initialized")

            arena_path = "/workspace/teko/documents/CAD/USD/arena.usd"
            arena_prim_path = "/World/Arena"
            stage.DefinePrim(Sdf.Path(arena_prim_path), "Xform")
            stage.GetPrimAtPath(arena_prim_path).GetReferences().AddReference(arena_path)
            print(f"✅ Arena carregada via omni.usd: {arena_path}")
        except Exception as e:
            print(f"⚠️ Erro ao carregar arena: {e}")
            print("Carregando plano básico no lugar...")
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Spawn TEKO
        from teko.robots.teko import TEKO_CONFIGURATION
        self.robot = Articulation(TEKO_CONFIGURATION.replace(prim_path="/World/Robot"))
        self.scene.articulations["robot"] = self.robot

        # Antes de iniciar a física, coloca o prim em pose aleatória via USD (garante variação a cada execução)
        self._usd_randomize_robot_pose()

        # Finaliza a cena
        self.scene.clone_environments(copy_from_source=False)

    # ------------------------------------------------------------------
    def _usd_randomize_robot_pose(self):
        """Define uma pose aleatória via USD (funciona pré-física em 0.47.1)."""
        try:
            stage = get_context().get_stage()
            robot_prim = stage.GetPrimAtPath("/World/Robot")
            if not robot_prim:
                print("⚠️ USD: prim do robô não encontrado para randomização inicial.")
                return

            # área 8x8 m
            x = random.uniform(-4.0, 4.0)
            y = random.uniform(-4.0, 4.0)
            z = 0.15
            yaw_deg = random.uniform(-180.0, 180.0)

            xf = UsdGeom.Xformable(robot_prim)
            # Limpa a ordem e reescreve ops (translate + rotateZ)
            xf.ClearXformOpOrder()
            t_op = xf.AddTranslateOp()
            r_op = xf.AddRotateZOp()  # graus
            t_op.Set(Gf.Vec3d(x, y, z))
            r_op.Set(yaw_deg)
            print(f"🎯 Spawn inicial (USD): x={x:.2f} y={y:.2f} z={z:.2f} yaw={yaw_deg:.1f}°")
        except Exception as e:
            print(f"⚠️ Falha na randomização USD do robô: {e}")

    # ------------------------------------------------------------------
    def _lazy_init_articulation(self):
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
        return {}

    def _get_rewards(self):
        return torch.zeros(1, device=self.device)

    def _get_dones(self):
        return (
            torch.zeros(1, dtype=torch.bool, device=self.device),
            torch.zeros(1, dtype=torch.bool, device=self.device),
        )

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        """Tentativa de randomizar pose a cada reset (primeiro via dynamic_control; se não, avisa)."""
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()

        # Amostras aleatórias dentro de 8×8 m
        num_envs = len(env_ids)
        x = torch.empty(num_envs, device=self.device).uniform_(-4.0, 4.0)
        y = torch.empty(num_envs, device=self.device).uniform_(-4.0, 4.0)
        z = torch.full((num_envs,), 0.15, device=self.device)

        yaw = torch.empty(num_envs, device=self.device).uniform_(-math.pi, math.pi)
        qw = torch.cos(yaw / 2)
        qz = torch.sin(yaw / 2)
        # quaternions (x,y,z,w) = (0,0,sin/2,cos/2) — aqui vamos precisar (w,x,y,z) para DC
        qx = torch.zeros_like(qw)
        qy = torch.zeros_like(qw)

        # 1) Tenta via dynamic_control (mais compatível com builds antigas)
        try:
            import omni.isaac.dynamic_control as dc
            dci = dc.acquire_dynamic_control_interface()
            # o root do TEKO é /World/Robot/teko_urdf (geralmente); se for só /World/Robot, dci ainda encontra
            art = dci.get_articulation("/World/Robot")
            if art is None:
                # tenta pelo filho comum do importador URDF
                art = dci.get_articulation("/World/Robot/teko_urdf")
            if art is not None:
                for i in range(num_envs):
                    pose = dc.Transform()
                    pose.p = dc.Vector3(float(x[i].item()), float(y[i].item()), float(z[i].item()))
                    # dynamic_control usa quaternion (w, x, y, z)
                    pose.r = dc.Quaternion(float(qw[i].item()),
                                           float(qx[i].item()),
                                           float(qy[i].item()),
                                           float(qz[i].item()))
                    dci.set_articulation_root_pose(art, pose)
                return
            else:
                print("⚠️ dynamic_control: articulação não encontrada por caminho. Mantendo pose atual.")
        except Exception as e:
            print(f"⚠️ dynamic_control indisponível p/ reset aleatório: {e}")
        # 2) Se dynamic_control falhar, mantemos a pose atual (não quebramos o episódio)
        #    O spawn continuará aleatório no início (via USD).
