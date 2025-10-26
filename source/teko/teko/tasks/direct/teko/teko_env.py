# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment — Isaac Lab 0.47.1
-----------------------------------
Simula o robô TEKO num ambiente estático e retorna observações RGB
a partir de uma câmera já existente no USD (/World/teko_urdf/RearCamera).

Compatível com Gymnasium e skrl.
"""

from __future__ import annotations
import random
from typing import Tuple
import numpy as np
import torch

# Omniverse / USD API
from omni.usd import get_context
from pxr import Sdf, UsdGeom, Gf, UsdLux

# Isaac Lab / Simulação
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# Câmera (API nativa simples)
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

# Configurações
from .teko_env_cfg import TekoEnvCfg
from .robots.teko import TEKO_CONFIGURATION


# --------------------------------------------------------------------------- #
# Utilitário simples para lidar com rotações
# --------------------------------------------------------------------------- #
def _as_quat(rot_any) -> np.ndarray:
    """Aceita (w,x,y,z) ou (x,y,z,w) ou Euler (graus) e retorna quat (w,x,y,z)."""
    if rot_any is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    arr = np.asarray(rot_any, dtype=np.float32).flatten()
    if arr.size == 4:
        if abs(arr[-1]) > abs(arr[0]):  # assume (x,y,z,w)
            return np.array([arr[3], arr[0], arr[1], arr[2]], dtype=np.float32)
        return arr
    elif arr.size == 3:
        q = rot_utils.euler_angles_to_quats(arr, degrees=True).astype(np.float32)
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Ambiente principal
# --------------------------------------------------------------------------- #
class TekoEnv(DirectRLEnv):
    """Ambiente RL de um único robô TEKO com observações RGB."""

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        # define atributos antes do construtor base (pois ele chama _setup_scene)
        self._cam_res: Tuple[int, int] = (640, 480)
        self.actions = None
        self._max_wheel_speed = None
        self.dof_idx = None
        self.camera: Camera | None = None
        

        super().__init__(cfg, render_mode, **kwargs)
        self._max_wheel_speed = 10.0
    # ------------------------------------------------------------------ #
    # Cena e spawn
    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        stage = get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is not initialized")

        # Arena
        try:
            arena_path = "/workspace/teko/documents/CAD/USD/stage_arena.usd"
            arena_prim_path = "/World/StageArena"
            stage.DefinePrim(Sdf.Path(arena_prim_path), "Xform")
            arena_prim = stage.GetPrimAtPath(arena_prim_path)
            arena_prim.GetReferences().AddReference(arena_path)
            UsdGeom.Xformable(arena_prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        except Exception:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Luzes básicas
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr(3000.0)
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SunLight"))
        sun.CreateIntensityAttr(500.0)
        UsdGeom.Xformable(sun).AddRotateXOp().Set(-45.0)
        UsdGeom.Xformable(sun).AddRotateYOp().Set(30.0)

        # Robô principal
        self.robot = Articulation(TEKO_CONFIGURATION.replace(prim_path="/World/Robot"))
        self.scene.articulations["robot"] = self.robot
        self._usd_randomize_robot_pose()
        self.scene.clone_environments(copy_from_source=False)

        # ------------------------------------------------------------------ #
        # Câmera já existente no USD
        # ------------------------------------------------------------------ #
        sim = SimulationContext.instance()
        cam_path = "/World/teko_urdf/RearCamera"

        if not sim.stage.GetPrimAtPath(cam_path).IsValid():
            print(f"[WARN] Camera prim not found at {cam_path}, defining one.")
            UsdGeom.Camera.Define(sim.stage, Sdf.Path(cam_path))
        else:
            print(f"[INFO] Reusing existing camera at {cam_path}")

        pos = (0.0, 0.0, 0.0)
        rot = (0.0, 0.0, 0.0)
        quat = _as_quat(rot)
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)

        self.camera = Camera(
            prim_path=cam_path,
            position=np.array(pos, dtype=np.float32),
            orientation=quat_xyzw,
            resolution=self._cam_res,
            frequency=30,
        )
        self.camera.initialize()

        # render inicial
        for _ in range(3):
            sim.step(render=True)

        try:
            rgba = self.camera.get_rgba()
            if isinstance(rgba, np.ndarray) and rgba.size > 0:
                mn, mx = int(rgba[..., :3].min()), int(rgba[..., :3].max())
                print(f"[INFO] Camera RGBA ready | shape={rgba.shape} | min={mn} | max={mx}")
            else:
                print("[WARN] Camera returned empty buffer on first check.")
        except Exception as e:
            print(f"[WARN] Camera first read failed: {e}")

    # ------------------------------------------------------------------ #
    # Pose aleatória do robô
    # ------------------------------------------------------------------ #
    def _usd_randomize_robot_pose(self):
        stage = get_context().get_stage()
        robot_prim = stage.GetPrimAtPath("/World/Robot")
        if not robot_prim:
            return
        x = random.uniform(-1.4, 1.4)
        y = random.uniform(-1.9, 1.9)
        z = 0.43
        yaw = random.uniform(-180.0, 180.0)
        xf = UsdGeom.Xformable(robot_prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        xf.AddRotateZOp().Set(yaw)

    # ------------------------------------------------------------------ #
    # Controle do robô
    # ------------------------------------------------------------------ #
    def _lazy_init_articulation(self):
        if self.dof_idx is not None or getattr(self.robot, "root_physx_view", None) is None:
            return
        name_to_idx = {n: i for i, n in enumerate(self.robot.joint_names)}
        self.dof_idx = torch.tensor(
            [name_to_idx[n] for n in self.cfg.dof_names if n in name_to_idx],
            dtype=torch.long,
            device=self.device,
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        if self.dof_idx is None:
            return
        left, right = self.actions[0]
        targets = torch.tensor([left, right, left, right], device=self.device).unsqueeze(0) * self._max_wheel_speed
        polarity = torch.tensor(self.cfg.wheel_polarity, device=self.device).unsqueeze(0)
        self.robot.set_joint_velocity_target(
            targets * polarity, env_ids=torch.tensor([0], device=self.device), joint_ids=self.dof_idx
        )

    # ------------------------------------------------------------------ #
    # Observações e recompensas
    # ------------------------------------------------------------------ #
    def _get_observations(self):
        """Retorna imagem RGB da câmera como tensor normalizado [0,1]."""
        if self.camera is None:
            h, w = self._cam_res[1], self._cam_res[0]
            rgb = torch.zeros((1, 3, h, w), device=self.device, dtype=torch.float32)
            return {"policy": rgb}

        rgba = self.camera.get_rgba()
        if not isinstance(rgba, np.ndarray) or rgba.size == 0:
            h, w = self._cam_res[1], self._cam_res[0]
            rgb = torch.zeros((1, 3, h, w), device=self.device, dtype=torch.float32)
            return {"policy": rgb}

        rgb_np = rgba[..., :3]
        rgb = torch.from_numpy(rgb_np).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return {"policy": rgb}

    def _get_rewards(self):
        return torch.zeros(1, device=self.device)

    def _get_dones(self):
        return (
            torch.zeros(1, dtype=torch.bool, device=self.device),
            torch.zeros(1, dtype=torch.bool, device=self.device),
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()
