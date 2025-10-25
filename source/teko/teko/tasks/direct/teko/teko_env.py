# SPDX-License-Identifier: BSD-3-Clause
"""Environment to simulate the TEKO robot in a static arena, returning real RGB observations."""

from __future__ import annotations

import math
import random
import torch

from omni.usd import get_context
from pxr import Usd, Sdf, UsdGeom, Gf

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .teko_env_cfg import TekoEnvCfg
from .robots.teko import TEKO_CONFIGURATION
from .sensors.camera import TekoCamera
from pxr import UsdLux


class TekoEnv(DirectRLEnv):
    """Single-robot Isaac Lab environment for TEKO using a real RGB camera."""

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Controle e robot params
        self.actions = torch.zeros((1, 2), device=self.device)
        self._max_wheel_speed = float(cfg.max_wheel_speed)
        self.dof_idx = None

        # Instancia a câmera real
        self.camera = TekoCamera(cfg.tiled_camera)

        print("[INFO] Forcing camera render once...")
        self.camera.update()
        rgb = self.camera.get_rgb()
        if rgb is not None:
            print(f"[DEBUG] Camera frame captured | min={rgb.min()} | max={rgb.max()}")
        else:
            print("[WARN] No camera frame yet.")

    # -------------------------------------------------------------------- #
    #  Cena e spawn
    # -------------------------------------------------------------------- #
    def _setup_scene(self):
        stage = get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is not initialized")

        # ------------------------------------------------------------------ #
        # Arena
        # ------------------------------------------------------------------ #
        try:
            arena_path = "/workspace/teko/documents/CAD/USD/stage_arena.usd"
            arena_prim_path = "/World/StageArena"
            stage.DefinePrim(Sdf.Path(arena_prim_path), "Xform")
            arena_prim = stage.GetPrimAtPath(arena_prim_path)
            arena_prim.GetReferences().AddReference(arena_path)
            UsdGeom.Xformable(arena_prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        except Exception:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # ------------------------------------------------------------------ #
        # Lights
        # ------------------------------------------------------------------ #
        dome_path = Sdf.Path("/World/DomeLight")
        dome = UsdLux.DomeLight.Define(stage, dome_path)
        dome.CreateIntensityAttr(3000.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        print("[INFO] Dome light spawned successfully (USD).")

        sun_path = Sdf.Path("/World/SunLight")
        sun = UsdLux.DistantLight.Define(stage, sun_path)
        sun.CreateIntensityAttr(500.0)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.9))
        sun.CreateAngleAttr(0.53)
        UsdGeom.Xformable(sun).AddRotateXOp().Set(-45.0)
        UsdGeom.Xformable(sun).AddRotateYOp().Set(30.0)
        print("[INFO] Sun light spawned successfully (USD).")

        # ------------------------------------------------------------------ #
        # Robot
        # ------------------------------------------------------------------ #
        self.robot = Articulation(TEKO_CONFIGURATION.replace(prim_path="/World/Robot"))
        self.scene.articulations["robot"] = self.robot

        # ------------------------------------------------------------------ #
        # Randomize robot pose e clones
        # ------------------------------------------------------------------ #
        self._usd_randomize_robot_pose()
        self.scene.clone_environments(copy_from_source=False)

        # ------------------------------------------------------------------ #
        # ✅ Ativa renderização RTX da câmera
        # ------------------------------------------------------------------ #
        try:
            from omni.isaac.core.utils.prims import get_prim_at_path
            from omni.isaac.sensor import Camera

            cam_prim_path = "/World/Robot/RearCamera"
            cam_prim = get_prim_at_path(cam_prim_path)
            if cam_prim.IsValid():
                camera_sensor = Camera(cam_prim_path, frequency=30)
                camera_sensor.initialize_rtx()
                camera_sensor.set_resolution(640, 480)
                print(f"[INFO] RTX camera initialized at {cam_prim_path}")
            else:
                print(f"[WARN] Camera prim not found at {cam_prim_path}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize RTX camera: {e}")

    # -------------------------------------------------------------------- #
    #  Randomização de pose
    # -------------------------------------------------------------------- #
    def _usd_randomize_robot_pose(self):
        """Randomize robot position and yaw inside the arena."""
        stage = get_context().get_stage()
        robot_prim = stage.GetPrimAtPath("/World/Robot")
        if not robot_prim:
            return

        x = random.uniform(-1.4, 1.4)
        y = random.uniform(-1.9, 1.9)
        z = 0.43
        yaw_deg = random.uniform(-180.0, 180.0)

        xf = UsdGeom.Xformable(robot_prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        xf.AddRotateZOp().Set(yaw_deg)

    # -------------------------------------------------------------------- #
    #  Controle
    # -------------------------------------------------------------------- #
    def _lazy_init_articulation(self):
        """Cache DOF indices from joint names only once."""
        if self.dof_idx is not None or getattr(self.robot, "root_physx_view", None) is None:
            return
        joint_names = list(self.robot.joint_names)
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        dof_list = [name_to_idx[n] for n in self.cfg.dof_names if n in name_to_idx]
        self.dof_idx = torch.tensor(dof_list, dtype=torch.long, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Receive 2D action (left/right) from agent."""
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        """Convert 2D action into 4-wheel velocity targets (single env)."""
        if self.dof_idx is None:
            return

        left, right = self.actions[0]
        targets = torch.tensor([left, right, left, right], device=self.device).unsqueeze(0) * self._max_wheel_speed
        polarity = torch.tensor(self.cfg.wheel_polarity, device=self.device).unsqueeze(0)

        self.robot.set_joint_velocity_target(
            targets * polarity,
            env_ids=torch.tensor([0], device=self.device),
            joint_ids=self.dof_idx,
        )

    # -------------------------------------------------------------------- #
    #  Observações e rewards
    # -------------------------------------------------------------------- #
    def _get_observations(self):
        """Return real camera image (RGB) as observation."""
        self.camera.update()
        rgb = self.camera.get_rgb()

        if rgb is None:
            rgb = torch.zeros((1, self.cfg.tiled_camera.height, self.cfg.tiled_camera.width, 3), device=self.device)

        rgb = rgb.permute(0, 3, 1, 2).float() / 255.0
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
