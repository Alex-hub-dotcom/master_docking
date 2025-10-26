# SPDX-License-Identifier: BSD-3-Clause
"""Environment to simulate the TEKO robot in a static arena, returning RGB observations from Isaac Lab TiledCamera."""

from __future__ import annotations
import random
import numpy as np
import torch

from omni.usd import get_context
from pxr import Sdf, UsdGeom, Gf, UsdLux

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import TiledCamera
from isaaclab.sim import SimulationContext
from isaaclab.sim import utils as sim_utils

from .teko_env_cfg import TekoEnvCfg
from .robots.teko import TEKO_CONFIGURATION


class TekoEnv(DirectRLEnv):
    """Single-robot Isaac Lab environment for TEKO using RTX TiledCamera as observation."""

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((1, 2), device=self.device)
        self._max_wheel_speed = float(cfg.max_wheel_speed)
        self.dof_idx = None
        self.camera_sensor: TiledCamera | None = None

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

        # Lighting
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

        spot_path = Sdf.Path("/World/DebugLight")
        spot = UsdLux.SphereLight.Define(stage, spot_path)
        spot.CreateIntensityAttr(2000.0)
        spot.CreateRadiusAttr(0.5)
        UsdGeom.Xformable(spot).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.5))
        print("[INFO] Debug light spawned above robot (for camera exposure test).")

        # Robot
        self.robot = Articulation(TEKO_CONFIGURATION.replace(prim_path="/World/Robot"))
        self.scene.articulations["robot"] = self.robot
        self._usd_randomize_robot_pose()
        self.scene.clone_environments(copy_from_source=False)

        # ------------------------------------------------------------------ #
        # Camera setup (Isaac Lab 0.47.1 fallback)
        # ------------------------------------------------------------------ #
        sim = SimulationContext.instance()
        stage_sim = sim.stage
        cam_path = self.cfg.tiled_camera.prim_path

        # Cria prim se necessário
        prim = stage_sim.GetPrimAtPath(cam_path)
        if not prim or not prim.IsValid():
            print(f"[INFO] Spawning RTX camera prim at {cam_path}")
            sim_utils.spawn_camera_from_cfg(
                prim_path=cam_path,
                cfg=self.cfg.tiled_camera.spawn,
                translation=self.cfg.tiled_camera.offset.pos,
                orientation=self.cfg.tiled_camera.offset.rot,
            )
        else:
            print(f"[INFO] Existing camera found at {cam_path}")

        # Instancia o sensor
        self.camera_sensor = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["rgb_camera"] = self.camera_sensor

        # Inicializa manualmente (API interna)
        if not hasattr(self.camera_sensor, "_is_outdated"):
            print("[INFO] Forcing internal initialization of TiledCamera buffers.")
            try:
                self.camera_sensor._initialize()  # método privado em 0.47.1
            except Exception as e:
                print(f"[WARN] Internal _initialize() failed: {e}")

        # Warm-up de render
        print("[INFO] Performing warm-up renders...")
        from omni.kit.viewport.utility import get_active_viewport_window

        try:
            vp_win = get_active_viewport_window()
            if vp_win:
                vp = vp_win.viewport_api
                vp.set_texture_resolution((self.cfg.tiled_camera.width, self.cfg.tiled_camera.height))
                vp.set_active_camera(self.cfg.tiled_camera.prim_path)
                vp.set_render_mode("PathTracing")
                vp.set_post_process_option("EnableAutoExposure", True)
                vp.set_post_process_option("EnableRayTracedAmbientOcclusion", True)
                vp.set_post_process_option("EnableReflections", True)
                print("[INFO] RTX PathTracing enabled with auto-exposure.")
            else:
                print("[WARN] No active viewport window found.")
        except Exception as e:
            print(f"[WARN] Could not enable RTX rendering: {e}")
        for _ in range(3):
            sim.render()
            sim.step(render=True)

        # Teste do feed RGB
        try:
            data = self.camera_sensor.data.output.get("rgb", None)
            if data is not None and np.any(data):
                print(f"[INFO] RGB feed active | shape={data.shape} | min={data.min():.3f} | max={data.max():.3f}")
            else:
                print("[WARN] RGB feed still zero — check lighting or exposure.")
        except Exception as e:
            print(f"[WARN] Could not read RGB buffer: {e}")


    def _usd_randomize_robot_pose(self):
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

    def _lazy_init_articulation(self):
        if self.dof_idx is not None or getattr(self.robot, "root_physx_view", None) is None:
            return
        joint_names = list(self.robot.joint_names)
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        dof_list = [name_to_idx[n] for n in self.cfg.dof_names if n in name_to_idx]
        self.dof_idx = torch.tensor(dof_list, dtype=torch.long, device=self.device)

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
            targets * polarity,
            env_ids=torch.tensor([0], device=self.device),
            joint_ids=self.dof_idx,
        )

    def _get_observations(self):
        data = None
        if self.camera_sensor is not None:
            data = self.camera_sensor.data.output.get("rgb", None)

        if data is None:
            rgb = torch.zeros(
                (1, self.cfg.tiled_camera.height, self.cfg.tiled_camera.width, 3),
                device=self.device,
                dtype=torch.uint8,
            )
        else:
            if isinstance(data, np.ndarray):
                rgb = torch.from_numpy(data)
            else:
                rgb = data
            rgb = rgb.to(self.device)
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)
            if rgb.dtype != torch.uint8:
                if rgb.dtype.is_floating_point:
                    rgb = (rgb.clamp(0, 1) * 255.0).to(torch.uint8)
                else:
                    rgb = rgb.to(torch.uint8)

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
