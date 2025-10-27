# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment — Isaac Lab 0.47.1
-----------------------------------
Single-robot environment for TEKO with:
 - RGB pinhole camera (Raspberry Pi V2 emulation)
 - RTX LiDAR prim preconfigured in the USD: /World/teko_urdf/Lidar
"""

from __future__ import annotations
import random
import numpy as np
import torch
from typing import Tuple

# Omniverse / USD
from omni.usd import get_context
from pxr import Sdf, Usd, UsdGeom, UsdLux, Gf

# Isaac Lab
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# Sensors
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.rtx import LidarRtx
import isaacsim.core.utils.numpy.rotations as rot_utils

# TEKO config
from .teko_env_cfg import TekoEnvCfg
from .robots.teko import TEKO_CONFIGURATION


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _as_quat(rot_any) -> np.ndarray:
    """Convert Euler (°) or quaternion tuples to (w, x, y, z)."""
    if rot_any is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    arr = np.asarray(rot_any, dtype=np.float32).flatten()
    if arr.size == 4:
        return arr if abs(arr[0]) > abs(arr[-1]) else np.array([arr[3], arr[0], arr[1], arr[2]], dtype=np.float32)
    if arr.size == 3:
        q = rot_utils.euler_angles_to_quats(arr, degrees=True).astype(np.float32)
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
class TekoEnv(DirectRLEnv):
    """Single-robot RL environment with RGB camera and RTX LiDAR."""

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        self._cam_res: Tuple[int, int] = (cfg.camera.width, cfg.camera.height)
        self.actions = None
        self._max_wheel_speed = 10.0
        self.dof_idx = None
        self.camera: Camera | None = None
        self.lidar: LidarRtx | None = None
        super().__init__(cfg, render_mode, **kwargs)

    # -----------------------------------------------------------------
    # Scene setup
    # -----------------------------------------------------------------
    def _setup_scene(self):
        stage = get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage not initialized")

        # Arena / ground
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
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr(3000.0)
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SunLight"))
        sun.CreateIntensityAttr(500.0)
        sun_xf = UsdGeom.Xformable(sun)
        sun_xf.AddRotateXOp().Set(-45.0)
        sun_xf.AddRotateYOp().Set(30.0)

        # Robot
        self.robot = Articulation(TEKO_CONFIGURATION.replace(prim_path="/World/Robot"))
        self.scene.articulations["robot"] = self.robot
        self._usd_randomize_robot_pose()
        self.scene.clone_environments(copy_from_source=False)

        # -----------------------------------------------------------------
        # Camera (USD predefinida)
        # -----------------------------------------------------------------
        sim = SimulationContext.instance()
        cam_path = self.cfg.camera.prim_path
        prim = sim.stage.GetPrimAtPath(cam_path)
        if not prim.IsValid():
            raise RuntimeError(f"[ERROR] Camera prim not found at {cam_path}")
        print(f"[INFO] Using existing camera at {cam_path}")

        self.camera = Camera(
            prim_path=cam_path,
            resolution=self._cam_res,
            frequency=self.cfg.camera.frequency_hz,
        )
        self.camera.initialize()

        try:
            usd_cam = UsdGeom.Camera(prim)
            usd_cam.CreateFocalLengthAttr(3.04)
            usd_cam.CreateHorizontalApertureAttr(4.6)
            usd_cam.CreateVerticalApertureAttr(2.76)
            usd_cam.CreateFStopAttr(32.0)
            usd_cam.CreateFocusDistanceAttr(10.0)
            usd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
            print("[INFO] RearCamera set to Raspberry Pi V2 pinhole model.")
        except Exception as e:
            print(f"[WARN] Could not apply camera optics: {e}")

        # -----------------------------------------------------------------
        # LiDAR RTX — já presente no USD
        # -----------------------------------------------------------------
        lidar_prim_path = "/World/Robot/teko_urdf/TEKO_Body/LidarMount/Lidar"
        if not stage.GetPrimAtPath(lidar_prim_path):
            print(f"[ERROR] LiDAR prim not found at {lidar_prim_path}")
            self.lidar = None
        else:
            self.lidar = LidarRtx(lidar_prim_path)
            self.lidar.initialize()
            print(f"[INFO] RTX LiDAR found and initialized at {lidar_prim_path}")

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------
    def _usd_randomize_robot_pose(self):
        stage = get_context().get_stage()
        robot_prim = stage.GetPrimAtPath("/World/Robot")
        if not robot_prim:
            return
        x, y, z = random.uniform(-1.4, 1.4), random.uniform(-1.9, 1.9), 0.43
        yaw = random.uniform(-180.0, 180.0)
        xf = UsdGeom.Xformable(robot_prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        xf.AddRotateZOp().Set(yaw)

    # -----------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------
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
            targets * polarity,
            env_ids=torch.tensor([0], device=self.device),
            joint_ids=self.dof_idx,
        )

    # -----------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------
    def _get_observations(self):
        obs = {}

        # RGB
        try:
            rgba = self.camera.get_rgba()
            if isinstance(rgba, np.ndarray) and rgba.size > 0:
                rgb = torch.from_numpy(rgba[..., :3]).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                obs["rgb"] = rgb
            else:
                h, w = self._cam_res[1], self._cam_res[0]
                obs["rgb"] = torch.zeros((1, 3, h, w), device=self.device)
        except Exception:
            h, w = self._cam_res[1], self._cam_res[0]
            obs["rgb"] = torch.zeros((1, 3, h, w), device=self.device)

        # LiDAR
        try:
            if self.lidar is None:
                obs["lidar"] = torch.zeros((1, 3), device=self.device)
            else:
                data = self.lidar.get_point_cloud_data()
                pts = data.get("points") if data and "points" in data else None
                if pts is None:
                    obs["lidar"] = torch.zeros((1, 3), device=self.device)
                else:
                    obs["lidar"] = torch.from_numpy(pts).float().to(self.device)
        except Exception as e:
            print(f"[WARN] LiDAR read failed: {e}")
            obs["lidar"] = torch.zeros((1, 3), device=self.device)
        return obs

    # -----------------------------------------------------------------
    # RL placeholders
    # -----------------------------------------------------------------
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
