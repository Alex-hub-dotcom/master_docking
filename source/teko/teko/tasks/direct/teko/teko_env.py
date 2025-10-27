# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment — Isaac Lab 0.47.1
-----------------------------------
Single-robot environment for TEKO, using an RGB pinhole camera
configured to emulate the Raspberry Pi Camera V2 (Sony IMX219).

Compatible with Gymnasium and skrl.
"""

from __future__ import annotations
import random
from typing import Tuple
import numpy as np
import torch

# Omniverse / USD API
from omni.usd import get_context
from pxr import Sdf, UsdGeom, Gf, UsdLux

# Isaac Lab / Simulation
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# Isaac Sim Camera API
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

# Config
from .teko_env_cfg import TekoEnvCfg
from .robots.teko import TEKO_CONFIGURATION



# Quaternion utility

def _as_quat(rot_any) -> np.ndarray:
    """Converts Euler or quaternion tuples to (w, x, y, z) format."""
    if rot_any is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    arr = np.asarray(rot_any, dtype=np.float32).flatten()
    if arr.size == 4:
        if abs(arr[-1]) > abs(arr[0]):  # assumes (x, y, z, w)
            return np.array([arr[3], arr[0], arr[1], arr[2]], dtype=np.float32)
        return arr
    elif arr.size == 3:
        q = rot_utils.euler_angles_to_quats(arr, degrees=True).astype(np.float32)
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)



# Main environment

class TekoEnv(DirectRLEnv):
    """Single-robot RL environment with RGB camera observations."""

    cfg: TekoEnvCfg

    
    # Initialization
    
    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize environment and default parameters."""
        self._cam_res: Tuple[int, int] = (cfg.camera.width, cfg.camera.height)
        self.actions = None
        self._max_wheel_speed = None
        self.dof_idx = None
        self.camera: Camera | None = None

        super().__init__(cfg, render_mode, **kwargs)
        self._max_wheel_speed = 10.0

    
    # Scene setup
    
    def _setup_scene(self):
        """Load arena, lights, robot, and connect the existing camera."""
        stage = get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is not initialized")

        # Load arena or fallback to ground plane
        try:
            arena_path = "/workspace/teko/documents/CAD/USD/stage_arena.usd"
            arena_prim_path = "/World/StageArena"
            stage.DefinePrim(Sdf.Path(arena_prim_path), "Xform")
            arena_prim = stage.GetPrimAtPath(arena_prim_path)
            arena_prim.GetReferences().AddReference(arena_path)
            UsdGeom.Xformable(arena_prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        except Exception:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Basic lighting
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr(3000.0)
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SunLight"))
        sun.CreateIntensityAttr(500.0)
        UsdGeom.Xformable(sun).AddRotateXOp().Set(-45.0)
        UsdGeom.Xformable(sun).AddRotateYOp().Set(30.0)

        # Robot articulation
        self.robot = Articulation(TEKO_CONFIGURATION.replace(prim_path="/World/Robot"))
        self.scene.articulations["robot"] = self.robot
        self._usd_randomize_robot_pose()
        self.scene.clone_environments(copy_from_source=False)

        
        # Existing camera (keep pose, apply Pi V2 optical model)
        
        sim = SimulationContext.instance()
        cam_path = self.cfg.camera.prim_path
        prim = sim.stage.GetPrimAtPath(cam_path)

        if not prim.IsValid():
            raise RuntimeError(f"[ERROR] Camera prim not found at {cam_path}")
        print(f"[INFO] Reusing existing camera at {cam_path}")

        # Wrap camera without changing pose
        self.camera = Camera(
            prim_path=cam_path,
            resolution=self._cam_res,
            frequency=self.cfg.camera.frequency_hz,
        )
        self.camera.initialize()

        # Raspberry Pi Camera V2 optical parameters (RGB pinhole)
        try:
            usd_cam = UsdGeom.Camera(prim)
            usd_cam.CreateFocalLengthAttr(3.04)            # mm
            usd_cam.CreateHorizontalApertureAttr(4.6)      # mm
            usd_cam.CreateVerticalApertureAttr(2.76)       # mm
            usd_cam.CreateFStopAttr(32.0)                  # disable DOF blur
            usd_cam.CreateFocusDistanceAttr(10.0)          # focus to infinity
            usd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
            print("[INFO] RearCamera set to Raspberry Pi V2 pinhole model.")
        except Exception as e:
            print(f"[WARN] Could not apply camera optics: {e}")

        # Initial render sync
        for _ in range(3):
            sim.step(render=True)

        # Validate camera output
        try:
            rgba = self.camera.get_rgba()
            if isinstance(rgba, np.ndarray) and rgba.size > 0:
                mn, mx = int(rgba[..., :3].min()), int(rgba[..., :3].max())
                print(f"[INFO] Camera RGBA ready | shape={rgba.shape} | min={mn} | max={mx}")
            else:
                print("[WARN] Camera returned empty buffer.")
        except Exception as e:
            print(f"[WARN] Camera read failed: {e}")

    
    # Randomize robot pose
    
    def _usd_randomize_robot_pose(self):
        """Randomize robot position and yaw inside the arena."""
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

    
    # Robot control
    
    def _lazy_init_articulation(self):
        """Initialize joint indices once."""
        if self.dof_idx is not None or getattr(self.robot, "root_physx_view", None) is None:
            return
        name_to_idx = {n: i for i, n in enumerate(self.robot.joint_names)}
        self.dof_idx = torch.tensor(
            [name_to_idx[n] for n in self.cfg.dof_names if n in name_to_idx],
            dtype=torch.long,
            device=self.device,
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store actions before physics update."""
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        """Convert normalized actions into wheel velocity targets."""
        if self.dof_idx is None:
            return
        left, right = self.actions[0]
        targets = torch.tensor([left, right, left, right], device=self.device).unsqueeze(0) * self._max_wheel_speed
        polarity = torch.tensor(self.cfg.wheel_polarity, device=self.device).unsqueeze(0)
        self.robot.set_joint_velocity_target(
            targets * polarity, env_ids=torch.tensor([0], device=self.device), joint_ids=self.dof_idx
        )

    
    # Observations and rewards
    
    def _get_observations(self):
        """Return normalized [0,1] RGB tensor from the camera."""
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
        """Return zero reward (placeholder for RL)."""
        return torch.zeros(1, device=self.device)

    def _get_dones(self):
        """Return episode termination flags (placeholder for RL)."""
        return (
            torch.zeros(1, dtype=torch.bool, device=self.device),
            torch.zeros(1, dtype=torch.bool, device=self.device),
        )

    def _reset_idx(self, env_ids):
        """Reset environment and reinitialize articulations."""
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()
