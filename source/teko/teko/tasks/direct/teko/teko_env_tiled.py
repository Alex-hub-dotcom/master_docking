# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment with TiledCamera for Efficient Rendering
==========================================================
/home/schux00/teko/source/teko/teko/tasks/direct/teko/teko_env_tiled.py
Uses TiledCamera instead of individual cameras for better GPU memory efficiency.
Supports 150+ parallel environments on RTX 3090.

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from omni.usd import get_context
from pxr import Sdf, UsdGeom, UsdLux, Gf, UsdPhysics

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, TiledCameraCfg

from .teko_env_cfg import TekoEnvCfg
from .rewards.reward_teko import compute_total_reward
from .curriculum.curriculum_teko import (
    reset_environment_curriculum,
    set_curriculum_level,
)
from .utils.logging_utils import collect_episode_stats
from .robots.teko_static import TEKOStatic


class TekoEnvTiled(DirectRLEnv):
    """
    TEKO environment with TiledCamera for efficient batched rendering.
    Supports 150+ parallel environments.
    """

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        # Camera resolution
        self._cam_res = (cfg.tiled_camera.width, cfg.tiled_camera.height)

        # Frame stacking
        self.num_frame_stack = getattr(cfg, "num_frame_stack", 4)
        self.frame_stack = None
        self.frame_counts = None

        # Torque scaling
        self._max_wheel_torque = cfg.max_wheel_torque

        # Arena limits
        self._arena_half_x = float(cfg.arena_half_x)
        self._arena_half_y = float(cfg.arena_half_y)

        # Body footprints
        self._active_body_length = float(cfg.active_body_length)
        self._active_body_width = float(cfg.active_body_width)
        self._static_body_length = float(cfg.static_body_length)
        self._static_body_width = float(cfg.static_body_width)

        # Placeholders
        self.actions = None
        self.dof_idx = None
        self.tiled_camera = None  # TiledCamera instead of list
        self.goal_positions = None
        self.num_agents = 1
        self._polarity = None

        # Curriculum
        self.curriculum_level = 0

        # State tracking
        self.prev_robot_pos = None
        self.prev_distance = None
        self.prev_actions = None
        self.step_count = None
        self._last_success = None

        # Episode stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.reward_components = {
            "distance": [],
            "progress": [],
            "alignment": [],
            "approach_bonus": [],
            "collision_penalty": [],
            "boundary_penalty": [],
            "success_bonus": [],
            "time_penalty": [],
        }

        super().__init__(cfg, render_mode, **kwargs)

    def _init_observation_space(self):
        """Define observation space (stack of K grayscale frames)."""
        import gymnasium as gym

        num_channels = self.num_frame_stack
        frame_shape = (num_channels, self.cfg.tiled_camera.height, self.cfg.tiled_camera.width)

        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=frame_shape,
                dtype=np.float32,
            )
        })
        print(f"[INFO] TiledCamera mode: observation space = {frame_shape}")

    def _setup_scene(self):
        stage = get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage not initialized")

        if not stage.GetPrimAtPath("/World/envs/env_0"):
            stage.DefinePrim("/World/envs/env_0", "Xform")

        self._setup_global_lighting(stage)

        # Active robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Clone environments
        self.scene.clone_environments(copy_from_source=True)

        # Arena, goal robot, ground plane
        self._setup_per_environment_assets(stage)

        # TiledCamera (batched rendering)
        self._setup_tiled_camera()

        # Cache goal positions
        self._cache_goal_transforms()

    def _setup_global_lighting(self, stage):
        """Simple dome + sun lighting."""
        if stage.GetPrimAtPath("/World/DomeLight"):
            stage.RemovePrim("/World/DomeLight")

        ambient = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/AmbientLight"))
        ambient.CreateIntensityAttr(4000.0)
        ambient.CreateColorAttr(Gf.Vec3f(0.95, 0.95, 0.95))

        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SunLight"))
        sun.CreateIntensityAttr(2000.0)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
        UsdGeom.Xformable(sun).AddRotateXOp().Set(-50.0)
        UsdGeom.Xformable(sun).AddRotateYOp().Set(30.0)

    def _spawn_ground_plane(self, stage, env_idx: int):
        """Create ground plane for environment."""
        env_root = f"/World/envs/env_{env_idx}"
        ground_path = f"{env_root}/Ground"

        cube = UsdGeom.Cube.Define(stage, Sdf.Path(ground_path))
        xf = UsdGeom.Xformable(cube)
        xf.ClearXformOpOrder()

        floor_z = 0.185
        thickness = 0.02

        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, floor_z))
        hx = float(self._arena_half_x)
        hy = float(self._arena_half_y)
        xf.AddScaleOp().Set(Gf.Vec3d(hx, hy, thickness * 0.5))

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdGeom.Gprim(cube).CreateDisplayColorAttr([Gf.Vec3f(0.4, 0.4, 0.4)])

    def _setup_per_environment_assets(self, stage):
        """Setup arena, ground, robots for each environment."""
        num_envs = self.scene.cfg.num_envs
        ARENA_USD_PATH = "/workspace/teko/documents/CAD/USD/stage_arena.usd"

        for env_idx in range(num_envs):
            env_path = f"/World/envs/env_{env_idx}"

            try:
                arena_prim = stage.DefinePrim(f"{env_path}/Arena", "Xform")
                arena_prim.GetReferences().AddReference(ARENA_USD_PATH)
            except Exception as e:
                print(f"[WARN] Arena failed for env_{env_idx}: {e}")

            self._spawn_ground_plane(stage, env_idx)

            robot_prim = stage.GetPrimAtPath(f"{env_path}/Robot")
            if robot_prim.IsValid():
                xf_robot = UsdGeom.Xformable(robot_prim)
                xf_robot.ClearXformOpOrder()
                xf_robot.AddTranslateOp().Set(Gf.Vec3d(0.3, 0.0, 0.4))
                xf_robot.AddRotateZOp().Set(180.0)

            try:
                TEKOStatic(prim_path=f"{env_path}/RobotGoal")
            except Exception as e:
                print(f"[WARN] Failed to create static TEKO in env_{env_idx}: {e}")

        print(f"[INFO] Created {num_envs} environments.")

    def _setup_tiled_camera(self):
        """Setup TiledCamera for batched rendering."""
        cam_cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/teko_urdf/TEKO_Body/TEKO_WallBack/TEKO_Camera/RearCamera",
            update_period=0,
            height=self._cam_res[1],
            width=self._cam_res[0],
            data_types=["rgb"],
            spawn=None,
        )

        self.tiled_camera = TiledCamera(cfg=cam_cfg)
        print(f"[INFO] TiledCamera initialized for {self.scene.cfg.num_envs} envs")

    def _cache_goal_transforms(self):
        """Precompute goal positions."""
        num_envs = self.scene.cfg.num_envs
        self.goal_positions = torch.zeros((num_envs, 3), device=self.device)
        for env_idx, origin in enumerate(self.scene.env_origins):
            local_goal = torch.tensor([1.5, 0.0, 0.40], device=self.device)
            self.goal_positions[env_idx] = origin + local_goal
        print(f"[INFO] Cached {num_envs} goal positions.")


    @staticmethod
    def _quat_rotate_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate vector(s) by quaternion(s). Quat format: [x, y, z, w] (Isaac Lab convention)."""
        if vec.dim() == 1:
            v = vec.unsqueeze(0).expand(quat.shape[0], 3)
        else:
            v = vec
        q_xyz = quat[:, :3]
        q_w = quat[:, 3:4]
        t = 2.0 * torch.cross(q_xyz, v, dim=-1)
        return v + q_w * t + torch.cross(q_xyz, t, dim=-1)
    def get_sphere_distances_from_physics(self):
        """Compute connector distances with correct offsets and rotation."""
        FEMALE_OFFSET = torch.tensor([0.24, 0.0, -0.08], device=self.device)
        MALE_OFFSET = torch.tensor([-0.22667, -0.00144, -0.08815], device=self.device)

        active_pos = self.robot.data.root_pos_w
        static_pos = self.goal_positions

        female_pos = active_pos + FEMALE_OFFSET.unsqueeze(0).expand(active_pos.shape[0], 3)
        male_pos = static_pos + MALE_OFFSET.unsqueeze(0).expand(static_pos.shape[0], 3)

        diff = female_pos - male_pos
        dist_3d = torch.norm(diff, dim=-1)
        dist_xy = torch.norm(diff[:, :2], dim=-1)

        R_FEMALE = 0.005
        R_MALE = 0.005
        surface_3d = torch.clamp(dist_3d - (R_FEMALE + R_MALE), min=0.0)
        surface_xy = torch.clamp(dist_xy - (R_FEMALE + R_MALE), min=0.0)

        return female_pos, male_pos, surface_xy, surface_3d

    def _lazy_init_articulation(self):
        """Initialize joint indices once robot is loaded."""
        if self.dof_idx is not None or getattr(self.robot, "root_physx_view", None) is None:
            return

        name_to_idx = {n: i for i, n in enumerate(self.robot.joint_names)}
        indices = [name_to_idx[n] for n in self.cfg.dof_names if n in name_to_idx]
        if not indices:
            raise RuntimeError(f"No valid DOF names: {self.robot.joint_names}")

        self.dof_idx = torch.tensor(indices, dtype=torch.long, device=self.device)
        print(f"[INFO] DOF indices: {self.dof_idx}")

        if self._polarity is None:
            self._polarity = torch.tensor(
                self.cfg.wheel_polarity, device=self.device
            ).unsqueeze(0)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        """Convert RL actions into wheel torques."""
        if self.dof_idx is None or self.actions is None:
            return

        num_envs = self.scene.cfg.num_envs
        v_cmd = self.actions[:, 0]
        w_cmd = self.actions[:, 1]
        v = -v_cmd * 1.0  # Convenção: v- = ré
        w = w_cmd * 1.0

        k = 3.0
        left = torch.clamp(v + k * w, -1.0, 1.0)
        right = torch.clamp(v - k * w, -1.0, 1.0)

        torque_targets = (
            torch.stack([-left, right, -left, right], dim=1) * self._max_wheel_torque
        )
        # torque_targets = torque_targets * self._polarity  # REMOVIDO

        env_ids = torch.arange(num_envs, device=self.device)
        self.robot.set_joint_effort_target(
            torque_targets, env_ids=env_ids, joint_ids=self.dof_idx
        )

    def _get_observations(self) -> dict:
        """Capture RGB from TiledCamera, convert to grayscale, stack frames."""
        num_envs = self.scene.cfg.num_envs
        h, w = self._cam_res[1], self._cam_res[0]

        if self.frame_stack is None:
            self.frame_stack = torch.zeros(
                (num_envs, self.num_frame_stack, h, w),
                device=self.device, dtype=torch.float32,
            )
        if self.frame_counts is None:
            self.frame_counts = torch.zeros(
                num_envs, device=self.device, dtype=torch.int32
            )

        # Update TiledCamera
        self.tiled_camera.update(dt=0.0)

        # Get batched RGB data: [N, H, W, C]
        rgb_data = self.tiled_camera.data.output["rgb"]

        if rgb_data is None or rgb_data.numel() == 0:
            return {"rgb": self.frame_stack}

        # Handle RGBA -> RGB
        if rgb_data.shape[-1] == 4:
            rgb_data = rgb_data[..., :3]

        # Convert to [N, C, H, W] and normalize
        rgb = rgb_data.permute(0, 3, 1, 2).float() / 255.0

        # Resize if needed
        if rgb.shape[2] != h or rgb.shape[3] != w:
            rgb = F.interpolate(rgb, size=(h, w), mode="bilinear", align_corners=False)

        # Convert to grayscale: [N, 1, H, W] -> [N, H, W]
        gray = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]

        # Update frame stack
        reset_mask = self.frame_counts == 0
        if reset_mask.any():
            idx = reset_mask.nonzero(as_tuple=False).squeeze(-1)
            self.frame_stack[idx] = gray[idx].unsqueeze(1).expand(-1, self.num_frame_stack, -1, -1)
            self.frame_counts[idx] = self.num_frame_stack

        non_reset_mask = ~reset_mask
        if non_reset_mask.any():
            idx = non_reset_mask.nonzero(as_tuple=False).squeeze(-1)
            self.frame_stack[idx, :-1] = self.frame_stack[idx, 1:].clone()
            self.frame_stack[idx, -1] = gray[idx]

        return {"rgb": self.frame_stack}

    def _get_rewards(self):
        """Delegate to reward function."""
        return compute_total_reward(self)

    def _get_dones(self):
        """Episode termination logic."""
        _, _, surface_xy, _ = self.get_sphere_distances_from_physics()

        min_success_steps = 5
        min_collision_steps = 10

        success_thresh = getattr(self, "_success_threshold", 0.03)
        raw_success = surface_xy < success_thresh
        success = raw_success & (self.episode_length_buf >= min_success_steps)

        robot_pos_global = self.robot.data.root_pos_w
        env_origins = self.scene.env_origins
        robot_pos_local = robot_pos_global - env_origins

        hx = float(self._arena_half_x)
        hy = float(self._arena_half_y)

        out_of_bounds = (
            (robot_pos_local[:, 0].abs() > hx) |
            (robot_pos_local[:, 1].abs() > hy)
        )

        lin_vel = self.robot.data.root_lin_vel_w
        speed = torch.norm(lin_vel[:, :2], dim=-1)

        static_root_pos = self.goal_positions
        diff = robot_pos_global - static_root_pos
        dx = diff[:, 0]
        dy = diff[:, 1]

        static_half_len = 0.5 * self._static_body_length
        static_half_wid = 0.5 * self._static_body_width
        active_half_len = 0.5 * self._active_body_length
        active_half_wid = 0.5 * self._active_body_width

        boxes_overlap = (
            (dx.abs() < (static_half_len + active_half_len)) &
            (dy.abs() < (static_half_wid + active_half_wid))
        )

        collision = (
            boxes_overlap &
            (speed > 0.4) &
            ~raw_success &
            (self.episode_length_buf >= min_collision_steps)
        )

        self._last_success = success
        terminated = success | out_of_bounds | collision
        time_out = self.episode_length_buf >= self.max_episode_length

        if success.any():
            # print(f"[SUCCESS] {int(success.sum().item())} dockings!")
            pass
            pass

        return terminated, time_out

    def _reset_idx(self, env_ids):
        """Reset environments."""
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()

        num_envs = self.scene.cfg.num_envs
        if self.prev_distance is None:
            self.prev_distance = torch.zeros(num_envs, device=self.device)
        if self.prev_actions is None:
            self.prev_actions = torch.zeros((num_envs, 2), device=self.device)
        if self.step_count is None:
            self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

        self.prev_actions[env_ids] = 0.0
        self.step_count[env_ids] = 0

        if self.frame_counts is not None:
            self.frame_counts[env_ids] = 0

        reset_environment_curriculum(self, env_ids)

        _, _, surface_xy, _ = self.get_sphere_distances_from_physics()
        self.prev_distance[env_ids] = surface_xy[env_ids]

    def set_curriculum_level(self, level: int):
        """Set curriculum level."""
        set_curriculum_level(self, level)

    def get_episode_statistics(self):
        """Collect episode statistics."""
        return collect_episode_stats(self)
