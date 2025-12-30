# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment - TiledCamera + Frame Stacking (v9.3 - SUCCESS FLAG FIX)
=========================================================================
Key features:
- TiledCamera for efficient batched rendering (single GPU pass)
- 84x84 grayscale observations with 4-frame stacking
- Supports multi-stage curriculum (28 stages)
- Asymmetric actor-critic (vision + privileged state)
- Optimized for many parallel environments

CRITICAL FIX (v9.3):
- Added _last_success tracking in _get_dones() BEFORE reset happens
- This fixes SSR calculation which was broken (3-7% instead of 80%+)

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations

import numpy as np
import torch

from omni.usd import get_context
from pxr import Sdf, UsdGeom, UsdLux, Gf, UsdPhysics

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera

from .teko_env_cfg import TekoEnvCfg
from .rewards.reward_functions import compute_total_reward
from .curriculum.curriculum_manager import (
    reset_environment_curriculum,
    set_curriculum_level,
)
from .utils.logging_utils import collect_episode_stats
from .robots.teko_static import TEKOStatic


class TekoEnv(DirectRLEnv):
    """
    Torque-driven TEKO environment with TiledCamera and curriculum learning.
    """

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        # Camera resolution from TiledCamera config
        self._cam_width = int(cfg.tiled_camera.width)
        self._cam_height = int(cfg.tiled_camera.height)

        # Keep a local dt for sensor updates (robust across Isaac Lab versions)
        self._dt = float(cfg.sim.dt)

        # Frame stacking configuration
        self.num_frame_stack = int(cfg.num_frame_stack)
        self.frame_stack: torch.Tensor | None = None
        self.frame_counts: torch.Tensor | None = None

        # Torque scaling
        self._max_wheel_torque = float(cfg.max_wheel_torque)

        # Arena limits
        self._arena_half_x = float(cfg.arena_half_x)
        self._arena_half_y = float(cfg.arena_half_y)

        # Body footprints
        self._active_body_length = float(cfg.active_body_length)
        self._active_body_width = float(cfg.active_body_width)
        self._static_body_length = float(cfg.static_body_length)
        self._static_body_width = float(cfg.static_body_width)

        # Placeholders
        self.actions: torch.Tensor | None = None
        self.dof_idx: torch.Tensor | None = None
        self.tiled_camera: TiledCamera | None = None
        self.goal_positions: torch.Tensor | None = None
        self.num_agents = 1
        self._polarity: torch.Tensor | None = None

        # Curriculum
        self.curriculum_level = 0

        # State tracking
        self.prev_distance: torch.Tensor | None = None
        self.prev_actions: torch.Tensor | None = None
        self.step_count: torch.Tensor | None = None

        # ============================================================
        # CRITICAL: Success flag tracking (must be set BEFORE reset)
        # ============================================================
        self._last_success: torch.Tensor | None = None

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

    # ================================================================
    # OBSERVATION SPACE
    # ================================================================
    def _init_observation_space(self):
        """Define the observation space (stack of K grayscale frames)."""
        import gymnasium as gym

        frame_shape = (self.num_frame_stack, self._cam_height, self._cam_width)

        obs_dict = {
            "rgb": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=frame_shape,
                dtype=np.float32,
            )
        }

        # Add privileged state if asymmetric critic enabled
        if getattr(self.cfg, "asymmetric_critic", False):
            obs_dict["privileged"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7,),
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(obs_dict)
        print(
            f"[INFO] Observation space: rgb={frame_shape} "
            f"(K={self.num_frame_stack} grayscale frames), range [0, 1]"
        )
        if getattr(self.cfg, "asymmetric_critic", False):
            print("[INFO] Privileged state: (7,) [dx, dy, dz, yaw_err, vx, vy, w]")

    # ================================================================
    # Scene setup
    # ================================================================
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

        # TiledCamera (single instance for all environments)
        self._setup_tiled_camera()

        # Cache static robot positions
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
        print("[INFO] Global lighting setup complete.")

    def _spawn_ground_plane(self, stage, env_idx: int):
        """Create a static ground plane inside env_{idx}."""
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
        """Setup arena, ground, and static goal robot for each environment."""
        num_envs = int(self.scene.cfg.num_envs)
        arena_usd = "/workspace/teko/documents/CAD/USD/stage_arena.usd"

        for env_idx in range(num_envs):
            env_path = f"/World/envs/env_{env_idx}"

            # Arena
            try:
                arena_prim = stage.DefinePrim(f"{env_path}/Arena", "Xform")
                arena_prim.GetReferences().AddReference(arena_usd)
            except Exception as e:
                print(f"[WARN] Arena failed for env_{env_idx}: {e}")

            # Ground
            self._spawn_ground_plane(stage, env_idx)

            # Optional: enforce consistent starting transform for the active robot prim if it exists
            robot_prim = stage.GetPrimAtPath(f"{env_path}/Robot")
            if robot_prim.IsValid():
                xf_robot = UsdGeom.Xformable(robot_prim)
                xf_robot.ClearXformOpOrder()
                xf_robot.AddTranslateOp().Set(Gf.Vec3d(0.3, 0.0, 0.4))
                xf_robot.AddRotateZOp().Set(180.0)

            # Static goal robot
            try:
                TEKOStatic(prim_path=f"{env_path}/RobotGoal")
                if env_idx % 50 == 0:
                    print(f"[INFO] Spawned static TEKO goals... (env_{env_idx})")
            except Exception as e:
                print(f"[WARN] Failed to create static TEKO goal in env_{env_idx}: {e}")

        print(f"[INFO] Created {num_envs} environments.")

    def _setup_tiled_camera(self):
        """Setup TiledCamera for efficient batched rendering."""
        self.tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self.tiled_camera

        print(
            "[INFO] TiledCamera initialized: "
            f"{int(self.scene.cfg.num_envs)} cameras @ {self._cam_width}x{self._cam_height} "
            "(single batched render pass)"
        )

    def _cache_goal_transforms(self):
        """Precompute goal positions (world frame)."""
        num_envs = int(self.scene.cfg.num_envs)
        self.goal_positions = torch.zeros((num_envs, 3), device=self.device)

        for env_idx, origin in enumerate(self.scene.env_origins):
            local_goal = torch.tensor([1.0, 0.0, 0.40], device=self.device)
            self.goal_positions[env_idx] = origin + local_goal

        print(f"[INFO] Cached {num_envs} goal positions.")

    # ------------------------------------------------------------------
    # Quaternion rotation helper
    # ------------------------------------------------------------------
    def _rotate_vector_by_quat(self, vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """
        Rotate a vector by a quaternion.

        Args:
            vec: [3] local offset vector OR [N,3]
            quat: [N,4] quaternions in (w, x, y, z) format (Isaac Lab convention)

        Returns:
            [N,3] rotated vectors in world frame
        """
        num_envs = quat.shape[0]

        if vec.dim() == 1:
            vec = vec.unsqueeze(0).expand(num_envs, 3)

        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]

        # t = 2 * cross(q.xyz, v)
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)

        # v' = v + w * t + cross(q.xyz, t)
        rx = vx + qw * tx + (qy * tz - qz * ty)
        ry = vy + qw * ty + (qz * tx - qx * tz)
        rz = vz + qw * tz + (qx * ty - qy * tx)

        return torch.stack([rx, ry, rz], dim=-1)

    # ------------------------------------------------------------------
    # Sphere distance computation
    # ------------------------------------------------------------------
    def get_sphere_distances_from_physics(self):
        """
        Compute distance between male/female connector spheres.

        Female offset is rotated by the active robot orientation.
        """
        if self.goal_positions is None:
            raise RuntimeError("Goal positions not cached yet.")

        # Female connector offset in robot LOCAL frame
        female_offset_local = torch.tensor([-0.24, 0.0, -0.08], device=self.device)

        # Male connector offset for the static goal robot (goal does not rotate)
        male_offset = torch.tensor([0.22667, -0.00144, -0.08815], device=self.device)

        active_pos = self.robot.data.root_pos_w           # [N, 3]
        active_quat = self.robot.data.root_quat_w         # [N, 4] (w,x,y,z)
        static_pos = self.goal_positions                  # [N, 3]

        female_offset_world = self._rotate_vector_by_quat(female_offset_local, active_quat)

        female_pos = active_pos + female_offset_world
        male_pos = static_pos + male_offset.unsqueeze(0).expand(static_pos.shape[0], 3)

        diff = female_pos - male_pos
        dist_3d = torch.norm(diff, dim=-1)
        dist_xy = torch.norm(diff[:, :2], dim=-1)

        # Surface-to-surface distance (accounting for sphere radii)
        r_female = 0.005
        r_male = 0.005
        surface_3d = torch.clamp(dist_3d - (r_female + r_male), min=0.0)
        surface_xy = torch.clamp(dist_xy - (r_female + r_male), min=0.0)

        return female_pos, male_pos, surface_xy, surface_3d

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _lazy_init_articulation(self):
        """Initialize joint indices once robot is loaded."""
        if self.dof_idx is not None or getattr(self.robot, "root_physx_view", None) is None:
            return

        name_to_idx = {n: i for i, n in enumerate(self.robot.joint_names)}
        indices = [name_to_idx[n] for n in self.cfg.dof_names if n in name_to_idx]
        if not indices:
            raise RuntimeError(f"No valid DOF names found. Robot joints: {self.robot.joint_names}")

        self.dof_idx = torch.tensor(indices, dtype=torch.long, device=self.device)
        print(f"[INFO] DOF indices: {self.dof_idx.tolist()}")

        if self._polarity is None:
            self._polarity = torch.tensor(self.cfg.wheel_polarity, device=self.device).unsqueeze(0)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        """Convert RL actions into wheel torques."""
        if self.dof_idx is None or self.actions is None:
            return

        num_envs = int(self.scene.cfg.num_envs)
        v_cmd = self.actions[:, 0]
        w_cmd = self.actions[:, 1]

        # Clamp incoming commands defensively (env saturation should match policy)
        v_cmd = torch.clamp(v_cmd, -1.0, 1.0)
        w_cmd = torch.clamp(w_cmd, -1.0, 1.0)

        # Simple differential drive mixing
        k = 0.5
        left = torch.clamp(v_cmd - k * w_cmd, -1.0, 1.0)
        right = torch.clamp(v_cmd + k * w_cmd, -1.0, 1.0)

        torque_targets = torch.stack([left, right, left, right], dim=1) * self._max_wheel_torque
        torque_targets = torque_targets * self._polarity

        env_ids = torch.arange(num_envs, device=self.device)
        self.robot.set_joint_effort_target(torque_targets, env_ids=env_ids, joint_ids=self.dof_idx)

    # ------------------------------------------------------------------
    # Observations (TiledCamera + Frame Stacking)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """
        Capture grayscale frames from TiledCamera and build a K-frame stack.
        """
        if self.tiled_camera is None:
            raise RuntimeError("TiledCamera not initialized.")
        if self.goal_positions is None:
            raise RuntimeError("Goal positions not cached yet.")

        num_envs = int(self.scene.cfg.num_envs)
        h, w = self._cam_height, self._cam_width

        # Ensure the sensor refreshes its buffers on this step (prevents stale/empty frames)
        try:
            self.tiled_camera.update(dt=self._dt)
        except TypeError:
            # Some versions expose update() without dt
            self.tiled_camera.update()

        # Allocate frame buffers once
        if self.frame_stack is None:
            self.frame_stack = torch.zeros(
                (num_envs, self.num_frame_stack, h, w),
                device=self.device,
                dtype=torch.float16,
            )
        if self.frame_counts is None:
            self.frame_counts = torch.zeros(num_envs, device=self.device, dtype=torch.int32)

        # Read camera output
        rgb_data = self.tiled_camera.data.output.get("rgb", None)

        # Build current grayscale frame (float16 in [0,1])
        if rgb_data is None or rgb_data.numel() == 0:
            gray_current = torch.zeros((num_envs, h, w), device=self.device, dtype=torch.float16)
        else:
            # Expected shapes:
            # - [N, H, W, 4] uint8 (RGBA)
            # - [N, H, W, 3] uint8 (RGB)
            # - Sometimes float already in [0,1]
            if rgb_data.shape[-1] == 4:
                rgb_data = rgb_data[..., :3]

            rgb = rgb_data

            if rgb.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                rgb = rgb.float() / 255.0
            else:
                rgb = rgb.float()

            # Convert RGB -> grayscale
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            gray_current = torch.clamp(gray, 0.0, 1.0).to(torch.float16)

            # Defensive: if N mismatch, pad/trim (rare, but prevents crashes)
            if gray_current.shape[0] != num_envs:
                n = min(gray_current.shape[0], num_envs)
                tmp = torch.zeros((num_envs, h, w), device=self.device, dtype=torch.float16)
                tmp[:n] = gray_current[:n]
                gray_current = tmp

        # Frame stacking
        reset_mask = self.frame_counts == 0
        if reset_mask.any():
            idx = reset_mask.nonzero(as_tuple=False).squeeze(-1)
            self.frame_stack[idx] = gray_current[idx].unsqueeze(1).expand(-1, self.num_frame_stack, -1, -1)
            self.frame_counts[idx] = self.num_frame_stack

        non_reset_mask = ~reset_mask
        if non_reset_mask.any():
            idx = non_reset_mask.nonzero(as_tuple=False).squeeze(-1)
            # In-place shift to avoid extra allocations
            self.frame_stack[idx, :-1].copy_(self.frame_stack[idx, 1:])
            self.frame_stack[idx, -1].copy_(gray_current[idx])

        stacked = self.frame_stack.float()  # return float32 to the agent

        # Privileged state for asymmetric critic
        if getattr(self.cfg, "asymmetric_critic", False):
            robot_pos = self.robot.data.root_pos_w
            goal_pos = self.goal_positions
            robot_quat = self.robot.data.root_quat_w
            robot_vel = self.robot.data.root_lin_vel_w
            robot_angvel = self.robot.data.root_ang_vel_w

            diff = goal_pos - robot_pos
            dx, dy, dz = diff[:, 0], diff[:, 1], diff[:, 2]

            robot_yaw = self._extract_yaw(robot_quat)
            vec_to_goal = goal_pos - robot_pos
            goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])

            # Rear-facing yaw (robot is docking with its rear)
            rear_yaw = robot_yaw + torch.pi
            yaw_error = torch.atan2(torch.sin(rear_yaw - goal_yaw), torch.cos(rear_yaw - goal_yaw))

            privileged = torch.stack(
                [dx, dy, dz, yaw_error, robot_vel[:, 0], robot_vel[:, 1], robot_angvel[:, 2]],
                dim=-1,
            )

            return {"rgb": stacked, "privileged": privileged}

        return {"rgb": stacked}

    def _extract_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw from quaternion (w, x, y, z)."""
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return torch.atan2(siny_cosp, cosy_cosp)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self):
        """Delegate reward computation to the reward module."""
        return compute_total_reward(self)

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self):
        """Episode termination logic."""
        _, _, surface_xy, _ = self.get_sphere_distances_from_physics()

        min_success_steps = 5
        min_collision_steps = 10

        raw_success = surface_xy < 0.03
        success = raw_success & (self.episode_length_buf >= min_success_steps)

        # ============================================================
        # CRITICAL FIX: Store success flags BEFORE reset happens
        # This allows train_optuna_v3.py to read correct success status
        # ============================================================
        if self._last_success is None:
            self._last_success = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        self._last_success.copy_(success)

        # Out-of-bounds based on arena limits in local env coordinates
        robot_pos_global = self.robot.data.root_pos_w
        env_origins = self.scene.env_origins
        robot_pos_local = robot_pos_global - env_origins

        hx = float(self._arena_half_x)
        hy = float(self._arena_half_y)

        out_of_bounds = (robot_pos_local[:, 0].abs() > hx) | (robot_pos_local[:, 1].abs() > hy)

        # Collision proxy: AABB overlap + high speed + not already successful
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

        boxes_overlap = (dx.abs() < (static_half_len + active_half_len)) & (dy.abs() < (static_half_wid + active_half_wid))

        collision = (
            boxes_overlap
            & (speed > 0.4)
            & ~raw_success
            & (self.episode_length_buf >= min_collision_steps)
        )

        terminated = success | out_of_bounds | collision
        time_out = self.episode_length_buf >= self.max_episode_length

        return terminated, time_out

    # ------------------------------------------------------------------
    # Public accessor for success flags (used by train_optuna_v3.py)
    # ------------------------------------------------------------------
    def get_last_success(self) -> torch.Tensor:
        """
        Return the success flags from the last _get_dones() call.
        This is called by the training script AFTER step() but BEFORE
        the environment state is read again.
        """
        if self._last_success is None:
            # Fallback: return all False
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._last_success

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        """Reset selected environments."""
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()

        num_envs = int(self.scene.cfg.num_envs)
        if self.prev_distance is None:
            self.prev_distance = torch.zeros(num_envs, device=self.device)
        if self.prev_actions is None:
            self.prev_actions = torch.zeros((num_envs, 2), device=self.device)
        if self.step_count is None:
            self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

        self.prev_actions[env_ids] = 0.0
        self.step_count[env_ids] = 0

        # Force frame stack re-initialization for these envs
        if self.frame_counts is not None:
            self.frame_counts[env_ids] = 0

        # Curriculum reset
        reset_environment_curriculum(self, env_ids)

        # Cache starting distance (for progress reward etc.)
        _, _, surface_xy, _ = self.get_sphere_distances_from_physics()
        self.prev_distance[env_ids] = surface_xy[env_ids]

    # ------------------------------------------------------------------
    # Curriculum helpers
    # ------------------------------------------------------------------
    def set_curriculum_level(self, level: int):
        """Set curriculum level."""
        set_curriculum_level(self, level)

    def get_episode_statistics(self):
        """Collect episode statistics."""
        return collect_episode_stats(self)