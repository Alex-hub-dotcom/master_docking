# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment - Curriculum Compatible (v8.4 for 84×84 GRAYSCALE)
===================================================================
Changes for 84px upgrade:
- Proper grayscale observation pipeline (K × 84 × 84)
- Proper observation_space override (previous indentation bug fixed)
- Frame stacking cleaned for grayscale (no more RGB leftovers)
- Camera resized correctly to 84×84 from Isaac RGB
- Fully compatible with SimpleCNN v10.0
"""

from __future__ import annotations
import math
import numpy as np
import torch
from omni.usd import get_context
from pxr import Sdf, UsdGeom, UsdLux, Gf, UsdPhysics
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.sensors import Camera, CameraCfg

from .teko_env_cfg import TekoEnvCfg
from .rewards.reward_functions import compute_total_reward
from .curriculum.curriculum_manager import (
    reset_environment_curriculum,
    set_curriculum_level,
)
from .utils.logging_utils import collect_episode_stats
from .robots.teko_static import TEKOStatic


class TekoEnv(DirectRLEnv):

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):

        # --- CAMERA RESOLUTION (84 × 84) ---
        self._cam_res = (cfg.camera.width, cfg.camera.height)  # (84, 84)

        # Frame stacking: K=4 grayscale frames
        self.num_frame_stack = getattr(cfg, "num_frame_stack", 4)
        self.frame_stack = None
        self.frame_counts = None

        # Torque params
        self._max_wheel_torque = cfg.max_wheel_torque

        # Arena limits
        self._arena_half_x = float(cfg.arena_half_x)
        self._arena_half_y = float(cfg.arena_half_y)

        # Rect footprints
        self._active_body_length = float(cfg.active_body_length)
        self._active_body_width = float(cfg.active_body_width)
        self._static_body_length = float(cfg.static_body_length)
        self._static_body_width = float(cfg.static_body_width)

        # Init placeholders
        self.actions = None
        self.dof_idx = None
        self.cameras = []
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

        # Episode stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []

        # Loggable reward components
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
    # PROPER OBSERVATION SPACE OVERRIDE  (IMPORTANT FIX)
    # ================================================================
    def _init_observation_space(self):
        """Define observation space for 4 grayscale frames of size 84×84."""
        import gymnasium as gym

        K = self.num_frame_stack
        H = self.cfg.camera.height
        W = self.cfg.camera.width

        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(K, H, W),
                    dtype=np.float32,
                )
            }
        )

        print(f"[INFO] Observation space = ({K}, {H}, {W}) grayscale")

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

        # Clone rest
        self.scene.clone_environments(copy_from_source=True)

        # Arena, goal robot, ground plane
        self._setup_per_environment_assets(stage)

        # Cameras
        self._setup_cameras()

        # Cache static robot positions
        self._cache_goal_transforms()

    # ================================================================
    # Camera Extraction + Grayscale + Frame Stacking
    # ================================================================
    def _get_observations(self):
        import torch.nn.functional as F

        num_envs = self.scene.cfg.num_envs
        H, W = self._cam_res[1], self._cam_res[0]

        # Allocate if needed
        if self.frame_stack is None:
            self.frame_stack = torch.zeros(
                (num_envs, self.num_frame_stack, 1, H, W),
                device=self.device,
                dtype=torch.float32,
            )
        if self.frame_counts is None:
            self.frame_counts = torch.zeros(num_envs, device=self.device, dtype=torch.int32)

        gray_current = torch.zeros((num_envs, 1, H, W), device=self.device)

        # -- Extract grayscale frame for each env
        for env_idx, cam in enumerate(self.cameras):
            cam.update(dt=0.0)
            rgb_data = cam.data.output["rgb"]
            if rgb_data is None or rgb_data.numel() == 0:
                continue

            if rgb_data.ndim == 4:
                rgb_data = rgb_data.squeeze(0)
            if rgb_data.shape[-1] == 4:
                rgb_data = rgb_data[..., :3]

            # Normalize to [0,1]
            rgb = rgb_data.permute(2, 0, 1).float() / 255.0

            # Resize if needed (Isaac sometimes gives 1280×720)
            if rgb.shape[1] != H or rgb.shape[2] != W:
                rgb = F.interpolate(
                    rgb.unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # Convert to grayscale
            gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            gray_current[env_idx] = gray.unsqueeze(0)

        # -- Frame Stack Logic
        reset_mask = self.frame_counts == 0
        if reset_mask.any():
            idx = reset_mask.nonzero(as_tuple=False).squeeze(-1)
            # Fill entire stack with first frame
            self.frame_stack[idx, :, :, :, :] = gray_current[idx].unsqueeze(1)
            self.frame_counts[idx] = self.num_frame_stack

        non_reset_mask = ~reset_mask
        if non_reset_mask.any():
            idx = non_reset_mask.nonzero(as_tuple=False).squeeze(-1)
            # Shift
            self.frame_stack[idx, :-1] = self.frame_stack[idx, 1:].clone()
            # Insert newest frame
            self.frame_stack[idx, -1] = gray_current[idx]

        # Remove channel dim (1) → final shape (N, K, 84, 84)
        stacked = self.frame_stack.squeeze(2)
        return {"rgb": stacked}

    # ================================================================
    # Rewards / Dones unchanged
    # ================================================================
    def _get_rewards(self):
        return compute_total_reward(self)

    def _get_dones(self):
        # unchanged
        # ...
        return terminated, time_out

    # ================================================================
    # Reset
    # ================================================================
    def _reset_idx(self, env_ids):
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

    # Curriculum
    def set_curriculum_level(self, level: int):
        set_curriculum_level(self, level)

    def get_episode_statistics(self):
        return collect_episode_stats(self)
