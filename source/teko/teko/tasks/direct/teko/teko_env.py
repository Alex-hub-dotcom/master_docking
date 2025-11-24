# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment - Curriculum Compatible
========================================
- Supports multi-stage curriculum
- Nuclear penalties (-500 collision/boundary)
- Survival bonus (+0.3/step)
- Anti-crash exploit (min_collision_steps=10)
- Frame stacking: returns stacked RGB frames [3 * K, H, W]

This environment was created for the TEKO vision-based docking project.

Author: Alexandre Schleier Neves da Silva
For questions or collaboration, feel free to contact:
    alexandre.schleiernevesdasilva@uni-hohenheim.de
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
    """
    Torque-driven TEKO environment with curriculum.

    This class connects:
    - the TEKO robot model,
    - the static goal robot,
    - the RGB camera,
    - and the reward / curriculum logic
    into a reinforcement learning environment.
    """

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        # Camera resolution (used for observation tensor shape)
        self._cam_res = (cfg.camera.width, cfg.camera.height)

        # Frame stacking configuration
        # If cfg does not define num_frame_stack, default to 4
        self.num_frame_stack = getattr(cfg, "num_frame_stack", 4)
        self.frame_stack = None  # will hold [num_envs, K, 3, H, W] on device
        self.frame_counts = None  # used to reinitialize stack after resets

        # Torque scaling for the wheels
        self._max_wheel_torque = cfg.max_wheel_torque

        # Arena logical half-extents (for OOB + debug walls)
        self._arena_half_x = float(cfg.arena_half_x)
        self._arena_half_y = float(cfg.arena_half_y)

        # Rectangular body footprints (meters)
        self._active_body_length = float(cfg.active_body_length)
        self._active_body_width = float(cfg.active_body_width)
        self._static_body_length = float(cfg.static_body_length)
        self._static_body_width = float(cfg.static_body_width)

        # These will be initialized later
        self.actions = None
        self.dof_idx = None
        self.cameras = []
        self.goal_positions = None
        self.num_agents = 1
        self._polarity = None  # wheel direction (left/right) will be cached here

        # Curriculum learning (stage index; actual logic in curriculum_manager)
        self.curriculum_level = 0

        # State tracking (used by reward functions)
        self.prev_robot_pos = None
        self.prev_distance = None
        self.prev_actions = None
        self.step_count = None

        # Episode statistics (for logging and analysis)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.reward_components = {
            "distance": [],
            "progress": [],
            "alignment": [],
            "velocity_penalty": [],
            "oscillation_penalty": [],
            "collision_penalty": [],
            "wall_penalty": [],
        }

        # Call Isaac Lab base class constructor
        super().__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        stage = get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage not initialized")

        # Root Xform for the first environment (others will be cloned)
        if not stage.GetPrimAtPath("/World/envs/env_0"):
            stage.DefinePrim("/World/envs/env_0", "Xform")

        self._setup_global_lighting(stage)

        # Active robot (articulation)
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Clone environments (env_1, env_2, ...) from env_0
        self.scene.clone_environments(copy_from_source=True)

        # Arena, static robot, etc.
        self._setup_per_environment_assets(stage)

        # Cameras and goal positions
        self._setup_cameras()
        self._cache_goal_transforms()

    def _init_observation_space(self):
        """Define the observation space (here: stacked RGB images)."""
        import gymnasium as gym

        num_channels = 3 * self.num_frame_stack
        frame_shape = (num_channels, self.cfg.camera.height, self.cfg.camera.width)
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=frame_shape,
                    dtype=np.float32,
                )
            }
        )
        print(
            f"[INFO] Observation space set to {frame_shape} (frame stack K={self.num_frame_stack}), range [0, 1]"
        )

    def _setup_global_lighting(self, stage):
        """Simple dome + sun lighting to make the scene visible on camera."""
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
        """
        Create a simple static ground plane inside env_{idx}.

        - Centered at (0, 0, floor_z) in local env coordinates.
        - Size matches the logical arena (±arena_half_x, ±arena_half_y).
        - Has collision enabled via UsdPhysics.CollisionAPI.
        """
        env_root = f"/World/envs/env_{env_idx}"
        ground_path = f"{env_root}/Ground"

        cube = UsdGeom.Cube.Define(stage, Sdf.Path(ground_path))
        xf = UsdGeom.Xformable(cube)
        xf.ClearXformOpOrder()

        floor_z = 0.185     # height of arena floor
        thickness = 0.02    # 2 cm thick slab

        # Place at center of env, at given Z
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, floor_z))

        # Match arena extents
        hx = float(self._arena_half_x)
        hy = float(self._arena_half_y)
        # Cube's base size is 2, so scale uses half-lengths
        xf.AddScaleOp().Set(Gf.Vec3d(hx, hy, thickness * 0.5))

        # Enable collision (static collider — no rigid body)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        # Neutral color (you can hide it later if needed)
        UsdGeom.Gprim(cube).CreateDisplayColorAttr([Gf.Vec3f(0.4, 0.4, 0.4)])
        #UsdGeom.Gprim(cube).CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.8, 0.2)]) #for debugging use color for visual assistance

        print(f"[DEBUG] Spawned ground plane for env_{env_idx} at z={floor_z}")

    def _setup_per_environment_assets(self, stage):
        """
        For each environment (env_0, env_1, ...) we:
        - add the arena,
        - create a dedicated ground plane,
        - position the active robot,
        - spawn the static docking target with ArUco marker,
        - optionally spawn debug visualization (arena boundaries, robot body boxes).
        """
        num_envs = self.scene.cfg.num_envs
        ARENA_USD_PATH = "/workspace/teko/documents/CAD/USD/stage_arena.usd"
        ARUCO_IMG_PATH = "/workspace/teko/documents/Aruco/test_marker.png"

        for env_idx in range(num_envs):
            env_path = f"/World/envs/env_{env_idx}"

            # Arena visual mesh
            try:
                arena_prim = stage.DefinePrim(f"{env_path}/Arena", "Xform")
                arena_prim.GetReferences().AddReference(ARENA_USD_PATH)
            except Exception as e:
                print(f"[WARN] Arena failed for env_{env_idx}: {e}")

            # Dedicated ground collider matching arena extents
            self._spawn_ground_plane(stage, env_idx)

            # Active robot start pose (approximate)
            robot_prim = stage.GetPrimAtPath(f"{env_path}/Robot")
            if robot_prim.IsValid():
                xf_robot = UsdGeom.Xformable(robot_prim)
                xf_robot.ClearXformOpOrder()
                xf_robot.AddTranslateOp().Set(Gf.Vec3d(0.3, 0.0, 0.4))
                xf_robot.AddRotateZOp().Set(180.0)

            # Static goal robot
            try:
                TEKOStatic(
                    prim_path=f"{env_path}/RobotGoal",
                    #aruco_path=ARUCO_IMG_PATH,             # <- disable marker
                )
                print(f"[INFO] Spawned static TEKO goal in env_{env_idx}")
            except Exception as e:
                print(
                    f"[WARN] Failed to create static TEKO goal in env_{env_idx}: {e}"
                )

            # Debug visualization: arena boundaries + robot body boxes
            if getattr(self.cfg, "debug_boundaries", False):
                self._spawn_arena_boundaries(stage, env_idx)
            if getattr(self.cfg, "debug_robot_boxes", False):
                self._spawn_robot_debug_boxes(stage, env_idx)

        print(f"[INFO] Created {num_envs} environments.")

    def _setup_cameras(self):
        """Attach one RGB camera to the back of the active robot in each env."""
        for env_idx in range(self.scene.cfg.num_envs):
            cam_path = (
                f"/World/envs/env_{env_idx}/Robot/teko_urdf/TEKO_Body/"
                "TEKO_WallBack/TEKO_Camera/RearCamera"
            )

            cam_cfg = CameraCfg(
                prim_path=cam_path,
                update_period=0,
                height=self._cam_res[1],
                width=self._cam_res[0],
                data_types=["rgb"],
                spawn=None,
            )

            camera = Camera(cfg=cam_cfg)
            self.cameras.append(camera)

        print(f"[INFO] Initialized {len(self.cameras)} cameras.")

    def _cache_goal_transforms(self):
        """
        Precompute the global positions of the docking target
        for each environment. Used by reward and reset code.
        """
        num_envs = self.scene.cfg.num_envs
        self.goal_positions = torch.zeros((num_envs, 3), device=self.device)
        for env_idx, origin in enumerate(self.scene.env_origins):
            # The goal is always 1 m in +x and 0.40 m high in local env frame
            local_goal = torch.tensor([1.0, 0.0, 0.40], device=self.device)
            self.goal_positions[env_idx] = origin + local_goal
        print(f"[INFO] Cached {num_envs} goal positions.")

    # ------------------------------------------------------------------
    # Debug visualization: arena boundaries + robot body boxes
    # ------------------------------------------------------------------
    def _spawn_arena_boundaries(self, stage, env_idx: int):
        """
        Draw thin red 'walls' at the logical arena limits used in _get_dones.

        These use the same half-extents as the out-of-bounds check:

            |x| > arena_half_x  -> out of bounds
            |y| > arena_half_y  -> out of bounds

        Visual-only: no rigid body, no collision.
        """
        env_path = f"/World/envs/env_{env_idx}/Debug"
        stage.DefinePrim(env_path, "Xform")

        hx = float(self._arena_half_x)
        hy = float(self._arena_half_y)

        def make_wall(name: str, pos, scale):
            prim_path = f"{env_path}/{name}"
            cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
            xf = UsdGeom.Xformable(cube)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xf.AddScaleOp().Set(Gf.Vec3d(*scale))
            UsdGeom.Gprim(cube).CreateDisplayColorAttr(
                [Gf.Vec3f(1.0, 0.0, 0.0)]
            )

        # Slightly above ground so they are visible
        z = 0.4

        # X walls (along Y, at x = ±hx)
        make_wall(
            "Boundary_Xmin",
            pos=(-hx, 0.0, z),
            scale=(0.02, 2.0 * hy, 0.01),
        )
        make_wall(
            "Boundary_Xmax",
            pos=(hx, 0.0, z),
            scale=(0.02, 2.0 * hy, 0.01),
        )

        # Y walls (along X, at y = ±hy)
        make_wall(
            "Boundary_Ymin",
            pos=(0.0, -hy, z),
            scale=(2.0 * hx, 0.02, 0.01),
        )
        make_wall(
            "Boundary_Ymax",
            pos=(0.0, hy, z),
            scale=(2.0 * hx, 0.02, 0.01),
        )

        print(f"[DEBUG] Spawned arena boundaries for env_{env_idx}")

    def _spawn_robot_debug_boxes(self, stage, env_idx: int):
        """
        Attach rectangular debug boxes to TEKO_Body of active/static robots.
        """
        env_root = f"/World/envs/env_{env_idx}"
        active_root = f"{env_root}/Robot/teko_urdf/TEKO_Body"
        static_root = f"{env_root}/RobotGoal/teko_urdf/TEKO_Body"

        # Active robot box (green)
        self._make_debug_box(
            stage=stage,
            parent_path=active_root,
            name="DebugChassisActive",
            length=self._active_body_length,
            width=self._active_body_width,
            color=Gf.Vec3f(0.0, 1.0, 0.0),
        )

        # Static robot box (red)
        self._make_debug_box(
            stage=stage,
            parent_path=static_root,
            name="DebugChassisStatic",
            length=self._static_body_length,
            width=self._static_body_width,
            color=Gf.Vec3f(1.0, 0.0, 0.0),
        )

        print(f"[DEBUG] Spawned robot debug boxes for env_{env_idx}")

    def _make_debug_box(
        self,
        stage,
        parent_path: str,
        name: str,
        length: float,
        width: float,
        color: Gf.Vec3f,
    ):
        """
        Helper to create a thin rectangular box (footprint) attached
        to a given parent prim (Robot or RobotGoal).
        """
        box_path = f"{parent_path}/{name}"
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(box_path))
        xf = UsdGeom.Xformable(cube)
        xf.ClearXformOpOrder()

        # Local center at (0, 0, z); we keep it low, close to chassis
        z_local = 0.05
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_local))

        # UsdGeom.Cube has size 2 by default, so scale is half-lengths
        half_len = 0.5 * float(length)
        half_wid = 0.5 * float(width)
        xf.AddScaleOp().Set(Gf.Vec3d(half_len, half_wid, 0.01))

        UsdGeom.Gprim(cube).CreateDisplayColorAttr([color])

    # ------------------------------------------------------------------
    # Sphere position computation (docking measurement)
    # ------------------------------------------------------------------
    def get_sphere_distances_from_physics(self):
        """
        Compute the distance between the male/female connector spheres.
        """
        FEMALE_OFFSET = torch.tensor([0.24, 0.0, -0.08], device=self.device)
        MALE_OFFSET = torch.tensor([0.22667, -0.00144, -0.08815], device=self.device)

        active_pos = self.robot.data.root_pos_w
        static_pos = self.goal_positions

        female_pos = active_pos + FEMALE_OFFSET.unsqueeze(0).expand(
            active_pos.shape[0], 3
        )
        male_pos = static_pos + MALE_OFFSET.unsqueeze(0).expand(
            static_pos.shape[0], 3
        )

        diff = female_pos - male_pos
        dist_3d = torch.norm(diff, dim=-1)
        dist_xy = torch.norm(diff[:, :2], dim=-1)

        R_FEMALE = 0.005
        R_MALE = 0.005
        surface_3d = torch.clamp(dist_3d - (R_FEMALE + R_MALE), min=0.0)
        surface_xy = torch.clamp(dist_xy - (R_FEMALE + R_MALE), min=0.0)

        return female_pos, male_pos, surface_xy, surface_3d

    # ------------------------------------------------------------------
    # Actions (Torque control)
    # ------------------------------------------------------------------
    def _lazy_init_articulation(self):
        """
        Initialize joint indices once we know the robot is fully loaded.
        """
        if self.dof_idx is not None or getattr(self.robot, "root_physx_view", None) is None:
            return

        # Map joint names -> indices
        name_to_idx = {n: i for i, n in enumerate(self.robot.joint_names)}
        indices = [name_to_idx[n] for n in self.cfg.dof_names if n in name_to_idx]
        if not indices:
            raise RuntimeError(
                f"No valid DOF names found: {self.robot.joint_names}"
            )
        self.dof_idx = torch.tensor(indices, dtype=torch.long, device=self.device)
        print(f"[INFO] DOF indices: {self.dof_idx}")

        # Cache wheel polarity tensor on the correct device (only once)
        if self._polarity is None:
            self._polarity = torch.tensor(
                self.cfg.wheel_polarity, device=self.device
            ).unsqueeze(0)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Store actions for this step and ensure articulation is initialized
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        """
        Convert RL actions into wheel torques.

        Action space: [v_cmd, w_cmd] in [-1, 1]
          v_cmd -> forward/backward command
          w_cmd -> turning command
        """
        if self.dof_idx is None or self.actions is None:
            return

        num_envs = self.scene.cfg.num_envs

        # actions in [-1, 1]: [v_cmd, w_cmd]
        v_cmd = self.actions[:, 0]
        w_cmd = self.actions[:, 1]

        # Scaling factors for linear and angular commands
        v_max = 1.0
        w_max = 1.0

        v = v_cmd * v_max
        w = w_cmd * w_max

        # Differential-drive mapping: left/right from linear + angular components
        k = 0.5
        left = v - k * w
        right = v + k * w

        # Clamp before torque scaling so we stay in [-1, 1]
        left = torch.clamp(left, -1.0, 1.0)
        right = torch.clamp(right, -1.0, 1.0)

        # [FL, FR, BL, BR] torques (before polarity)
        torque_targets = (
            torch.stack([left, right, left, right], dim=1) * self._max_wheel_torque
        )

        # Apply wheel polarity (handles opposite rotation directions)
        torque_targets = torque_targets * self._polarity

        env_ids = torch.arange(num_envs, device=self.device)
        self.robot.set_joint_effort_target(
            torque_targets, env_ids=env_ids, joint_ids=self.dof_idx
        )

    # ------------------------------------------------------------------
    # Observations (with frame stacking)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """
        Capture RGB images from each camera, update the frame stack,
        and return stacked observations of shape
        [num_envs, 3 * num_frame_stack, H, W] in [0, 1].
        """
        import torch.nn.functional as F

        num_envs = self.scene.cfg.num_envs
        h, w = self._cam_res[1], self._cam_res[0]

        # Lazy initialization of frame stack buffers
        if self.frame_stack is None:
            self.frame_stack = torch.zeros(
                (num_envs, self.num_frame_stack, 3, h, w),
                device=self.device,
                dtype=torch.float32,
            )
        if self.frame_counts is None:
            self.frame_counts = torch.zeros(
                num_envs, device=self.device, dtype=torch.int32
            )

        # Current RGB frames for all envs: [num_envs, 3, H, W]
        rgb_current = torch.zeros((num_envs, 3, h, w), device=self.device)

        for env_idx, cam in enumerate(self.cameras):
            cam.update(dt=0.0)

            rgb_data = cam.data.output["rgb"]
            if rgb_data is None or rgb_data.numel() == 0:
                continue

            # Isaac Lab sometimes returns [1, H, W, C]
            if rgb_data.ndim == 4:
                rgb_data = rgb_data.squeeze(0)

            # Drop alpha channel if present (RGBA -> RGB)
            if rgb_data.shape[-1] == 4:
                rgb_data = rgb_data[..., :3]

            # Convert to [C, H, W] and normalize to [0, 1]
            rgb = rgb_data.permute(2, 0, 1).float() / 255.0

            # Make sure resolution is exactly (H, W)
            if rgb.shape[1] != h or rgb.shape[2] != w:
                rgb = F.interpolate(
                    rgb.unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            rgb_current[env_idx] = rgb

        # Env episodes that have just been reset:
        # first observation after reset repeats the same frame K times.
        reset_mask = self.frame_counts == 0
        if reset_mask.any():
            idx = reset_mask.nonzero(as_tuple=False).squeeze(-1)
            self.frame_stack[idx, :, :, :, :] = rgb_current[idx].unsqueeze(1)
            self.frame_counts[idx] = self.num_frame_stack

        # Non-reset envs: shift the stack and append the latest frame
        non_reset_mask = ~reset_mask
        if non_reset_mask.any():
            idx = non_reset_mask.nonzero(as_tuple=False).squeeze(-1)
            # Shift frames to the left and insert new one at the end
            self.frame_stack[idx, :-1] = self.frame_stack[idx, 1:].clone()
            self.frame_stack[idx, -1] = rgb_current[idx]

        # Collapse frame dimension into channels: [N, 3 * K, H, W]
        stacked_rgb = self.frame_stack.view(num_envs, 3 * self.num_frame_stack, h, w)

        return {"rgb": stacked_rgb}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self):
        """Delegate reward computation to an external function."""
        return compute_total_reward(self)

    # ------------------------------------------------------------------
    # Dones (NUCLEAR PENALTIES: -500)
    # ------------------------------------------------------------------
    def _get_dones(self):
        """
        Episode termination logic.

        Episode ends when:
        - docking success is achieved,
        - robot leaves the allowed arena (out of bounds),
        - a high-speed chassis collision (AABB overlap) occurs,
        - or the max episode length is reached.
        """
        _, _, surface_xy, _ = self.get_sphere_distances_from_physics()

        # Anti-exploit gates
        min_success_steps = 5
        min_collision_steps = 10  # helps avoid "crash immediately for reward"

        # Raw geometric success (within 3 cm of connector)
        raw_success = surface_xy < 0.03

        # Only count success as terminal after a few steps
        success = raw_success & (self.episode_length_buf >= min_success_steps)

        # ------------------------------------------------------------------
        # OUT OF BOUNDS (-500 penalty in rewards)
        # ------------------------------------------------------------------
        robot_pos_global = self.robot.data.root_pos_w
        env_origins = self.scene.env_origins
        robot_pos_local = robot_pos_global - env_origins

        hx = float(self._arena_half_x)
        hy = float(self._arena_half_y)

        out_of_bounds = (
            (robot_pos_local[:, 0].abs() > hx) |
            (robot_pos_local[:, 1].abs() > hy)
        )

        # ------------------------------------------------------------------
        # STATIC BODY COLLISION (AABB vs AABB, high speed)
        # ------------------------------------------------------------------
        lin_vel = self.robot.data.root_lin_vel_w
        speed = torch.norm(lin_vel[:, :2], dim=-1)

        # Static robot root positions (global)
        static_root_pos = self.goal_positions

        # Active center relative to static center (env axes)
        diff = robot_pos_global - static_root_pos
        dx = diff[:, 0]
        dy = diff[:, 1]

        # Static robot box half-sizes
        static_half_len = 0.5 * self._static_body_length
        static_half_wid = 0.5 * self._static_body_width

        # Active robot box half-sizes
        active_half_len = 0.5 * self._active_body_length
        active_half_wid = 0.5 * self._active_body_width

        # AABB overlap in XY (axis-aligned boxes)
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

        # ------------------------------------------------------------------
        # Termination & timeout
        # ------------------------------------------------------------------
        terminated = success | out_of_bounds | collision
        time_out = self.episode_length_buf >= self.max_episode_length

        if success.any():
            print(f"[SUCCESS] {int(success.sum().item())} dockings!")

        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset (USES CURRICULUM)
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        """
        Reset a subset of environments.

        This calls the curriculum reset function, which places
        the active robot at a position/orientation depending on
        the current curriculum stage.
        """
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()

        # Lazy init buffers
        num_envs = self.scene.cfg.num_envs
        if self.prev_distance is None:
            self.prev_distance = torch.zeros(num_envs, device=self.device)
        if self.prev_actions is None:
            self.prev_actions = torch.zeros((num_envs, 2), device=self.device)
        if self.step_count is None:
            self.step_count = torch.zeros(
                num_envs, dtype=torch.int32, device=self.device
            )

        # Reset per-episode state
        self.prev_actions[env_ids] = 0.0
        self.step_count[env_ids] = 0

        # Mark these envs so that the frame stack is reinitialized
        # on the next call to _get_observations.
        if self.frame_counts is not None:
            self.frame_counts[env_ids] = 0

        # Curriculum-based spawn (different stages = different initial poses)
        reset_environment_curriculum(self, env_ids)

        # Recompute initial distance for progress-based reward terms
        _, _, surface_xy, _ = self.get_sphere_distances_from_physics()
        self.prev_distance[env_ids] = surface_xy[env_ids]

    def set_curriculum_level(self, level: int):
        """Set curriculum level."""
        set_curriculum_level(self, level)

    def get_episode_statistics(self):
        """Collect statistics (mean reward, success rate, etc.)."""
        return collect_episode_stats(self)
