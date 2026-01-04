# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment Configuration (TiledCamera + Frame Stacking) NOVO
--------------------------------------------------------------
Optimized for:
- TiledCamera for efficient parallel rendering
- 84×84 grayscale observations with 4-frame stacking
- Asymmetric actor-critic (vision + privileged state)
- 200-500+ parallel environments on RTX 3090

Compatible with Isaac Lab 0.47.1 / Isaac Sim 5.0.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from teko.tasks.direct.teko.robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Environment configuration for torque-driven TEKO robot with TiledCamera."""

    # ------------------------------------------------------------------
    # General parameters
    # ------------------------------------------------------------------
    decimation = 2
    episode_length_s = 15.0
    enable_curriculum = True

    debug_boundaries: bool = False
    debug_robot_boxes: bool = False

    # ------------------------------------------------------------------
    # Asymmetric actor-critic flag
    # ------------------------------------------------------------------
    asymmetric_critic: bool = True

    # ------------------------------------------------------------------
    # Frame stacking
    # ------------------------------------------------------------------
    num_frame_stack: int = 4

    # ------------------------------------------------------------------
    # Arena limits
    # ------------------------------------------------------------------
    arena_half_x: float = 1.8
    arena_half_y: float = 2.4

    # ------------------------------------------------------------------
    # Body footprints
    # ------------------------------------------------------------------
    active_body_length: float = 0.35
    active_body_width: float = 0.20
    static_body_length: float = 0.35
    static_body_width: float = 0.20

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True,
    )

    # ------------------------------------------------------------------
    # Scene - INCREASED num_envs for TiledCamera efficiency
    # ------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=120,  # TiledCamera allows many more envs
        env_spacing=6.0,
        replicate_physics=True,
    )

    # ------------------------------------------------------------------
    # Spawn offset
    # ------------------------------------------------------------------
    robot_spawn_z_offset = 0.03

    # ------------------------------------------------------------------
    # Active robot configuration
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    dof_names = [
        "TEKO_Chassi_JointWheelFrontLeft",
        "TEKO_Chassi_JointWheelFrontRight",
        "TEKO_Chassi_JointWheelBackLeft",
        "TEKO_Chassi_JointWheelBackRight",
    ]

    # ------------------------------------------------------------------
    # Actuation
    # ------------------------------------------------------------------
    action_scale = 1.0
    max_wheel_torque = 5.0
    wheel_polarity = [-1.0, 1.0, -1.0, 1.0]

    # ------------------------------------------------------------------
    # TiledCamera Configuration (replaces per-env Camera)
    # ------------------------------------------------------------------
    # Uses regex pattern to match all camera prims across environments
    # Returns batched tensor [num_envs, H, W, C] in single render pass
    # ------------------------------------------------------------------
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/teko_urdf/TEKO_Body/TEKO_WallBack/TEKO_Camera/RearCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=None,  # <-- CHANGE THIS: Don't spawn, camera already exists in URDF
        width=128,
        height=128,
    )

    # ------------------------------------------------------------------
    # Static goal robot
    # ------------------------------------------------------------------
    @configclass
    class GoalCfg:
        """Static robot used as docking target."""
        usd_path = "/workspace/teko/documents/CAD/USD/teko.usd"
        prim_path = "/World/envs/env_.*/RobotGoal"
        aruco_texture = "/workspace/teko/documents/Aruco/test_marker.png"
        position = (1.0, 0.0, 0.40)
        aruco_offset = (0.17, 0.0, -0.045)
        aruco_size = 0.05

    goal = GoalCfg()

    # ------------------------------------------------------------------
    # Observation and Action Spaces
    # ------------------------------------------------------------------
    action_space = (2,)

    # Frame-stacked grayscale: [num_frame_stack, H, W]
    observation_space = {
        "rgb": (4, 128, 128),          # 4-frame stack of grayscale
        "privileged": (7,),          # [dx, dy, dz, yaw_err, vx, vy, w]
    }