# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment Configuration (Torque-driven, Modular)
-------------------------------------------------------
Optimized for:
- 84×84 grayscale observations (4-frame stack)
- 60 parallel environments

Compatible with Isaac Lab 0.47.1 / Isaac Sim 5.0.
"""

from __future__ import annotations
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from teko.tasks.direct.teko.robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Environment configuration for torque-driven TEKO robot."""

    # ------------------------------------------------------------------
    # General parameters
    # ------------------------------------------------------------------
    decimation = 2
    episode_length_s = 15.0
    enable_curriculum = False

    debug_boundaries: bool = False
    debug_robot_boxes: bool = False

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
    # Scene
    # ------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=60,
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
    max_wheel_torque = 1.2
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]

    # ------------------------------------------------------------------
    # Camera Configuration (84×84)
    # ------------------------------------------------------------------
    @configclass
    class CameraCfg:
        """Rear grayscale camera for RL docking."""

        prim_path = (
            "/World/envs/env_.*/Robot/teko_urdf/TEKO_Body/"
            "TEKO_WallBack/TEKO_Camera/RearCamera"
        )

        width = 84
        height = 84

        frequency_hz = 15
        focal_length = 3.6
        horiz_aperture = 4.8
        vert_aperture = 3.6
        f_stop = 16.0
        focus_distance = 2.0

        grayscale = True

    camera = CameraCfg()

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

    observation_space = {
        "rgb": (4, 84, 84),
    }