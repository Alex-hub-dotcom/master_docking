# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment Configuration (Torque-driven, Modular)
-------------------------------------------------------
Compatible with Isaac Lab 0.47.1 / Isaac Sim 5.0.

Provides configuration for:
- Active torque-driven robot (spawn offset added to prevent floor clipping)
- Static goal robot (with ArUco marker)
- Camera setup
- Arena limits
- Rectangular body footprints for active & static robots
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
    episode_length_s = 15.0          # shorter episodes: 15 seconds
    enable_curriculum = False

    # Visual debug helpers (used by TekoEnv)
    # - red arena boundaries at |x| = arena_half_x, |y| = arena_half_y
    # - green/red rectangular boxes attached to active & static robots
    debug_boundaries: bool = False
    debug_robot_boxes: bool = False

    # ------------------------------------------------------------------
    # Arena limits (env-local coordinates, meters)
    # ------------------------------------------------------------------
    # These are used BOTH for:
    #   - out-of-bounds check in TekoEnv._get_dones
    #   - red debug boundary walls in TekoEnv._spawn_arena_boundaries
    arena_half_x: float = 1.8   # short side (tuned to match wood)
    arena_half_y: float = 2.4   # long side  (tuned to match wood)

    # ------------------------------------------------------------------
    # Rectangular body footprints (meters)
    # ------------------------------------------------------------------
    # Used for:
    #   - debug chassis boxes (green = active, red = static)
    #   - static-box collision in TekoEnv._get_dones
    #
    # 35 cm x 20 cm as you measured.
    active_body_length: float = 0.35
    active_body_width: float = 0.20
    static_body_length: float = 0.35
    static_body_width: float = 0.20

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True,
    )

    # ------------------------------------------------------------------
    # Scene configuration
    # ------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=6.0,
        replicate_physics=True,
    )

    # ------------------------------------------------------------------
    # Spawn offset (active robot only)
    # ------------------------------------------------------------------
    # Prevents the robot's wheels from spawning intersecting with the ground.
    robot_spawn_z_offset = 0.03

    # ------------------------------------------------------------------
    # Active robot configuration
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Wheel joints used for torque control
    dof_names = [
        "TEKO_Chassi_JointWheelFrontLeft",
        "TEKO_Chassi_JointWheelFrontRight",
        "TEKO_Chassi_JointWheelBackLeft",
        "TEKO_Chassi_JointWheelBackRight",
    ]

    # ------------------------------------------------------------------
    # Actuation parameters
    # ------------------------------------------------------------------
    action_scale = 1.0
    max_wheel_torque = 1.2
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]  # Left/Right differential polarity

    # ------------------------------------------------------------------
    # Camera configuration (rear RGB camera)
    # ------------------------------------------------------------------
    @configclass
    class CameraCfg:
        """RGB camera mounted on the back wall of TEKO."""
        prim_path = (
            "/World/envs/env_.*/Robot/teko_urdf/TEKO_Body/"
            "TEKO_WallBack/TEKO_Camera/RearCamera"
        )
        width = 128
        height = 128
        frequency_hz = 15
        focal_length = 3.6
        horiz_aperture = 4.8
        vert_aperture = 3.6
        f_stop = 16.0
        focus_distance = 2.0

    camera = CameraCfg()

    # ------------------------------------------------------------------
    # Static goal robot configuration (with ArUco marker)
    # ------------------------------------------------------------------
    @configclass
    class GoalCfg:
        """Static goal robot used as docking target."""
        usd_path = "/workspace/teko/documents/CAD/USD/teko.usd"
        prim_path = "/World/envs/env_.*/RobotGoal"
        aruco_texture = "/workspace/teko/documents/Aruco/test_marker.png"
        position = (1.0, 0.0, 0.40)
        aruco_offset = (0.17, 0.0, -0.045)
        aruco_size = 0.05  # meters

    goal = GoalCfg()

    # ------------------------------------------------------------------
    # Observation and action spaces
    # ------------------------------------------------------------------
    # Action space is 2D: [v, w] = [forward/back, turn], each in [-1, 1].
    action_space = (2,)
    observation_space = {
    "rgb": (4, 128, 128),     # 4 grayscale frames (stack): [K=4, H=64, W=64]
    }

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    # Actions in [-1, 1] are interpreted as:
    #   v = actions[0] → forward/backward command
    #   w = actions[1] → turning command
    #
    # Inside the environment, these are mapped to wheel torques with:
    #   left  = v - k * w
    #   right = v + k * w
    # and then scaled by max_wheel_torque.