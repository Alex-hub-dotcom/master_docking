# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment Configuration
------------------------------
Configuration container for the single-robot TEKO environment.

Compatible with Isaac Lab 0.47.1 and DirectRLEnv.
"""

from __future__ import annotations

# Isaac Lab imports
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# TEKO robot configuration
from .robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Configuration for the TEKO single-robot environment."""

    # ------------------------------------------------------------------
    # 1. Timing and simulation parameters
    # ------------------------------------------------------------------
    decimation = 2                      # Number of physics steps per env step
    episode_length_s = 30.0             # Maximum episode duration (s)

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,                     # Physics timestep (s)
        render_interval=decimation,     # Render every N physics steps
        gravity=(0.0, 0.0, -9.81),      # Standard gravity
        use_fabric=True,                # Enable PhysX fabric solver
    )

    # ------------------------------------------------------------------
    # 2. Robot configuration
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/Robot"
    )

    # ------------------------------------------------------------------
    # 3. Scene configuration
    # ------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,                     # Number of parallel environments
        env_spacing=0.0,                # Distance between environments
        replicate_physics=True,         # Shared physics context
    )

    # ------------------------------------------------------------------
    # 4. Control parameters
    # ------------------------------------------------------------------
    dof_names = [                       # Motorized wheel joint names
        "TEKO_Chassi_JointWheelFrontLeft",
        "TEKO_Chassi_JointWheelFrontRight",
        "TEKO_Chassi_JointWheelBackLeft",
        "TEKO_Chassi_JointWheelBackRight",
    ]

    action_scale = 1.0                  # Scale factor for normalized actions
    max_wheel_speed = 6.0               # Max wheel angular velocity (rad/s)
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]  # Direction convention per wheel

    # ------------------------------------------------------------------
    # 5. Camera metadata
    # ------------------------------------------------------------------
    class CameraCfg:
        """Metadata for the pre-defined RearCamera prim."""
        prim_path = "/World/Robot/teko_urdf/TEKO_Body/TEKO_WallBack/TEKO_Camera/RearCamera"
        width = 640                     # Image width (px)
        height = 480                    # Image height (px)
        frequency_hz = 30               # Capture frequency (Hz)
        # Optical parameters — Raspberry Pi Camera V2 (Sony IMX219)
        focal_length = 3.04             # mm
        horiz_aperture = 4.6            # mm
        vert_aperture = 2.76            # mm
        f_stop = 32.0                   # Disable depth-of-field blur
        focus_distance = 10.0           # Focus to infinity

    camera = CameraCfg()                # Instantiate camera metadata

    # ------------------------------------------------------------------
    # 6. LiDAR metadata (RTX virtual LiDAR)
    # ------------------------------------------------------------------
    class LidarCfg:
        """Metadata for the RTX LiDAR sensor (already in USD)."""
        prim_path = "/World/Robot/teko_urdf/TEKO_Body/LidarMount/Lidar"
        horizontal_fov = 360.0
        vertical_fov = 30.0
        horizontal_resolution = 1024
        vertical_resolution = 32
        max_distance = 20.0
        rotation_rate_hz = 10.0

    lidar = LidarCfg()                  # Instantiate LiDAR metadata

    # ------------------------------------------------------------------
    # 7. Observation / action space (Isaac Lab compatible)
    # ------------------------------------------------------------------
    action_space = (2,)                 # Two continuous control inputs (L/R)

    observation_space = {               # Dictionary of observation dimensions
        "rgb": (3, 480, 640),           # RGB image (C, H, W)
        "lidar": (360, 3),              # LiDAR point cloud (N×3)
    }
