# SPDX-License-Identifier: BSD-3-Clause
"""
TekoEnvCfg — environment configuration for TEKO with RGB camera.
Compatible with Isaac Lab 0.47.1 / Isaac Sim 5.0.
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Configuration for TEKO environment (one active robot + static RobotGoal)."""

    # --- General parameters -------------------------------------------
    decimation = 2
    episode_length_s = 30.0

    # --- Simulation ---------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True,
    )

    # --- Active robot -------------------------------------------------
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/Robot"
    )

    # --- Scene configuration ------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # --- Degrees of freedom -------------------------------------------
    dof_names = [
        "TEKO_Chassi_JointWheelFrontLeft",
        "TEKO_Chassi_JointWheelFrontRight",
        "TEKO_Chassi_JointWheelBackLeft",
        "TEKO_Chassi_JointWheelBackRight",
    ]

    # --- Action configuration -----------------------------------------
    action_scale = 1.0
    max_wheel_speed = 6.0
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]

    # --- Camera configuration -----------------------------------------
    class CameraCfg:
        prim_path = (
            "/World/Robot/teko_urdf/TEKO_Body/TEKO_WallBack/TEKO_Camera/RearCamera"
        )
        width = 640
        height = 480
        frequency_hz = 30
        focal_length = 3.6
        horiz_aperture = 4.8
        vert_aperture = 3.6
        f_stop = 32.0
        focus_distance = 10.0


    camera = CameraCfg()

    # --- Goal robot configuration -------------------------------------
    class GoalCfg:
        usd_path = "/workspace/teko/documents/CAD/USD/teko_goal.usd"
        prim_path = "/World/RobotGoal"
        aruco_texture = "/workspace/teko/documents/Aruco/4x4_1000-1.png"
        position = (1.0, 0.0, 0.0)
        aruco_offset = (0.1675, 0.0, -0.025)
        aruco_size = 0.05

    goal = GoalCfg()

    # --- Observation and action spaces --------------------------------
    action_space = (2,)
    observation_space = {
        "rgb": (3, 480, 640),
    }
