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
    """Configuration for TEKO environment (active robot + static RobotGoal)."""

    # --- General parameters -------------------------------------------
    decimation = 2
    episode_length_s = 30.0

    # --- Simulation ---------------------------------------------------
    #  PhysX GPU ainda instável com múltiplos robôs → CPU = seguro
    use_gpu_physics: bool = False

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True,
    )

    # --- Active robot -------------------------------------------------
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/Robot"
    )

    # --- Scene configuration ------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10,          # múltiplos ambientes paralelos
        env_spacing=10.0,     # distância entre arenas
        replicate_physics=True,
    )

    # --- Degrees of freedom (rodas) -----------------------------------
    dof_names = [
        "TEKO_Chassi_JointWheelFrontLeft",
        "TEKO_Chassi_JointWheelFrontRight",
        "TEKO_Chassi_JointWheelBackLeft",
        "TEKO_Chassi_JointWheelBackRight",
    ]

    # --- Ações ---------------------------------------------------------
    action_scale = 1.0
    max_wheel_speed = 6.0
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]

    # --- Camera --------------------------------------------------------
    class CameraCfg:
        prim_path = (
            "/Robot/teko_urdf/TEKO_Body/TEKO_WallBack/TEKO_Camera/RearCamera"
        )
        width = 640
        height = 480
        frequency_hz = 15
        focal_length = 3.6
        horiz_aperture = 4.8
        vert_aperture = 3.6
        f_stop = 16.0
        focus_distance = 2.0

    camera = CameraCfg()

    # --- Goal robot (com ArUco emissivo) -------------------------------
    class GoalCfg:
        usd_path = "/workspace/teko/documents/CAD/USD/teko_goal.usd"
        prim_path = "/RobotGoal"
        aruco_texture = "/workspace/teko/documents/Aruco/4x4_1000-1.png"
        position = (1.0, 0.0, 0.0)
        aruco_offset = (0.1675, 0.0, -0.025)
        aruco_size = 0.05

    goal = GoalCfg()

    # --- Spaces --------------------------------------------------------
    action_space = (2,)  # [left, right]
    observation_space = {"rgb": (3, 480, 640)}

