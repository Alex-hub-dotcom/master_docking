# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for TEKO robot in a custom arena using the simple Isaac Sim Camera (no TiledCamera)."""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from gym import spaces
import numpy as np

from .robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Single-robot environment configuration for TEKO in a fixed arena."""

    # ------------------------------------------------------------------ #
    # Timing and simulation parameters
    # ------------------------------------------------------------------ #
    decimation = 2
    episode_length_s = 30.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True,
    )

    # ------------------------------------------------------------------ #
    # Robot
    # ------------------------------------------------------------------ #
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/Robot"
    )

    # ------------------------------------------------------------------ #
    # Scene
    # ------------------------------------------------------------------ #
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #
    dof_names = [
        "TEKO_Chassi_JointWheelFrontLeft",
        "TEKO_Chassi_JointWheelFrontRight",
        "TEKO_Chassi_JointWheelBackLeft",
        "TEKO_Chassi_JointWheelBackRight",
    ]

    action_scale = 1.0
    max_wheel_speed = 6.0
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]

    # ------------------------------------------------------------------ #
    # Camera configuration (for simple Camera class)
    # ------------------------------------------------------------------ #
    class CameraCfg:
        prim_path = "/World/Robot/RearCamera"
        position = (-0.179, 0.0, 0.0)  # offset relativo ao robô
        rotation = (0.0, 0.0, 0.0)     # Euler em graus
        width = 640
        height = 480
        frequency_hz = 30

    camera = CameraCfg()

    # ------------------------------------------------------------------ #
    # Observation/action space
    # ------------------------------------------------------------------ #
    action_space = 2
    observation_space = [3, 480, 640]
