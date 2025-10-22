# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Simple config for TEKO single-robot environment (compatible with Isaac Lab 0.47.1)."""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from teko.robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Configuration for TEKO robot in a custom arena."""

    decimation = 2
    episode_length_s = 30.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True,
    )

    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/Robot",
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=0.0,
        replicate_physics=True,
    )

    dof_names = [
        "TEKO_Chassi_JointWheelFrontLeft",
        "TEKO_Chassi_JointWheelFrontRight",
        "TEKO_Chassi_JointWheelBackLeft",
        "TEKO_Chassi_JointWheelBackRight",
    ]

    action_scale = 1.0
    max_wheel_speed = 6.0
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]

    action_space = 2
    observation_space = 1
    state_space = 0
