# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Simple config for TEKO single-robot environment."""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from teko.robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Minimal configuration to load TEKO robot and ground."""

    # Timing
    decimation = 2
    episode_length_s = 30.0

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Robot
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(prim_path="/World/Robot")

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.0)

    # DOF names (must match the USD joint names)
    dof_names = [
        "TEKO_Chassi_JointWheelFrontLeft",   
        "TEKO_Chassi_JointWheelFrontRight",  
        "TEKO_Chassi_JointWheelBackLeft",    
        "TEKO_Chassi_JointWheelBackRight",
    ]

    # Control parameters
    action_scale = 1.0
    max_wheel_speed = 6.0
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]
    # Required placeholders for DirectRLEnv (even if not using RL)
    action_space = 2          # [left, right] for differential drive
    observation_space = 1     # dummy value (unused)
    state_space = 0           # optional, safe default
