# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for TEKO robot in a custom arena (Isaac Lab 0.47.1 compatible)."""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from gym import spaces

import numpy as np
from .robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Single-robot environment configuration for TEKO in a fixed arena."""

    # Timing and episode length
    decimation = 2
    episode_length_s = 30.0

    # Simulation parameters
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True,
    )

    # Robot articulation configuration
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/Robot"
    )

    # Scene setup
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # Degrees of freedom to control
    dof_names = [
        "TEKO_Chassi_JointWheelFrontLeft",
        "TEKO_Chassi_JointWheelFrontRight",
        "TEKO_Chassi_JointWheelBackLeft",
        "TEKO_Chassi_JointWheelBackRight",
    ]

    # Action scaling and motor polarity
    action_scale = 1.0
    max_wheel_speed = 6.0
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]
    

    # Camera Setup
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/teko_urdf/RearCamera",
        data_types=["rgb"],
        update_period=1.0 / 30.0,  # 30 FPS (≈ Raspberry Pi V2)
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.6,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),    
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.179, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0), # quartenions remember this! 
            convention="ros",
        ),
        width=640,
        height=480,
    )

    action_space = 2
    observation_space = [3, 480, 640]

    # Observation and action spaces (required by DirectRLEnv)
 
    # observation_space = {
    #     "policy": spaces.Box(
    #         low=0.0,
    #         high=1.0,
    #         shape=(3, 480, 640),   # C, H, W da tua câmera
    #         dtype=np.float32,
    #     )
    # }

    # action_space = spaces.Box(
    #     low=-1.0,
    #     high=1.0,
    #     shape=(2,),                # [left, right] motores
    #     dtype=np.float32,
    # )


    #When working with rendering, make sure to add the --enable_cameras argument when launching the environment. For example:
    #python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras

    # # Observation and state space for the CNN policy
    # @configclass
    # class env:
    #     action_space = 2
    #     observation_space = [3, 64, 64]
    #     state_space = 0
