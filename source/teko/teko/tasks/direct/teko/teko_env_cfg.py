# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from teko.robots.teko import TEKO_CONFIGURATION  # must resolve


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Config for TEKO: differential tracks (2 actions) or 4 independent wheels."""

    # Timing
    decimation = 2
    episode_length_s = 5.0

    # Spaces
    action_space = 2                  # set to 4 if independent_wheels = True
    observation_space = 3             # [dot, cross_z, forward_speed]
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Robot
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # NEW joint names in order [FL, FR, RL, RR]
    dof_names = [
        "teko_body_Revolucionar_31",  # FL
        "teko_body_Revolucionar_32",  # FR
        "teko_body_Revolucionar_33",  # RL   <- was 34
        "teko_body_Revolucionar_34",  # RR   <- was 33
    ]

    # Control
    independent_wheels = False
    action_scale = 18.0
    wheel_polarity = -1.0  # flip all if +left/+right moved backward before

    # Drive (velocity-like behavior uses damping as gain)
    drive_damping = 30.0
    drive_max_force = 400.0

    # Spawn height offset above ground to avoid initial penetration
    spawn_height = 0.05  # try 0.03–0.08 depending on your USD

    # Legacy placeholders (unused)
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    initial_pole_angle_range = [-0.25, 0.25]
    max_cart_pos = 3.0
