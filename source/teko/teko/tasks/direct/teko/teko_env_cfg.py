# /workspace/teko/source/teko/teko/tasks/direct/teko/teko_env_cfg.py
# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from teko.robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Minimal config for TEKO: ground plane only, linear forward test."""

    # Timing
    decimation = 2
    # Long episode so demos don't auto-reset mid-run
    episode_length_s = 300.0

    # Spaces
    action_space = 2               # [left, right]
    observation_space = 1          # forward speed
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )

    # Robot
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Scene: 8 envs with enough spacing for 8x8 m arenas + margin
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs= 9,
        env_spacing=12.0,   # increase to 14.0 if you ever see overlap
        replicate_physics=True,
    )

    # Joint names (order: [FL, FR, RL, RR])
    dof_names = [
        "teko_body_front_left",
        "teko_body_front_right",
        "teko_body_back_left",
        "teko_body_back_right",
    ]

    # Control
    independent_wheels = False
    action_scale = 8.0
    wheel_polarity = -1.0   # flip to +1.0 if forward is inverted for your USD

    # Drive (velocity-like via damping)
    drive_damping = 35.0
    drive_max_force = 25.0

    # Spawn / safety
    spawn_height = 0.08
    max_wheel_speed = 12.0
    max_wheel_accel = 60.0

    # Feed-forward torque
    ff_torque = 8.0