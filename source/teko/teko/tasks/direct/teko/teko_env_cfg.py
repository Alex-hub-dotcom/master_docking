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
    episode_length_s = 5.0

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

    # Scene (demo may sobrescrever para 1 env)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=10.0,
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
   # Control
    action_scale = 8.0
    wheel_polarity = -1.0     # mantenha se “frente” já está correta

    # Drive
    drive_damping = 40.0
    drive_max_force = 30.0    # mais força nas rodas

    # Safety / limits
    max_wheel_speed = 10.0    # rad/s
    max_wheel_accel = 25.0    # rad/s^2

    # (opcional) ajuda a sair da inércia se sentir “travado”
    ff_torque = 6.0
