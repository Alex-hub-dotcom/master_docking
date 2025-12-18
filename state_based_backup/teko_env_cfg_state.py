# SPDX-License-Identifier: BSD-3-Clause
"""
State-Based Environment Config for TEKO (FINAL – Learnable)
===========================================================

Configuração mínima e consistente para treino PPO state-based
com observação ground-truth NORMALIZADA.

Usar ESTE CFG para:
- sanity check
- validação do reward
- validação do PPO

NÃO usar curriculum
NÃO usar ruído
NÃO usar visão

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from teko.tasks.direct.teko.robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfgState(DirectRLEnvCfg):
    """
    FINAL state-based config — MUST learn fast (>90% SSR).
    """

    # =========================================================================
    # ENV / RL BASICS
    # =========================================================================
    episode_length_s = 15.0
    decimation = 4

    num_envs = 256              # seguro para debugging
    num_observations = 3        # [dx_norm, dy_norm, yaw_norm]
    num_actions = 2             # [linear_vel, angular_vel]

    observation_space = {"policy": (3,)}
    action_space = (2,)

    use_privileged_obs = False
    asymmetric_critic = False
    num_frame_stack = 1

    # =========================================================================
    # SIMULATION
    # =========================================================================
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # =========================================================================
    # SCENE
    # =========================================================================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=4.0,
        replicate_physics=True,
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # =========================================================================
    # ACTION SCALING
    # =========================================================================
    action_scale = 1.0
    action_scale_linear = 0.3    # m/s
    action_scale_angular = 1.5   # rad/s

    # =========================================================================
    # ARENA
    # =========================================================================
    arena_half_x = 1.5
    arena_half_y = 1.0
    robot_spawn_z_offset = 0.03

    # =========================================================================
    # ROBOT DIMENSIONS (para lógica de colisão)
    # =========================================================================
    active_body_length = 0.34
    active_body_width = 0.26
    static_body_length = 0.34
    static_body_width = 0.26

    # =========================================================================
    # CURRICULUM — DESLIGADO (CRÍTICO)
    # =========================================================================
    enable_curriculum = False
    curriculum_level = 0
    stage_thresholds = {}
    min_stage_steps = 0
    max_stage_steps = 0

    # =========================================================================
    # CAMERA DUMMY (exigência da classe base, não usada)
    # =========================================================================
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/DUMMY_CAMERA",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=None,
        width=1,
        height=1,
    )
