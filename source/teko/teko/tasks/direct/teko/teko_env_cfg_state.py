# SPDX-License-Identifier: BSD-3-Clause
"""
State-Based Environment Config for TEKO (Debugging)
====================================================

Configuration for state-based training using ground truth observations.

Key differences from vision config:
- No camera/vision
- 4D state observations: [dx, dy, dz, yaw_error]
- Position noise ±2cm, rotation noise ±3°
- 200+ environments (no vision overhead)

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from teko.teko.tasks.direct.teko.robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfgState(DirectRLEnvCfg):
    """
    State-based configuration for TEKO docking.
    
    Uses ground truth pose instead of vision for debugging.
    """
    
    # =========================================================================
    # SIMULATION SETTINGS
    # =========================================================================
    
    # Environment settings
    episode_length_s = 15.0
    decimation = 4
    num_envs = 1000  # Can scale higher without vision overhead
    num_observations = 4  # [dx, dy, dz, yaw_error]
    num_actions = 2  # [linear_vel, angular_vel]
    
    # Observation and action spaces
    observation_space = {"policy": (4,)}
    action_space = (2,)
    
    # State observation mode
    use_privileged_obs = True
    
    # Noise parameters (for robustness)
    state_noise_pos = 0.02      # ±2cm position noise
    state_noise_rot = 0.05      # ±~3° rotation noise (radians)
    state_noise_vel = 0.01      # ±1cm/s velocity noise
    
    # Physics settings
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
    # SCENE CONFIGURATION
    # =========================================================================
    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=4.0,
        replicate_physics=True,
    )
    
    # Ground plane
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
    
    # Active robot (controllable)
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Action/torque settings
    max_wheel_torque = 1.2
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]
    dof_names = [
        "TEKO_Chassi_JointWheelFrontLeft",
        "TEKO_Chassi_JointWheelFrontRight",
        "TEKO_Chassi_JointWheelBackLeft",
        "TEKO_Chassi_JointWheelBackRight",
    ]
    
    # Debug flags
    debug_boundaries = False
    debug_robot_boxes = False
    
    # Spawn offset
    robot_spawn_z_offset = 0.03
    
    # NOTE: Static robot spawned manually in env code (TEKOStatic class)
    
    # =========================================================================
    # ARENA CONFIGURATION
    # =========================================================================
    
    arena_half_x = 1.5
    arena_half_y = 1.0
    
    # =========================================================================
    # ACTION SCALING
    # =========================================================================
    
    action_scale = 0.5
    action_scale_linear = 0.3   # Max 0.3 m/s
    action_scale_angular = 1.5  # Max 1.5 rad/s
    
    # =========================================================================
    # ROBOT DIMENSIONS (for collision detection)
    # =========================================================================
    
    active_body_length = 0.34
    active_body_width = 0.26
    static_body_length = 0.34
    static_body_width = 0.26
    
    # =========================================================================
    # CURRICULUM SETTINGS
    # =========================================================================
    
    enable_curriculum = True
    curriculum_level = 0
    
    # Stage thresholds (SUCCESS RATE required to advance)
    stage_thresholds = {
        0: 0.75,   # S0 (0°, forward)
        1: 0.70,   # S1 (±5°)
        2: 0.70,   # S2 (±10°)
        3: 0.65,   # S3 (±15°)
        4: 0.65,   # S4 (±20°)
        5: 0.60,   # S5 (±30°)
        6: 0.60,   # S6 (±45°)
        7: 0.55,   # S7 (±60°)
        8: 0.55,   # S8 (±90°) - critical stage
        9: 0.50,   # S9 (±105°)
        10: 0.50,  # S10 (±120°)
        11: 0.45,  # S11 (±135°)
        12: 0.45,  # S12 (±150°)
        13: 0.40,  # S13 (±165°)
        14: 0.40,  # S14 (±180°)
        15: 0.40,  # S15 (full exploration)
        16: 0.35,  # S16 (mastery)
    }
    
    # Minimum steps per stage before advancing
    min_stage_steps = 50_000
    
    # Safety valve (force advance after max steps)
    max_stage_steps = 1_000_000