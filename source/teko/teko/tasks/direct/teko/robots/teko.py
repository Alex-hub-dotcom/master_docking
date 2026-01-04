# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Robot Configuration — clean and simulation-safe
====================================================

This file defines the TEKO mobile robot as an Isaac Lab ArticulationCfg.
It is used by the TEKO RL environments for torque-driven control.

Author: Alexandre Schleier Neves da Silva
For questions or collaboration, contact:
    alexandre.schleiernevesdasilva@uni-hohenheim.de
"""

from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Path to the TEKO USD robot model in your workspace
TEKO_PATH = "/workspace/teko/documents/CAD/USD/teko.usd"

# Wheel joint names as defined in the USD / URDF
WHEEL_JOINTS = [
    "TEKO_Chassi_JointWheelFrontLeft",
    "TEKO_Chassi_JointWheelFrontRight",
    "TEKO_Chassi_JointWheelBackLeft",
    "TEKO_Chassi_JointWheelBackRight",
]

# Main robot configuration used by Isaac Lab
TEKO_CONFIGURATION = ArticulationCfg(
    # How to spawn the TEKO robot into the scene
    spawn=sim_utils.UsdFileCfg(
        usd_path=TEKO_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,          # simpler + more stable
            solver_position_iteration_count=16,     # more iterations = more stable contacts
            solver_velocity_iteration_count=4,
            sleep_threshold=0.001,
            stabilization_threshold=0.0001,
        ),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.3,      # resistance to linear motion (was 0.1)
            angular_damping=0.3,     # resistance to rotation (was 0.15)
            max_linear_velocity=2.0,
            max_angular_velocity=10.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,    # small offset to help contacts be stable
            rest_offset=0.0,
            torsional_patch_radius=0.003,
        ),
    ),

    # Initial joint state (all wheels at 0 position and velocity)
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={joint: 0.0 for joint in WHEEL_JOINTS},
        joint_vel={joint: 0.0 for joint in WHEEL_JOINTS},
    ),

    # Actuator configuration: torque-controlled wheels
    actuators={
        "wheel_actuators": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINTS,  # which joints this actuator controls
            effort_limit_sim=3.0,           # max torque (kept low to reduce crazy jumps)
            stiffness=0.0,                  # 0 = pure torque control (no spring)
            damping=0.5,                    # how much the joint resists fast motion
            friction=0.1,                   # less joint friction → smoother rolling
            armature=0.001,                 # small rotational inertia on joints
        ),
    },
)
