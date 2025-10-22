# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the TEKO robot (wheel joints emulate tracks)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Path to the converted USD
TEKO_PATH = "/workspace/teko/documents/CAD/USD/teko.usd"

# Wheel joint names in the USD (prefix TEKO_Chassi_)
# Convention: [Front-Left, Front-Right, Rear-Left, Rear-Right]
WHEEL_JOINTS = [
    "TEKO_Chassi_JointWheelFrontLeft",   # FL
    "TEKO_Chassi_JointWheelFrontRight",  # FR
    "TEKO_Chassi_JointWheelBackLeft",    # RL
    "TEKO_Chassi_JointWheelBackRight",   # RR
]

# Articulation configuration
TEKO_CONFIGURATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TEKO_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={joint: 0.0 for joint in WHEEL_JOINTS},
    ),
    actuators={
        # Implicit actuator — behaves like velocity control when stiffness=0 and damping>0.
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINTS,   # explicit list, no regex ambiguity
            effort_limit_sim=6.0,            # Nm per wheel
            stiffness=0.0,
            damping=30.0,                    # stable damping for ~6kg robot
        ),
    },
)
