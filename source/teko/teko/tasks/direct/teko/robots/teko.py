# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for the TEKO robot (wheel joints emulate continuous tracks)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Path to the TEKO robot USD file (converted from URDF/Fusion)
TEKO_PATH = "/workspace/teko/documents/CAD/USD/teko.usd"

# Explicit joint names for the 4 wheel joints (used for tracked locomotion)
WHEEL_JOINTS = [
    "TEKO_Chassi_JointWheelFrontLeft",   # Front-left
    "TEKO_Chassi_JointWheelFrontRight",  # Front-right
    "TEKO_Chassi_JointWheelBackLeft",    # Rear-left
    "TEKO_Chassi_JointWheelBackRight",   # Rear-right
]

# Full Articulation configuration for the TEKO robot
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
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINTS,
            effort_limit_sim=6.0,   # Max torque per joint (Nm)
            stiffness=0.0,          # No spring force (acts as velocity control)
            damping=30.0,           # Damping tuned for small tracked robot (~6kg)
        ),
    },
)
