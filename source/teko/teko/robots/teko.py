# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Teko robot"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Path to your converted USD
TEKO_PATH = "/workspace/teko/documents/teko_bot.usd"

# Wheel joint names (new, after URDF → USD):
# Order we will use everywhere: [FL, FR, RL, RR]
WHEEL_JOINTS = [
    "teko_body_Revolucionar_31",  # FL
    "teko_body_Revolucionar_32",  # FR
    "teko_body_Revolucionar_33",  # RL
    "teko_body_Revolucionar_34",  # RR
]

TEKO_CONFIGURATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TEKO_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Keep only names that exist in the USD to avoid regex errors
            "teko_body_Revolucionar_31": 0.0,
            "teko_body_Revolucionar_32": 0.0,
            "teko_body_Revolucionar_33": 0.0,
            "teko_body_Revolucionar_34": 0.0,
            # If this slider exists in your USD, keep it; otherwise remove the line below
            "teko_connect_male_Deslizador_19": 0.0,
        }
    ),
    actuators={
        # Soft velocity-like behavior via high damping, zero stiffness.
        # Use explicit list to avoid regex mismatches.
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINTS,
            effort_limit_sim=120.0,   # torque limit per wheel
            stiffness=0.0,
            damping=40.0,             # try 30–60; increase if too “loose”, decrease if jittery
        ),
    },
)
