# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Teko robot (wheel joints emulate tracks)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Path to the converted USD
TEKO_PATH = "/workspace/teko/documents/teko.usd"

# Wheel joint names in a consistent order used across the codebase:
# Convention: [Front-Left, Front-Right, Rear-Left, Rear-Right]
WHEEL_JOINTS = [
    "teko_body_front_left",   # FL
    "teko_body_front_right",  # FR
    "teko_body_back_left",    # RL
    "teko_body_back_right",   # RR
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
            "teko_body_front_left": 0.0,
            "teko_body_front_right": 0.0,
            "teko_body_back_left": 0.0,
            "teko_body_back_right": 0.0,
        }
    ),
    actuators={
        # Implicit actuator: velocity-like behavior when stiffness=0 and damping>0.
        # Tuned mild to avoid violent impulses on a ~6 kg platform.
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINTS,  # explicit list prevents regex mismatches
            effort_limit_sim=6.0,           # ~Nm per wheel (XL430-scale, with headroom)
            stiffness=0.0,
            damping=30.0,                   # moderate damping; runtime drive gains refine it
        ),
    },
)
