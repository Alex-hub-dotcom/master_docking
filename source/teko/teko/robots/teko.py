# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Teko robot."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # kept for compatibility, not used directly

# Absolute path to the USD of the robot
TEKO_PATH = "/workspace/teko/documents/teko_bot.usd"

# IMPORTANT:
# Joint names were updated after URDF->USD conversion:
#   - front-right  : teko_body_Revolucionar_32
#   - front-left   : teko_body_Revolucionar_31
#   - rear-right   : teko_body_Revolucionar_33
#   - rear-left    : teko_body_Revolucionar_34
#   - slider (aruco pin or mast): teko_connect_male_Deslizador_19
#
# We initialize joint positions to zero for the four wheel joints and the slider,
# and we attach an implicit actuator only to the four wheel joints. The slider is NOT actuated.

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
            # wheels (new names)
            "teko_body_Revolucionar_31": 0.0,  # FL
            "teko_body_Revolucionar_32": 0.0,  # FR
            "teko_body_Revolucionar_34": 0.0,  # RL
            "teko_body_Revolucionar_33": 0.0,  # RR
            # slider / mast (not actuated)
            "teko_connect_male_Deslizador_19": 0.0,
        }
    ),
    actuators={
        # Attach one implicit actuator to the four wheel joints only
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[
                "teko_body_Revolucionar_31",
                "teko_body_Revolucionar_32",
                "teko_body_Revolucionar_33",
                "teko_body_Revolucionar_34",
            ],
            # Effort limit in simulation (tune alongside your env drive limits if needed)
            effort_limit_sim=400.0,
            # Velocity-mode style: keep stiffness low (0) and use damping as gain
            stiffness=0.0,
            damping=30.0,
        )
    },
)
