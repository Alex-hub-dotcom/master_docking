"""Configuration for the Teko robot """

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

TEKO_PATH = "/workspace/teko/documents/teko_bot.usd"

TEKO_CONFIGURATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TEKO_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # base
            "teko_body_Revolucionar_10": 0.0,
            "teko_body_Revolucionar_11": 0.0,
            "teko_body_Revolucionar_8": 0.0,
            "teko_body_Revolucionar_9": 0.0,
            "teko_connect_male_Deslizador_19": 0.0
        }
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=["teko_body_.*"],
            effort_limit_sim=1000.0,
            stiffness=0.0,
            damping=1e5,
        )
    },
)

