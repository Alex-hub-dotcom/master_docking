#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app = AppLauncher(args)

import torch
import numpy as np
import sys
sys.path.insert(0, "/workspace/teko/source/teko")

from teko.tasks.direct.teko.teko_env_state import TekoEnvState
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

cfg = TekoEnvCfg()
cfg.scene.num_envs = 1
cfg.enable_curriculum = False

env = TekoEnvState(cfg=cfg)
device = env.device

env.reset()

print("=" * 60)
print("DOF MAPPING TEST")
print(f"Joint names: {env.robot.joint_names}")
print(f"DOF indices: {env.dof_idx}")
print(f"Polarity: {env._polarity}")
print("=" * 60)

# Teste: aplicar torque +5 em cada slot individualmente
for slot in range(4):
    env.reset()
    for _ in range(20):
        env.step(torch.tensor([[0.0, 0.0]], device=device))
    
    pos_init = env.robot.data.root_pos_w[0].clone()
    q = env.robot.data.root_quat_w[0]
    yaw_init = float(torch.atan2(2*(q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]*q[1] + q[2]*q[2])))
    
    # Aplicar torque só neste slot
    for _ in range(100):
        torques = torch.zeros((1, 4), device=device)
        torques[0, slot] = 5.0  # Torque positivo
        env.robot.set_joint_effort_target(torques, env_ids=torch.tensor([0], device=device), joint_ids=env.dof_idx)
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(env.cfg.sim.dt)
    
    pos_final = env.robot.data.root_pos_w[0]
    q = env.robot.data.root_quat_w[0]
    yaw_final = float(torch.atan2(2*(q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]*q[1] + q[2]*q[2])))
    
    dx = float(pos_final[0] - pos_init[0])
    dy = float(pos_final[1] - pos_init[1])
    dyaw = np.rad2deg(yaw_final - yaw_init)
    
    joint_name = env.cfg.dof_names[slot]
    print(f"Slot {slot} ({joint_name}): dx={dx:+.3f}m, dy={dy:+.3f}m, dyaw={dyaw:+.1f}°")

env.close()
app.app.close()
