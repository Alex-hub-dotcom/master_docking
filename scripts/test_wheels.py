#!/usr/bin/env python3
"""Teste individual de cada roda com torques +/-"""

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


def main():
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 1
    cfg.enable_curriculum = False
    
    env = TekoEnvState(cfg=cfg)
    device = env.device
    
    env.reset()
    
    print("=" * 60)
    print("TESTE INDIVIDUAL DE RODAS (+/-)")
    print(f"DOF indices: {env.dof_idx}")
    print(f"Max torque: {env._max_wheel_torque}")
    print("=" * 60)
    
    # Testar cada roda com torque positivo e negativo
    for wheel_idx in range(4):
        for sign in [+1, -1]:
            env.reset()
            
            for _ in range(30):
                action = torch.tensor([[0.0, 0.0]], device=device)
                env.step(action)
            
            pos_inicial = env.robot.data.root_pos_w[0].clone()
            q = env.robot.data.root_quat_w[0]
            yaw_inicial = float(torch.atan2(2*(q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]*q[1] + q[2]*q[2])))
            
            torques = torch.zeros((1, 4), device=device)
            torques[0, wheel_idx] = sign * env._max_wheel_torque
            
            for _ in range(200):
                env.robot.set_joint_effort_target(
                    torques,
                    env_ids=torch.tensor([0], device=device),
                    joint_ids=env.dof_idx
                )
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(env.cfg.sim.dt)
            
            pos_depois = env.robot.data.root_pos_w[0]
            q = env.robot.data.root_quat_w[0]
            yaw_depois = float(torch.atan2(2*(q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]*q[1] + q[2]*q[2])))
            
            delta_x = float(pos_depois[0] - pos_inicial[0])
            delta_yaw = np.rad2deg(yaw_depois - yaw_inicial)
            
            sign_str = "+" if sign > 0 else "-"
            print(f"Slot {wheel_idx} (DOF {env.dof_idx[wheel_idx].item()}) torque={sign_str}: dx={delta_x:+.3f}m, dyaw={delta_yaw:+.1f}°")
    
    print("\n" + "=" * 60)
    env.close()
    app.app.close()


if __name__ == "__main__":
    main()
