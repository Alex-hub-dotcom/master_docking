#!/usr/bin/env python3
"""Teste de rotação pura - o robô consegue girar?"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app = AppLauncher(args)
sim = app.app

import torch
import numpy as np
import sys
sys.path.insert(0, "/workspace/teko/source/teko")

from teko.tasks.direct.teko.teko_env_state import TekoEnvState
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg


def get_yaw(quat):
    """Extract yaw from quaternion [x,y,z,w]"""
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    return float(torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))


def main():
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 1
    cfg.enable_curriculum = False
    
    env = TekoEnvState(cfg=cfg)
    device = env.device
    
    env.reset()
    
    # Dar tempo para estabilizar
    for _ in range(50):
        action = torch.tensor([[0.0, 0.0]], device=device)
        env.step(action)
    
    print("=" * 60)
    print("TESTE DE ROTAÇÃO - k =", end=" ")
    
    # Ler k atual do env
    # O k está hardcoded, vamos testar com ações
    print("(ver _apply_action)")
    print("=" * 60)
    
    pos_inicial = env.robot.data.root_pos_w[0].clone()
    yaw_inicial = get_yaw(env.robot.data.root_quat_w[0])
    print(f"\nPosição inicial: x={pos_inicial[0]:.3f}, y={pos_inicial[1]:.3f}")
    print(f"Yaw inicial: {np.rad2deg(yaw_inicial):.1f}°")
    
    # Teste 1: Só rotação positiva (w=+1, v=0)
    print("\n--- Teste 1: w=+1, v=0 por 200 steps ---")
    for _ in range(200):
        action = torch.tensor([[0.0, 1.0]], device=device)
        env.step(action)
    
    pos_depois = env.robot.data.root_pos_w[0]
    yaw_depois = get_yaw(env.robot.data.root_quat_w[0])
    delta_yaw = np.rad2deg(yaw_depois - yaw_inicial)
    delta_x = float(pos_depois[0] - pos_inicial[0])
    delta_y = float(pos_depois[1] - pos_inicial[1])
    print(f"Delta yaw: {delta_yaw:+.1f}°")
    print(f"Delta pos: x={delta_x:+.3f}m, y={delta_y:+.3f}m")
    
    # Reset posição
    env.reset()
    for _ in range(50):
        action = torch.tensor([[0.0, 0.0]], device=device)
        env.step(action)
    
    pos_inicial = env.robot.data.root_pos_w[0].clone()
    yaw_inicial = get_yaw(env.robot.data.root_quat_w[0])
    
    # Teste 2: Só rotação negativa (w=-1, v=0)
    print("\n--- Teste 2: w=-1, v=0 por 200 steps ---")
    for _ in range(200):
        action = torch.tensor([[0.0, -1.0]], device=device)
        env.step(action)
    
    pos_depois = env.robot.data.root_pos_w[0]
    yaw_depois = get_yaw(env.robot.data.root_quat_w[0])
    delta_yaw = np.rad2deg(yaw_depois - yaw_inicial)
    delta_x = float(pos_depois[0] - pos_inicial[0])
    delta_y = float(pos_depois[1] - pos_inicial[1])
    print(f"Delta yaw: {delta_yaw:+.1f}°")
    print(f"Delta pos: x={delta_x:+.3f}m, y={delta_y:+.3f}m")
    
    # Reset
    env.reset()
    for _ in range(50):
        action = torch.tensor([[0.0, 0.0]], device=device)
        env.step(action)
    
    pos_inicial = env.robot.data.root_pos_w[0].clone()
    yaw_inicial = get_yaw(env.robot.data.root_quat_w[0])
    
    # Teste 3: Só frente (v=+1, w=0)
    print("\n--- Teste 3: v=+1, w=0 por 200 steps ---")
    for _ in range(200):
        action = torch.tensor([[1.0, 0.0]], device=device)
        env.step(action)
    
    pos_depois = env.robot.data.root_pos_w[0]
    yaw_depois = get_yaw(env.robot.data.root_quat_w[0])
    delta_yaw = np.rad2deg(yaw_depois - yaw_inicial)
    delta_x = float(pos_depois[0] - pos_inicial[0])
    delta_y = float(pos_depois[1] - pos_inicial[1])
    print(f"Delta yaw: {delta_yaw:+.1f}°")
    print(f"Delta pos: x={delta_x:+.3f}m, y={delta_y:+.3f}m")
    
    # Reset
    env.reset()
    for _ in range(50):
        action = torch.tensor([[0.0, 0.0]], device=device)
        env.step(action)
    
    pos_inicial = env.robot.data.root_pos_w[0].clone()
    yaw_inicial = get_yaw(env.robot.data.root_quat_w[0])
    
    # Teste 4: Só trás (v=-1, w=0)
    print("\n--- Teste 4: v=-1, w=0 por 200 steps ---")
    for _ in range(200):
        action = torch.tensor([[-1.0, 0.0]], device=device)
        env.step(action)
    
    pos_depois = env.robot.data.root_pos_w[0]
    yaw_depois = get_yaw(env.robot.data.root_quat_w[0])
    delta_yaw = np.rad2deg(yaw_depois - yaw_inicial)
    delta_x = float(pos_depois[0] - pos_inicial[0])
    delta_y = float(pos_depois[1] - pos_inicial[1])
    print(f"Delta yaw: {delta_yaw:+.1f}°")
    print(f"Delta pos: x={delta_x:+.3f}m, y={delta_y:+.3f}m")
    
    print("\n" + "=" * 60)
    print("RESUMO:")
    print("- Se rotação dá <10° em 200 steps: k muito baixo ou torque insuficiente")
    print("- Se movimento linear dá <0.1m em 200 steps: torque muito baixo")
    print("- Ideal: rotação ~30-60°, linear ~0.3-0.5m em 200 steps")
    print("=" * 60)
    
    env.close()
    sim.close()


if __name__ == "__main__":
    main()
