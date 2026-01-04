#!/usr/bin/env python3
"""Teste de estabilidade - robô pula?"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app = AppLauncher(args)

import torch
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
    print("TESTE DE ESTABILIDADE")
    print(f"Max torque: {env._max_wheel_torque}")
    print("=" * 60)
    
    z_min, z_max = 1000, -1000
    
    # Aplicar ações aleatórias por 500 steps
    for i in range(500):
        action = torch.tensor([[1.0, 1.0]], device=device)  # Frente + rotação máxima
        env.step(action)
        
        z = float(env.robot.data.root_pos_w[0, 2])
        z_min = min(z_min, z)
        z_max = max(z_max, z)
        
        if i % 100 == 0:
            pos = env.robot.data.root_pos_w[0]
            print(f"Step {i}: z={z:.3f}m, x={pos[0]:.2f}, y={pos[1]:.2f}")
    
    print(f"\nZ range: {z_min:.3f} - {z_max:.3f}m")
    print(f"Delta Z: {z_max - z_min:.3f}m")
    
    if z_max - z_min > 0.1:
        print("⚠️ INSTÁVEL! Robô está pulando/quicando!")
    else:
        print("✅ Estável - robô no chão")
    
    env.close()
    app.app.close()


if __name__ == "__main__":
    main()
