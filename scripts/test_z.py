#!/usr/bin/env python3
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

cfg = TekoEnvCfg()
cfg.scene.num_envs = 1
cfg.enable_curriculum = False

env = TekoEnvState(cfg=cfg)
env.reset()

print("=" * 50)
print("TESTE DE POSIÇÃO Z")
print("=" * 50)

for i in range(100):
    action = torch.tensor([[0.0, 0.0]], device=env.device)
    env.step(action)
    
    pos = env.robot.data.root_pos_w[0]
    if i % 20 == 0:
        print(f"Step {i}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")

print("\nAplicando rotação w=1...")
for i in range(100):
    action = torch.tensor([[0.0, 1.0]], device=env.device)
    env.step(action)
    
    pos = env.robot.data.root_pos_w[0]
    if i % 20 == 0:
        print(f"Step {i}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")

env.close()
app.app.close()
