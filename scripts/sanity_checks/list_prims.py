#!/usr/bin/env python3
"""List all prims containing 'Sphere' in their path."""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args)

import omni.usd

# Setup minimal env to get prims
import sys
import os
for p in ("/workspace/teko/source/teko", "/home/schux00/teko/source/teko"):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from teko.tasks.direct.teko.teko_env import TekoEnv
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
import torch

cfg = TekoEnvCfg()
cfg.scene.num_envs = 1
env = TekoEnv(cfg=cfg)
env.reset()

# Step a few times
for _ in range(5):
    env.step(torch.zeros((1, 2), device=env.device))

stage = omni.usd.get_context().get_stage()

print("\n" + "="*80)
print("PRIMS containing 'Sphere' or 'Connector':")
print("="*80)

for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "Sphere" in path or "Connector" in path:
        print(path)

print("\n" + "="*80)
print("Full Robot hierarchy (env_0):")
print("="*80)

def print_tree(prim, indent=0):
    path = str(prim.GetPath())
    if "/World/envs/env_0/Robot" in path or "/World/envs/env_0/RobotGoal" in path:
        print("  " * indent + prim.GetName())
        for child in prim.GetChildren():
            print_tree(child, indent + 1)

root = stage.GetPrimAtPath("/World/envs/env_0")
if root:
    for child in root.GetChildren():
        print_tree(child)

env.close()
app.app.close()
