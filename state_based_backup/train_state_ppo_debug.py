#!/usr/bin/env python3
"""State-Based PPO Training - DEBUG VERSION"""

print("[DEBUG] Starting script...", flush=True)

from isaaclab.app import AppLauncher

print("[DEBUG] Launching Isaac Sim...", flush=True)

app_launcher = AppLauncher({
    "headless": True,
    "enable_cameras": False,
})
simulation_app = app_launcher.app

print("[DEBUG] Isaac Sim loaded!", flush=True)

import sys
import torch
sys.path.insert(0, "/workspace/teko/source/teko")

print("[DEBUG] Importing environment...", flush=True)

from teko.tasks.direct.teko.teko_env_state import TekoEnvState
from teko.tasks.direct.teko.teko_env_cfg_state import TekoEnvCfgState

print("[DEBUG] Creating config...", flush=True)

cfg = TekoEnvCfgState()
cfg.scene.num_envs = 16  # Start small for debugging
cfg.num_envs = 16

print(f"[DEBUG] Config created: num_envs={cfg.scene.num_envs}", flush=True)
print("[DEBUG] Creating environment...", flush=True)

try:
    env = TekoEnvState(cfg=cfg)
    print("[DEBUG] Environment created successfully!", flush=True)
    
    print("[DEBUG] Resetting environment...", flush=True)
    obs, _ = env.reset()
    print(f"[DEBUG] Reset done. Obs keys: {obs.keys()}", flush=True)
    print(f"[DEBUG] Policy obs shape: {obs['policy'].shape}", flush=True)
    
    # Run a few steps
    for i in range(5):
        action = torch.zeros((16, 2), device=env.device)
        obs, reward, term, trunc, info = env.step(action)
        print(f"[DEBUG] Step {i}: reward_mean={reward.mean():.3f}", flush=True)
    
    print("[DEBUG] SUCCESS! Environment works.", flush=True)
    env.close()
    
except Exception as e:
    print(f"[ERROR] {e}", flush=True)
    import traceback
    traceback.print_exc()

simulation_app.close()
print("[DEBUG] Done.", flush=True)
