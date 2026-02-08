"""
TEKO Full Behavior Validation
==============================
Tests: motion, rotation, approach, docking success, reward signal.
Saves camera frames at key moments.
"""
import argparse, sys, os, time
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

from isaaclab.app import AppLauncher
launcher_args = argparse.Namespace(headless=args.headless, enable_cameras=True)
app_launcher = AppLauncher(launcher_args)
simulation_app = app_launcher.app

import torch
import numpy as np
from PIL import Image
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.teko_env_tiled_imu import TekoEnvTiledIMU

cfg = TekoEnvCfg()
cfg.scene.num_envs = args.num_envs
env = TekoEnvTiledIMU(cfg=cfg)

out_dir = f"/workspace/teko/logs/full_debug/{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(out_dir, exist_ok=True)

def get_state():
    rp = env.robot.data.root_pos_w[0].cpu().numpy()
    rq = env.robot.data.root_quat_w[0].cpu().numpy()
    gp = env.goal_positions[0].cpu().numpy()
    x, y, z, w = rq
    yaw = np.rad2deg(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    fp, mp, sxy, s3d = env.get_sphere_distances_from_physics()
    return {
        "robot_pos": rp, "yaw": yaw, "goal_pos": gp,
        "female": fp[0].cpu().numpy(), "male": mp[0].cpu().numpy(),
        "dist_xy": sxy[0].item(), "dist_3d": s3d[0].item(),
    }

def save_frame(name):
    env.tiled_camera.update(dt=0.0)
    raw = env.tiled_camera.data.output["rgb"][0]
    if raw.shape[-1] == 4:
        raw = raw[..., :3]
    Image.fromarray(raw.cpu().numpy().astype("uint8")).save(f"{out_dir}/{name}.png")

def print_state(label, state):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Robot pos:  ({state['robot_pos'][0]:.4f}, {state['robot_pos'][1]:.4f}, {state['robot_pos'][2]:.4f})")
    print(f"  Robot yaw:  {state['yaw']:.1f}°")
    print(f"  Goal pos:   ({state['goal_pos'][0]:.4f}, {state['goal_pos'][1]:.4f}, {state['goal_pos'][2]:.4f})")
    print(f"  Female:     ({state['female'][0]:.4f}, {state['female'][1]:.4f}, {state['female'][2]:.4f})")
    print(f"  Male:       ({state['male'][0]:.4f}, {state['male'][1]:.4f}, {state['male'][2]:.4f})")
    print(f"  Dist XY:    {state['dist_xy']:.4f}m")
    print(f"  Dist 3D:    {state['dist_3d']:.4f}m")

results = []

# ============================================================
# TEST 1: Initial spawn
# ============================================================
obs, _ = env.reset()
for _ in range(10):
    env.step(torch.zeros((1, 2), device=env.device))

s0 = get_state()
print_state("TEST 1: INITIAL SPAWN", s0)
save_frame("01_initial_spawn")
results.append(("Initial spawn", "OK" if s0['dist_xy'] < 1.0 else "FAIL"))

# ============================================================
# TEST 2: Forward motion (action=[1, 0])
# ============================================================
total_reward = 0.0
for i in range(100):
    obs, reward, term, trunc, _ = env.step(torch.tensor([[1.0, 0.0]], device=env.device))
    total_reward += reward[0].item()
    if (i+1) % 25 == 0:
        s = get_state()
        save_frame(f"02_forward_step{i+1}")
        print(f"  Step {i+1}: pos_x={s['robot_pos'][0]:.4f}, dist_xy={s['dist_xy']:.4f}m, reward={reward[0].item():.2f}")

s1 = get_state()
print_state("TEST 2: AFTER 100 FORWARD STEPS", s1)
dx = s1['robot_pos'][0] - s0['robot_pos'][0]
dist_delta = s0['dist_xy'] - s1['dist_xy']
print(f"  Robot moved dx={dx:.4f}m")
print(f"  Distance decreased by {dist_delta:.4f}m")
print(f"  Total reward: {total_reward:.2f}")
results.append(("Forward motion", "OK" if abs(dx) > 0.01 else "FAIL - no movement"))
results.append(("Distance decreasing", "OK" if dist_delta > 0 else "FAIL - distance increased"))
results.append(("Reward positive on approach", "OK" if total_reward > 0 else "WARN - negative reward"))

# ============================================================
# TEST 3: Rotation (action=[0, 1])
# ============================================================
env.reset()
for _ in range(10):
    env.step(torch.zeros((1, 2), device=env.device))
s_pre_rot = get_state()

for i in range(50):
    env.step(torch.tensor([[0.0, 1.0]], device=env.device))

s_post_rot = get_state()
yaw_delta = abs(s_post_rot['yaw'] - s_pre_rot['yaw'])
print_state("TEST 3: AFTER 50 ROTATION STEPS", s_post_rot)
print(f"  Yaw changed by {yaw_delta:.1f}°")
save_frame("03_after_rotation")
results.append(("Rotation works", "OK" if yaw_delta > 5.0 else "FAIL - no rotation"))

# ============================================================
# TEST 4: Drive to docking (many forward steps)
# ============================================================
env.reset()
for _ in range(10):
    env.step(torch.zeros((1, 2), device=env.device))

min_dist = 999.0
success_detected = False
for i in range(500):
    obs, reward, term, trunc, _ = env.step(torch.tensor([[1.0, 0.0]], device=env.device))
    s = get_state()
    min_dist = min(min_dist, s['dist_xy'])

    if term[0].item() or trunc[0].item():
        if hasattr(env, '_last_success') and env._last_success[0].item():
            success_detected = True
            save_frame(f"04_success_at_step{i}")
            print(f"\n  SUCCESS detected at step {i}! dist_xy={s['dist_xy']:.4f}m")
        else:
            save_frame(f"04_term_at_step{i}")
            print(f"\n  Terminated at step {i} (not success). dist_xy={s['dist_xy']:.4f}m")
        break

    if (i+1) % 100 == 0:
        print(f"  Step {i+1}: dist_xy={s['dist_xy']:.4f}m, reward={reward[0].item():.2f}")

print(f"  Minimum distance reached: {min_dist:.4f}m")
results.append(("Can reach docking distance", "OK" if min_dist < 0.05 else f"WARN - min_dist={min_dist:.4f}m"))
results.append(("Success detection", "OK" if success_detected else "NOT TRIGGERED (may need more steps or alignment)"))

# ============================================================
# TEST 5: Camera updates between frames
# ============================================================
env.reset()
for _ in range(10):
    env.step(torch.zeros((1, 2), device=env.device))

frames = []
for i in range(3):
    env.step(torch.tensor([[0.5, 0.3]], device=env.device))
    env.tiled_camera.update(dt=0.0)
    raw = env.tiled_camera.data.output["rgb"][0]
    frames.append(raw.float().cpu())

diffs = []
for i in range(len(frames)-1):
    mad = (frames[i+1] - frames[i]).abs().mean().item()
    diffs.append(mad)
    print(f"  Frame {i} -> {i+1} MAD: {mad:.4f}")

results.append(("Camera updates between frames", "OK" if all(d > 0.1 for d in diffs) else "WARN - low frame diff"))

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
for name, status in results:
    icon = "✅" if status == "OK" else ("⚠️" if "WARN" in status else "❌")
    print(f"  {icon} {name}: {status}")
print(f"\n  Images saved to: {out_dir}")
print(f"{'='*60}")

env.close()
simulation_app.close()