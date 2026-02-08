import argparse, sys, os, time
parser = argparse.ArgumentParser(description="TEKO Sanity Check (Tiled)")
parser.add_argument("--test", type=str, default="camera")
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
out_dir = f"/workspace/teko/logs/sanity_tiled/{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(out_dir, exist_ok=True)
env.reset()
for _ in range(10):
    env.step(torch.zeros((env.num_envs, 2), device=env.device))
for i in range(5):
    obs, _, _, _, _ = env.step(torch.zeros((env.num_envs, 2), device=env.device))
    rgb = obs["rgb"][0]
    frame = rgb[-1].clamp(0, 1)
    img = (frame * 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(img, mode="L").save(f"{out_dir}/gray_frame_{i}.png")
    print(f"Frame {i}: min={frame.min():.3f} max={frame.max():.3f} mean={frame.mean():.3f}")

# Connector distances
female_pos, male_pos, surface_xy, surface_3d = env.get_sphere_distances_from_physics()
fp = female_pos[0].cpu().numpy()
mp = male_pos[0].cpu().numpy()
print(f"\n=== CONNECTOR POSITIONS ===")
print(f"Female (active rear): ({fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f})")
print(f"Male (static front):  ({mp[0]:.4f}, {mp[1]:.4f}, {mp[2]:.4f})")
print(f"Surface dist XY: {surface_xy[0].item():.4f}m")
print(f"Surface dist 3D: {surface_3d[0].item():.4f}m")

# Robot info
rp = env.robot.data.root_pos_w[0].cpu().numpy()
rq = env.robot.data.root_quat_w[0].cpu().numpy()
gp = env.goal_positions[0].cpu().numpy()
x,y,z,w = rq
yaw = np.rad2deg(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
print(f"\nRobot pos: ({rp[0]:.4f},{rp[1]:.4f},{rp[2]:.4f}) yaw={yaw:.1f}")
print(f"Goal pos:  ({gp[0]:.4f},{gp[1]:.4f},{gp[2]:.4f})")
print(f"Saved to: {out_dir}")
env.close()
simulation_app.close()
