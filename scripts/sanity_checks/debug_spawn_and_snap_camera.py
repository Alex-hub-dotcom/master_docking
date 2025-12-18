#!/usr/bin/env python3
"""
TEKO Sanity - Spawn & Snap Camera
=================================
Headless camera sanity script:
- Resets the env
- Runs warmup steps (camera init)
- Captures stacked grayscale frames and saves PNGs
- Checks whether the camera is updating (frame delta)
- Logs robot_pose, goal_pose, rel, dist_xy, yaw_error to a .txt file

Typical usage (inside the container):
  python /workspace/teko/scripts/sanity_scripts/debug_spawn_and_snap_camera.py --headless

Tip: choose an output_dir that is mounted so you can copy files back easily.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------------
# AppLauncher MUST be created before importing Isaac Sim-dependent modules
# -----------------------------------------------------------------------------
from isaaclab.app import AppLauncher


def maybe_add_teko_to_syspath() -> str | None:
    """Best-effort add TEKO source path to sys.path."""
    candidates = [
        "/workspace/teko/source/teko",
        "/home/schux00/teko/source/teko",
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            return p
    return None


def wrap_angle(x):
    """Wrap angle to [-pi, pi] using atan2(sin, cos)."""
    import torch
    return torch.atan2(torch.sin(x), torch.cos(x))


def main():
    parser = argparse.ArgumentParser(description="TEKO camera sanity snapshot (headless).")
    AppLauncher.add_app_launcher_args(parser)

    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments for debugging.")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Steps to let the camera initialize.")
    parser.add_argument("--capture_steps", type=int, default=3, help="How many steps to capture after warmup.")
    parser.add_argument("--move_linear", type=float, default=0.0, help="Linear action during capture (action units).")
    parser.add_argument("--move_angular", type=float, default=0.0, help="Angular action during capture (action units).")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/teko/scripts/sanity_scripts/out/camera",
        help="Directory to save PNGs and logs (must be creatable).",
    )

    args = parser.parse_args()
    args.enable_cameras = True  # CRITICAL

    # Launch Isaac Sim
    app = AppLauncher(args)
    sim_app = app.app

    # Ensure TEKO import path
    added = maybe_add_teko_to_syspath()
    if added:
        print(f"[INFO] Added TEKO to sys.path: {added}")
    else:
        print("[WARN] Could not auto-add TEKO path. Make sure PYTHONPATH is set correctly.")

    # Safe to import after AppLauncher
    import torch
    from PIL import Image

    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    # Output folders
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs

    # (Optional) disable curriculum for pure sensor sanity
    if hasattr(cfg, "enable_curriculum"):
        cfg.enable_curriculum = False

    env = TekoEnv(cfg=cfg)

    # Reset
    obs, _ = env.reset()

    print("\n" + "=" * 70)
    print("TEKO CAMERA SANITY SNAPSHOT")
    print("=" * 70)
    print(f"num_envs      : {args.num_envs}")
    print(f"warmup_steps  : {args.warmup_steps}")
    print(f"capture_steps : {args.capture_steps}")
    print(f"output        : {run_dir}")
    print("=" * 70)

    # Warmup steps (let camera initialize)
    for _ in range(args.warmup_steps):
        action = torch.zeros((args.num_envs, 2), device=env.device)
        obs, _, _, _, _ = env.step(action)

    def get_rgb_stack(obs_dict):
        """Fetch rgb stack from observation dict (best-effort)."""
        if "rgb" not in obs_dict:
            raise KeyError(f"obs_dict has no 'rgb'. Keys: {list(obs_dict.keys())}")
        return obs_dict["rgb"]

    prev_last = None
    log_lines = []

    # Capture loop
    for k in range(args.capture_steps):
        action = torch.zeros((args.num_envs, 2), device=env.device)
        action[:, 0] = float(args.move_linear)
        action[:, 1] = float(args.move_angular)

        obs, reward, terminated, truncated, info = env.step(action)

        rgb = get_rgb_stack(obs)

        # Use env 0 for printing/logging by default
        env_i = 0
        stack = rgb[env_i]  # (S, H, W) or (S, C, H, W)

        # If shape is (S, C, H, W), convert to grayscale by selecting channel 0
        if stack.ndim == 4:
            stack_gray = stack[:, 0, :, :]
        else:
            stack_gray = stack

        last_frame = stack_gray[-1]  # (H, W)

        # Check camera updates (mean absolute diff)
        mad = None
        if prev_last is not None:
            mad = (last_frame - prev_last).abs().mean().item()
        prev_last = last_frame.detach().clone()

        # Print quick stats
        print(f"\n[Capture step {k}]")
        print(f"  rgb shape         : {rgb.shape}")
        print(f"  last frame min/max: {last_frame.min().item():.4f} / {last_frame.max().item():.4f}")
        print(f"  last frame mean   : {last_frame.mean().item():.4f}")
        if mad is not None:
            print(f"  frame MAD vs prev : {mad:.6f}  (if ~0 always, the camera may be frozen)")

        # Save stacked frames as PNG (env0)
        for s in range(stack_gray.shape[0]):
            frame = stack_gray[s].clamp(0, 1)
            frame_u8 = (frame * 255.0).to(torch.uint8).cpu().numpy()
            Image.fromarray(frame_u8, mode="L").save(run_dir / f"env{env_i}_step{k}_frame{s}.png")

        # Save last frame as preview
        last_u8 = (last_frame.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        Image.fromarray(last_u8, mode="L").save(run_dir / f"env{env_i}_step{k}_preview_last.png")

        # Log robot/goal geometry (best-effort; depends on env internals)
        try:
            robot_pos = env.robot.data.root_pos_w[env_i].detach()
            robot_quat = env.robot.data.root_quat_w[env_i].detach()
            goal_pos = env.goal_positions[env_i].detach()

            rel = goal_pos - robot_pos
            dist_xy = torch.norm(rel[:2]).item()

            # yaw: use env helper if available, else compute here
            if hasattr(env, "_extract_yaw"):
                robot_yaw = env._extract_yaw(env.robot.data.root_quat_w)[env_i].item()
            else:
                qx, qy, qz, qw = robot_quat[0], robot_quat[1], robot_quat[2], robot_quat[3]
                siny_cosp = 2.0 * (qw * qz + qx * qy)
                cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                robot_yaw = torch.atan2(siny_cosp, cosy_cosp).item()

            goal_yaw = torch.atan2(rel[1], rel[0]).item()
            yaw_err = wrap_angle(torch.tensor(goal_yaw - robot_yaw)).abs().item()

            line = (
                f"step={k} | dist_xy={dist_xy:.4f} | yaw_err={yaw_err:.4f} | "
                f"robot_pos=({robot_pos[0].item():.3f},{robot_pos[1].item():.3f},{robot_pos[2].item():.3f}) | "
                f"goal_pos=({goal_pos[0].item():.3f},{goal_pos[1].item():.3f},{goal_pos[2].item():.3f}) | "
                f"rel=({rel[0].item():.3f},{rel[1].item():.3f},{rel[2].item():.3f}) | "
                f"reward={reward[env_i].item():.4f} | term={bool(terminated[env_i].item())} | trunc={bool(truncated[env_i].item())}"
            )
            print("  " + line)
            log_lines.append(line)

        except Exception as e:
            print(f"  [WARN] Could not log robot/goal geometry: {repr(e)}")

    # Write log file
    log_path = run_dir / "debug_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TEKO camera sanity snapshot log\n")
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"num_envs: {args.num_envs}\n")
        f.write(f"warmup_steps: {args.warmup_steps}\n")
        f.write(f"capture_steps: {args.capture_steps}\n")
        f.write(f"move_linear: {args.move_linear}\n")
        f.write(f"move_angular: {args.move_angular}\n\n")
        for line in log_lines:
            f.write(line + "\n")

    print("\n" + "=" * 70)
    print(f"✅ Saved frames + log to: {run_dir}")
    print(f"   Log file: {log_path}")
    print("=" * 70)

    # Cleanup
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
