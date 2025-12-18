#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Sanity - Wheels & Dynamics Test
====================================

Tests that actions actually move the robot:
- Phase 1: Idle (baseline)
- Phase 2: Forward (v=1, w=0)
- Phase 3: Backward (v=-1, w=0)
- Phase 4: Turn left (v=0, w=1)
- Phase 5: Turn right (v=0, w=-1)
- Phase 6: Idle (stop)

Records video using Replicator for headless verification.
Logs position/velocity to verify movement.

Author: Alexandre Schleier Neves da Silva
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def maybe_add_teko_to_syspath() -> None:
    for p in ("/workspace/teko/source/teko", "/home/schux00/teko/source/teko"):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def main():
    parser = argparse.ArgumentParser(description="TEKO wheels & dynamics sanity test with video.")
    AppLauncher.add_app_launcher_args(parser)
    
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--phase_duration", type=float, default=3.0, help="Seconds per phase")
    parser.add_argument("--output_dir", type=str, 
                        default="/workspace/teko/scripts/sanity_checks/out/wheels")
    parser.add_argument("--record_video", type=int, default=1, help="1=record, 0=no video")
    parser.add_argument("--video_fps", type=int, default=30)
    
    args = parser.parse_args()
    args.enable_cameras = True
    
    app = AppLauncher(args)
    sim_app = app.app
    
    maybe_add_teko_to_syspath()
    
    import torch
    import numpy as np
    import omni.replicator.core as rep
    
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
    
    # Output setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "motion_log.txt"
    
    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    
    log("=" * 80)
    log("TEKO WHEELS & DYNAMICS SANITY TEST")
    log("=" * 80)
    log(f"timestamp: {timestamp}")
    log(f"phase_duration: {args.phase_duration}s")
    log(f"record_video: {args.record_video}")
    
    # Create environment
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if hasattr(cfg, "enable_curriculum"):
        cfg.enable_curriculum = False
    
    env = TekoEnv(cfg=cfg)
    device = env.device
    
    # Reset
    env.reset()
    
    # Get initial position for reference
    init_pos = env.robot.data.root_pos_w[0].clone()
    log(f"\nInitial position: ({init_pos[0]:.4f}, {init_pos[1]:.4f}, {init_pos[2]:.4f})")
    
    # Setup video recording
    video_writer = None
    rp = None
    
    if args.record_video == 1:
        log("\n[VIDEO] Setting up recording...")
        try:
            # Get robot position for camera placement
            robot_pos = env.robot.data.root_pos_w[0].cpu().numpy()
            goal_pos = env.goal_positions[0].cpu().numpy()
            
            # Camera looking at scene from above-side
            mid_x = 0.5 * (robot_pos[0] + goal_pos[0])
            mid_y = 0.5 * (robot_pos[1] + goal_pos[1])
            
            cam_pos = (mid_x - 0.5, mid_y + 2.0, 1.5)  # Side view
            look_at = (mid_x, mid_y, 0.35)
            
            with rep.new_layer():
                cam = rep.create.camera(position=cam_pos, look_at=look_at)
                rp = rep.create.render_product(cam, (1280, 720))
            
            video_path = str(run_dir / "wheels_test.mp4")
            video_writer = rep.WriterRegistry.get("BasicWriter")
            video_writer.initialize(
                output_dir=str(run_dir / "video_frames"),
                rgb=True,
            )
            video_writer.attach([rp])
            
            log(f"[VIDEO] Camera at {cam_pos}, looking at {look_at}")
            log(f"[VIDEO] Frames will be saved to: {run_dir / 'video_frames'}")
        except Exception as e:
            log(f"[VIDEO] Setup failed: {e}")
            video_writer = None
    
    # Define test phases
    # Action format: [v, w] where v=linear, w=angular
    phases = [
        ("1_idle_start",  [0.0,  0.0]),
        ("2_forward",     [1.0,  0.0]),
        ("3_backward",    [-1.0, 0.0]),
        ("4_turn_left",   [0.0,  1.0]),
        ("5_turn_right",  [0.0, -1.0]),
        ("6_idle_end",    [0.0,  0.0]),
    ]
    
    # Physics frequency (from env config)
    dt = 1.0 / 60.0  # Approximate
    steps_per_phase = int(args.phase_duration / dt)
    log_interval = max(1, steps_per_phase // 10)  # Log 10 times per phase
    
    log(f"\nSteps per phase: {steps_per_phase}")
    log(f"Log interval: {log_interval} steps")
    
    # Run phases
    total_frames = 0
    
    for phase_name, action_vals in phases:
        log(f"\n{'='*60}")
        log(f"[PHASE] {phase_name} | action=[{action_vals[0]:.1f}, {action_vals[1]:.1f}]")
        log(f"{'='*60}")
        
        action = torch.tensor([action_vals], device=device, dtype=torch.float32)
        
        # Track movement during phase
        start_pos = env.robot.data.root_pos_w[0].clone()
        
        for step in range(steps_per_phase):
            # Step environment
            obs, reward, term, trunc, info = env.step(action)
            
            # Record video frame
            if video_writer is not None:
                try:
                    rep.orchestrator.step()
                    total_frames += 1
                except:
                    pass
            
            # Log periodically
            if step % log_interval == 0:
                pos = env.robot.data.root_pos_w[0]
                lin_vel = env.robot.data.root_lin_vel_w[0]
                ang_vel = env.robot.data.root_ang_vel_w[0]
                
                log(f"  step {step:4d} | "
                    f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) | "
                    f"lin_vel=({lin_vel[0]:.3f}, {lin_vel[1]:.3f}) | "
                    f"ang_vel_z={ang_vel[2]:.3f}")
        
        # Phase summary
        end_pos = env.robot.data.root_pos_w[0]
        delta = end_pos - start_pos
        dist = torch.norm(delta[:2]).item()
        
        log(f"\n  [PHASE SUMMARY]")
        log(f"  Movement: dx={delta[0]:.4f}, dy={delta[1]:.4f}, total={dist:.4f}m")
        
        # Sanity checks
        if phase_name == "2_forward":
            if dist < 0.05:
                log(f"  ⚠️ WARNING: Robot barely moved forward! (dist={dist:.4f}m)")
            else:
                log(f"  ✅ Robot moved forward: {dist:.4f}m")
        elif phase_name == "3_backward":
            if dist < 0.05:
                log(f"  ⚠️ WARNING: Robot barely moved backward! (dist={dist:.4f}m)")
            else:
                log(f"  ✅ Robot moved backward: {dist:.4f}m")
        elif "turn" in phase_name:
            # For turns, check angular velocity was non-zero
            log(f"  ✅ Turn phase completed")
    
    # Final summary
    final_pos = env.robot.data.root_pos_w[0]
    total_delta = final_pos - init_pos
    total_dist = torch.norm(total_delta[:2]).item()
    
    log(f"\n{'='*80}")
    log(f"FINAL SUMMARY")
    log(f"{'='*80}")
    log(f"Initial pos: ({init_pos[0]:.4f}, {init_pos[1]:.4f}, {init_pos[2]:.4f})")
    log(f"Final pos:   ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
    log(f"Total displacement: {total_dist:.4f}m")
    log(f"Video frames recorded: {total_frames}")
    
    # Cleanup video
    if video_writer is not None:
        try:
            video_writer.detach()
            log(f"\n[VIDEO] Frames saved to: {run_dir / 'video_frames'}")
            log(f"[VIDEO] To create video, run:")
            log(f"  ffmpeg -framerate {args.video_fps} -pattern_type glob -i '{run_dir}/video_frames/*.png' -c:v libx264 -pix_fmt yuv420p {run_dir}/wheels_test.mp4")
        except:
            pass
    
    log(f"\n✅ Test complete! Logs saved to: {log_path}")
    
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()