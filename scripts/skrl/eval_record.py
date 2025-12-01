#!/usr/bin/env python3
"""
Headless evaluation - records video of robot docking.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import imageio

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 for cleaner video")
parser.add_argument("--stage", type=int, default=24, help="Curriculum stage to test")
parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to record")
parser.add_argument("--output", type=str, default="eval_video.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app = AppLauncher(args)
sim = app.app

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = create_visual_encoder("simple", 256, False)
        self.actor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Tanh())
        self.critic = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
        self.log_std_v = nn.Parameter(torch.tensor(0.0))
        self.log_std_w = nn.Parameter(torch.tensor(0.0))

    def act(self, obs):
        feat = self.encoder(obs)
        mean = self.actor(feat)
        return mean

def main():
    device = torch.device("cuda:0")

    # Setup environment
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_curriculum = True
    env = TekoEnv(cfg=cfg)

    # Load policy
    policy = Policy().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    print(f"\n{'='*60}")
    print(f"🎥 RECORDING EVALUATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    ckpt_stage = ckpt.get('curriculum_level', 'unknown')
    print(f"Checkpoint stage: {ckpt_stage}")
    print(f"Testing stage: S{args.stage}")
    print(f"Mastered: {ckpt.get('mastered_stages', [])}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    # Set curriculum stage
    env.set_curriculum_level(args.stage)

    frames = []
    successes = 0
    total = 0

    try:
        obs_dict, _ = env.reset()
        obs = obs_dict["rgb"].to(device)
        
        for ep in range(args.episodes):
            print(f"Recording episode {ep+1}/{args.episodes}...")
            done_all = False
            step = 0
            ep_frames = []
            
            while not done_all and step < 500:
                # Get camera frame for video (use last frame of stack)
                # obs is [N, 4, H, W] grayscale - take last frame
                frame = obs[0, -1].cpu().numpy()  # [H, W]
                
                # Convert to RGB for video (grayscale -> RGB)
                frame_rgb = np.stack([frame, frame, frame], axis=-1)  # [H, W, 3]
                
                # Normalize to 0-255
                frame_rgb = (frame_rgb * 255).astype(np.uint8)
                
                # Upscale for better visibility (64x64 -> 256x256)
                frame_rgb = np.repeat(np.repeat(frame_rgb, 4, axis=0), 4, axis=1)
                
                ep_frames.append(frame_rgb)
                
                with torch.no_grad():
                    action = policy.act(obs)
                
                obs_dict, reward, term, trunc, info = env.step(action)
                obs = obs_dict["rgb"].to(device)
                done = term | trunc
                
                done_all = done.all()
                step += 1
            
            # Check result
            total += 1
            # Use info dict for success check if available
            success = reward[0].item() > 50
            if success:
                successes += 1
                print(f"  ✅ SUCCESS! (steps={step}, reward={reward[0]:.1f})")
            else:
                print(f"  ❌ Failed (steps={step}, reward={reward[0]:.1f})")
            
            frames.extend(ep_frames)
            
            # Add separator frames between episodes
            if ep < args.episodes - 1:
                separator = np.zeros((256, 256, 3), dtype=np.uint8)
                for _ in range(10):
                    frames.append(separator)
            
            # Reset for next episode
            obs_dict, _ = env.reset()
            obs = obs_dict["rgb"].to(device)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")

    finally:
        # Save video
        if frames:
            output_path = Path(args.output)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            print(f"\nSaving video ({len(frames)} frames)...")
            imageio.mimsave(str(output_path), frames, fps=30)
            print(f"✅ Video saved: {output_path}")
        
        print(f"\n{'='*60}")
        print(f"📊 RESULTS: {successes}/{total} successful ({100*successes/max(1,total):.1f}%)")
        print(f"{'='*60}")
        
        env.close()
        sim.close()

if __name__ == "__main__":
    main()