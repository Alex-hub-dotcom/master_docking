#!/usr/bin/env python3
"""
Visualize checkpoint with GUI - watch the robot dock in real-time.
"""
import argparse
import torch
import torch.nn as nn
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--stage", type=int, default=22, help="Curriculum stage to test")
parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

# Launch WITH GUI (no --headless)
app = AppLauncher(args)
sim = app.app

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder

class Policy(nn.Module):
    LOG_STD_V_MIN, LOG_STD_V_MAX = -1.5, 0.2
    LOG_STD_W_MIN, LOG_STD_W_MAX = -1.0, 0.6

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
        return mean  # Deterministic action (no sampling)

def main():
    device = torch.device("cuda:0")

    # Setup environment
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_curriculum = True
    env = TekoEnv(cfg=cfg)

    # Load policy
    policy = Policy().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    print(f"\n{'='*60}")
    print(f"🎬 VISUAL EVALUATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Stage: {ckpt.get('curriculum_level', 'unknown')} (testing S{args.stage})")
    print(f"Mastered: {ckpt.get('mastered_stages', [])}")
    print(f"{'='*60}\n")

    # Set curriculum stage
    env.set_curriculum_level(args.stage)

    successes = 0
    total = 0

    try:
        obs_dict, _ = env.reset()
        obs = obs_dict["rgb"].to(device)
        
        for ep in range(args.episodes):
            done_all = False
            step = 0
            
            while not done_all:
                with torch.no_grad():
                    action = policy.act(obs)
                
                obs_dict, reward, term, trunc, _ = env.step(action)
                obs = obs_dict["rgb"].to(device)
                done = term | trunc
                
                # Check for successes
                for i in range(args.num_envs):
                    if done[i]:
                        total += 1
                        if reward[i] > 50:  # Success threshold
                            successes += 1
                            print(f"✅ SUCCESS! (Episode {total}, reward={reward[i]:.1f})")
                        else:
                            print(f"❌ Failed (Episode {total}, reward={reward[i]:.1f})")
                
                done_all = done.all()
                step += 1
                
                # Slow down for visualization
                time.sleep(0.01)
            
            # Reset for next episode
            obs_dict, _ = env.reset()
            obs = obs_dict["rgb"].to(device)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")

    finally:
        print(f"\n{'='*60}")
        print(f"📊 RESULTS: {successes}/{total} successful ({100*successes/max(1,total):.1f}%)")
        print(f"{'='*60}")
        env.close()
        sim.close()

if __name__ == "__main__":
    main()