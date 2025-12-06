#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
State-Based PPO Training for TEKO (Debugging)
==============================================

Train with ground truth state observations to validate:
- Curriculum progression
- Reward function effectiveness
- Training stability

Expected: 1000 envs, 6-8 hours → Stage 14+ with 60%+ SSR

Usage:
    python scripts/train_state_ppo.py

Author: Alexandre Schleier Neves da Silva
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from pathlib import Path

# Isaac Lab imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app


# TEKO imports
import sys
sys.path.insert(0, "/workspace/teko/source")

from teko.teko.tasks.direct.teko.teko_env_cfg_state import TekoEnvCfgState
from teko.teko.tasks.direct.teko.teko_env_state import TekoEnvState
from teko.teko.tasks.direct.teko.teko_brain.state_policy import StateMLP
from teko.teko.tasks.direct.teko.curriculum.curriculum_manager import (
    NUM_STAGES, STAGE_NAMES, set_curriculum_level
)


# =============================================================================
# PPO HYPERPARAMETERS
# =============================================================================
PPO_CONFIG = {
    "total_timesteps": 150_000_000,
    "rollout_steps": 256,
    "batch_size": 2048,
    "epochs": 6,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_loss_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "log_interval": 10,
    "save_interval": 30_000,
}


# =============================================================================
# CURRICULUM MANAGER
# =============================================================================

class CurriculumManager:
    """Manages curriculum progression based on success rate."""
    
    def __init__(self, env):
        self.env = env
        self.current_stage = 0
        self.stage_steps = 0
        self.stage_successes = []
        self.success_window = 50
        
        # Stage thresholds (same as vision-based)
        self.thresholds = {
            0: 0.75, 1: 0.70, 2: 0.70,           # Forward
            3: 0.65, 4: 0.65, 5: 0.65, 6: 0.65,  # Small offsets
            7: 0.60, 8: 0.60, 9: 0.60, 10: 0.55, # Medium offsets + turns
            11: 0.45, 12: 0.45, 13: 0.45,        # Blind stages
            14: 0.45, 15: 0.45, 16: 0.45,
        }
        
        self.min_steps = 50_000
        self.max_steps = {
            i: 800_000 if i < 8 else 1_200_000 if i < 11 else 2_000_000
            for i in range(NUM_STAGES)
        }
    
    def update(self, done: torch.Tensor, success: torch.Tensor):
        """Update curriculum based on episode completions."""
        # Count episodes (not environment steps!)
        episodes_done = done.sum().item()
        self.stage_steps += episodes_done
        
        # Track successes for completed episodes
        for is_done, is_success in zip(done.cpu(), success.cpu()):
            if is_done:
                self.stage_successes.append(int(is_success))
                if len(self.stage_successes) > self.success_window:
                    self.stage_successes.pop(0)
        
        # Check advancement
        if len(self.stage_successes) >= self.success_window:
            ssr = np.mean(self.stage_successes)
            threshold = self.thresholds[self.current_stage]
            max_steps = self.max_steps[self.current_stage]
            
            can_advance = (
                (ssr >= threshold and self.stage_steps >= self.min_steps) or
                (self.stage_steps >= max_steps)
            )
            
            if can_advance and self.current_stage < NUM_STAGES - 1:
                reason = "SSR met" if ssr >= threshold else "Safety valve"
                print(f"\n{'='*70}")
                print(f"✅ {STAGE_NAMES[self.current_stage]} COMPLETE ({reason})")
                print(f"   SSR: {ssr:.1%} (threshold: {threshold:.1%})")
                print(f"   Episodes: {self.stage_steps:,}")
                print(f"{'='*70}\n")
                
                self.current_stage += 1
                self.stage_steps = 0
                self.stage_successes = []
                set_curriculum_level(self.env, self.current_stage)
    
    def get_stats(self) -> dict:
        ssr = np.mean(self.stage_successes) if self.stage_successes else 0.0
        return {
            "stage": self.current_stage,
            "ssr": ssr,
            "episodes": self.stage_steps,
            "threshold": self.thresholds[self.current_stage],
            "max_episodes": self.max_steps[self.current_stage],
        }


# =============================================================================
# PPO AGENT (WITH PROPER STOCHASTIC POLICY)
# =============================================================================

class PPOAgent:
    """PPO with Gaussian policy for continuous control."""
    
    def __init__(self, policy: StateMLP, cfg: dict, device: str):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["learning_rate"])
        
        # Learnable log_std
        self.log_std = nn.Parameter(torch.zeros(2, device=device))  # [v, w]
        self.optimizer.add_param_group({"params": [self.log_std]})
        
        self.rollout_buffer = {
            "states": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
        }
    
    def act(self, state: torch.Tensor):
        """Sample action from policy."""
        with torch.no_grad():
            mean, value = self.policy(state)
            std = torch.exp(self.log_std)
            
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Clamp actions
            action = torch.clamp(action, -1.0, 1.0)
            
        return action, value.squeeze(-1), log_prob
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition."""
        self.rollout_buffer["states"].append(state)
        self.rollout_buffer["actions"].append(action)
        self.rollout_buffer["rewards"].append(reward)
        self.rollout_buffer["values"].append(value)
        self.rollout_buffer["log_probs"].append(log_prob)
        self.rollout_buffer["dones"].append(done)
    
    def compute_returns(self, next_value: torch.Tensor):
        """Compute GAE returns."""
        rewards = torch.stack(self.rollout_buffer["rewards"])
        values = torch.stack(self.rollout_buffer["values"])
        dones = torch.stack(self.rollout_buffer["dones"])
        
        T, N = rewards.shape
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = values[t + 1]
            
            delta = rewards[t] + self.cfg["gamma"] * next_val * next_non_terminal - values[t]
            gae = delta + self.cfg["gamma"] * self.cfg["gae_lambda"] * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        return returns, advantages
    
    def update(self, next_state: torch.Tensor):
        """PPO update with proper gradient flow."""
        # Compute returns
        with torch.no_grad():
            _, next_value = self.policy(next_state)
            next_value = next_value.squeeze(-1)
        
        returns, advantages = self.compute_returns(next_value)
        
        # Flatten
        states = torch.stack(self.rollout_buffer["states"]).view(-1, 4)
        actions = torch.stack(self.rollout_buffer["actions"]).view(-1, 2)
        old_log_probs = torch.stack(self.rollout_buffer["log_probs"]).view(-1)
        returns = returns.view(-1)
        advantages = advantages.view(-1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch updates
        batch_size = self.cfg["batch_size"]
        total_samples = len(states)
        
        policy_losses = []
        value_losses = []
        entropies = []
        
        for _ in range(self.cfg["epochs"]):
            indices = torch.randperm(total_samples, device=self.device)
            
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Forward pass
                mean, value = self.policy(batch_states)
                value = value.squeeze(-1)
                std = torch.exp(self.log_std)
                
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Policy loss
                ratio = torch.exp(log_prob - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.cfg["clip_epsilon"], 1 + self.cfg["clip_epsilon"]) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(value, batch_returns)
                
                # Total loss
                loss = (
                    policy_loss +
                    self.cfg["value_loss_coef"] * value_loss -
                    self.cfg["entropy_coef"] * entropy
                )
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + [self.log_std],
                    self.cfg["max_grad_norm"]
                )
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
        
        # Clear buffer
        self.rollout_buffer = {k: [] for k in self.rollout_buffer}
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
        }


# =============================================================================
# DIAGNOSTIC MODE
# =============================================================================

DIAGNOSTIC_MODE = False  # Set True to debug


def run_diagnostics(env, policy, device):
    """Run comprehensive diagnostics."""
    print("\n" + "="*70)
    print("🔬 DIAGNOSTIC MODE")
    print("="*70)
    
    obs_dict, _ = env.reset()
    state = obs_dict["policy"]
    
    print(f"\n1️⃣ OBSERVATION CHECK:")
    print(f"   Shape: {state.shape}")
    print(f"   Device: {state.device}")
    print(f"   Range: [{state.min().item():.3f}, {state.max().item():.3f}]")
    print(f"   Sample states (first 3):")
    for i in range(min(3, state.shape[0])):
        print(f"      Env {i}: [{state[i,0]:.3f}, {state[i,1]:.3f}, {state[i,2]:.3f}, {state[i,3]:.3f}]")
    
    print(f"\n2️⃣ DISTANCE CHECK:")
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    print(f"   Surface XY: [{surface_xy.min().item():.4f}, {surface_xy.max().item():.4f}]")
    print(f"   Initial successes: {(surface_xy < 0.03).sum().item()}/{state.shape[0]}")
    
    print(f"\n3️⃣ REWARD CHECK (10 steps):")
    for step_i in range(10):
        action = torch.randn(state.shape[0], 2, device=device) * 0.3
        obs_dict, reward, term, trunc, _ = env.step(action)
        done = term | trunc
        
        _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
        success = surface_xy < 0.03
        
        print(f"   Step {step_i:2d}: R=[{reward.min().item():6.1f}, {reward.max().item():6.1f}] "
              f"Dist=[{surface_xy.min().item():.3f}, {surface_xy.max().item():.3f}] "
              f"Succ={success.sum().item():3d}")
    
    print(f"\n4️⃣ CURRICULUM:")
    print(f"   Stage: {env.curriculum_level}")
    print(f"   Name: {STAGE_NAMES[env.curriculum_level]}")
    
    print(f"\n5️⃣ POLICY CHECK:")
    with torch.no_grad():
        mean, value = policy(state)
    print(f"   Mean shape: {mean.shape}, range: [{mean.min().item():.3f}, {mean.max().item():.3f}]")
    print(f"   Value shape: {value.shape}, range: [{value.min().item():.2f}, {value.max().item():.2f}]")
    
    print("\n" + "="*70)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*70 + "\n")
    
    exit()


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = TekoEnvCfgState()
    env = TekoEnvState(cfg=cfg)
    device = env.device
    
    policy = StateMLP(state_dim=4, action_dim=2, hidden_dim=128).to(device)
    agent = PPOAgent(policy, PPO_CONFIG, device)
    curriculum = CurriculumManager(env)
    
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(f"teko_state_debug/{run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # DIAGNOSTIC MODE
    if DIAGNOSTIC_MODE:
        run_diagnostics(env, policy, device)
    
    # Training
    obs, _ = env.reset()
    state = obs["policy"]
    
    total_steps = 0
    episode_rewards = []
    
    print(f"\n🚀 State-based training")
    print(f"   Envs: {cfg.num_envs}")
    print(f"   Total steps: {PPO_CONFIG['total_timesteps']:,}")
    print(f"   Checkpoints: {ckpt_dir}\n")
    
    while total_steps < PPO_CONFIG["total_timesteps"]:
        # Collect rollout
        for _ in range(PPO_CONFIG["rollout_steps"]):
            action, value, log_prob = agent.act(state)
            next_obs, reward, term, trunc, _ = env.step(action)
            next_state = next_obs["policy"]
            done = term | trunc
            
            # Compute success
            _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
            success = surface_xy < 0.03
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            curriculum.update(done, success)
            
            state = next_state
            total_steps += cfg.num_envs
            
            if done.any():
                episode_rewards.extend(reward[done].cpu().tolist())
        
        # Update
        losses = agent.update(state)
        
        # Log
        if total_steps % (PPO_CONFIG["log_interval"] * PPO_CONFIG["rollout_steps"] * cfg.num_envs) < cfg.num_envs:
            stats = curriculum.get_stats()
            avg_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
            
            print(f"[{total_steps:>9}] S{stats['stage']:02d} | "
                  f"SSR={stats['ssr']*100:4.1f}% | "
                  f"R={avg_r:6.1f} | "
                  f"VL={losses['value_loss']:6.2f} | "
                  f"Ent={losses['entropy']:.3f} | "
                  f"Ep={stats['episodes']:>6}/{stats['max_episodes']}")
        
        # Save
        if total_steps % PPO_CONFIG["save_interval"] < cfg.num_envs:
            torch.save({
                "policy": policy.state_dict(),
                "log_std": agent.log_std,
                "optimizer": agent.optimizer.state_dict(),
                "total_steps": total_steps,
                "stage": curriculum.current_stage,
            }, ckpt_dir / f"ckpt_{total_steps}.pt")
            print(f"💾 Checkpoint saved")
    
    print(f"\n✅ Training complete! Final stage: {curriculum.current_stage}/{NUM_STAGES-1}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()