#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Asymmetric Actor-Critic PPO Training for TEKO Vision-Based Docking
===================================================================
- Actor uses ONLY vision (84×84 grayscale, 4 frames) → deployable
- Critic uses vision + privileged state [dx, dy, dz, yaw, vx, vy, w]
- Shared vision encoder between actor/critic
- 100 parallel environments on RTX 3090
- Curriculum learning through 17 stages

Author: Alexandre Schleier Neves da Silva
Date: December 2024
"""
# Isaac Lab launcher (MUST come first!)
from isaaclab.app import AppLauncher
# AFTER (correct):
app_launcher = AppLauncher({
    "headless": True,
    "enable_cameras": True
})
simulation_app = app_launcher.app

import os
import sys
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Add TEKO to path
sys.path.insert(0, "/workspace/teko/source/teko")

from teko.tasks.direct.teko.teko_env import TekoEnv
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.asymmetric_policy import create_asymmetric_policy


class AsymmetricPPOTrainer:
    """PPO trainer for asymmetric actor-critic."""
    
    def __init__(
        self,
        env: TekoEnv,
        policy: torch.nn.Module,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 256,
        batch_size: int = 2048,
        epochs: int = 6,
        device: str = "cuda",
    ):
        self.env = env
        self.policy = policy
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=learning_rate
        )
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        
    def collect_rollout(self):
        """Collect rollout data with asymmetric observations."""
        num_envs = self.env.num_envs
        
        # Storage
        obs_rgb = []
        obs_privileged = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        # Get initial observation
        if not hasattr(self, '_obs_cache'):
            self._obs_cache, _ = self.env.reset()
        
        # Collect rollout
        for step in range(self.rollout_steps):
            obs = self._obs_cache
            
            # Forward pass (no gradients during rollout)
            with torch.no_grad():
                action, log_prob = self.policy.sample_action(obs, deterministic=False)
                _, _, value = self.policy(obs)
            
            # Store
            obs_rgb.append(obs["rgb"].clone())
            obs_privileged.append(obs["privileged"].clone())
            actions.append(action.clone())
            log_probs.append(log_prob.clone())
            values.append(value.squeeze(-1).clone())
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated
            
            # Store rewards and dones
            rewards.append(reward.clone())
            dones.append(done.clone())
            
            # Cache next observation
            self._obs_cache = next_obs
            
            # Update stats
            self.total_steps += num_envs
            self.total_episodes += done.sum().item()
        
        # Get final value for GAE
        with torch.no_grad():
            _, _, final_value = self.policy(self._obs_cache)
            final_value = final_value.squeeze(-1)
        
        # Stack tensors
        obs_rgb = torch.stack(obs_rgb)  # (T, N, 4, 84, 84)
        obs_privileged = torch.stack(obs_privileged)  # (T, N, 7)
        actions = torch.stack(actions)  # (T, N, 2)
        log_probs = torch.stack(log_probs)  # (T, N)
        rewards = torch.stack(rewards)  # (T, N)
        dones = torch.stack(dones)  # (T, N)
        values = torch.stack(values)  # (T, N)
        
        # Compute advantages and returns with GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = torch.zeros(num_envs, device=self.device)
        next_value = final_value
        
        for t in reversed(range(self.rollout_steps)):
            next_non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]
        
        return {
            "obs_rgb": obs_rgb,
            "obs_privileged": obs_privileged,
            "actions": actions,
            "old_log_probs": log_probs,
            "advantages": advantages,
            "returns": returns,
        }
        
    def update_policy(self, rollout_data):
        """Update policy with PPO."""
        # Flatten batch dimensions
        T, N = rollout_data["obs_rgb"].shape[:2]
        total_samples = T * N
        
        obs_rgb = rollout_data["obs_rgb"].view(total_samples, 4, 84, 84)
        obs_privileged = rollout_data["obs_privileged"].view(total_samples, 7)
        actions = rollout_data["actions"].view(total_samples, 2)
        old_log_probs = rollout_data["old_log_probs"].view(total_samples)
        advantages = rollout_data["advantages"].view(total_samples)
        returns = rollout_data["returns"].view(total_samples)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Statistics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        num_updates = 0
        
        # Multiple epochs over the data
        for epoch in range(self.epochs):
            # Random permutation
            indices = torch.randperm(total_samples, device=self.device)
            
            # Mini-batch updates
            for start_idx in range(0, total_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                batch_obs = {
                    "rgb": obs_rgb[batch_indices],
                    "privileged": obs_privileged[batch_indices],
                }
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                mean, std, values = self.policy(batch_obs)
                
                # Compute log probs
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Policy loss (PPO clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = 0.5 * ((values.squeeze(-1) - batch_returns) ** 2).mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.value_loss_coef * value_loss 
                    - self.entropy_coef * entropy
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_approx_kl += ((ratio - 1.0) - torch.log(ratio)).mean().item()
                num_updates += 1
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_approx_kl / num_updates,
        }
    
    def train(self, total_rollouts: int, log_interval: int = 2, save_interval: int = 50):
        """Main training loop."""
        print(f"\n{'='*80}")
        print(f"ASYMMETRIC ACTOR-CRITIC PPO TRAINING")
        print(f"{'='*80}")
        print(f"Environments: {self.env.num_envs}")
        print(f"Rollout steps: {self.rollout_steps}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs per rollout: {self.epochs}")
        print(f"Total rollouts: {total_rollouts}")
        print(f"{'='*80}\n")
        
        for rollout_idx in range(total_rollouts):
            # Collect rollout
            rollout_data = self.collect_rollout()
            
            # Update policy
            update_stats = self.update_policy(rollout_data)
            
            # Logging
            if (rollout_idx + 1) % log_interval == 0:
                self._log_progress(rollout_idx + 1, rollout_data, update_stats)
            
            # Save checkpoint
            if (rollout_idx + 1) % save_interval == 0:
                self._save_checkpoint(rollout_idx + 1)
    
    def _log_progress(self, rollout_idx, rollout_data, update_stats):
        """Log training progress."""
        # Compute episode stats
        returns = rollout_data["returns"]
        mean_return = returns.mean().item()
        max_return = returns.max().item()
        min_return = returns.min().item()
        
        # Get curriculum info
        stage = self.env.curriculum_level
        stage_steps = getattr(self.env, 'stage_steps', 0)
        
        print(f"\n{'='*80}")
        print(f"Rollout {rollout_idx} | Stage S{stage} | Steps: {stage_steps:,}")
        print(f"{'-'*80}")
        print(f"Total steps: {self.total_steps:,} | Total episodes: {self.total_episodes:,}")
        print(f"Policy loss: {update_stats['policy_loss']:.4f} | Value loss: {update_stats['value_loss']:.4f}")
        print(f"Entropy: {update_stats['entropy']:.4f} | Approx KL: {update_stats['approx_kl']:.4f}")
        print(f"Return: {mean_return:.2f} (min: {min_return:.2f}, max: {max_return:.2f})")
        print(f"{'='*80}\n")
    
    def _save_checkpoint(self, rollout_idx):
        """Save model checkpoint."""
        checkpoint = {
            "rollout": rollout_idx,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "curriculum_level": self.env.curriculum_level,
        }
        
        save_path = Path(f"checkpoints/asymmetric_ppo_rollout_{rollout_idx}.pt")
        save_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"[CHECKPOINT] Saved to {save_path}")


def main():
    """Main training entry point."""
    
    # Environment config
    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = 50
    env_cfg.asymmetric_critic = True  # Enable asymmetric observations
    env_cfg.camera.width = 84
    env_cfg.camera.height = 84
    
    print("[INFO] Creating environment...")
    env = TekoEnv(cfg=env_cfg)
    
    print("[INFO] Creating asymmetric policy...")
    policy = create_asymmetric_policy(
        vision_shape=(4, 84, 84),
        privileged_dim=7,
        action_dim=2,
        device="cuda"
    )
    
    print("[INFO] Creating trainer...")
    trainer = AsymmetricPPOTrainer(
        env=env,
        policy=policy,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        rollout_steps=256,
        batch_size=2048,
        epochs=6,
        device="cuda",
    )
    
    print("[INFO] Starting training...")
    trainer.train(
        total_rollouts=10000,
        log_interval=2,
        save_interval=50,
    )
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()