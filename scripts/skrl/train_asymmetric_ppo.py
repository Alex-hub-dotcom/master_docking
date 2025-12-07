#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Asymmetric Actor-Critic PPO Training for TEKO Vision-Based Docking (v2.0)
==========================================================================
FIXED VERSION with:
- Proper curriculum advancement based on SSR
- Success rate tracking
- Entropy monitoring and fixing
- Detailed logging

Author: Alexandre Schleier Neves da Silva
Date: December 2024
"""
# Isaac Lab launcher (MUST come first!)
from isaaclab.app import AppLauncher

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
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Add TEKO to path
sys.path.insert(0, "/workspace/teko/source/teko")

from teko.tasks.direct.teko.teko_env import TekoEnv
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.curriculum.curriculum_manager import NUM_STAGES


# =============================================================================
# ASYMMETRIC POLICY (FIXED LOG_STD)
# =============================================================================

class SimpleCNN(torch.nn.Module):
    """CNN for 84x84 grayscale with LayerNorm."""
    
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        import math
        
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=6, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.ln1 = torch.nn.LayerNorm([32, 27, 27])
        self.ln2 = torch.nn.LayerNorm([64, 13, 13])
        self.ln3 = torch.nn.LayerNorm([128, 7, 7])
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(6272, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, feature_dim),
            torch.nn.ReLU(inplace=True),
        )
        
        self.feature_dim = feature_dim
        self._init_weights()
    
    def _init_weights(self):
        import math
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        
        x = self.conv3(x)
        x = self.ln3(x)
        x = torch.relu(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class AsymmetricActorCritic(torch.nn.Module):
    """
    Asymmetric actor-critic with FIXED log_std handling.
    """
    
    def __init__(
        self,
        vision_shape=(4, 84, 84),
        privileged_dim=7,
        action_dim=2,
        hidden_dim=256,
        init_log_std=-0.5,  # Start with moderate exploration
    ):
        super().__init__()
        import math
        
        self.vision_shape = vision_shape
        self.privileged_dim = privileged_dim
        self.action_dim = action_dim
        
        # Shared vision encoder
        self.vision_encoder = SimpleCNN(
            in_channels=vision_shape[0],
            feature_dim=hidden_dim
        )
        
        # Actor head (vision-only)
        self.actor_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, action_dim),
            torch.nn.Tanh(),
        )
        
        # FIXED: Separate learnable log_std with wider bounds
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), init_log_std))
        
        # Bounds for log_std (wider range for better exploration)
        self.LOG_STD_MIN = -2.0
        self.LOG_STD_MAX = 0.5
        
        # State encoder for privileged info
        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(privileged_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace=True),
        )
        
        # Critic head: vision(256) + state(128) → value
        self.critic_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + 128, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 1),
        )
        
        self._init_heads()
    
    def _init_heads(self):
        import math
        for module in [self.actor_head, self.state_encoder, self.critic_head]:
            for m in module.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        
        # Small init for final actor layer (important for stable training)
        torch.nn.init.orthogonal_(self.actor_head[-2].weight, gain=0.01)
    
    def _get_std(self):
        """Get clamped std (not log_std)."""
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return torch.exp(log_std)
    
    def forward_actor(self, vision):
        """Actor forward (vision-only)."""
        features = self.vision_encoder(vision)
        mean = self.actor_head(features)
        std = self._get_std().unsqueeze(0).expand(mean.shape[0], -1)
        return mean, std
    
    def forward_critic(self, vision, privileged):
        """Critic forward (vision + privileged state)."""
        vision_features = self.vision_encoder(vision)
        state_features = self.state_encoder(privileged)
        fused = torch.cat([vision_features, state_features], dim=-1)
        value = self.critic_head(fused)
        return value
    
    def forward(self, obs):
        """Full forward for training."""
        vision = obs["rgb"]
        privileged = obs["privileged"]
        
        mean, std = self.forward_actor(vision)
        value = self.forward_critic(vision, privileged)
        
        return mean, std, value
    
    def sample_action(self, obs, deterministic=False):
        """Sample action for rollout collection."""
        mean, std = self.forward_actor(obs["rgb"])
        
        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device)
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        
        return action, log_prob


# =============================================================================
# PPO TRAINER WITH CURRICULUM
# =============================================================================

class AsymmetricPPOTrainer:
    """PPO trainer with proper curriculum advancement."""
    
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
        # Curriculum settings
        ssr_threshold: float = 0.70,
        min_episodes_for_advancement: int = 500,
        ssr_window_size: int = 1000,
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
        
        # Curriculum settings
        self.ssr_threshold = ssr_threshold
        self.min_episodes_for_advancement = min_episodes_for_advancement
        
        # Optimizer (include log_std in optimization!)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=learning_rate
        )
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        
        # Curriculum tracking
        self.stage_successes = 0
        self.stage_episodes = 0
        self.stage_steps = 0
        
        # Rolling SSR window for more stable estimates
        self.ssr_window = deque(maxlen=ssr_window_size)
        
        # Episode tracking for current rollout
        self.rollout_successes = 0
        self.rollout_episodes = 0
        
        # Success threshold (reward indicating success)
        self.success_reward_threshold = 350.0  # Success bonus is 400
        
        # TensorBoard
        self.writer = SummaryWriter(f"runs/vision_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    def collect_rollout(self):
        """Collect rollout data with success tracking."""
        num_envs = self.env.num_envs
        
        # Storage
        obs_rgb = []
        obs_privileged = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        # Reset rollout counters
        self.rollout_successes = 0
        self.rollout_episodes = 0
        
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
            
            # ============================================================
            # TRACK SUCCESSES FOR CURRICULUM
            # ============================================================
            if done.any():
                done_indices = done.nonzero(as_tuple=False).squeeze(-1)
                done_rewards = reward[done_indices]
                
                # Count successes (high reward indicates success)
                successes = (done_rewards > self.success_reward_threshold).sum().item()
                num_done = done_indices.shape[0]
                
                self.rollout_successes += successes
                self.rollout_episodes += num_done
                
                # Add to rolling window (1 for success, 0 for failure)
                for i in range(num_done):
                    is_success = done_rewards[i].item() > self.success_reward_threshold
                    self.ssr_window.append(1 if is_success else 0)
            
            # Cache next observation
            self._obs_cache = next_obs
            
            # Update stats
            self.total_steps += num_envs
            self.stage_steps += num_envs
        
        # Update curriculum tracking
        self.stage_successes += self.rollout_successes
        self.stage_episodes += self.rollout_episodes
        self.total_episodes += self.rollout_episodes
        
        # Get final value for GAE
        with torch.no_grad():
            _, _, final_value = self.policy(self._obs_cache)
            final_value = final_value.squeeze(-1)
        
        # Stack tensors
        obs_rgb = torch.stack(obs_rgb)
        obs_privileged = torch.stack(obs_privileged)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        values = torch.stack(values)
        
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
            "rewards": rewards,
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
    
    def check_curriculum_advancement(self):
        """Check if we should advance to next curriculum stage."""
        if self.stage_episodes < self.min_episodes_for_advancement:
            return False
        
        # Compute SSR for current stage
        if self.stage_episodes > 0:
            stage_ssr = self.stage_successes / self.stage_episodes
        else:
            stage_ssr = 0.0
        
        # Also compute rolling SSR
        if len(self.ssr_window) > 0:
            rolling_ssr = sum(self.ssr_window) / len(self.ssr_window)
        else:
            rolling_ssr = 0.0
        
        # Use stage SSR for advancement decision
        if stage_ssr >= self.ssr_threshold:
            current_stage = self.env.curriculum_level
            next_stage = min(current_stage + 1, NUM_STAGES - 1)
            
            if next_stage > current_stage:
                print(f"\n{'='*80}")
                print(f"🎉 CURRICULUM ADVANCEMENT!")
                print(f"   Stage {current_stage} → Stage {next_stage}")
                print(f"   Stage SSR: {stage_ssr:.1%} (threshold: {self.ssr_threshold:.1%})")
                print(f"   Rolling SSR: {rolling_ssr:.1%}")
                print(f"   Episodes at stage: {self.stage_episodes}")
                print(f"{'='*80}\n")
                
                self.env.set_curriculum_level(next_stage)
                
                # Reset stage counters
                self.stage_successes = 0
                self.stage_episodes = 0
                self.stage_steps = 0
                self.ssr_window.clear()
                
                return True
        
        return False
    
    def train(self, total_rollouts: int, log_interval: int = 2, save_interval: int = 50):
        """Main training loop with curriculum."""
        print(f"\n{'='*80}")
        print(f"ASYMMETRIC ACTOR-CRITIC PPO TRAINING (v2.0)")
        print(f"{'='*80}")
        print(f"Environments: {self.env.num_envs}")
        print(f"Rollout steps: {self.rollout_steps}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs per rollout: {self.epochs}")
        print(f"Total rollouts: {total_rollouts}")
        print(f"SSR threshold for advancement: {self.ssr_threshold:.1%}")
        print(f"Min episodes per stage: {self.min_episodes_for_advancement}")
        print(f"{'='*80}\n")
        
        for rollout_idx in range(total_rollouts):
            # Collect rollout
            rollout_data = self.collect_rollout()
            
            # Update policy
            update_stats = self.update_policy(rollout_data)
            
            # Check curriculum advancement
            self.check_curriculum_advancement()
            
            # Logging
            if (rollout_idx + 1) % log_interval == 0:
                self._log_progress(rollout_idx + 1, rollout_data, update_stats)
            
            # Save checkpoint
            if (rollout_idx + 1) % save_interval == 0:
                self._save_checkpoint(rollout_idx + 1)
    
    def _log_progress(self, rollout_idx, rollout_data, update_stats):
        """Log training progress with SSR."""
        # Compute episode stats
        rewards = rollout_data["rewards"]
        returns = rollout_data["returns"]
        mean_return = returns.mean().item()
        max_return = returns.max().item()
        min_return = returns.min().item()
        mean_reward = rewards.mean().item()
        
        # Get curriculum info
        stage = self.env.curriculum_level
        
        # Compute SSR
        if self.stage_episodes > 0:
            stage_ssr = self.stage_successes / self.stage_episodes
        else:
            stage_ssr = 0.0
        
        if len(self.ssr_window) > 0:
            rolling_ssr = sum(self.ssr_window) / len(self.ssr_window)
        else:
            rolling_ssr = 0.0
        
        # Get log_std values
        log_std = self.policy.log_std.data.cpu().numpy()
        std = np.exp(np.clip(log_std, -2.0, 0.5))
        
        print(f"\n{'='*80}")
        print(f"Rollout {rollout_idx} | Stage S{stage} | Stage Steps: {self.stage_steps:,}")
        print(f"{'-'*80}")
        print(f"Total steps: {self.total_steps:,} | Total episodes: {self.total_episodes:,}")
        print(f"Stage episodes: {self.stage_episodes} | Stage successes: {self.stage_successes}")
        print(f"Stage SSR: {stage_ssr:.1%} | Rolling SSR: {rolling_ssr:.1%} | Threshold: {self.ssr_threshold:.1%}")
        print(f"{'-'*80}")
        print(f"Policy loss: {update_stats['policy_loss']:.4f} | Value loss: {update_stats['value_loss']:.4f}")
        print(f"Entropy: {update_stats['entropy']:.4f} | Approx KL: {update_stats['approx_kl']:.4f}")
        print(f"Log_std: [{log_std[0]:.3f}, {log_std[1]:.3f}] | Std: [{std[0]:.3f}, {std[1]:.3f}]")
        print(f"Mean reward: {mean_reward:.2f} | Return: {mean_return:.2f} (min: {min_return:.2f}, max: {max_return:.2f})")
        print(f"{'='*80}\n")
        
        # TensorBoard
        self.writer.add_scalar("curriculum/stage", stage, self.total_steps)
        self.writer.add_scalar("curriculum/stage_ssr", stage_ssr, self.total_steps)
        self.writer.add_scalar("curriculum/rolling_ssr", rolling_ssr, self.total_steps)
        self.writer.add_scalar("loss/policy", update_stats['policy_loss'], self.total_steps)
        self.writer.add_scalar("loss/value", update_stats['value_loss'], self.total_steps)
        self.writer.add_scalar("policy/entropy", update_stats['entropy'], self.total_steps)
        self.writer.add_scalar("policy/kl", update_stats['approx_kl'], self.total_steps)
        self.writer.add_scalar("policy/std_v", std[0], self.total_steps)
        self.writer.add_scalar("policy/std_w", std[1], self.total_steps)
        self.writer.add_scalar("reward/mean", mean_reward, self.total_steps)
        self.writer.add_scalar("reward/return_mean", mean_return, self.total_steps)
    
    def _save_checkpoint(self, rollout_idx):
        """Save model checkpoint."""
        checkpoint = {
            "rollout": rollout_idx,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "curriculum_level": self.env.curriculum_level,
            "stage_successes": self.stage_successes,
            "stage_episodes": self.stage_episodes,
            "stage_steps": self.stage_steps,
        }
        
        save_dir = Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        
        save_path = save_dir / f"asymmetric_ppo_v2_rollout_{rollout_idx}.pt"
        torch.save(checkpoint, save_path)
        print(f"[CHECKPOINT] Saved to {save_path}")
        
        # Also save "latest" for easy resumption
        latest_path = save_dir / "asymmetric_ppo_v2_latest.pt"
        torch.save(checkpoint, latest_path)


def create_policy(device="cuda"):
    """Create the asymmetric policy."""
    policy = AsymmetricActorCritic(
        vision_shape=(4, 84, 84),
        privileged_dim=7,
        action_dim=2,
        hidden_dim=256,
        init_log_std=-0.5,  # Start with moderate exploration
    ).to(device)
    
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"[INFO] AsymmetricActorCritic: {total_params:,} parameters")
    
    return policy


def main():
    """Main training entry point."""
    
    # Environment config
    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = 50
    env_cfg.asymmetric_critic = True
    env_cfg.camera.width = 84
    env_cfg.camera.height = 84
    
    print("[INFO] Creating environment...")
    env = TekoEnv(cfg=env_cfg)
    
    print("[INFO] Creating asymmetric policy...")
    policy = create_policy(device="cuda")
    
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
        # Curriculum settings
        ssr_threshold=0.70,           # 70% success rate to advance
        min_episodes_for_advancement=500,  # Min episodes before checking
        ssr_window_size=1000,         # Rolling window for SSR
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