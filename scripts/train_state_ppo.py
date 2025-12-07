#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
State-Based PPO Training for TEKO Docking (Debugging)
======================================================
Uses ground truth state [dx, dy, dz, yaw_error] instead of vision.
Same reward function and curriculum as vision-based training.

Purpose: Validate that curriculum + rewards work before vision debugging.

If this works → reward/curriculum are fine, vision is the problem
If this fails → fix reward/curriculum first

Author: Alexandre Schleier Neves da Silva
Date: December 2024
"""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({
    "headless": True,
    "enable_cameras": False,  # No cameras needed!
})
simulation_app = app_launcher.app

import sys
import torch
import numpy as np
from pathlib import Path
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.insert(0, "/workspace/teko/source/teko")

from teko.tasks.direct.teko.teko_env_state import TekoEnvState
from teko.tasks.direct.teko.teko_env_cfg_state import TekoEnvCfgState
from teko.tasks.direct.teko.curriculum.curriculum_manager import NUM_STAGES


# =============================================================================
# STATE-BASED MLP POLICY
# =============================================================================

class StatePolicy(torch.nn.Module):
    """Simple MLP for state-based control."""
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=256, init_log_std=-0.5):
        super().__init__()
        
        # Shared feature extractor
        self.features = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
        )
        
        # Actor head
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
            torch.nn.Tanh(),
        )
        
        # Critic head
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        
        # Learnable log_std
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), init_log_std))
        self.LOG_STD_MIN = -2.0
        self.LOG_STD_MAX = 0.5
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        # Small init for actor output
        torch.nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)
    
    def _get_std(self):
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return torch.exp(log_std)
    
    def forward(self, state):
        features = self.features(state)
        mean = self.actor(features)
        value = self.critic(features)
        std = self._get_std().unsqueeze(0).expand(mean.shape[0], -1)
        return mean, std, value
    
    def sample_action(self, state, deterministic=False):
        mean, std, _ = self.forward(state)
        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


# =============================================================================
# PPO TRAINER
# =============================================================================

class StatePPOTrainer:
    """PPO trainer for state-based debugging."""
    
    def __init__(
        self,
        env,
        policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        rollout_steps=128,
        batch_size=2048,
        epochs=6,
        device="cuda",
        ssr_threshold=0.70,
        min_episodes=500,
    ):
        self.env = env
        self.policy = policy
        self.device = device
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.ssr_threshold = ssr_threshold
        self.min_episodes = min_episodes
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # Stats
        self.total_steps = 0
        self.total_episodes = 0
        self.stage_successes = 0
        self.stage_episodes = 0
        self.stage_steps = 0
        self.ssr_window = deque(maxlen=1000)
        
        self.success_threshold = 350.0
        
        # TensorBoard
        self.writer = SummaryWriter(f"runs/state_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    def collect_rollout(self):
        num_envs = self.env.num_envs
        
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        rollout_successes = 0
        rollout_episodes = 0
        
        if not hasattr(self, '_obs'):
            self._obs, _ = self.env.reset()
        
        for _ in range(self.rollout_steps):
            state = self._obs["policy"]
            
            with torch.no_grad():
                action, log_prob = self.policy.sample_action(state)
                _, _, value = self.policy(state)
            
            states.append(state.clone())
            actions.append(action.clone())
            log_probs.append(log_prob.clone())
            values.append(value.squeeze(-1).clone())
            
            self._obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated
            
            rewards.append(reward.clone())
            dones.append(done.clone())
            
            # Track successes
            if done.any():
                done_idx = done.nonzero(as_tuple=False).squeeze(-1)
                done_rewards = reward[done_idx]
                successes = (done_rewards > self.success_threshold).sum().item()
                num_done = done_idx.shape[0]
                
                rollout_successes += successes
                rollout_episodes += num_done
                
                for i in range(num_done):
                    self.ssr_window.append(1 if done_rewards[i].item() > self.success_threshold else 0)
            
            self.total_steps += num_envs
            self.stage_steps += num_envs
        
        self.stage_successes += rollout_successes
        self.stage_episodes += rollout_episodes
        self.total_episodes += rollout_episodes
        
        # Final value
        with torch.no_grad():
            _, _, final_value = self.policy(self._obs["policy"])
            final_value = final_value.squeeze(-1)
        
        # Stack
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        values = torch.stack(values)
        
        # GAE
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(num_envs, device=self.device)
        next_value = final_value
        
        for t in reversed(range(self.rollout_steps)):
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]
        
        returns = advantages + values
        
        return {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "advantages": advantages,
            "returns": returns,
            "rewards": rewards,
        }
    
    def update_policy(self, data):
        T, N = data["states"].shape[:2]
        total = T * N
        
        states = data["states"].view(total, -1)
        actions = data["actions"].view(total, -1)
        old_log_probs = data["log_probs"].view(total)
        advantages = data["advantages"].view(total)
        returns = data["returns"].view(total)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "kl": 0}
        n_updates = 0
        
        for _ in range(self.epochs):
            idx = torch.randperm(total, device=self.device)
            
            for start in range(0, total, self.batch_size):
                end = min(start + self.batch_size, total)
                b_idx = idx[start:end]
                
                mean, std, values = self.policy(states[b_idx])
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(actions[b_idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                
                ratio = torch.exp(log_prob - old_log_probs[b_idx])
                surr1 = ratio * advantages[b_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[b_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * ((values.squeeze(-1) - returns[b_idx]) ** 2).mean()
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["kl"] += ((ratio - 1) - torch.log(ratio)).mean().item()
                n_updates += 1
        
        return {k: v / n_updates for k, v in stats.items()}
    
    def check_advancement(self):
        if self.stage_episodes < self.min_episodes:
            return False
        
        ssr = self.stage_successes / self.stage_episodes if self.stage_episodes > 0 else 0
        
        if ssr >= self.ssr_threshold:
            current = self.env.curriculum_level
            next_stage = min(current + 1, NUM_STAGES - 1)
            
            if next_stage > current:
                print(f"\n{'='*70}")
                print(f"🎉 ADVANCEMENT: Stage {current} → {next_stage}")
                print(f"   SSR: {ssr:.1%} | Episodes: {self.stage_episodes}")
                print(f"{'='*70}\n")
                
                self.env.set_curriculum_level(next_stage)
                self.stage_successes = 0
                self.stage_episodes = 0
                self.stage_steps = 0
                self.ssr_window.clear()
                return True
        return False
    
    def train(self, total_rollouts, log_interval=5, save_interval=100):
        print(f"\n{'='*70}")
        print("STATE-BASED PPO TRAINING (Debugging)")
        print(f"{'='*70}")
        print(f"Envs: {self.env.num_envs} | Rollout: {self.rollout_steps}")
        print(f"SSR threshold: {self.ssr_threshold:.0%} | Min episodes: {self.min_episodes}")
        print(f"{'='*70}\n")
        
        for rollout in range(total_rollouts):
            data = self.collect_rollout()
            stats = self.update_policy(data)
            self.check_advancement()
            
            if (rollout + 1) % log_interval == 0:
                self._log(rollout + 1, data, stats)
            
            if (rollout + 1) % save_interval == 0:
                self._save(rollout + 1)
    
    def _log(self, rollout, data, stats):
        stage = self.env.curriculum_level
        ssr = self.stage_successes / self.stage_episodes if self.stage_episodes > 0 else 0
        rolling = sum(self.ssr_window) / len(self.ssr_window) if self.ssr_window else 0
        
        log_std = self.policy.log_std.data.cpu().numpy()
        std = np.exp(np.clip(log_std, -2, 0.5))
        
        mean_ret = data["returns"].mean().item()
        mean_rew = data["rewards"].mean().item()
        
        print(f"\n{'='*70}")
        print(f"Rollout {rollout} | Stage S{stage} | Steps: {self.stage_steps:,}")
        print(f"{'-'*70}")
        print(f"Total: {self.total_steps:,} steps | {self.total_episodes:,} episodes")
        print(f"Stage: {self.stage_episodes} eps | {self.stage_successes} success | SSR: {ssr:.1%}")
        print(f"Rolling SSR: {rolling:.1%} | Threshold: {self.ssr_threshold:.0%}")
        print(f"{'-'*70}")
        print(f"Policy: {stats['policy_loss']:.4f} | Value: {stats['value_loss']:.4f}")
        print(f"Entropy: {stats['entropy']:.4f} | KL: {stats['kl']:.4f}")
        print(f"Log_std: [{log_std[0]:.3f}, {log_std[1]:.3f}] | Std: [{std[0]:.3f}, {std[1]:.3f}]")
        print(f"Mean reward: {mean_rew:.2f} | Return: {mean_ret:.2f}")
        print(f"{'='*70}\n")
        
        # TensorBoard
        self.writer.add_scalar("curriculum/stage", stage, self.total_steps)
        self.writer.add_scalar("curriculum/stage_ssr", ssr, self.total_steps)
        self.writer.add_scalar("curriculum/rolling_ssr", rolling, self.total_steps)
        self.writer.add_scalar("loss/policy", stats['policy_loss'], self.total_steps)
        self.writer.add_scalar("loss/value", stats['value_loss'], self.total_steps)
        self.writer.add_scalar("policy/entropy", stats['entropy'], self.total_steps)
        self.writer.add_scalar("policy/kl", stats['kl'], self.total_steps)
        self.writer.add_scalar("policy/std_v", std[0], self.total_steps)
        self.writer.add_scalar("policy/std_w", std[1], self.total_steps)
        self.writer.add_scalar("reward/mean", mean_rew, self.total_steps)
        self.writer.add_scalar("reward/return_mean", mean_ret, self.total_steps)
    
    def _save(self, rollout):
        path = Path("checkpoints")
        path.mkdir(exist_ok=True)
        
        torch.save({
            "rollout": rollout,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "stage": self.env.curriculum_level,
            "total_steps": self.total_steps,
        }, path / f"state_ppo_rollout_{rollout}.pt")
        
        print(f"[SAVE] checkpoints/state_ppo_rollout_{rollout}.pt")


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = TekoEnvCfgState()
    cfg.scene.num_envs = 500  # No vision = many more envs!
    
    print("[INFO] Creating state-based environment...")
    env = TekoEnvState(cfg=cfg)
    
    print("[INFO] Creating policy...")
    policy = StatePolicy(state_dim=4, action_dim=2, hidden_dim=256).to("cuda")
    print(f"[INFO] Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    print("[INFO] Creating trainer...")
    trainer = StatePPOTrainer(
        env=env,
        policy=policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        rollout_steps=128,
        batch_size=4096,
        epochs=6,
        ssr_threshold=0.70,
        min_episodes=500,
    )
    
    print("[INFO] Starting training...")
    trainer.train(total_rollouts=5000, log_interval=5, save_interval=100)
    
    print("\n✅ State-based training complete!")


if __name__ == "__main__":
    main()