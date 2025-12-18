#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Optuna Hyperparameter Optimization for TEKO State-Based Docking
================================================================

State-based version for FAST hyperparameter search.
Uses ground truth [dx, dy, dz, yaw_error] instead of vision.

FIXED: Uses shared environment to prevent Isaac Sim hanging between trials.

Author: Alexandre Schleier Neves da Silva
Date: December 2024
"""

# Isaac Lab launcher (MUST come first!)
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({
    "headless": True,
    "enable_cameras": False,
})
simulation_app = app_launcher.app

import os
import sys
import json
import torch
import optuna
import numpy as np
import traceback
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

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
        
        self.features = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
        )
        
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
            torch.nn.Tanh(),
        )
        
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        
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
# OPTUNA PPO TRAINER (STATE-BASED)
# =============================================================================

class OptunaStatePPOTrainer:
    """PPO trainer with Optuna integration for state-based training."""
    
    def __init__(
        self,
        env: TekoEnvState,
        policy: torch.nn.Module,
        trial: optuna.Trial,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 128,
        batch_size: int = 4096,
        epochs: int = 6,
        device: str = "cuda",
        ssr_threshold: float = 0.70,
        min_episodes_for_advancement: int = 500,
        ssr_window_size: int = 1000,
    ):
        self.env = env
        self.policy = policy
        self.trial = trial
        self.device = device
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.ssr_threshold = ssr_threshold
        self.min_episodes_for_advancement = min_episodes_for_advancement
        
        cfg = getattr(env, "cfg", None)
        self.stage_thresholds = getattr(cfg, "stage_thresholds", None)
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Statistics - reset for each trial
        self.total_steps = 0
        self.total_episodes = 0
        self.stage_successes = 0
        self.stage_episodes = 0
        self.stage_steps = 0
        
        self.ssr_window = deque(maxlen=ssr_window_size)
        self.entropy_window = deque(maxlen=50)
        
        self.max_stage_reached = 0
        self.total_successes = 0
        self.total_steps_to_success = 0
        self.success_count_for_avg = 0
        
        self.rollouts_without_stage_progress = 0
        self.last_stage_for_progress_check = 0
        
        # TensorBoard
        self.tb_dir = Path("/workspace/teko/experiments/optuna_state/tb") / f"trial_{trial.number}"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        
        # Clear obs cache for new trial
        self._obs_cache = None
        
        print(f"[TRIAL {trial.number}] Trainer initialized", flush=True)
    
    def collect_rollout(self):
        """Collect rollout data with success tracking."""
        num_envs = self.env.num_envs
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        rollout_successes = 0
        rollout_episodes = 0
        
        if self._obs_cache is None:
            self._obs_cache, _ = self.env.reset()
        
        for step in range(self.rollout_steps):
            state = self._obs_cache["policy"]
            
            with torch.no_grad():
                action, log_prob = self.policy.sample_action(state)
                _, _, value = self.policy(state)
            
            states.append(state.clone())
            actions.append(action.clone())
            log_probs.append(log_prob.clone())
            values.append(value.squeeze(-1).clone())
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated
            
            rewards.append(reward.clone())
            dones.append(done.clone())
            
            if done.any():
                done_idx = done.nonzero(as_tuple=False).squeeze(-1)
                
                success_buf = getattr(self.env, "_last_success", None)
                if success_buf is None:
                    success_flag = torch.zeros_like(done, dtype=torch.bool)
                else:
                    success_flag = success_buf
                
                successes = success_flag[done_idx].sum().item()
                num_done = done_idx.shape[0]
                
                rollout_successes += successes
                rollout_episodes += num_done
                
                if successes > 0:
                    success_indices = done_idx[success_flag[done_idx]]
                    for idx in success_indices:
                        steps_taken = self.env.episode_length_buf[idx].item()
                        self.total_steps_to_success += steps_taken
                        self.success_count_for_avg += 1
                
                for i in range(num_done):
                    is_success = success_flag[done_idx[i]].item()
                    self.ssr_window.append(1 if is_success else 0)
            
            self._obs_cache = next_obs
            self.total_steps += num_envs
            self.stage_steps += num_envs
        
        self.stage_successes += rollout_successes
        self.stage_episodes += rollout_episodes
        self.total_episodes += rollout_episodes
        self.total_successes += rollout_successes
        
        with torch.no_grad():
            _, _, final_value = self.policy(self._obs_cache["policy"])
            final_value = final_value.squeeze(-1)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        values = torch.stack(values)
        
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(num_envs, device=self.device)
        next_value = final_value
        
        for t in reversed(range(self.rollout_steps)):
            next_non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            next_value = values[t]
        
        returns = advantages + values
        
        return {
            "states": states,
            "actions": actions,
            "old_log_probs": log_probs,
            "advantages": advantages,
            "returns": returns,
            "rewards": rewards,
        }
    
    def update_policy(self, rollout_data):
        """Update policy with PPO."""
        T, N = rollout_data["states"].shape[:2]
        total_samples = T * N
        
        states = rollout_data["states"].view(total_samples, -1)
        actions = rollout_data["actions"].view(total_samples, -1)
        old_log_probs = rollout_data["old_log_probs"].view(total_samples)
        advantages = rollout_data["advantages"].view(total_samples)
        returns = rollout_data["returns"].view(total_samples)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_entropy = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_updates = 0
        
        for epoch in range(self.epochs):
            indices = torch.randperm(total_samples, device=self.device)
            
            for start_idx in range(0, total_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                mean, std, values = self.policy(batch_states)
                
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * ((values.squeeze(-1) - batch_returns) ** 2).mean()
                
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_entropy += entropy.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_updates += 1
        
        avg_entropy = total_entropy / num_updates
        self.entropy_window.append(avg_entropy)
        
        log_std = self.policy.log_std.data.cpu().numpy()
        std = np.exp(np.clip(log_std, -2.0, 0.5))
        
        return {
            "entropy": avg_entropy,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "std_v": std[0],
            "std_w": std[1],
        }
    
    def check_curriculum_advancement(self) -> bool:
        """Check if we should advance to next curriculum stage."""
        if self.stage_episodes < self.min_episodes_for_advancement:
            return False
        
        stage_ssr = self.stage_successes / self.stage_episodes if self.stage_episodes > 0 else 0.0
        current_stage = int(self.env.curriculum_level)
        
        if self.stage_thresholds is not None:
            required = self.stage_thresholds.get(current_stage, self.ssr_threshold)
        else:
            required = self.ssr_threshold
        
        if stage_ssr >= required:
            next_stage = min(current_stage + 1, NUM_STAGES - 1)
            
            if next_stage > current_stage:
                print(f"[OPTUNA] Stage {current_stage} → {next_stage} (SSR: {stage_ssr:.1%})", flush=True)
                self.env.set_curriculum_level(next_stage)
                self.max_stage_reached = max(self.max_stage_reached, next_stage)
                
                self.stage_successes = 0
                self.stage_episodes = 0
                self.stage_steps = 0
                self.ssr_window.clear()
                
                self.rollouts_without_stage_progress = 0
                self.last_stage_for_progress_check = next_stage
                
                return True
        
        return False
    
    def should_prune(self, rollout_idx: int) -> Tuple[bool, str]:
        """Check if trial should be pruned."""
        current_stage = self.env.curriculum_level
        
        if rollout_idx > 30 and len(self.entropy_window) >= 20:
            recent_entropy = np.mean(list(self.entropy_window)[-20:])
            if recent_entropy < 0.1:
                return True, f"Entropy collapse: {recent_entropy:.4f}"
        
        if current_stage == self.last_stage_for_progress_check:
            self.rollouts_without_stage_progress += 1
        else:
            self.rollouts_without_stage_progress = 0
            self.last_stage_for_progress_check = current_stage
        
        max_rollouts_per_stage = 100 if current_stage >= 6 else 80
        
        if self.rollouts_without_stage_progress > max_rollouts_per_stage:
            stage_ssr = self.stage_successes / self.stage_episodes if self.stage_episodes > 0 else 0.0
            return True, f"Stuck at Stage {current_stage} for {self.rollouts_without_stage_progress} rollouts (SSR: {stage_ssr:.1%})"
        
        if rollout_idx > 50 and self.total_episodes > 300:
            overall_ssr = self.total_successes / self.total_episodes if self.total_episodes > 0 else 0
            if overall_ssr < 0.2 and current_stage < 2:
                return True, f"Overall SSR too low: {overall_ssr:.1%}"
        
        return False, ""
    
    def get_metrics(self) -> Tuple[float, float, int]:
        """Get metrics for Optuna objectives."""
        ssr = self.total_successes / self.total_episodes if self.total_episodes > 0 else 0.0
        avg_steps = self.total_steps_to_success / self.success_count_for_avg if self.success_count_for_avg > 0 else float('inf')
        return ssr, avg_steps, self.max_stage_reached
    
    def train(self, max_rollouts: int) -> Tuple[float, float, int]:
        """Training loop with pruning."""
        print(f"[TRIAL {self.trial.number}] Starting training for {max_rollouts} rollouts", flush=True)
        
        for rollout_idx in range(max_rollouts):
            rollout_data = self.collect_rollout()
            update_stats = self.update_policy(rollout_data)
            
            self.check_curriculum_advancement()
            
            should_prune, reason = self.should_prune(rollout_idx)
            if should_prune:
                print(f"[OPTUNA PRUNE] Trial pruned at rollout {rollout_idx}: {reason}", flush=True)
                raise optuna.TrialPruned(reason)
            
            if (rollout_idx + 1) % 25 == 0:
                ssr, avg_steps, max_stage = self.get_metrics()
                stage = int(self.env.curriculum_level)
                entropy = update_stats["entropy"]
                print(f"[R{rollout_idx+1}] S{stage} | SSR: {ssr:.1%} | "
                      f"Ent: {entropy:.3f} | MaxS: {max_stage}", flush=True)
            
            self._log_tensorboard(rollout_idx, update_stats)
        
        self.writer.close()
        print(f"[TRIAL {self.trial.number}] Training complete", flush=True)
        return self.get_metrics()
    
    def _log_tensorboard(self, rollout_idx: int, update_stats: dict):
        """Log metrics to TensorBoard."""
        step = self.total_steps
        
        self.writer.add_scalar("curriculum/stage", self.env.curriculum_level, step)
        self.writer.add_scalar("curriculum/max_stage", self.max_stage_reached, step)
        
        ssr = self.total_successes / self.total_episodes if self.total_episodes > 0 else 0.0
        stage_ssr = self.stage_successes / self.stage_episodes if self.stage_episodes > 0 else 0.0
        self.writer.add_scalar("ssr/overall", ssr, step)
        self.writer.add_scalar("ssr/stage", stage_ssr, step)
        
        if len(self.ssr_window) > 0:
            rolling_ssr = sum(self.ssr_window) / len(self.ssr_window)
            self.writer.add_scalar("ssr/rolling", rolling_ssr, step)
        
        self.writer.add_scalar("loss/policy", update_stats["policy_loss"], step)
        self.writer.add_scalar("loss/value", update_stats["value_loss"], step)
        self.writer.add_scalar("policy/entropy", update_stats["entropy"], step)
        self.writer.add_scalar("policy/std_v", update_stats["std_v"], step)
        self.writer.add_scalar("policy/std_w", update_stats["std_w"], step)


# =============================================================================
# OPTUNA OBJECTIVE FUNCTION
# =============================================================================

def create_objective(shared_env, num_envs: int = 512, max_rollouts: int = 200, target_stage: int = 10):
    """Factory function to create Optuna objective using shared environment."""
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function using shared environment."""
        
        # Hyperparameter search space
        entropy_coef = trial.suggest_float("entropy_coef", 0.005, 0.15, log=True)
        init_log_std = trial.suggest_float("init_log_std", -1.5, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.4)
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.99)
        
        print(f"\n{'='*60}", flush=True)
        print(f"[TRIAL {trial.number}] State-Based HPO", flush=True)
        print(f"  entropy_coef: {entropy_coef:.4f}", flush=True)
        print(f"  init_log_std: {init_log_std:.3f}", flush=True)
        print(f"  learning_rate: {learning_rate:.6f}", flush=True)
        print(f"  clip_epsilon: {clip_epsilon:.3f}", flush=True)
        print(f"  gamma: {gamma:.4f}", flush=True)
        print(f"  gae_lambda: {gae_lambda:.4f}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # Use shared environment - reset to stage 0
        env = shared_env
        env.set_curriculum_level(0)
        env.reset()
        
        trainer = None
        
        try:
            print(f"[TRIAL {trial.number}] Using shared environment ({num_envs} envs)", flush=True)
            
            # Create NEW policy for each trial
            policy = StatePolicy(
                state_dim=4,
                action_dim=2,
                hidden_dim=256,
                init_log_std=init_log_std,
            ).to("cuda")
            
            # Create trainer
            trainer = OptunaStatePPOTrainer(
                env=env,
                policy=policy,
                trial=trial,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_epsilon=clip_epsilon,
                entropy_coef=entropy_coef,
                value_loss_coef=0.5,
                max_grad_norm=0.5,
                rollout_steps=128,
                batch_size=4096,
                epochs=6,
                device="cuda",
                ssr_threshold=0.70,
                min_episodes_for_advancement=500,
                ssr_window_size=1000,
            )
            
            # Train
            ssr, avg_steps, max_stage = trainer.train(max_rollouts=max_rollouts)
            
        except optuna.TrialPruned:
            if trainer:
                ssr, avg_steps, max_stage = trainer.get_metrics()
            else:
                ssr, avg_steps, max_stage = 0.0, 10000, 0
            raise
            
        except Exception as e:
            print(f"[TRIAL {trial.number}] ERROR: {e}", flush=True)
            traceback.print_exc()
            ssr, avg_steps, max_stage = 0.0, 10000, 0
            
        finally:
            # Close TensorBoard writer but NOT the environment
            if trainer and hasattr(trainer, 'writer'):
                try:
                    trainer.writer.close()
                except:
                    pass
            # Clear CUDA cache
            torch.cuda.empty_cache()
        
        # Score: max_stage + SSR bonus
        score = max_stage + ssr
        
        print(f"\n[TRIAL {trial.number}] COMPLETE", flush=True)
        print(f"  Max Stage: {max_stage}", flush=True)
        print(f"  Final SSR: {ssr:.1%}", flush=True)
        print(f"  Score: {score:.2f}", flush=True)
        
        if max_stage >= target_stage:
            print(f"  ✅ TARGET STAGE {target_stage} REACHED!", flush=True)
        
        return score
    
    return objective


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for State-Based Optuna HPO."""
    
    STUDY_NAME = "teko_state_hpo_v2"
    STORAGE_PATH = "/workspace/teko/experiments/optuna_state"
    DB_PATH = f"sqlite:///{STORAGE_PATH}/teko_state_optuna_v2.db"
    
    NUM_ENVS = 512
    MAX_ROLLOUTS = 200
    N_TRIALS = 100
    TARGET_STAGE = 10
    
    Path(STORAGE_PATH).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}", flush=True)
    print("TEKO STATE-BASED OPTUNA HPO (Fixed - Shared Env)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Study: {STUDY_NAME}", flush=True)
    print(f"Storage: {DB_PATH}", flush=True)
    print(f"Environments: {NUM_ENVS}", flush=True)
    print(f"Max rollouts/trial: {MAX_ROLLOUTS}", flush=True)
    print(f"Total trials: {N_TRIALS}", flush=True)
    print(f"Target stage: {TARGET_STAGE}", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    # Create shared environment ONCE
    print("[GLOBAL] Creating shared environment...", flush=True)
    env_cfg = TekoEnvCfgState()
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.num_envs = NUM_ENVS
    shared_env = TekoEnvState(cfg=env_cfg)
    print(f"[GLOBAL] Shared environment created with {NUM_ENVS} envs", flush=True)
    
    # Create sampler
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        multivariate=True,
    )
    
    # Create study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        sampler=sampler,
        direction="maximize",
        storage=DB_PATH,
        load_if_exists=True,
    )
    
    # Create objective with shared environment
    objective = create_objective(
        shared_env=shared_env,
        num_envs=NUM_ENVS,
        max_rollouts=MAX_ROLLOUTS,
        target_stage=TARGET_STAGE,
    )
    
    try:
        # Run optimization
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            catch=(Exception,),
            show_progress_bar=True,
        )
    finally:
        # Close environment at the very end
        print("[GLOBAL] Closing shared environment...", flush=True)
        shared_env.close()
    
    # Print results
    print(f"\n{'='*70}", flush=True)
    print("OPTIMIZATION COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
    
    if study.best_trial:
        print(f"\n🏆 Best Trial: {study.best_trial.number}", flush=True)
        print(f"   Score: {study.best_trial.value:.2f}", flush=True)
        print(f"   Params:", flush=True)
        for key, value in study.best_trial.params.items():
            print(f"     {key}: {value}", flush=True)
        
        # Save best config
        results_path = Path(STORAGE_PATH) / "best_config_v2.json"
        best_config = {
            "trial": study.best_trial.number,
            "score": study.best_trial.value,
            "params": study.best_trial.params,
        }
        
        with open(results_path, "w") as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\n💾 Best config saved to: {results_path}", flush=True)
    
    print(f"\n📈 TensorBoard: tensorboard --logdir {STORAGE_PATH}/tb", flush=True)
    print(f"🔍 Optuna Dashboard: optuna-dashboard {DB_PATH}", flush=True)


if __name__ == "__main__":
    main()