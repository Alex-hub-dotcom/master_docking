# SPDX-License-Identifier: BSD-3-Clause
"""
Asymmetric Actor-Critic Policy for TEKO Vision-Based Docking
==============================================================
- Actor uses ONLY vision (84×84 grayscale, 4 frames) → deployable
- Critic uses vision + privileged state [dx, dy, dz, yaw, vx, vy, w]
- Shared vision encoder (SimpleCNN v9.6) between actor/critic

Author: Alexandre Schleier Neves da Silva
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple


class SimpleCNN(nn.Module):
    """CNN optimized for robot shape recognition at 84x84 with LayerNorm (v9.6)."""
    
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=6, stride=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # LayerNorm for each conv output
        self.ln1 = nn.LayerNorm([32, 27, 27])
        self.ln2 = nn.LayerNorm([64, 13, 13])
        self.ln3 = nn.LayerNorm([128, 7, 7])
        
        # FC head: 128*7*7 = 6272
        self.fc = nn.Sequential(
            nn.Linear(6272, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )
        
        self.feature_dim = feature_dim
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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


class AsymmetricActorCritic(nn.Module):
    """
    Asymmetric actor-critic with shared vision encoder.
    
    - Actor: vision → 256 → action (deployable)
    - Critic: vision + privileged → value (training only)
    """
    
    def __init__(
        self,
        vision_shape: Tuple[int, int, int] = (4, 84, 84),
        privileged_dim: int = 7,
        action_dim: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.vision_shape = vision_shape
        self.privileged_dim = privileged_dim
        self.action_dim = action_dim
        
        # Shared vision encoder (SimpleCNN v9.6)
        self.vision_encoder = SimpleCNN(
            in_channels=vision_shape[0],
            feature_dim=hidden_dim
        )
        
        # Actor head (vision-only)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )
        
        # Learnable log_std
        self.log_std_v = nn.Parameter(torch.tensor(0.0))
        self.log_std_w = nn.Parameter(torch.tensor(0.0))
        
        # Log_std bounds (from your working Policy)
        self.LOG_STD_V_MIN, self.LOG_STD_V_MAX = -1.5, 0.2
        self.LOG_STD_W_MIN, self.LOG_STD_W_MAX = -1.0, 0.6
        
        # State encoder for privileged info
        self.state_encoder = nn.Sequential(
            nn.Linear(privileged_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        
        # Critic head: vision(256) + state(128) → value
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        
        self._init_heads()
    
    def _init_heads(self):
        for module in [self.actor_head, self.state_encoder, self.critic_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Small init for final actor layer
        nn.init.orthogonal_(self.actor_head[-2].weight, gain=0.01)
    
    def _get_log_std(self):
        log_std_v = self.log_std_v.clamp(self.LOG_STD_V_MIN, self.LOG_STD_V_MAX)
        log_std_w = self.log_std_w.clamp(self.LOG_STD_W_MIN, self.LOG_STD_W_MAX)
        return torch.stack([log_std_v, log_std_w])
    
    def forward_actor(self, vision: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Actor forward (vision-only, deployable)."""
        features = self.vision_encoder(vision)
        mean = self.actor_head(features)
        log_std = self._get_log_std()
        std = log_std.exp().unsqueeze(0).expand(mean.shape[0], -1)
        return mean, std
    
    def forward_critic(self, vision: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        """Critic forward (vision + privileged state)."""
        vision_features = self.vision_encoder(vision)
        state_features = self.state_encoder(privileged)
        fused = torch.cat([vision_features, state_features], dim=-1)
        value = self.critic_head(fused)
        return value
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward for training."""
        vision = obs["rgb"]
        privileged = obs["privileged"]
        
        mean, std = self.forward_actor(vision)
        value = self.forward_critic(vision, privileged)
        
        return mean, std, value
    
    def sample_action(
        self, 
        obs: Dict[str, torch.Tensor], 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action (for trainer's collect_rollout)."""
        mean, std = self.forward_actor(obs["rgb"])
        
        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device)
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        
        return action, log_prob
    
    def act(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (compatible with your PPO trainer)."""
        mean, std = self.forward_actor(obs["rgb"])
        value = self.forward_critic(obs["rgb"], obs["privileged"])
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions (for PPO update)."""
        mean, std = self.forward_actor(obs["rgb"])
        value = self.forward_critic(obs["rgb"], obs["privileged"])
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return log_prob, value.squeeze(-1), entropy


def create_asymmetric_policy(
    vision_shape: Tuple[int, int, int] = (4, 84, 84),
    privileged_dim: int = 7,
    action_dim: int = 2,
    device: str = "cuda"
) -> AsymmetricActorCritic:
    """Factory function."""
    policy = AsymmetricActorCritic(
        vision_shape=vision_shape,
        privileged_dim=privileged_dim,
        action_dim=action_dim
    ).to(device)
    
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"[INFO] AsymmetricActorCritic: {total_params:,} params")
    
    return policy