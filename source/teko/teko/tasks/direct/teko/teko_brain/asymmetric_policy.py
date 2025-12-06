# SPDX-License-Identifier: BSD-3-Clause
"""
Asymmetric Actor-Critic Policy for TEKO Vision-Based Docking
==============================================================
- Actor uses ONLY vision (84×84 grayscale, 4 frames) → deployable
- Critic uses vision + privileged state [dx, dy, dz, yaw, vx, vy, w]
- Shared vision encoder (SimpleCNN) between actor/critic
- Proven approach from legged locomotion (ANYmal, MIT Cheetah)

Author: Alexandre Schleier Neves da Silva
Date: December 2024
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class AsymmetricActorCritic(nn.Module):
    """
    Asymmetric actor-critic with shared vision encoder.
    
    Architecture:
    - Vision encoder (SimpleCNN): 84×84×4 → 256 features (shared)
    - Actor: 256 → 128 → 2 actions [v, w]
    - Critic: concat(256 vision, 128 state) → 128 → value
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
        
        # ============================================================
        # SHARED VISION ENCODER (SimpleCNN v9.6)
        # ============================================================
        self.vision_encoder = SimpleCNN(
            in_channels=vision_shape[0],
            feature_dim=hidden_dim
        )
        
        # ============================================================
        # ACTOR (Vision-only → deployable)
        # ============================================================
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, action_dim)
        )
        
        # Learnable log_std for stochastic policy
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # ============================================================
        # CRITIC (Vision + privileged state)
        # ============================================================
        # State encoder: [dx, dy, dz, yaw_error, vx, vy, w] → 128
        self.state_encoder = nn.Sequential(
            nn.Linear(privileged_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
        )
        
        # Fusion: concat(256 vision, 128 state) → value
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
        # ============================================================
        # WEIGHT INITIALIZATION
        # ============================================================
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Orthogonal initialization for stable training."""
        for module in [self.actor_head, self.state_encoder, self.critic_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Small initialization for final actor layer (smooth actions)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        
    def forward_actor(self, vision: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Actor forward pass (vision-only).
        
        Args:
            vision: (B, 4, 84, 84) grayscale frames
            
        Returns:
            mean: (B, 2) action means [v, w]
            std: (B, 2) action stds
        """
        # Encode vision
        vision_features = self.vision_encoder(vision)  # (B, 256)
        
        # Actor head
        mean = torch.tanh(self.actor_head(vision_features))  # (B, 2) ∈ [-1, 1]
        
        # Expand log_std to batch
        std = self.log_std.exp().unsqueeze(0).expand(mean.shape[0], -1)  # (B, 2)
        
        return mean, std
    
    def forward_critic(
        self, 
        vision: torch.Tensor, 
        privileged: torch.Tensor
    ) -> torch.Tensor:
        """
        Critic forward pass (vision + privileged state).
        
        Args:
            vision: (B, 4, 84, 84) grayscale frames
            privileged: (B, 7) [dx, dy, dz, yaw_error, vx, vy, w]
            
        Returns:
            value: (B, 1) state value
        """
        # Encode vision (shared encoder)
        vision_features = self.vision_encoder(vision)  # (B, 256)
        
        # Encode privileged state
        state_features = self.state_encoder(privileged)  # (B, 128)
        
        # Fuse and predict value
        fused = torch.cat([vision_features, state_features], dim=-1)  # (B, 384)
        value = self.critic_head(fused)  # (B, 1)
        
        return value
    
    def forward(
        self, 
        obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.
        
        Args:
            obs: {"rgb": (B, 4, 84, 84), "privileged": (B, 7)}
            
        Returns:
            mean: (B, 2) action means
            std: (B, 2) action stds
            value: (B, 1) state value
        """
        vision = obs["rgb"]
        privileged = obs["privileged"]
        
        # Actor (vision-only)
        mean, std = self.forward_actor(vision)
        
        # Critic (vision + privileged)
        value = self.forward_critic(vision, privileged)
        
        return mean, std, value
    
    def sample_action(
        self, 
        obs: Dict[str, torch.Tensor], 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: {"rgb": (B, 4, 84, 84), "privileged": (B, 7)}
            deterministic: If True, return mean action
            
        Returns:
            action: (B, 2) sampled action
            log_prob: (B,) log probability
        """
        mean, std = self.forward_actor(obs["rgb"])
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob


class SimpleCNN(nn.Module):
    """
    SimpleCNN v9.6 for 84×84 grayscale observations.
    
    Architecture:
    - Conv1: 4→32, kernel=6, stride=3 → 84→27
    - Conv2: 32→64, kernel=4, stride=2 → 27→13
    - Conv3: 64→128, kernel=3, stride=2 → 13→7
    - FC: 6272→512→256
    
    Parameters: ~500K (vs ResNet18 11M)
    """
    
    def __init__(self, in_channels: int = 4, feature_dim: int = 256):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv1: 84×84×4 → 27×27×32
            nn.Conv2d(in_channels, 32, kernel_size=6, stride=3, padding=0),
            nn.LayerNorm([32, 27, 27]),
            nn.ELU(),
            
            # Conv2: 27×27×32 → 13×13×64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.LayerNorm([64, 13, 13]),
            nn.ELU(),
            
            # Conv3: 13×13×64 → 7×7×128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([128, 7, 7]),
            nn.ELU(),
        )
        
        # Flatten size: 128 * 7 * 7 = 6272
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ELU(),
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Orthogonal initialization for conv and fc layers."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, 4, 84, 84) grayscale frames
            
        Returns:
            features: (B, 256) encoded features
        """
        x = self.conv_layers(x)  # (B, 128, 7, 7)
        x = x.flatten(start_dim=1)  # (B, 6272)
        x = self.fc_layers(x)  # (B, 256)
        return x


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_asymmetric_policy(
    vision_shape: Tuple[int, int, int] = (4, 84, 84),
    privileged_dim: int = 7,
    action_dim: int = 2,
    device: str = "cuda"
) -> AsymmetricActorCritic:
    """
    Factory function to create asymmetric policy.
    
    Args:
        vision_shape: (C, H, W) for vision input
        privileged_dim: Dimension of privileged state
        action_dim: Dimension of action space
        device: Device to place model on
        
    Returns:
        policy: AsymmetricActorCritic model
    """
    policy = AsymmetricActorCritic(
        vision_shape=vision_shape,
        privileged_dim=privileged_dim,
        action_dim=action_dim
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    print(f"[INFO] AsymmetricActorCritic created")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Vision encoder: SimpleCNN (84×84)")
    print(f"  - Privileged dim: {privileged_dim}")
    print(f"  - Action dim: {action_dim}")
    
    return policy


if __name__ == "__main__":
    """Test the asymmetric policy."""
    
    # Create policy
    policy = create_asymmetric_policy(device="cpu")
    
    # Test forward pass
    batch_size = 16
    obs = {
        "rgb": torch.randn(batch_size, 4, 84, 84),
        "privileged": torch.randn(batch_size, 7)
    }
    
    # Full forward
    mean, std, value = policy(obs)
    print(f"\n[TEST] Forward pass:")
    print(f"  - Action mean: {mean.shape}")
    print(f"  - Action std: {std.shape}")
    print(f"  - Value: {value.shape}")
    
    # Sample action
    action, log_prob = policy.sample_action(obs)
    print(f"\n[TEST] Sample action:")
    print(f"  - Action: {action.shape}")
    print(f"  - Log prob: {log_prob.shape}")
    
    # Actor-only forward (deployment mode)
    mean_deploy, std_deploy = policy.forward_actor(obs["rgb"])
    print(f"\n[TEST] Actor-only (deployment):")
    print(f"  - Action mean: {mean_deploy.shape}")
    print(f"  - Action std: {std_deploy.shape}")
    
    print("\n✅ All tests passed!")