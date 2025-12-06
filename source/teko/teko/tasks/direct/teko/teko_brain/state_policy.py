# SPDX-License-Identifier: BSD-3-Clause
"""
State-Based MLP Policy for TEKO (Debugging)
============================================

Simple MLP for state-based control using ground truth observations.
Used to validate curriculum + rewards without vision bottleneck.

Observation: [dx, dy, dz, yaw_error] (4D state vector)
Actions: [linear_vel, angular_vel]

Author: Alexandre Schleier Neves da Silva
"""

import torch
import torch.nn as nn


class StateMLP(nn.Module):
    """
    Simple MLP policy for state-based control.
    
    Architecture:
    - Input: 4D state [dx, dy, dz, yaw_error]
    - Hidden: 2 layers × 128 units
    - Output: 2D actions [v, ω]
    """
    
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor):
        """
        Forward pass.
        
        Args:
            state: [N, 4] state tensor
        
        Returns:
            actions: [N, 2] action tensor
            values: [N, 1] value estimates
        """
        actions = self.actor(state)
        values = self.critic(state)
        return actions, values
    
    def act(self, state: torch.Tensor, deterministic: bool = False):
        """Get actions (for inference)."""
        actions, _ = self.forward(state)
        return actions
    
    def evaluate(self, state: torch.Tensor):
        """Get value estimates."""
        _, values = self.forward(state)
        return values