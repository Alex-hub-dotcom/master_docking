# SPDX-License-Identifier: BSD-3-Clause
"""
CNN Feature Extractor for TEKO Docking (v9.6 - SHAPE-BASED 84x84 + LayerNorm)
-----------------------------------------------------------------------------
- Input: 4 grayscale frames [B, 4, 84, 84]
- Optimized for learning robot geometry, NOT ArUco markers
- LayerNorm after each conv for training stability
- Orthogonal initialization for RL

Author: Alexandre Schleier Neves da Silva
"""

import torch
import torch.nn as nn
import math


class SimpleCNN(nn.Module):
    """CNN optimized for robot shape recognition at 84x84 with LayerNorm."""

    def __init__(self, feature_dim=256, num_frame_stack=4, input_h=84, input_w=84):
        super().__init__()
        self.num_frame_stack = num_frame_stack

        # Conv layers (no norm in Sequential - we'll apply manually for LayerNorm)
        self.conv1 = nn.Conv2d(num_frame_stack, 32, kernel_size=6, stride=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Compute intermediate sizes for LayerNorm
        # Conv1: 84 -> 27, Conv2: 27 -> 13, Conv3: 13 -> 6
        self.ln1 = nn.LayerNorm([32, 27, 27])
        self.ln2 = nn.LayerNorm([64, 13, 13])
        self.ln3 = nn.LayerNorm([128, 7, 7])

        # Dynamic flatten size (safety check)
        with torch.no_grad():
            dummy = torch.zeros(1, num_frame_stack, input_h, input_w)
            n_flat = self._forward_conv(dummy).view(1, -1).shape[1]

        # FC head
        self.fc = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.feature_dim = feature_dim

        # Initialize weights (orthogonal is standard for RL)
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization - proven for RL stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_conv(self, x):
        """Forward through conv layers with LayerNorm."""
        x = self.conv1(x)
        x = self.ln1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.ln2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = self.ln3(x)
        x = torch.relu(x)

        return x

    def forward(self, x):
        """
        Args:
            x: [B, K, H, W] grayscale frame stack
        Returns:
            features: [B, feature_dim]
        """
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_visual_encoder(
    architecture="simple", 
    feature_dim=256, 
    pretrained=False,
    num_frame_stack=4,
    input_h=84,
    input_w=84,
):
    """Factory function for the TEKO CNN encoder."""
    return SimpleCNN(
        feature_dim=feature_dim,
        num_frame_stack=num_frame_stack,
        input_h=input_h,
        input_w=input_w,
    )