# SPDX-License-Identifier: BSD-3-Clause
"""
CNN Feature Extractor for TEKO Docking (v9.2 - GRAYSCALE 128x128)
--------------------------------------------------------------
- Input: 4 grayscale frames [B, 4, 128, 128]
- Dynamic flatten size detection (no assumptions about resolution)
- Very lightweight for 32–64 parallel envs

Author: Alexandre Schleier Neves da Silva
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Lightweight CNN for TEKO grayscale docking input."""

    def __init__(self, feature_dim=256, num_frame_stack=4, input_h=128, input_w=128):
        super().__init__()

        self.num_frame_stack = num_frame_stack

        # Convolutional backbone (for 64×64)
        self.features = nn.Sequential(
            nn.Conv2d(num_frame_stack, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Dynamically compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, num_frame_stack, input_h, input_w)
            n_flat = self.features(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: [B, K, H, W] grayscale frame stack
            
        Returns:
            features: [B, feature_dim]
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_visual_encoder(
    architecture="simple", 
    feature_dim=256, 
    pretrained=False,
    num_frame_stack=4,
    input_h=128,
    input_w=128,
):
    """Factory function for the TEKO CNN encoder."""
    return SimpleCNN(
        feature_dim=feature_dim,
        num_frame_stack=num_frame_stack,
        input_h=input_h,
        input_w=input_w,
    )
