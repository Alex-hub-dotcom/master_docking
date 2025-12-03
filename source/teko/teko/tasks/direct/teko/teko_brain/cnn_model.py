# SPDX-License-Identifier: BSD-3-Clause
"""
CNN Feature Extractor for TEKO Docking (v10.0 - Optimized GRAYSCALE 84x84)
--------------------------------------------------------------------------
This version keeps the SAME CLASS NAME (SimpleCNN) and SAME factory name,
but internally uses a much stronger and more stable architecture based on
the DeepMind IMPALA-style encoder.

- Input: 4 grayscale frames [B, 4, 84, 84]
- Stronger receptive field (8x8 → 4x4 → 3x3)
- Perfect for PPO with 60 parallel environments (RTX 3090)
- Fully drop-in replacement for previous SimpleCNN

Author: Alexandre Schleier Neves da Silva
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Optimized CNN for TEKO grayscale docking with 4-frame stack."""

    def __init__(self, feature_dim=256, num_frame_stack=4, input_h=84, input_w=84):
        super().__init__()

        self.num_frame_stack = num_frame_stack

        # ---- New optimized convolutional backbone ----
        self.conv1 = nn.Conv2d(num_frame_stack, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # ---- Dynamic flatten size computation ----
        with torch.no_grad():
            dummy = torch.zeros(1, num_frame_stack, input_h, input_w)
            conv_out = self._forward_conv(dummy).view(1, -1).shape[1]

        # ---- Fully connected head ----
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.feature_dim = feature_dim

    # Helper: pass only through conv stack
    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    # Main forward
    def forward(self, x):
        """
        Args:
            x: [B, K, H, W] grayscale stacked frames
        Returns:
            features: [B, feature_dim]
        """
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---- Factory function (kept identical) ----
def create_visual_encoder(
    architecture="simple",
    feature_dim=256,
    pretrained=False,
    num_frame_stack=4,
    input_h=84,
    input_w=84,
):
    """Factory for the TEKO CNN encoder (keeps original interface unchanged)."""
    return SimpleCNN(
        feature_dim=feature_dim,
        num_frame_stack=num_frame_stack,
        input_h=input_h,
        input_w=input_w,
    )
