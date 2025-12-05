# SPDX-License-Identifier: BSD-3-Clause
"""
CNN Feature Extractor for TEKO Docking (v9.7 - OPTIMIZED 64x64 + FP16)
-----------------------------------------------------------------------
- Input: 4 grayscale frames [B, 4, 64, 64]
- Optimized for learning robot geometry with minimal VRAM
- LayerNorm after each conv for training stability
- Orthogonal initialization for RL
- Compatible with mixed precision training

Author: Alexandre Schleier Neves da Silva
"""

import torch
import torch.nn as nn
import math


class SimpleCNN(nn.Module):
    """CNN optimized for robot shape recognition at 64x64 with LayerNorm."""
    
    def __init__(self, feature_dim=256, num_frame_stack=4, input_h=64, input_w=64):
        super().__init__()
        
        self.num_frame_stack = num_frame_stack
        
        # Conv layers (adjusted for 64x64 input)
        self.conv1 = nn.Conv2d(num_frame_stack, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Compute intermediate sizes for LayerNorm
        # Conv1: 64 -> 32, Conv2: 32 -> 16, Conv3: 16 -> 8
        self.ln1 = nn.LayerNorm([32, 32, 32])
        self.ln2 = nn.LayerNorm([64, 16, 16])
        self.ln3 = nn.LayerNorm([128, 8, 8])
        
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
    input_h=64,
    input_w=64,
):
    """Factory function for the TEKO CNN encoder."""
    return SimpleCNN(
        feature_dim=feature_dim,
        num_frame_stack=num_frame_stack,
        input_h=input_h,
        input_w=input_w,
    )