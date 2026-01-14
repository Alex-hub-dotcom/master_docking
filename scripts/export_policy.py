#!/usr/bin/env python3
"""
Export trained policy to TorchScript for deployment
"""

import torch
import torch.nn as nn
import math
import sys
import os

# Adicionar path
sys.path.insert(0, "/home/schux00/teko/scripts")

# =====================================================
# COPIAR A ARQUITETURA DO SCRIPT DE TREINO
# =====================================================

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x * attn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        y = x.view(b, c, -1).mean(-1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class VisionEncoderAttentionYaw(nn.Module):
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        import torch.nn.functional as F
        
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        self.channel_attn = ChannelAttention(64)
        self.spatial_attn = SpatialAttention(64)
        
        self.gn1 = nn.GroupNorm(8, 32)
        self.gn2 = nn.GroupNorm(8, 64)
        self.gn3 = nn.GroupNorm(8, 64)
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            flat_size = self._forward_conv(dummy).shape[1]
        
        self.fc = nn.Linear(flat_size, feature_dim)
        
        self.yaw_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        
        self.feature_dim = feature_dim
    
    def _forward_conv(self, x):
        import torch.nn.functional as F
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x.flatten(1)
    
    def forward(self, x):
        import torch.nn.functional as F
        x = self._forward_conv(x)
        features = F.relu(self.fc(x))
        return features


class VisionIMUAttentionYawPolicy(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
    
    def __init__(self, vis_dim=256, imu_dim=6, hidden=256, action_dim=2):
        super().__init__()
        
        self.vision_encoder = VisionEncoderAttentionYaw(in_channels=4, feature_dim=vis_dim)
        
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
        )
        
        fused_dim = vis_dim + 64
        
        self.actor_head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        
        priv_dim = 7
        self.critic_head = nn.Sequential(
            nn.Linear(fused_dim + priv_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(True),
            nn.Linear(hidden // 2, 1),
        )


class DeployPolicy(nn.Module):
    """Deployable policy - actor only, no critic."""
    def __init__(self, full_policy):
        super().__init__()
        self.vision_encoder = full_policy.vision_encoder
        self.imu_encoder = full_policy.imu_encoder
        self.actor_head = full_policy.actor_head
    
    def forward(self, rgb, imu):
        """
        Args:
            rgb: [1, 4, 128, 128] - 4 stacked grayscale frames, normalized [0,1]
            imu: [1, 6] - linear vel (x,y,z) + angular vel (x,y,z)
        Returns:
            action: [1, 2] - [v_cmd, omega_cmd] in [-1, 1]
        """
        vis_feat = self.vision_encoder(rgb)
        imu_feat = self.imu_encoder(imu)
        feat = torch.cat([vis_feat, imu_feat], dim=-1)
        return torch.tanh(self.actor_head(feat))


def export_policy(checkpoint_path, output_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Create policy and load weights
    policy = VisionIMUAttentionYawPolicy()
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    
    print(f"  Stage: {ckpt.get('stage', '?')}")
    print(f"  Max Stage: {ckpt.get('max_stage', '?')}")
    print(f"  Steps: {ckpt.get('step', '?'):,}")
    
    # Create deployable version
    deploy = DeployPolicy(policy)
    deploy.eval()
    
    # Trace
    dummy_rgb = torch.zeros(1, 4, 128, 128)
    dummy_imu = torch.zeros(1, 6)
    
    traced = torch.jit.trace(deploy, (dummy_rgb, dummy_imu))
    traced.save(output_path)
    
    print(f"\n[OK] Exported to: {output_path}")
    print(f"     Input RGB: [1, 4, 128, 128] float32 [0,1]")
    print(f"     Input IMU: [1, 6] float32")
    print(f"     Output: [1, 2] float32 (v_cmd, omega_cmd)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="/home/schux00/teko_policy_S27.pt")
    args = parser.parse_args()
    
    export_policy(args.checkpoint, args.output)
