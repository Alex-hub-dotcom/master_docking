#!/usr/bin/env python3
"""
Export Vision Optimal S41 Model
===============================
Exports the trained policy to ONNX/TorchScript for deployment.

Architecture: VisionIMUAttentionYawPolicy
- Input: RGB 4x128x128 (frame stack) + IMU 6D
- Output: Action 2D (v_cmd, w_cmd)

Author: Alexandre Schleier Neves da Silva
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from datetime import datetime


# =============================================================================
# ARCHITECTURE (must match training exactly)
# =============================================================================

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))


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
        return x * self.fc(y).view(b, c, 1, 1)


class VisionEncoderAttentionYaw(nn.Module):
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.channel_attn = ChannelAttention(64)
        self.spatial_attn = SpatialAttention(64)
        self.gn1 = nn.GroupNorm(8, 32)
        self.gn2 = nn.GroupNorm(8, 64)
        self.gn3 = nn.GroupNorm(8, 64)
        
        # Calculate flat size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            flat_size = self._forward_conv(dummy).shape[1]
        
        self.fc = nn.Linear(flat_size, feature_dim)
        
        self.yaw_head = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(True),
            nn.Linear(64, 32), nn.ReLU(True),
            nn.Linear(32, 1), nn.Tanh()
        )
        self.feature_dim = feature_dim
    
    def _forward_conv(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x.flatten(1)
    
    def forward(self, x):
        return F.relu(self.fc(self._forward_conv(x)))
    
    def predict_yaw(self, features):
        return self.yaw_head(features) * math.pi


class VisionIMUAttentionYawPolicy(nn.Module):
    """Full policy as trained."""
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
    
    def __init__(self, vis_dim=256, imu_dim=6, hidden=256, action_dim=2):
        super().__init__()
        self.vision_encoder = VisionEncoderAttentionYaw(in_channels=4, feature_dim=vis_dim)
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, 64), nn.ReLU(True),
            nn.Linear(64, 64), nn.ReLU(True),
        )
        fused_dim = vis_dim + 64
        self.actor_head = nn.Sequential(
            nn.Linear(fused_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        priv_dim = 7
        self.critic_head = nn.Sequential(
            nn.Linear(fused_dim + priv_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden // 2), nn.ReLU(True),
            nn.Linear(hidden // 2, 1),
        )
    
    def _std(self):
        return torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
    
    def forward(self, rgb, imu):
        """Forward pass for inference (deterministic action)."""
        vis_feat = self.vision_encoder(rgb)
        imu_feat = self.imu_encoder(imu)
        fused = torch.cat([vis_feat, imu_feat], dim=-1)
        mean = self.actor_head(fused)
        return torch.tanh(mean)
    
    def forward_with_yaw(self, rgb, imu):
        """Forward pass with yaw prediction (for analysis)."""
        vis_feat = self.vision_encoder(rgb)
        imu_feat = self.imu_encoder(imu)
        fused = torch.cat([vis_feat, imu_feat], dim=-1)
        mean = self.actor_head(fused)
        yaw_pred = self.vision_encoder.predict_yaw(vis_feat)
        return torch.tanh(mean), yaw_pred


class DeploymentPolicy(nn.Module):
    """
    Simplified policy for deployment (no critic, no privileged info).
    
    Inputs:
        rgb: [B, 4, 128, 128] - 4 stacked grayscale frames
        imu: [B, 6] - IMU data [vx, vy, vz, wx, wy, wz]
    
    Outputs:
        action: [B, 2] - [v_cmd, w_cmd] in [-1, 1]
    """
    def __init__(self, full_policy):
        super().__init__()
        self.vision_encoder = full_policy.vision_encoder
        self.imu_encoder = full_policy.imu_encoder
        self.actor_head = full_policy.actor_head
    
    def forward(self, rgb, imu):
        vis_feat = self.vision_encoder(rgb)
        imu_feat = self.imu_encoder(imu)
        fused = torch.cat([vis_feat, imu_feat], dim=-1)
        mean = self.actor_head(fused)
        return torch.tanh(mean)


def load_checkpoint(path):
    """Load checkpoint and print info."""
    print(f"Loading: {path}")
    ckpt = torch.load(path, map_location='cpu')
    
    print(f"  Step: {ckpt.get('step', 'N/A'):,}")
    print(f"  Stage: {ckpt.get('stage', 'N/A')}")
    print(f"  Max Stage: {ckpt.get('max_stage', 'N/A')}")
    
    if 'config' in ckpt:
        cfg = ckpt['config']
        print(f"  LR: {cfg.get('learning_rate', 'N/A')}")
        print(f"  Entropy: {cfg.get('entropy_coef', 'N/A')}")
    
    return ckpt


def export_onnx(policy, output_path):
    """Export to ONNX format."""
    policy.eval()
    
    dummy_rgb = torch.randn(1, 4, 128, 128)
    dummy_imu = torch.randn(1, 6)
    
    print(f"\nExporting ONNX: {output_path}")
    
    torch.onnx.export(
        policy,
        (dummy_rgb, dummy_imu),
        output_path,
        input_names=['rgb', 'imu'],
        output_names=['action'],
        dynamic_axes={
            'rgb': {0: 'batch'},
            'imu': {0: 'batch'},
            'action': {0: 'batch'},
        },
        opset_version=11,
        do_constant_folding=True,
    )
    print(f"  Done!")


def export_torchscript(policy, output_path):
    """Export to TorchScript format."""
    policy.eval()
    
    dummy_rgb = torch.randn(1, 4, 128, 128)
    dummy_imu = torch.randn(1, 6)
    
    print(f"\nExporting TorchScript: {output_path}")
    
    scripted = torch.jit.trace(policy, (dummy_rgb, dummy_imu))
    scripted.save(output_path)
    print(f"  Done!")


def main():
    parser = argparse.ArgumentParser(description='Export Vision Optimal S41')
    parser.add_argument('--checkpoint', type=str, 
                        default='/home/schux00/checkpoints/vision_optimal_FINAL_S41.pt',
                        help='Checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/home/schux00/exported_models',
                        help='Output directory')
    parser.add_argument('--formats', type=str, nargs='+',
                        default=['onnx', 'torchscript', 'weights'],
                        help='Export formats')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint)
    
    # Create and load policy
    policy = VisionIMUAttentionYawPolicy()
    policy.load_state_dict(ckpt['policy'])
    policy.eval()
    
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\nPolicy loaded: {n_params:,} parameters")
    
    # Create deployment policy (no critic)
    deploy_policy = DeploymentPolicy(policy)
    deploy_policy.eval()
    
    n_deploy_params = sum(p.numel() for p in deploy_policy.parameters())
    print(f"Deployment policy: {n_deploy_params:,} parameters")
    
    exported = []
    
    # Export formats
    if 'onnx' in args.formats:
        path = os.path.join(args.output_dir, f'teko_vision_S41_{timestamp}.onnx')
        export_onnx(deploy_policy, path)
        exported.append(path)
    
    if 'torchscript' in args.formats:
        path = os.path.join(args.output_dir, f'teko_vision_S41_{timestamp}.pt')
        export_torchscript(deploy_policy, path)
        exported.append(path)
    
    if 'weights' in args.formats:
        path = os.path.join(args.output_dir, f'teko_vision_S41_{timestamp}_weights.pth')
        torch.save(policy.state_dict(), path)
        print(f"\nExporting weights: {path}")
        exported.append(path)
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    for f in exported:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f)}: {size_mb:.2f} MB")
    
    # Usage examples
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    print("""
[ONNX - Python]
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('teko_vision_S41.onnx')

# Inputs (normalize images to [0,1] or [-1,1] as trained)
rgb = np.random.randn(1, 4, 128, 128).astype(np.float32)  # 4 stacked frames
imu = np.random.randn(1, 6).astype(np.float32)  # [vx,vy,vz,wx,wy,wz]

# Inference
action = sess.run(['action'], {'rgb': rgb, 'imu': imu})[0]
v_cmd, w_cmd = action[0]  # Commands in [-1, 1]
```

[TorchScript - Python]
```python
import torch

model = torch.jit.load('teko_vision_S41.pt')
model.eval()

with torch.no_grad():
    rgb = torch.randn(1, 4, 128, 128)
    imu = torch.randn(1, 6)
    action = model(rgb, imu)
    v_cmd, w_cmd = action[0].tolist()
```

[TorchScript - C++/LibTorch]
```cpp
#include <torch/script.h>

torch::jit::script::Module model = torch::jit::load("teko_vision_S41.pt");

std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::randn({1, 4, 128, 128}));  // rgb
inputs.push_back(torch::randn({1, 6}));  // imu

auto output = model.forward(inputs).toTensor();
```
""")
    
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()