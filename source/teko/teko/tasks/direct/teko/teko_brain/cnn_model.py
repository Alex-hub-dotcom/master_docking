# SPDX-License-Identifier: BSD-3-Clause
"""
CNN Feature Extractor for TEKO Docking
--------------------------------------
Stripped-down build: only a lightweight custom CNN (SimpleCNN), trained from scratch.

Key points:
- No torchvision / MobileNet dependency.
- BatchNorm after each Conv for more stable training.
- NO ImageNet normalization (data comes in [0,1] from environment)
- Supports stacked frames: input [B, 3 * K, H, W] is reshaped to K RGB frames.
- Aggregation over time by mean-pooling the per-frame features.

Author: Alexandre Schleier Neves da Silva
Contact: alexandre.schleiernevesdasilva@uni-hohenheim.de
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for the TEKO docking task.

    Supports both:
      - single-frame input: [B, 3, H, W]
      - stacked frames:     [B, 3 * K, H, W]

    For stacked frames, the network:
      1) reshapes to [B, K, 3, H, W],
      2) runs each frame independently through the same CNN,
      3) mean-pools the resulting features over the K frames.
    
    ✅ IMPORTANT: Input data MUST be in [0, 1] range (already normalized in env).
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()

        # --- Convolutional feature extractor ---
        # Shapes (approx, assuming input [B, 3, 480, 640]):
        #   Conv1:  [B, 3, 480, 640]  -> [B, 32, 120, 160]
        #   Pool1:  [B, 32, 120, 160] -> [B, 32, 60, 80]
        #   Conv2:  [B, 32, 60, 80]   -> [B, 64, 30, 40]
        #   Pool2:  [B, 64, 30, 40]   -> [B, 64, 15, 20]
        #   Conv3:  [B, 64, 15, 20]   -> [B, 128, 15, 20]
        #   Pool3:  [B, 128, 15, 20]  -> [B, 128, 7, 10]
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # --- Automatically determine flattened size ---
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 480, 640)
            n_flat = self.features(dummy).view(1, -1).shape[1]

        # --- Fully connected projection ---
        self.fc = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.feature_dim = feature_dim

        # --- Initialize weights ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB images in [0, 1] range with shape:
                 - [B, 3, H, W]          (single frame), or
                 - [B, 3 * K, H, W]      (K stacked frames).

        Returns:
            features: [B, feature_dim]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got {x.shape}")

        b, c, h, w = x.shape
        if c % 3 != 0:
            raise ValueError(
                f"Expected channel dimension to be a multiple of 3 (RGB frames), "
                f"got C={c}"
            )

        num_frames = c // 3

        # Reshape to [B, K, 3, H, W] for per-frame processing
        x = x.view(b, num_frames, 3, h, w)

        # ✅ NO NORMALIZATION - data is already in [0, 1] from environment!
        # The environment already does: rgb = rgb_data.permute(2, 0, 1).float() / 255.0

        # Collapse time dimension into batch: [B * K, 3, H, W]
        x = x.view(b * num_frames, 3, h, w)

        # Standard CNN forward
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # [B * K, feature_dim]

        # Aggregate over frames (mean pooling over time): [B, feature_dim]
        x = x.view(b, num_frames, self.feature_dim).mean(dim=1)

        return x


# ================================================================
#  Factory Function
# ================================================================
def create_visual_encoder(
    architecture: str = "simple",
    feature_dim: int = 256,
    pretrained: bool = True,  # kept for API compatibility, but unused
) -> nn.Module:
    """
    Create a visual encoder for the TEKO docking task.

    In this stripped-down build, we always return SimpleCNN.

    Args:
        architecture: "simple" or "mobilenet" (both map to SimpleCNN here).
        feature_dim:  Output feature dimension for the RL policy.
        pretrained:   Ignored (kept so old code does not break).

    Returns:
        nn.Module: encoder instance (SimpleCNN)
    """
    arch = architecture.lower()
    if arch not in ("simple", "mobilenet"):
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"This build only supports SimpleCNN. Use architecture='simple'."
        )

    if arch == "mobilenet":
        print(
            "[TEKO][cnn_model] Warning: 'mobilenet' encoder requested, "
            "but stripped-down build only provides SimpleCNN. "
            "Using SimpleCNN instead."
        )

    return SimpleCNN(feature_dim=feature_dim)


# ----------------------------------------------------------------
# Backwards-compat alias: some code still imports DockingCNN.
# ----------------------------------------------------------------
DockingCNN = SimpleCNN


# ================================================================
#  Self-test (optional)
# ================================================================
if __name__ == "__main__":
    print("Testing SimpleCNN model...")

    # Single-frame input
    test_input_1 = torch.rand(4, 3, 480, 640)  # [0, 1] range
    s = SimpleCNN(feature_dim=256)
    out_1 = s(test_input_1)
    print(f"Single-frame input:  {test_input_1.shape} -> {out_1.shape}")
    print(f"  Output range: [{out_1.min().item():.3f}, {out_1.max().item():.3f}]")

    # Stacked-frame input (e.g., 4 frames -> 12 channels)
    test_input_4 = torch.rand(4, 12, 480, 640)  # [0, 1] range
    out_4 = s(test_input_4)
    print(f"Stacked-frame input: {test_input_4.shape} -> {out_4.shape}")
    print(f"  Output range: [{out_4.min().item():.3f}, {out_4.max().item():.3f}]")

    print("\n✓ SimpleCNN test passed!")