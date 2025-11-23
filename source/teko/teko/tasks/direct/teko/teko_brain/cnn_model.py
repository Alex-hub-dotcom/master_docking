# SPDX-License-Identifier: BSD-3-Clause
"""
CNN Feature Extractor for TEKO Docking
--------------------------------------
Stripped-down build: only a lightweight custom CNN (SimpleCNN), trained from scratch.

Key points:
- No torchvision / MobileNet dependency.
- BatchNorm after each Conv for more stable training.
- ImageNet-style normalization, stored as buffers (no allocations in forward).
- `DockingCNN` is kept as an alias of `SimpleCNN` for backwards compatibility.
- `create_visual_encoder(...)` accepts "simple" or "mobilenet" but both return SimpleCNN.

Author: Alexandre Schleier Neves da Silva
If you have questions or need support, contact:
  alexandre.schleiernevesdasilva@uni-hohenheim.de
"""

import torch
import torch.nn as nn


# ================================================================
#  SimpleCNN (Lightweight custom architecture)
# ================================================================
class SimpleCNN(nn.Module):
    """
    Lightweight CNN for the TEKO docking task.
    This model is always trained from scratch.
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
            # No dropout: keep behaviour deterministic for RL.
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.feature_dim = feature_dim

        # --- Register normalization buffers (no realloc on each forward) ---
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)

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
            x: RGB images [B, 3, H, W], values in [0, 1].

        Returns:
            features: [B, feature_dim]
        """
        # Normalize (same as ImageNet for consistency,
        # even though we don't use pretrained weights)
        x = (x - self.img_mean) / self.img_std

        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


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
        pretrained:   Ignored (kept so old code doesn't break).

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
# We map it to SimpleCNN so those imports continue to work.
# ----------------------------------------------------------------
DockingCNN = SimpleCNN


# ================================================================
#  Self-test (optional)
# ================================================================
if __name__ == "__main__":
    print("Testing SimpleCNN model...")

    test_input = torch.randn(4, 3, 480, 640)
    s = SimpleCNN(feature_dim=256)
    out = s(test_input)
    print(f"   Input:  {test_input.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Params: {sum(p.numel() for p in s.parameters()):,}")

    print("\n✓ SimpleCNN test passed!")
