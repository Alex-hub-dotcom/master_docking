# Copyright (c) 2022-2025, The TEKO Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Camera interface for TEKO environments using tiled rendering.
This module wraps Isaac Lab's TiledCamera to provide easy access
to RGB (and optionally depth/segmentation) data from each robot's camera.
"""

from __future__ import annotations

import torch
from isaaclab.sensors import TiledCamera, TiledCameraCfg



class TekoCamera:
    """High-level wrapper around Isaac Lab's TiledCamera for TEKO robots."""

    def __init__(self, cfg: TiledCameraCfg):
        """Initialize the camera sensor using a TiledCameraCfg."""
        self.cfg = cfg
        self.tiled_camera = TiledCamera(cfg)
        #Necessary due to outdate Isaac lab stuff
        self.tiled_camera._timestamp = 0.0                                                               #1
        self.tiled_camera._timestamp_last_update = 0.0                                                   #2
        self.tiled_camera._is_outdated = torch.zeros((1,), dtype=torch.bool, device="cuda")              #3

    def get_rgb(self) -> torch.Tensor:
        """Return the latest RGB image batch from all environments.

        Shape: (num_envs, height, width, 3)
        Dtype: torch.uint8
        """
         # Garante que a câmera está atualizada
        if self.tiled_camera.data.output is None:
            # força uma atualização inicial caso o buffer ainda não exista
            self.tiled_camera.update(0.0)

        data = self.tiled_camera.data.output
        if data is None or "rgb" not in data:
            return torch.zeros((1, self.cfg.height, self.cfg.width, 3), dtype=torch.uint8, device="cuda")

        return data["rgb"]

    def get_depth(self) -> torch.Tensor:
        """Return the latest depth image batch if available."""
        return self.tiled_camera.data.output.get("depth", None)

    def update(self, dt: float = 0.0):
        """Update the camera buffers (called once per simulation step)."""
        self.tiled_camera.update(dt)

    def reset(self):
        """Reset the internal buffers of the camera."""
        self.tiled_camera.reset()

    def __repr__(self) -> str:
        """Readable summary of the camera state."""
        info = (
            f"TekoCamera(num_envs={self.tiled_camera.num_cameras}, "
            f"resolution={self.cfg.width}x{self.cfg.height}, "
            f"data_types={self.cfg.data_types})"
        )
        return info
