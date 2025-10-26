# SPDX-License-Identifier: BSD-3-Clause
"""
Simple camera wrapper for the TEKO robot using isaacsim.sensors.camera.Camera
(compatible with Isaac Sim 5.0 / Isaac Lab 0.47.1).
"""

from __future__ import annotations
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.sensors.camera import Camera


class TekoCamera:
    """Safe wrapper around isaacsim.sensors.camera.Camera."""

    def __init__(
        self,
        prim_path: str = "/World/Robot/RearCamera",
        position=(0.0, 0.0, 0.6),
        rotation=(0.0, 0.0, 0.0),  # Euler degrees
        resolution=(640, 480),
        frequency_hz: int = 30,
    ):
        self.prim_path = prim_path
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.resolution = resolution
        self.frequency_hz = frequency_hz

        # Converte Euler → quat (x, y, z, w)
        quat_xyzw = rot_utils.euler_angles_to_quats(self.rotation, degrees=True)

        # Cria e inicializa a câmera
        self._camera = Camera(
            prim_path=self.prim_path,
            position=self.position,
            orientation=quat_xyzw,
            resolution=self.resolution,
            frequency=self.frequency_hz,
        )
        self._camera.initialize()

        print(f"[INFO] Camera created at {self.prim_path} | res={self.resolution} | freq={self.frequency_hz} Hz")

    # ------------------------------------------------------------------ #
    def get_rgba(self) -> np.ndarray:
        """Returns the full RGBA frame as uint8 numpy array (H, W, 4)."""
        return self._camera.get_rgba()

    def get_rgb(self) -> np.ndarray:
        """Returns only the RGB channels (H, W, 3)."""
        rgba = self._camera.get_rgba()
        if isinstance(rgba, np.ndarray) and rgba.shape[-1] == 4:
            return rgba[..., :3]
        return rgba

    def capture_frame(self) -> np.ndarray:
        """Alias for get_rgb(), for quick one-off captures."""
        return self.get_rgb()

    def __repr__(self):
        return f"<TekoCamera prim={self.prim_path}, res={self.resolution}, freq={self.frequency_hz} Hz>"
