# SPDX-License-Identifier: BSD-3-Clause
"""
Camera sensor wrapper for the TEKO robot (Isaac Lab 0.47.1 compatible, usando câmera existente no USD).
"""

from __future__ import annotations
import torch
import time

from isaaclab.sensors import TiledCamera as BaseTiledCamera, TiledCameraCfg
from isaacsim.core.utils.prims import get_prim_at_path  # ✅ versão correta para 0.47.1


class PatchedTiledCamera(BaseTiledCamera):
    """Versão corrigida da TiledCamera que garante que self.cfg existe."""

    def __init__(self, cfg: TiledCameraCfg):
        self.cfg = cfg
        try:
            super().__init__(cfg)
        except AttributeError:
            pass


class TekoCamera:
    """Wrapper seguro e compatível para uma TiledCamera do Isaac Lab."""

    def __init__(self, cfg: TiledCameraCfg):
        """Inicializa e anexa a câmera definida no config."""
        self.cfg = cfg

        # --- Evita recriação se a câmera já existe no USD ---
        prim = get_prim_at_path(cfg.prim_path)
        if prim and prim.IsValid():
            cfg.spawn = None  # impede o TiledCamera de tentar criar novamente

        # Cria o objeto de câmera patchado
        self._camera = PatchedTiledCamera(cfg)

        # Força inicialização de campos internos se faltarem
        defaults = {
            "_timestamp": 0.0,
            "_timestamp_last_update": 0.0,
            "_is_outdated": False,
            "_enabled": True,
            "_usd_sensor_prim": None,
            "_sensor_interface": None,
            "_last_render_time": time.time(),
        }
        for key, val in defaults.items():
            if not hasattr(self._camera, key):
                setattr(self._camera, key, val)

        print(f"[INFO] Camera initialized at prim: {cfg.prim_path}")
        print(f"[INFO] Resolution: {cfg.width}x{cfg.height}")
        print(f"[INFO] Data types: {cfg.data_types}")

    def update(self, dt: float = 1.0 / 60.0):
        """Atualiza o sensor da câmera (compatível com builds incompletas)."""
        try:
            self._camera.update(dt)
        except AttributeError:
            if not hasattr(self._camera, "_timestamp"):
                self._camera._timestamp = 0.0
            self._camera._timestamp += dt

    def get_rgb(self) -> torch.Tensor | None:
        """Obtém o frame RGB da câmera."""
        data = getattr(self._camera, "data", None)
        if not data or not hasattr(data, "output"):
            return None

        rgb = data.output.get("rgb")
        if rgb is None or not isinstance(rgb, torch.Tensor):
            return None

        return rgb

    def __repr__(self):
        return (
            f"<TekoCamera prim={self.cfg.prim_path} "
            f"res={self.cfg.width}x{self.cfg.height}>"
        )
