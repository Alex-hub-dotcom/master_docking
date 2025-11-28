"""
TEKO Brain Module
----------------
Contains CNN models for vision-based docking.
"""
from .cnn_model import create_visual_encoder, SimpleCNN

__all__ = [
    "create_visual_encoder",
    "SimpleCNN",
]
