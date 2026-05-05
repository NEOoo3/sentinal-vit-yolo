"""
Sentinel-ViT: Source Core
This module exposes the primary architecture and configuration components.
"""

from .model import YOLOv11ViTHybrid, build_model
from .config import Config, ModelConfig, get_default_config

__all__ = [
    "YOLOv11ViTHybrid",
    "build_model",
    "Config",
    "ModelConfig",
    "get_default_config"
]
