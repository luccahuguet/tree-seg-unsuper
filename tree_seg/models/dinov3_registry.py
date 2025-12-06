"""Registry and configs for DINOv3 adapter variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import sys
from pathlib import Path

# Ensure local dinov3 submodule is importable
DINOV3_PATH = Path(__file__).parent.parent.parent / "dinov3"
if str(DINOV3_PATH) not in sys.path:
    sys.path.insert(0, str(DINOV3_PATH))

import dinov3.hub.backbones as dinov3_backbones

AttentionOptions = Literal["q", "k", "v", "o", "none"]


class LoadingStrategy(Enum):
    """DINOv3 model loading strategies."""

    ORIGINAL_HUB = "original_hub"
    HUGGINGFACE = "huggingface"
    RANDOM_WEIGHTS = "random_weights"


@dataclass
class ModelConfig:
    """Configuration for a DINOv3 model variant."""

    hub_fn: callable
    hf_model: str
    feat_dim: int
    description: str
    params_count: str


MODEL_REGISTRY = {
    "dinov3_vits16": ModelConfig(
        hub_fn=dinov3_backbones.dinov3_vits16,
        hf_model="facebook/dinov3-vits16-pretrain-lvd1689m",
        feat_dim=384,
        description="Small model - good balance of speed and accuracy",
        params_count="21M",
    ),
    "dinov3_vitb16": ModelConfig(
        hub_fn=dinov3_backbones.dinov3_vitb16,
        hf_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        feat_dim=768,
        description="Base model - recommended for most use cases",
        params_count="86M",
    ),
    "dinov3_vitl16": ModelConfig(
        hub_fn=dinov3_backbones.dinov3_vitl16,
        hf_model="facebook/dinov3-vitl16-pretrain-lvd1689m",
        feat_dim=1024,
        description="Large model - higher accuracy, slower processing",
        params_count="304M",
    ),
    "dinov3_vith16plus": ModelConfig(
        hub_fn=dinov3_backbones.dinov3_vith16plus,
        hf_model="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        feat_dim=1280,
        description="Huge+ model - best accuracy, requires significant GPU memory",
        params_count="1.1B",
    ),
    "dinov3_vit7b16": ModelConfig(
        hub_fn=dinov3_backbones.dinov3_vit7b16,
        hf_model="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        feat_dim=4096,
        description="Mega model - satellite-grade accuracy, 40+ GB VRAM required",
        params_count="7B",
    ),
}
