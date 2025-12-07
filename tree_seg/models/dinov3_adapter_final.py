"""
DINOv3 Adapter - Final Implementation

Production-ready DINOv3 adapter incorporating insights from the official
Meta implementation and our debugging discoveries.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from .dinov3_loader import HuggingFaceWeightLoader, WeightLoadingError
from .dinov3_features import extract_backbone_features
from .dinov3_registry import (
    AttentionOptions,
    LoadingStrategy,
    MODEL_REGISTRY,
)

# Setup logging
logger = logging.getLogger(__name__)

# Add DINOv3 to Python path
DINOV3_PATH = Path(__file__).parent.parent.parent / "dinov3"
if str(DINOV3_PATH) not in sys.path:
    sys.path.insert(0, str(DINOV3_PATH))

# Import DINOv3 components
try:
    import dinov3.hub.backbones as dinov3_backbones  # noqa: F401
except ImportError as e:
    raise ImportError(f"Failed to import DINOv3. Ensure submodule is initialized: {e}")


class DINOv3Exception(Exception):
    """Base exception for DINOv3 adapter issues."""

    pass


class DINOv3Adapter(nn.Module):
    """
    Production-ready DINOv3 adapter for tree segmentation.

    Key features:
    - Multi-strategy loading (original hub -> HuggingFace -> random weights)
    - Automatic LinearKMaskedBias initialization fix
    - Robust error handling and logging
    - Optimized parameter mapping
    - Compatible with existing segmentation pipeline
    """

    def __init__(
        self,
        model_name: str,
        stride: int = 4,
        dtype: torch.dtype = torch.float32,
        track_grad: bool = False,
        hf_token: Optional[str] = None,
        force_strategy: Optional[LoadingStrategy] = None,
    ):
        super().__init__()

        # Validate model
        if model_name not in MODEL_REGISTRY:
            available_models = list(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {available_models}"
            )

        self.model_name = model_name
        self.model_config = MODEL_REGISTRY[model_name]
        self.stride = stride
        self.dtype = dtype
        self.track_grad = track_grad
        self.force_strategy = force_strategy

        # Load backbone model
        self.backbone, self.loading_strategy = self._load_backbone_with_strategy(
            hf_token
        )
        self.backbone.eval()

        # Apply dtype conversion
        if dtype != torch.float32:
            self.backbone = self.backbone.to(dtype)

        # Model properties from official implementation
        self.patch_size = getattr(self.backbone, "patch_size", 16)
        self.feat_dim = self.model_config.feat_dim

        self._log_initialization_success()

    def _load_backbone_with_strategy(
        self, hf_token: Optional[str]
    ) -> Tuple[nn.Module, LoadingStrategy]:
        """Load backbone using the best available strategy with smart caching."""
        # Skip original hub by default since it consistently fails with 403
        strategies = [LoadingStrategy.HUGGINGFACE, LoadingStrategy.RANDOM_WEIGHTS]

        # Only try original hub if explicitly forced or if we haven't seen it fail
        if self.force_strategy == LoadingStrategy.ORIGINAL_HUB:
            strategies = [
                LoadingStrategy.ORIGINAL_HUB,
                LoadingStrategy.HUGGINGFACE,
                LoadingStrategy.RANDOM_WEIGHTS,
            ]

        last_error = None

        for strategy in strategies:
            try:
                if strategy == LoadingStrategy.ORIGINAL_HUB:
                    backbone = self._load_from_original_hub()
                    return backbone, strategy

                elif strategy == LoadingStrategy.HUGGINGFACE:
                    backbone = self._load_from_huggingface(hf_token)
                    return backbone, strategy

                elif strategy == LoadingStrategy.RANDOM_WEIGHTS:
                    backbone = self._load_random_weights()
                    return backbone, strategy

            except Exception as e:
                last_error = e
                logger.warning(f"Strategy {strategy.value} failed: {e}")
                continue

        raise DINOv3Exception(
            f"All loading strategies failed. Last error: {last_error}"
        )

    def _load_from_original_hub(self) -> nn.Module:
        """Load from original DINOv3 hub (often blocked by permissions)."""
        logger.info("Attempting original DINOv3 hub...")
        backbone = self.model_config.hub_fn(pretrained=True)
        logger.info("✅ Loaded from original DINOv3 hub")
        return backbone

    def _load_from_huggingface(self, hf_token: Optional[str]) -> nn.Module:
        """Load architecture and apply HuggingFace pretrained weights."""
        logger.info("Loading from HuggingFace...")

        # Initialize weight loader
        weight_loader = HuggingFaceWeightLoader(self.model_config.hf_model, hf_token)

        if not weight_loader.load_weights():
            raise WeightLoadingError("Failed to download HuggingFace weights")

        # Load model architecture
        backbone = self.model_config.hub_fn(pretrained=False)

        # Essential initialization steps (from our debugging)
        logger.info("Initializing model parameters...")
        backbone.init_weights()  # Initialize LayerScale and other parameters

        logger.info("Fixing LinearKMaskedBias initialization...")
        self._fix_linear_k_masked_bias(backbone)  # Fix NaN bias_mask issue

        # Apply HuggingFace weights
        logger.info("Applying HuggingFace weights...")
        applied, total = weight_loader.apply_weights_to_model(backbone)

        logger.info(f"✅ HuggingFace loading complete ({applied}/{total} parameters)")
        return backbone

    def _load_random_weights(self) -> nn.Module:
        """Load architecture with random weights (fallback)."""
        logger.warning(
            "Loading with random weights - model performance will be limited!"
        )
        backbone = self.model_config.hub_fn(pretrained=False)
        backbone.init_weights()
        self._fix_linear_k_masked_bias(backbone)
        return backbone

    def _fix_linear_k_masked_bias(self, model: nn.Module) -> None:
        """
        Fix LinearKMaskedBias layers with NaN bias_mask.

        This was the root cause of our NaN feature issue - the bias_mask
        was initialized with NaN values, causing NaN propagation.
        """
        fixed_count = 0

        for name, module in model.named_modules():
            if hasattr(module, "bias_mask") and module.bias_mask is not None:
                if torch.isnan(module.bias_mask).any():
                    # Create proper mask: enable Q and V bias, disable K bias
                    bias_len = module.bias_mask.shape[0]
                    third_len = bias_len // 3

                    # Mask pattern: [Q: 1.0, K: 0.0, V: 1.0]
                    proper_mask = torch.ones_like(module.bias_mask)
                    proper_mask[third_len : 2 * third_len] = 0.0  # Disable K bias

                    module.bias_mask.data.copy_(proper_mask)
                    fixed_count += 1

        if fixed_count > 0:
            logger.info(f"✅ Fixed {fixed_count} LinearKMaskedBias layers")

    def _log_initialization_success(self):
        """Log successful initialization details."""
        strategy_msg = {
            LoadingStrategy.ORIGINAL_HUB: "Original DINOv3 hub",
            LoadingStrategy.HUGGINGFACE: f"HuggingFace ({self.model_config.hf_model})",
            LoadingStrategy.RANDOM_WEIGHTS: "Random weights",
        }

        logger.info("🎉 DINOv3Adapter initialized successfully!")
        logger.info(f"   Model: {self.model_name} ({self.model_config.params_count})")
        logger.info(f"   Strategy: {strategy_msg[self.loading_strategy]}")
        logger.info(f"   Features: {self.feat_dim}D, Patch size: {self.patch_size}")
        logger.info(f"   Description: {self.model_config.description}")

    def forward_sequential(
        self,
        x: torch.Tensor,
        attn_choice: AttentionOptions = "none",
        use_multi_layer: bool = False,
        layer_indices: tuple = (3, 6, 9, 12),
        feature_aggregation: str = "concat",
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features using DINOv3 backbone.

        Based on official adapter but simplified for tree segmentation.
        Uses the same interface as the original HighResDV2 adapter.

        Args:
            x: Input tensor
            attn_choice: Attention choice
            use_multi_layer: Extract features from multiple layers
            layer_indices: Which layers to extract from (1-indexed)
            feature_aggregation: How to combine features ("concat", "average", "weighted")
        """
        patch_features, outputs = extract_backbone_features(
            backbone=self.backbone,
            x=x,
            track_grad=self.track_grad,
            dtype=self.dtype,
            use_multi_layer=use_multi_layer,
            layer_indices=layer_indices,
            feature_aggregation=feature_aggregation,
            patch_size=self.patch_size,
        )

        outputs["x_patchattn"] = patch_features if attn_choice != "none" else None
        return outputs

    def get_n_patches(self, img_h: int, img_w: int) -> Tuple[int, int]:
        """Calculate number of patches for given image dimensions."""
        return img_h // self.patch_size, img_w // self.patch_size

    def eval(self):
        """Set model to evaluation mode."""
        self.backbone.eval()
        return self


def create_dinov3_model(
    model_name: str,
    stride: int = 4,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> DINOv3Adapter:
    """
    Factory function for creating DINOv3 adapters.

    Args:
        model_name: DINOv3 model variant
        stride: Stride (kept for API compatibility)
        device: Target device
        dtype: Model precision
        **kwargs: Additional DINOv3Adapter arguments
    """
    model = DINOv3Adapter(model_name=model_name, stride=stride, dtype=dtype, **kwargs)

    if device is not None:
        model = model.to(device)

    return model


# Utility functions for model discovery and information
def list_available_models() -> Dict[str, str]:
    """List all available DINOv3 models with descriptions."""
    return {
        name: f"{config.description} ({config.params_count})"
        for name, config in MODEL_REGISTRY.items()
    }


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_REGISTRY[model_name]
    return {
        "name": model_name,
        "feature_dim": config.feat_dim,
        "parameters": config.params_count,
        "description": config.description,
        "hf_model": config.hf_model,
    }


def print_model_info():
    """Print information about all available models."""
    print("🔍 Available DINOv3 Models:")
    print("=" * 60)

    for model_name, config in MODEL_REGISTRY.items():
        print(f"📦 {model_name}")
        print(f"   Parameters: {config.params_count}")
        print(f"   Features: {config.feat_dim}D")
        print(f"   Description: {config.description}")
        print(f"   HF Model: {config.hf_model}")
        print()


if __name__ == "__main__":
    # Demo usage
    print_model_info()
