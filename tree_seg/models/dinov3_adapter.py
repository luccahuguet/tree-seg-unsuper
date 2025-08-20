"""
DINOv3 Adapter for tree segmentation - replaces HighResDV2.

Provides a clean interface to DINOv3 models that matches the existing API.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Literal
from pathlib import Path

# Add DINOv3 to Python path
DINOV3_PATH = Path(__file__).parent.parent.parent / "dinov3"
if str(DINOV3_PATH) not in sys.path:
    sys.path.insert(0, str(DINOV3_PATH))

# Import DINOv3 components
try:
    import dinov3.hub.backbones as dinov3_backbones
    from dinov3.hub.utils import DINOV3_BASE_URL
except ImportError as e:
    raise ImportError(f"Failed to import DINOv3. Ensure submodule is initialized: {e}")

AttentionOptions = Literal["q", "k", "v", "o", "none"]


class DINOv3Adapter(nn.Module):
    """
    Adapter for DINOv3 models that provides the same interface as HighResDV2.
    
    This adapter:
    1. Loads DINOv3 backbone models
    2. Extracts features from multiple layers if needed
    3. Provides the same forward_sequential interface
    4. Handles feature upsampling and processing
    """
    
    def __init__(
        self,
        model_name: str,
        stride: int = 4,
        dtype: torch.dtype = torch.float32,
        track_grad: bool = False,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.stride = stride
        self.dtype = dtype
        self.track_grad = track_grad
        
        # Load the DINOv3 backbone
        self.backbone = self._load_backbone(model_name)
        self.backbone.eval()
        
        # Get model parameters
        self.patch_size = 16  # DINOv3 uses patch size 16
        self.feat_dim = self._get_feature_dim(model_name)
        
        # Set dtype
        if dtype != torch.float32:
            self.backbone = self.backbone.to(dtype)
            
        print(f"✅ DINOv3Adapter initialized: {model_name}")
        print(f"   Feature dimension: {self.feat_dim}")
        print(f"   Patch size: {self.patch_size}")
        print(f"   Stride: {stride}")
    
    def _load_backbone(self, model_name: str) -> nn.Module:
        """Load DINOv3 backbone model."""
        # Map model names to DINOv3 hub functions
        model_map = {
            "dinov3_vits16": dinov3_backbones.dinov3_vits16,
            "dinov3_vitb16": dinov3_backbones.dinov3_vitb16,
            "dinov3_vitl16": dinov3_backbones.dinov3_vitl16,
            "dinov3_vith16plus": dinov3_backbones.dinov3_vith16plus,
            "dinov3_vit7b16": dinov3_backbones.dinov3_vit7b16,
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown DINOv3 model: {model_name}. Available: {list(model_map.keys())}")
        
        # Load model using the hub function
        model_fn = model_map[model_name]
        try:
            backbone = model_fn(pretrained=True, weights="LVD1689M")
            print(f"📥 Loaded DINOv3 model: {model_name} (LVD1689M weights)")
        except Exception as e:
            # Fallback to SAT493M if available
            try:
                backbone = model_fn(pretrained=True, weights="SAT493M")
                print(f"📥 Loaded DINOv3 model: {model_name} (SAT493M weights - satellite optimized)")
            except Exception as e2:
                # Fallback to default
                backbone = model_fn(pretrained=True)
                print(f"📥 Loaded DINOv3 model: {model_name} (default weights)")
        
        return backbone
    
    def _get_feature_dim(self, model_name: str) -> int:
        """Get feature dimension for the model."""
        # DINOv3 feature dimensions
        dim_map = {
            "dinov3_vits16": 384,      # Small
            "dinov3_vitb16": 768,      # Base
            "dinov3_vitl16": 1024,     # Large
            "dinov3_vith16plus": 1280, # Huge+
            "dinov3_vit7b16": 1536,    # 7B
        }
        return dim_map.get(model_name, 768)  # Default to base
    
    def _get_image_patches_info(self, image_height: int, image_width: int) -> Dict[str, int]:
        """Calculate patch grid dimensions."""
        # DINOv3 uses patch size 16 with stride based on the model
        # For feature extraction, we use the model's default stride behavior
        h_patches = image_height // self.patch_size
        w_patches = image_width // self.patch_size
        
        return {
            "h_patches": h_patches,
            "w_patches": w_patches,
            "total_patches": h_patches * w_patches
        }
    
    def forward_sequential(
        self, 
        x: torch.Tensor, 
        attn_choice: AttentionOptions = "none"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass that extracts features and optionally attention.
        
        Args:
            x: Input image tensor (C, H, W)
            attn_choice: Type of attention to extract ("none", "o", etc.)
            
        Returns:
            Dictionary with extracted features
        """
        x.requires_grad = self.track_grad
        
        if self.dtype != torch.float32:
            x = x.to(self.dtype)
        
        # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # Get original image dimensions
        _, _, img_h, img_w = x.shape
        
        # Extract features using DINOv3
        with torch.no_grad():
            # Get patch tokens (excluding CLS token)
            features = self.backbone.forward_features(x)
            
            # DINOv3 returns features in format (B, N+1, D) where N is num_patches, +1 for CLS
            # We want just the patch tokens
            patch_features = features[:, 1:, :]  # Remove CLS token
            
            # Reshape to spatial format
            patch_info = self._get_image_patches_info(img_h, img_w)
            h_patches, w_patches = patch_info["h_patches"], patch_info["w_patches"]
            
            # Reshape from (B, N, D) to (B, H, W, D)
            patch_features = patch_features.view(1, h_patches, w_patches, -1)
            
            # Remove batch dimension and convert to (H, W, D)
            patch_features = patch_features.squeeze(0)
        
        # For v1.5 compatibility, we return both patch features and attention features
        # For now, we'll duplicate patch features as attention features
        # This maintains compatibility with existing segmentation code
        result = {
            "x_norm_patchtokens": patch_features,
            "x_patchattn": patch_features if attn_choice != "none" else None
        }
        
        return result
    
    def get_n_patches(self, img_h: int, img_w: int) -> tuple[int, int]:
        """Get number of patches for given image dimensions."""
        patch_info = self._get_image_patches_info(img_h, img_w)
        return patch_info["h_patches"], patch_info["w_patches"]
    
    def eval(self):
        """Set model to evaluation mode."""
        self.backbone.eval()
        return self


def create_dinov3_model(
    model_name: str,
    stride: int = 4,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> DINOv3Adapter:
    """
    Factory function to create DINOv3 adapter.
    
    Args:
        model_name: Name of the DINOv3 model
        stride: Stride parameter (kept for compatibility)
        device: Target device
        dtype: Model dtype
        
    Returns:
        Configured DINOv3Adapter
    """
    model = DINOv3Adapter(
        model_name=model_name,
        stride=stride,
        dtype=dtype
    )
    
    if device is not None:
        model = model.to(device)
    
    return model