"""
DINOv3 Adapter - Final Implementation

Production-ready DINOv3 adapter incorporating insights from the official
Meta implementation and our debugging discoveries.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Literal, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup logging
logger = logging.getLogger(__name__)

# Add DINOv3 to Python path
DINOV3_PATH = Path(__file__).parent.parent.parent / "dinov3"
if str(DINOV3_PATH) not in sys.path:
    sys.path.insert(0, str(DINOV3_PATH))

# Import DINOv3 components
try:
    import dinov3.hub.backbones as dinov3_backbones
except ImportError as e:
    raise ImportError(f"Failed to import DINOv3. Ensure submodule is initialized: {e}")

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


# Model registry - official DINOv3 variants
MODEL_REGISTRY = {
    "dinov3_vits16": ModelConfig(
        hub_fn=dinov3_backbones.dinov3_vits16,
        hf_model="facebook/dinov3-vits16-pretrain-lvd1689m",
        feat_dim=384,
        description="Small model - good balance of speed and accuracy",
        params_count="21M"
    ),
    "dinov3_vitb16": ModelConfig(
        hub_fn=dinov3_backbones.dinov3_vitb16,
        hf_model="facebook/dinov3-vitb16-pretrain-lvd1689m", 
        feat_dim=768,
        description="Base model - recommended for most use cases",
        params_count="86M"
    ),
    "dinov3_vitl16": ModelConfig(
        hub_fn=dinov3_backbones.dinov3_vitl16,
        hf_model="facebook/dinov3-vitl16-pretrain-lvd1689m",
        feat_dim=1024,
        description="Large model - higher accuracy, slower processing",
        params_count="304M"
    ),
    "dinov3_vith16plus": ModelConfig(
        hub_fn=dinov3_backbones.dinov3_vith16plus,
        hf_model="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        feat_dim=1280,
        description="Huge+ model - best accuracy, requires significant GPU memory",
        params_count="1.1B"
    ),
    "dinov3_vit7b16": ModelConfig(
        hub_fn=dinov3_backbones.dinov3_vit7b16,
        hf_model="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        feat_dim=4096,
        description="Mega model - satellite-grade accuracy, 40+ GB VRAM required",
        params_count="7B"
    ),
}


class DINOv3Exception(Exception):
    """Base exception for DINOv3 adapter issues."""
    pass


class WeightLoadingError(DINOv3Exception):
    """Exception for weight loading failures."""
    pass


class HuggingFaceWeightLoader:
    """
    Optimized HuggingFace weight loader with caching and robust error handling.
    
    Key improvements from debugging:
    - Handles missing K bias in attention layers
    - Proper QKV weight concatenation
    - Shape mismatch resolution (mask_token reshaping)
    - Comprehensive parameter mapping
    """
    
    def __init__(self, hf_model_id: str, hf_token: Optional[str] = None):
        self.hf_model_id = hf_model_id
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self._weights_cache: Optional[Dict[str, torch.Tensor]] = None
        
    def load_weights(self) -> bool:
        """Load HuggingFace safetensors weights with efficient caching."""
        try:
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
            
            if not self.hf_token:
                logger.warning("No HF_TOKEN found - some models may be inaccessible")
            
            # Check if already cached locally
            try:
                weights_path = hf_hub_download(
                    repo_id=self.hf_model_id,
                    filename='model.safetensors',
                    token=self.hf_token,
                    local_files_only=True  # Try local cache first
                )
                logger.info(f"✅ Using cached weights: {self.hf_model_id}")
            except:
                # Download if not in cache
                logger.info(f"⬇️ Downloading model weights: {self.hf_model_id}")
                weights_path = hf_hub_download(
                    repo_id=self.hf_model_id,
                    filename='model.safetensors',
                    token=self.hf_token
                )
            
            # Load all weights into memory for fast access
            self._weights_cache = {}
            with safe_open(weights_path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    self._weights_cache[key] = f.get_tensor(key)
            
            logger.info(f"Successfully loaded {len(self._weights_cache)} parameters")
            return True
            
        except Exception as e:
            logger.error(f"HuggingFace weight loading failed: {e}")
            return False
    
    def apply_weights_to_model(self, model: nn.Module) -> Tuple[int, int]:
        """Apply HuggingFace weights to DINOv3 model with optimized mapping."""
        if not self._weights_cache:
            raise WeightLoadingError("Weights not loaded. Call load_weights() first.")
        
        # Create parameter mapping
        parameter_map = self._create_parameter_mapping(model)
        
        # Apply weights
        applied_count = 0
        total_params = len(list(model.parameters()))
        
        for param_name, param in model.named_parameters():
            if param_name in parameter_map:
                mapping_info = parameter_map[param_name]
                
                if isinstance(mapping_info, str):
                    # Direct 1:1 mapping
                    if self._apply_direct_mapping(param, mapping_info, param_name):
                        applied_count += 1
                        
                elif isinstance(mapping_info, dict):
                    # Complex mapping (e.g., QKV concatenation)
                    if mapping_info['type'] == 'qkv_concat':
                        if self._apply_qkv_concatenation(param, mapping_info, param_name):
                            applied_count += 1
        
        success_rate = applied_count / total_params
        logger.info(f"Applied {applied_count}/{total_params} parameters ({success_rate:.1%})")
        
        if success_rate < 0.5:
            raise WeightLoadingError(f"Insufficient parameters loaded: {success_rate:.1%}")
        
        return applied_count, total_params
    
    def _create_parameter_mapping(self, model: nn.Module) -> Dict[str, Union[str, Dict]]:
        """Create comprehensive parameter mapping from DINOv3 to HuggingFace names."""
        dinov3_params = {name: param.shape for name, param in model.named_parameters()}
        mapping = {}
        
        # Basic parameter mappings
        basic_mappings = {
            # Patch embedding
            'patch_embed.proj.weight': 'embeddings.patch_embeddings.weight',
            'patch_embed.proj.bias': 'embeddings.patch_embeddings.bias',
            
            # Special tokens
            'cls_token': 'embeddings.cls_token',
            'storage_tokens': 'embeddings.register_tokens',
            'mask_token': 'embeddings.mask_token',
            
            # Final layer norm
            'norm.weight': 'norm.weight', 
            'norm.bias': 'norm.bias',
        }
        
        for dinov3_name, hf_name in basic_mappings.items():
            if dinov3_name in dinov3_params:
                mapping[dinov3_name] = hf_name
        
        # Transformer blocks - systematic mapping
        for param_name in dinov3_params:
            if param_name.startswith('blocks.'):
                block_mapping = self._map_transformer_block_param(param_name)
                if block_mapping:
                    mapping[param_name] = block_mapping
        
        logger.debug(f"Created mapping for {len(mapping)}/{len(dinov3_params)} parameters")
        return mapping
    
    def _map_transformer_block_param(self, param_name: str) -> Optional[Union[str, Dict]]:
        """Map individual transformer block parameters."""
        parts = param_name.split('.')
        if len(parts) < 3:
            return None
            
        block_idx = parts[1]
        component_path = '.'.join(parts[2:])
        
        # Layer normalization
        if component_path in ['norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']:
            return f'layer.{block_idx}.{component_path}'
        
        # Attention components
        if component_path == 'attn.proj.weight':
            return f'layer.{block_idx}.attention.o_proj.weight'
        elif component_path == 'attn.proj.bias':
            return f'layer.{block_idx}.attention.o_proj.bias'
        elif component_path in ['attn.qkv.weight', 'attn.qkv.bias']:
            # QKV requires concatenation
            weight_or_bias = component_path.split('.')[-1]
            return {
                'type': 'qkv_concat',
                'components': [
                    f'layer.{block_idx}.attention.q_proj.{weight_or_bias}',
                    f'layer.{block_idx}.attention.k_proj.{weight_or_bias}',
                    f'layer.{block_idx}.attention.v_proj.{weight_or_bias}',
                ]
            }
        
        # MLP components
        if component_path.startswith('mlp.'):
            mlp_component = component_path.replace('mlp.', '')
            # Map fc1->up_proj, fc2->down_proj for HF compatibility
            if mlp_component.startswith('fc1.'):
                hf_component = mlp_component.replace('fc1.', 'up_proj.')
            elif mlp_component.startswith('fc2.'):
                hf_component = mlp_component.replace('fc2.', 'down_proj.')
            else:
                hf_component = mlp_component
            
            return f'layer.{block_idx}.mlp.{hf_component}'
        
        return None
    
    def _apply_direct_mapping(self, param: nn.Parameter, hf_name: str, dinov3_name: str) -> bool:
        """Apply direct 1:1 parameter mapping with shape compatibility."""
        if hf_name not in self._weights_cache:
            logger.debug(f"HF parameter {hf_name} not found for {dinov3_name}")
            return False
        
        hf_tensor = self._weights_cache[hf_name]
        
        # Handle shape mismatches
        if param.shape != hf_tensor.shape:
            # Special case: mask_token reshape [1,1,D] -> [1,D] 
            if (dinov3_name == 'mask_token' and 
                len(hf_tensor.shape) == 3 and len(param.shape) == 2):
                hf_tensor = hf_tensor.squeeze(1)
                logger.debug(f"Reshaped {dinov3_name}: {self._weights_cache[hf_name].shape} -> {hf_tensor.shape}")
            else:
                logger.warning(f"Shape mismatch {dinov3_name}: {param.shape} vs {hf_tensor.shape}")
                return False
        
        param.data.copy_(hf_tensor)
        return True
    
    def _apply_qkv_concatenation(self, param: nn.Parameter, mapping_info: Dict, dinov3_name: str) -> bool:
        """Apply QKV concatenation with missing bias handling."""
        component_names = mapping_info['components']
        tensors = []
        missing_components = []
        
        # Collect available tensors, create zeros for missing ones
        for comp_name in component_names:
            if comp_name in self._weights_cache:
                tensors.append(self._weights_cache[comp_name])
            else:
                missing_components.append(comp_name)
                # Create appropriate zero tensor
                if tensors:
                    # Use shape of existing tensor
                    zero_tensor = torch.zeros_like(tensors[0])
                else:
                    # Infer shape from parameter (divide by 3 for QKV)
                    if param.dim() == 1:  # bias
                        feat_dim = param.shape[0] // 3
                        zero_tensor = torch.zeros(feat_dim, dtype=param.dtype)
                    else:  # weight
                        out_dim = param.shape[0] // 3
                        in_dim = param.shape[1]
                        zero_tensor = torch.zeros(out_dim, in_dim, dtype=param.dtype)
                
                tensors.append(zero_tensor)
        
        # Concatenate Q, K, V tensors
        concatenated = torch.cat(tensors, dim=0)
        
        if param.shape != concatenated.shape:
            logger.warning(f"QKV concat shape mismatch {dinov3_name}: {param.shape} vs {concatenated.shape}")
            return False
        
        param.data.copy_(concatenated)
        
        if missing_components:
            logger.debug(f"QKV concatenation {dinov3_name} (missing: {[c.split('.')[-2] for c in missing_components]})")
        
        return True


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
            raise ValueError(f"Unknown model '{model_name}'. Available: {available_models}")
        
        self.model_name = model_name
        self.model_config = MODEL_REGISTRY[model_name]
        self.stride = stride
        self.dtype = dtype
        self.track_grad = track_grad
        self.force_strategy = force_strategy
        
        # Load backbone model
        self.backbone, self.loading_strategy = self._load_backbone_with_strategy(hf_token)
        self.backbone.eval()
        
        # Apply dtype conversion
        if dtype != torch.float32:
            self.backbone = self.backbone.to(dtype)
        
        # Model properties from official implementation
        self.patch_size = getattr(self.backbone, 'patch_size', 16)
        self.feat_dim = self.model_config.feat_dim
        
        self._log_initialization_success()
    
    def _load_backbone_with_strategy(self, hf_token: Optional[str]) -> Tuple[nn.Module, LoadingStrategy]:
        """Load backbone using the best available strategy with smart caching."""
        # Skip original hub by default since it consistently fails with 403
        strategies = [LoadingStrategy.HUGGINGFACE, LoadingStrategy.RANDOM_WEIGHTS]
        
        # Only try original hub if explicitly forced or if we haven't seen it fail
        if self.force_strategy == LoadingStrategy.ORIGINAL_HUB:
            strategies = [LoadingStrategy.ORIGINAL_HUB, LoadingStrategy.HUGGINGFACE, LoadingStrategy.RANDOM_WEIGHTS]
        
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
        
        raise DINOv3Exception(f"All loading strategies failed. Last error: {last_error}")
    
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
        logger.warning("Loading with random weights - model performance will be limited!")
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
            if hasattr(module, 'bias_mask') and module.bias_mask is not None:
                if torch.isnan(module.bias_mask).any():
                    # Create proper mask: enable Q and V bias, disable K bias
                    bias_len = module.bias_mask.shape[0]
                    third_len = bias_len // 3
                    
                    # Mask pattern: [Q: 1.0, K: 0.0, V: 1.0]
                    proper_mask = torch.ones_like(module.bias_mask)
                    proper_mask[third_len:2*third_len] = 0.0  # Disable K bias
                    
                    module.bias_mask.data.copy_(proper_mask)
                    fixed_count += 1
        
        if fixed_count > 0:
            logger.info(f"✅ Fixed {fixed_count} LinearKMaskedBias layers")
    
    def _log_initialization_success(self):
        """Log successful initialization details."""
        strategy_msg = {
            LoadingStrategy.ORIGINAL_HUB: "Original DINOv3 hub",
            LoadingStrategy.HUGGINGFACE: f"HuggingFace ({self.model_config.hf_model})",
            LoadingStrategy.RANDOM_WEIGHTS: "Random weights"
        }
        
        logger.info(f"🎉 DINOv3Adapter initialized successfully!")
        logger.info(f"   Model: {self.model_name} ({self.model_config.params_count})")
        logger.info(f"   Strategy: {strategy_msg[self.loading_strategy]}")
        logger.info(f"   Features: {self.feat_dim}D, Patch size: {self.patch_size}")
        logger.info(f"   Description: {self.model_config.description}")
    
    def forward_sequential(
        self, 
        x: torch.Tensor, 
        attn_choice: AttentionOptions = "none"
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features using DINOv3 backbone.
        
        Based on official adapter but simplified for tree segmentation.
        Uses the same interface as the original HighResDV2 adapter.
        """
        x.requires_grad = self.track_grad
        
        if self.dtype != torch.float32:
            x = x.to(self.dtype)
        
        # Ensure batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        original_shape = x.shape[-2:]
        
        # Extract features using DINOv3
        with torch.no_grad():
            # Get features from backbone
            features = self.backbone.forward_features(x)
            
            # Handle different output formats (from official adapter insights)
            patch_features = self._extract_patch_features(features)
            
            # Reshape to spatial format
            patch_features = self._reshape_to_spatial(patch_features, original_shape)
        
        return {
            "x_norm_patchtokens": patch_features,
            "x_patchattn": patch_features if attn_choice != "none" else None
        }
    
    def _extract_patch_features(self, features: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """Extract patch tokens from backbone output."""
        if isinstance(features, dict):
            # Dictionary output - look for patch tokens
            if 'x_norm_patchtokens' in features:
                return features['x_norm_patchtokens']
            elif 'x_prenorm' in features:
                # Remove CLS token (first token)
                return features['x_prenorm'][:, 1:, :] 
            else:
                # Fallback to first tensor value
                tensor_features = list(features.values())[0]
                if tensor_features.dim() == 3 and tensor_features.shape[1] > 1:
                    return tensor_features[:, 1:, :]  # Remove CLS
                return tensor_features
        else:
            # Tensor output - remove CLS token
            return features[:, 1:, :] if features.shape[1] > 1 else features
    
    def _reshape_to_spatial(self, patch_features: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
        """Reshape linear patch features to spatial grid."""
        img_h, img_w = original_shape
        h_patches = img_h // self.patch_size
        w_patches = img_w // self.patch_size
        
        # Reshape from (B, N, D) to (H, W, D)
        batch_size = patch_features.shape[0]
        feat_dim = patch_features.shape[-1]
        
        spatial_features = patch_features.view(batch_size, h_patches, w_patches, feat_dim)
        return spatial_features.squeeze(0)  # Remove batch dimension
    
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
    **kwargs
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
    model = DINOv3Adapter(
        model_name=model_name,
        stride=stride,
        dtype=dtype,
        **kwargs
    )
    
    if device is not None:
        model = model.to(device)
    
    return model


# Utility functions for model discovery and information
def list_available_models() -> Dict[str, str]:
    """List all available DINOv3 models with descriptions."""
    return {name: f"{config.description} ({config.params_count})" 
            for name, config in MODEL_REGISTRY.items()}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODEL_REGISTRY[model_name]
    return {
        'name': model_name,
        'feature_dim': config.feat_dim,
        'parameters': config.params_count,
        'description': config.description,
        'hf_model': config.hf_model,
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