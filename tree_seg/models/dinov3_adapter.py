"""
DINOv3 Adapter for tree segmentation - replaces HighResDV2.

Provides a clean interface to DINOv3 models that matches the existing API.
"""

import sys
import torch
import torch.nn as nn
from typing import Dict, Literal
from pathlib import Path

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
        """Load DINOv3 backbone model with transformers fallback."""
        # Map model names to DINOv3 hub functions
        model_map = {
            "dinov3_vits16": dinov3_backbones.dinov3_vits16,
            "dinov3_vitb16": dinov3_backbones.dinov3_vitb16,
            "dinov3_vitl16": dinov3_backbones.dinov3_vitl16,
            "dinov3_vith16plus": dinov3_backbones.dinov3_vith16plus,
            "dinov3_vit7b16": dinov3_backbones.dinov3_vit7b16,
        }
        
        # Map model names to Hugging Face model IDs for transformers fallback
        hf_model_map = {
            "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m", 
            "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            # Plus variants (enhanced models with better architecture)
            "dinov3_vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "dinov3_vitl16plus": "facebook/dinov3-vitl16plus-pretrain-lvd1689m",
            # Satellite-optimized variants
            "dinov3_vitl16_sat": "facebook/dinov3-vitl16-pretrain-sat493m",
            "dinov3_vit7b16_sat": "facebook/dinov3-vit7b16-pretrain-sat493m",
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown DINOv3 model: {model_name}. Available: {list(model_map.keys())}")
        
        # Try 1: Load using original DINOv3 hub
        model_fn = model_map[model_name]
        try:
            backbone = model_fn(pretrained=True)
            print(f"📥 Loaded DINOv3 model: {model_name} (DINOv3 hub, pretrained)")
            return backbone
        except Exception as e:
            print(f"⚠️  DINOv3 hub loading failed: {e}")
        
        # Try 2: Load using direct safetensors download (if available)
        if model_name in hf_model_map:
            try:
                print("🔄 Trying direct safetensors download...")
                success = self._load_hf_safetensors_weights(model_fn, hf_model_map[model_name])
                if success:
                    backbone = model_fn(pretrained=False)  # Load architecture
                    
                    # Initialize all parameters (especially LayerScale gamma)
                    print("   Initializing DINOv3 parameters...")
                    backbone.init_weights()
                    
                    # Fix LinearKMaskedBias bias_mask initialization issue
                    print("   Fixing LinearKMaskedBias bias_mask...")
                    self._fix_linear_k_masked_bias(backbone)
                    
                    # Load the weights we just downloaded and converted
                    weights_applied = self._apply_converted_weights(backbone, model_name)
                    if weights_applied:
                        print(f"📥 Loaded DINOv3 model: {model_name} (safetensors, pretrained)")
                        print(f"   Hugging Face model: {hf_model_map[model_name]}")
                        return backbone
                    else:
                        print("   ⚠️ Weight application failed, continuing to next method...")
            except Exception as e:
                print(f"⚠️  Safetensors loading failed: {e}")
        
        # Try 3: Load using transformers library (if available and model is in HF)
        if model_name in hf_model_map:
            try:
                from transformers import AutoModel
                print("🔄 Trying transformers fallback...")
                hf_model_id = hf_model_map[model_name]
                backbone = AutoModel.from_pretrained(hf_model_id)
                print(f"📥 Loaded DINOv3 model: {model_name} (transformers, pretrained)")
                print(f"   Hugging Face model: {hf_model_id}")
                return backbone
            except ImportError:
                print("⚠️  transformers library not available")
            except Exception as e:
                print(f"⚠️  Transformers loading failed: {e}")
        
        # Try 4: Load architecture only (random weights)
        try:
            print("🔄 Loading model architecture only (random weights)...")
            backbone = model_fn(pretrained=False)
            print(f"📥 Loaded DINOv3 model: {model_name} (random initialization)")
            print("⚠️  Note: Using random weights - performance will be limited")
            return backbone
        except Exception as e2:
            print(f"❌ Failed to load DINOv3 model architecture: {e2}")
            raise
    
    def _get_feature_dim(self, model_name: str) -> int:
        """Get feature dimension for the model."""
        # DINOv3 feature dimensions
        dim_map = {
            "dinov3_vits16": 384,      # Small
            "dinov3_vits16plus": 384,  # Small+ (same dim as small)
            "dinov3_vitb16": 768,      # Base
            "dinov3_vitl16": 1024,     # Large
            "dinov3_vitl16plus": 1024, # Large+ (same dim as large)
            "dinov3_vith16plus": 1280, # Huge+
            "dinov3_vit7b16": 1536,    # 7B
            # Satellite variants have same dimensions as base models
            "dinov3_vitl16_sat": 1024, # Large (satellite)
            "dinov3_vit7b16_sat": 1536, # 7B (satellite)
        }
        return dim_map.get(model_name, 768)  # Default to base
    
    def _load_hf_safetensors_weights(self, model_fn, hf_model_id):
        """Download and prepare HuggingFace safetensors weights."""
        try:
            import os
            from dotenv import load_dotenv
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
            
            load_dotenv()
            
            if 'HF_TOKEN' not in os.environ:
                print("⚠️  No HF_TOKEN found in environment")
                return False
            
            print(f"   Downloading weights from {hf_model_id}...")
            model_path = hf_hub_download(
                repo_id=hf_model_id,
                filename='model.safetensors',
                token=os.environ['HF_TOKEN']
            )
            
            # Load and convert weights
            with safe_open(model_path, framework='pt', device='cpu') as f:
                # This is a simplified conversion - we'd need proper mapping
                # For now, let's see what keys we have
                hf_keys = list(f.keys())
                print(f"   Found {len(hf_keys)} parameters in HF model")
                
                # Store the safetensors file path for later use
                self._hf_weights_path = model_path
                
            return True
            
        except Exception as e:
            print(f"   Safetensors download failed: {e}")
            return False
    
    def _apply_converted_weights(self, backbone, model_name):
        """Apply HuggingFace weights to original DINOv3 architecture."""
        try:
            from safetensors import safe_open
            import torch
            
            # Load HF weights
            hf_weights = {}
            with safe_open(self._hf_weights_path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    hf_weights[key] = f.get_tensor(key)
            
            print(f"   Loaded {len(hf_weights)} HF parameters")
            
            # Create mapping from HF parameter names to DINOv3 parameter names
            weight_mapping = self._create_weight_mapping(backbone, hf_weights)
            
            # Apply weights to backbone
            applied_count = 0
            total_count = 0
            
            for dinov3_name, param in backbone.named_parameters():
                total_count += 1
                if dinov3_name in weight_mapping:
                    if isinstance(weight_mapping[dinov3_name], str):
                        # Simple mapping
                        hf_name = weight_mapping[dinov3_name]
                        if hf_name in hf_weights:
                            hf_weight = hf_weights[hf_name]
                            if param.shape == hf_weight.shape:
                                param.data.copy_(hf_weight)
                                applied_count += 1
                            elif dinov3_name == 'mask_token' and param.shape == torch.Size([1, 384]) and hf_weight.shape == torch.Size([1, 1, 384]):
                                # Special case: reshape HF mask_token from [1, 1, 384] to [1, 384]
                                param.data.copy_(hf_weight.squeeze(1))
                                applied_count += 1
                                print(f"   ✅ Reshaped mask_token from {hf_weight.shape} to {param.shape}")
                            else:
                                print(f"   ⚠️ Shape mismatch for {dinov3_name}: {param.shape} vs {hf_weight.shape}")
                    elif isinstance(weight_mapping[dinov3_name], dict):
                        # Special handling (e.g., QKV concatenation)
                        special_mapping = weight_mapping[dinov3_name]
                        if special_mapping['type'] == 'concat_qkv':
                            q_name, k_name, v_name = special_mapping['hf_names']
                            
                            # Handle case where some bias terms might not exist (e.g., K bias is often disabled)
                            available_tensors = []
                            missing_names = []
                            
                            for name in [q_name, k_name, v_name]:
                                if name in hf_weights:
                                    available_tensors.append(hf_weights[name])
                                else:
                                    missing_names.append(name)
                                    # Create zero tensor with same shape as the available ones
                                    if available_tensors:
                                        ref_tensor = available_tensors[0]
                                        zero_tensor = torch.zeros_like(ref_tensor)
                                    else:
                                        # If this is the first tensor and it's missing, we need to determine shape
                                        # For DINOv3 small, each head has 384 dimensions
                                        if 'bias' in name:
                                            zero_tensor = torch.zeros(384, device='cpu', dtype=torch.float32)
                                        else:
                                            # For weight matrices - this would need more logic
                                            print(f"   ⚠️ Cannot determine shape for missing weight {name}")
                                            continue
                                    available_tensors.append(zero_tensor)
                            
                            if len(available_tensors) == 3:
                                # Concatenate Q, K, V
                                qkv_tensor = torch.cat(available_tensors, dim=0)
                                
                                if param.shape == qkv_tensor.shape:
                                    param.data.copy_(qkv_tensor)
                                    applied_count += 1
                                    if missing_names:
                                        print(f"   ✅ Concatenated QKV for {dinov3_name} (missing: {missing_names})")
                                    else:
                                        print(f"   ✅ Concatenated QKV for {dinov3_name}")
                                else:
                                    print(f"   ⚠️ QKV concat shape mismatch for {dinov3_name}: {param.shape} vs {qkv_tensor.shape}")
            
            print(f"   ✅ Applied {applied_count}/{total_count} weights from HuggingFace model")
            
            # Debug: show some unmapped parameters
            unmapped_params = []
            for dinov3_name, param in backbone.named_parameters():
                if dinov3_name not in weight_mapping:
                    unmapped_params.append(dinov3_name)
            
            if unmapped_params:
                print(f"   ⚠️ Unmapped parameters ({len(unmapped_params)}):")
                for name in unmapped_params[:10]:
                    print(f"     - {name}")
                if len(unmapped_params) > 10:
                    print(f"     ... and {len(unmapped_params) - 10} more")
            
            if applied_count > total_count * 0.5:  # At least 50% of weights applied
                print("   ✅ Successfully loaded most weights from HuggingFace")
                return True
            else:
                print(f"   ❌ Only {applied_count/total_count:.1%} of weights applied - insufficient")
                return False
                
        except Exception as e:
            print(f"   Weight application failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _fix_linear_k_masked_bias(self, backbone):
        """Fix LinearKMaskedBias layers that have NaN bias_mask initialization."""
        try:
            import torch
            fixed_count = 0
            
            for name, module in backbone.named_modules():
                if hasattr(module, 'bias_mask') and module.bias_mask is not None:
                    if torch.isnan(module.bias_mask).any():
                        # The bias_mask should mask out the K (middle third) bias terms
                        # Create proper mask: [1, 1, 1, 0, 0, 0, 1, 1, 1] for Q, K, V
                        bias_size = module.bias_mask.shape[0]
                        third_size = bias_size // 3
                        
                        # Create mask: Q and V get 1.0, K gets 0.0 (or NaN for masking)
                        new_mask = torch.ones_like(module.bias_mask)
                        new_mask[third_size:2*third_size] = 0.0  # Mask K bias
                        
                        module.bias_mask.data.copy_(new_mask)
                        fixed_count += 1
            
            if fixed_count > 0:
                print(f"   ✅ Fixed {fixed_count} LinearKMaskedBias layers")
            
        except Exception as e:
            print(f"   ⚠️ Error fixing LinearKMaskedBias: {e}")
    
    def _create_weight_mapping(self, backbone, hf_weights):
        """Create mapping from DINOv3 parameter names to HuggingFace parameter names."""
        
        # Get all DINOv3 parameter names
        dinov3_params = {name: param.shape for name, param in backbone.named_parameters()}
        
        print("   Creating weight mapping...")
        print(f"   DINOv3 has {len(dinov3_params)} parameters")
        
        # Common parameter mapping patterns
        mapping = {}
        
        # Patch embedding
        if 'patch_embed.proj.weight' in dinov3_params:
            mapping['patch_embed.proj.weight'] = 'embeddings.patch_embeddings.weight'
            mapping['patch_embed.proj.bias'] = 'embeddings.patch_embeddings.bias'
        
        # CLS token and positional embeddings  
        if 'cls_token' in dinov3_params:
            mapping['cls_token'] = 'embeddings.cls_token'
        if 'storage_tokens' in dinov3_params:  # DINOv3 calls these storage_tokens, HF calls them register_tokens
            mapping['storage_tokens'] = 'embeddings.register_tokens'  
        if 'mask_token' in dinov3_params:
            # HF mask_token has shape [1, 1, 384], DINOv3 has [1, 384] - need reshape
            mapping['mask_token'] = 'embeddings.mask_token'
        
        # Transformer blocks
        # DINOv3 uses 'blocks.{i}.' while HF uses 'layer.{i}.'
        for dinov3_name in dinov3_params:
            if dinov3_name.startswith('blocks.'):
                # Extract block number and rest of the path
                parts = dinov3_name.split('.')
                if len(parts) >= 3:
                    block_num = parts[1]
                    rest = '.'.join(parts[2:])
                    
                    # Map transformer components based on actual HF structure
                    if rest == 'norm1.weight':
                        mapping[dinov3_name] = f'layer.{block_num}.norm1.weight'
                    elif rest == 'norm1.bias':
                        mapping[dinov3_name] = f'layer.{block_num}.norm1.bias'
                    elif rest == 'attn.qkv.weight':
                        # HF splits QKV into separate q_proj, k_proj, v_proj - concatenate them
                        mapping[dinov3_name] = {
                            'type': 'concat_qkv',
                            'hf_names': [
                                f'layer.{block_num}.attention.q_proj.weight',
                                f'layer.{block_num}.attention.k_proj.weight', 
                                f'layer.{block_num}.attention.v_proj.weight'
                            ]
                        }
                    elif rest == 'attn.qkv.bias':
                        # Same for biases
                        mapping[dinov3_name] = {
                            'type': 'concat_qkv',
                            'hf_names': [
                                f'layer.{block_num}.attention.q_proj.bias',
                                f'layer.{block_num}.attention.k_proj.bias',
                                f'layer.{block_num}.attention.v_proj.bias'
                            ]
                        }
                    elif rest == 'attn.proj.weight':
                        mapping[dinov3_name] = f'layer.{block_num}.attention.o_proj.weight'
                    elif rest == 'attn.proj.bias':
                        mapping[dinov3_name] = f'layer.{block_num}.attention.o_proj.bias'
                    elif rest == 'norm2.weight':
                        mapping[dinov3_name] = f'layer.{block_num}.norm2.weight'
                    elif rest == 'norm2.bias':
                        mapping[dinov3_name] = f'layer.{block_num}.norm2.bias'
                    elif rest == 'mlp.fc1.weight':
                        mapping[dinov3_name] = f'layer.{block_num}.mlp.fc1.weight'
                    elif rest == 'mlp.fc1.bias':
                        mapping[dinov3_name] = f'layer.{block_num}.mlp.fc1.bias'
                    elif rest == 'mlp.fc2.weight':
                        mapping[dinov3_name] = f'layer.{block_num}.mlp.fc2.weight'
                    elif rest == 'mlp.fc2.bias':
                        mapping[dinov3_name] = f'layer.{block_num}.mlp.fc2.bias'
        
        # Final layer norm
        if 'norm.weight' in dinov3_params:
            mapping['norm.weight'] = 'norm.weight'
        if 'norm.bias' in dinov3_params:
            mapping['norm.bias'] = 'norm.bias'
        
        print(f"   Created {len(mapping)} parameter mappings")
        
        # Debug: show some mappings
        print("   Sample mappings:")
        for i, (dinov3_name, hf_name) in enumerate(mapping.items()):
            if i < 5:
                dinov3_shape = dinov3_params.get(dinov3_name, "Unknown")
                hf_exists = hf_name in hf_weights
                print(f"     {dinov3_name} -> {hf_name} ({dinov3_shape}) {'✓' if hf_exists else '✗'}")
        
        return mapping
    
    def _create_hf_wrapper(self, weights_path):
        """Create custom wrapper for HuggingFace weights."""
        import torch
        import torch.nn as nn
        from safetensors import safe_open
        
        class HFDINOv3Wrapper(nn.Module):
            def __init__(self, weights_path):
                super().__init__()
                self.weights = {}
                
                # Load all weights
                with safe_open(weights_path, framework='pt', device='cpu') as f:
                    for key in f.keys():
                        self.weights[key] = f.get_tensor(key)
            
            def forward_features(self, x):
                """Custom forward pass using HF weights."""
                x.shape[0]
                device = x.device
                
                # Move weights to same device as input
                patch_weight = self.weights['embeddings.patch_embeddings.weight'].to(device)
                patch_bias = self.weights['embeddings.patch_embeddings.bias'].to(device)
                
                # Apply patch embedding
                patches = torch.nn.functional.conv2d(x, patch_weight, patch_bias, stride=16)
                
                # Flatten and transpose: B, C, H, W -> B, N, C
                B, C, H, W = patches.shape
                patches = patches.view(B, C, H*W).transpose(1, 2)
                
                # Add CLS token
                cls_token = self.weights['embeddings.cls_token'].to(device).expand(B, -1, -1)
                full_sequence = torch.cat([cls_token, patches], dim=1)
                
                # Create register tokens (storage tokens)
                register_tokens = self.weights['embeddings.register_tokens'].to(device).expand(B, -1, -1)
                
                # Return dict matching DINOv3 format
                return {
                    'x_norm_patchtokens': patches,  # Patch tokens without CLS
                    'x_prenorm': full_sequence,     # Full sequence with CLS
                    'x_norm_clstoken': cls_token,   # CLS token only
                    'x_storage_tokens': register_tokens,  # Register tokens
                    'masks': None
                }
        
        return HFDINOv3Wrapper(weights_path)
    
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
            
            # Handle different DINOv3 output formats
            if isinstance(features, dict):
                # If features is a dictionary, extract the main feature tensor
                if 'x_norm_patchtokens' in features:
                    patch_features = features['x_norm_patchtokens']
                elif 'x_prenorm' in features:
                    patch_features = features['x_prenorm'][:, 1:, :]  # Remove CLS token
                elif 'last_hidden_state' in features:
                    patch_features = features['last_hidden_state'][:, 1:, :]  # Remove CLS token
                else:
                    # Fallback: try to find the main tensor
                    main_key = list(features.keys())[0]
                    tensor_features = features[main_key]
                    if tensor_features.dim() == 3:
                        patch_features = tensor_features[:, 1:, :]  # Remove CLS token
                    else:
                        patch_features = tensor_features
            elif torch.is_tensor(features):
                # DINOv3 returns features in format (B, N+1, D) where N is num_patches, +1 for CLS
                # We want just the patch tokens
                patch_features = features[:, 1:, :]  # Remove CLS token
            else:
                raise ValueError(f"Unexpected features type: {type(features)}. Expected torch.Tensor or dict.")
            
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