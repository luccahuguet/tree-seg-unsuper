"""Weight-loading utilities for the DINOv3 adapter."""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WeightLoadingError(Exception):
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
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self._weights_cache: Optional[Dict[str, torch.Tensor]] = None

    def load_weights(self) -> bool:
        """Load HuggingFace safetensors weights with efficient caching."""
        try:
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open

            if not self.hf_token:
                logger.warning("No HF_TOKEN found - some models may be inaccessible")

            try:
                weights_path = hf_hub_download(
                    repo_id=self.hf_model_id,
                    filename="model.safetensors",
                    token=self.hf_token,
                    local_files_only=True,
                )
                logger.info(f"✅ Using cached weights: {self.hf_model_id}")
            except Exception:
                logger.info(f"⬇️ Downloading model weights: {self.hf_model_id}")
                weights_path = hf_hub_download(
                    repo_id=self.hf_model_id,
                    filename="model.safetensors",
                    token=self.hf_token,
                )

            self._weights_cache = {}
            with safe_open(weights_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self._weights_cache[key] = f.get_tensor(key)

            logger.info(f"Successfully loaded {len(self._weights_cache)} parameters")
            return True

        except Exception as exc:
            logger.error(f"HuggingFace weight loading failed: {exc}")
            return False

    def apply_weights_to_model(self, model: nn.Module) -> Tuple[int, int]:
        """Apply HuggingFace weights to DINOv3 model with optimized mapping."""
        if not self._weights_cache:
            raise WeightLoadingError("Weights not loaded. Call load_weights() first.")

        parameter_map = self._create_parameter_mapping(model)
        applied_count = 0
        total_params = len(list(model.parameters()))

        for param_name, param in model.named_parameters():
            if param_name in parameter_map:
                mapping_info = parameter_map[param_name]

                if isinstance(mapping_info, str):
                    if self._apply_direct_mapping(param, mapping_info, param_name):
                        applied_count += 1

                elif isinstance(mapping_info, dict):
                    if mapping_info["type"] == "qkv_concat":
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
        mapping: Dict[str, Union[str, Dict]] = {}

        basic_mappings = {
            "patch_embed.proj.weight": "embeddings.patch_embeddings.weight",
            "patch_embed.proj.bias": "embeddings.patch_embeddings.bias",
            "cls_token": "embeddings.cls_token",
            "storage_tokens": "embeddings.register_tokens",
            "mask_token": "embeddings.mask_token",
            "norm.weight": "norm.weight",
            "norm.bias": "norm.bias",
        }

        for dinov3_name, hf_name in basic_mappings.items():
            if dinov3_name in dinov3_params:
                mapping[dinov3_name] = hf_name

        for param_name in dinov3_params:
            if param_name.startswith("blocks."):
                block_mapping = self._map_transformer_block_param(param_name)
                if block_mapping:
                    mapping[param_name] = block_mapping

        logger.debug(f"Created mapping for {len(mapping)}/{len(dinov3_params)} parameters")
        return mapping

    def _map_transformer_block_param(self, param_name: str) -> Optional[Union[str, Dict]]:
        """Map individual transformer block parameters."""
        parts = param_name.split(".")
        if len(parts) < 3:
            return None

        block_idx = parts[1]
        component_path = ".".join(parts[2:])

        if component_path in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
            return f"layer.{block_idx}.{component_path}"

        if component_path == "attn.proj.weight":
            return f"layer.{block_idx}.attention.o_proj.weight"
        if component_path == "attn.proj.bias":
            return f"layer.{block_idx}.attention.o_proj.bias"
        if component_path in ["attn.qkv.weight", "attn.qkv.bias"]:
            weight_or_bias = component_path.split(".")[-1]
            return {
                "type": "qkv_concat",
                "components": [
                    f"layer.{block_idx}.attention.q_proj.{weight_or_bias}",
                    f"layer.{block_idx}.attention.k_proj.{weight_or_bias}",
                    f"layer.{block_idx}.attention.v_proj.{weight_or_bias}",
                ],
            }

        if component_path.startswith("mlp."):
            mlp_component = component_path.replace("mlp.", "")
            if mlp_component.startswith("fc1."):
                hf_component = mlp_component.replace("fc1.", "up_proj.")
            elif mlp_component.startswith("fc2."):
                hf_component = mlp_component.replace("fc2.", "down_proj.")
            else:
                hf_component = mlp_component

            return f"layer.{block_idx}.mlp.{hf_component}"

        return None

    def _apply_direct_mapping(self, param: nn.Parameter, hf_name: str, dinov3_name: str) -> bool:
        """Apply direct 1:1 parameter mapping with shape compatibility."""
        if hf_name not in self._weights_cache:
            logger.debug(f"HF parameter {hf_name} not found for {dinov3_name}")
            return False

        hf_tensor = self._weights_cache[hf_name]

        if param.shape != hf_tensor.shape:
            if dinov3_name == "mask_token" and len(hf_tensor.shape) == 3 and len(param.shape) == 2:
                hf_tensor = hf_tensor.squeeze(1)
                logger.debug(f"Reshaped {dinov3_name}: {self._weights_cache[hf_name].shape} -> {hf_tensor.shape}")
            else:
                logger.warning(f"Shape mismatch {dinov3_name}: {param.shape} vs {hf_tensor.shape}")
                return False

        param.data.copy_(hf_tensor)
        return True

    def _apply_qkv_concatenation(self, param: nn.Parameter, mapping_info: Dict, dinov3_name: str) -> bool:
        """Apply QKV concatenation with missing bias handling."""
        component_names = mapping_info["components"]
        tensors = []
        missing_components = []

        for comp_name in component_names:
            if comp_name in self._weights_cache:
                tensors.append(self._weights_cache[comp_name])
            else:
                missing_components.append(comp_name)
                if tensors:
                    zero_tensor = torch.zeros_like(tensors[0])
                else:
                    if param.dim() == 1:
                        feat_dim = param.shape[0] // 3
                        zero_tensor = torch.zeros(feat_dim, dtype=param.dtype)
                    else:
                        out_dim = param.shape[0] // 3
                        in_dim = param.shape[1]
                        zero_tensor = torch.zeros(out_dim, in_dim, dtype=param.dtype)

                tensors.append(zero_tensor)

        concatenated = torch.cat(tensors, dim=0)

        if param.shape != concatenated.shape:
            logger.warning(f"QKV concat shape mismatch {dinov3_name}: {param.shape} vs {concatenated.shape}")
            return False

        param.data.copy_(concatenated)

        if missing_components:
            logger.debug(f"QKV concatenation {dinov3_name} (missing: {[c.split('.')[-2] for c in missing_components]})")

        return True
