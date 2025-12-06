"""Feature extraction helpers for the DINOv3 adapter."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def extract_backbone_features(
    backbone,
    x: torch.Tensor,
    *,
    track_grad: bool,
    dtype: torch.dtype,
    use_multi_layer: bool,
    layer_indices: tuple,
    feature_aggregation: str,
    patch_size: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Extract patch features (spatial) from a DINOv3 backbone."""
    x.requires_grad = track_grad

    if dtype != torch.float32:
        x = x.to(dtype)

    if x.dim() == 3:
        x = x.unsqueeze(0)

    original_shape = x.shape[-2:]

    with torch.no_grad():
        if use_multi_layer and hasattr(backbone, "get_intermediate_layers"):
            layers_to_extract = [idx - 1 for idx in layer_indices]
            intermediate_outputs = backbone.get_intermediate_layers(
                x,
                n=layers_to_extract,
                reshape=False,
                return_class_token=False,
                norm=True,
            )

            layer_features_list = []
            for layer_output in intermediate_outputs:
                layer_features_list.append(layer_output)

            patch_counts = [f.shape[1] for f in layer_features_list]
            if len(set(patch_counts)) > 1:
                raise ValueError(f"Inconsistent patch counts across layers: {patch_counts}")

            if feature_aggregation == "concat":
                patch_features = torch.cat(layer_features_list, dim=-1)
            elif feature_aggregation == "average":
                stacked = torch.stack(layer_features_list, dim=0)
                patch_features = stacked.mean(dim=0)
            elif feature_aggregation == "weighted":
                weights = torch.linspace(0.5, 1.0, len(layer_features_list), device=x.device)
                weights = weights / weights.sum()
                weighted = torch.stack([f * weights[i] for i, f in enumerate(layer_features_list)], dim=0)
                patch_features = weighted.sum(dim=0)
            else:
                raise ValueError(f"Unknown aggregation: {feature_aggregation}")
        else:
            features = backbone.forward_features(x)
            patch_features = _extract_patch_features(features)

        patch_features = _reshape_to_spatial(patch_features, original_shape, patch_size)

    return patch_features, {"x_norm_patchtokens": patch_features}


def _extract_patch_features(features: torch.Tensor | Dict) -> torch.Tensor:
    """Extract patch tokens from backbone output."""
    if isinstance(features, dict):
        if "x_norm_patchtokens" in features:
            return features["x_norm_patchtokens"]
        if "x_prenorm" in features:
            return features["x_prenorm"][:, 1:, :]
        tensor_features = list(features.values())[0]
        if tensor_features.dim() == 3 and tensor_features.shape[1] > 1:
            return tensor_features[:, 1:, :]
        return tensor_features

    return features[:, 1:, :] if features.shape[1] > 1 else features


def _reshape_to_spatial(
    patch_features: torch.Tensor,
    original_shape: Tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    """Reshape linear patch features to spatial grid."""
    img_h, img_w = original_shape
    h_patches = img_h // patch_size
    w_patches = img_w // patch_size

    batch_size = patch_features.shape[0]
    num_patches = patch_features.shape[1]
    feat_dim = patch_features.shape[-1]

    expected_patches = h_patches * w_patches
    if num_patches != expected_patches:
        import math

        side = int(math.sqrt(num_patches))
        if side * side == num_patches:
            h_patches = w_patches = side
        else:
            raise ValueError(
                f"Patch count mismatch: got {num_patches} patches, expected {expected_patches} ({h_patches}x{w_patches})"
            )

    spatial_features = patch_features.view(batch_size, h_patches, w_patches, feat_dim)
    return spatial_features.squeeze(0)
