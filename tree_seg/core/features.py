"""Feature extraction utilities (tiling, pyramid, attention)."""
import cv2
import numpy as np
import torch
from PIL import Image

from tree_seg.models.tiling import TileManager, TileConfig


def extract_features(
    image_np: np.ndarray,
    model,
    preprocess,
    stride: int,
    device,
    use_attention_features: bool,
    use_multi_layer: bool,
    layer_indices: tuple,
    feature_aggregation: str,
    use_pyramid: bool,
    pyramid_scales: tuple,
    pyramid_aggregation: str,
    verbose: bool,
) -> tuple[np.ndarray, int, int]:
    """Extract patch (and optional attention) features with optional pyramid."""
    h, w = image_np.shape[:2]

    if use_pyramid:
        pyramid_feature_maps = []

        for scale_idx, scale in enumerate(pyramid_scales):
            if verbose:
                print(f"   Processing scale {scale}× ({scale_idx+1}/{len(pyramid_scales)})...")

            scaled_h = int(h * scale)
            scaled_w = int(w * scale)
            scaled_image_np = cv2.resize(image_np, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            image = Image.fromarray(scaled_image_np)
            image_tensor = preprocess(image).to(device)

            if verbose and scale_idx == 0:
                print(f"   Preprocessed tensor shape: {image_tensor.shape}")

            with torch.no_grad():
                attn_choice = "o" if use_attention_features else "none"
                features_out = model.forward_sequential(
                    image_tensor,
                    attn_choice=attn_choice,
                    use_multi_layer=use_multi_layer,
                    layer_indices=layer_indices,
                    feature_aggregation=feature_aggregation,
                )

                if isinstance(features_out, dict):
                    patch_features = features_out["x_norm_patchtokens"]
                    attn_features = features_out.get("x_patchattn", None) if use_attention_features else None
                else:
                    if hasattr(features_out, "dim") and features_out.dim() == 4:
                        features = features_out.mean(dim=0)
                    else:
                        features = features_out
                    H_scale = W_scale = 518 // stride
                    features = features.unsqueeze(0)
                    features = torch.nn.functional.interpolate(
                        features, size=(H_scale, W_scale), mode="bilinear", align_corners=False
                    ).squeeze(0)
                    features = features.permute(1, 2, 0)
                    patch_features = features
                    attn_features = None

                if patch_features.dim() == 2:
                    n_patches = patch_features.shape[0]
                    H_scale = W_scale = int(np.sqrt(n_patches))
                    patch_features = patch_features.view(H_scale, W_scale, -1)
                    if attn_features is not None:
                        attn_features = attn_features.view(H_scale, W_scale, -1)

                if attn_features is not None and use_attention_features:
                    combined_features = torch.cat([patch_features, attn_features], dim=-1)
                else:
                    combined_features = patch_features

                pyramid_feature_maps.append(combined_features.cpu().numpy())

                if verbose:
                    print(f"   Scale {scale}× features: {combined_features.shape}")

        reference_idx = pyramid_scales.index(1.0) if 1.0 in pyramid_scales else len(pyramid_scales) // 2
        ref_shape = pyramid_feature_maps[reference_idx].shape[:2]

        if verbose:
            print(f"   Resizing all feature maps to reference size: {ref_shape}")

        resized_features = []
        for feat_map in pyramid_feature_maps:
            if feat_map.shape[:2] != ref_shape:
                feat_resized = cv2.resize(feat_map, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_LINEAR)
                resized_features.append(feat_resized)
            else:
                resized_features.append(feat_map)

                if pyramid_aggregation == "concat":
                    features_np = np.concatenate(resized_features, axis=-1)
                    if verbose:
                        print(f"   Concatenated pyramid features: {features_np.shape}")
        else:
            features_np = np.mean(resized_features, axis=0)
            if verbose:
                print(f"   Averaged pyramid features: {features_np.shape}")

        H, W = features_np.shape[:2]
    else:
        image = Image.fromarray(image_np)
        image_tensor = preprocess(image).to(device)

        if verbose:
            print(f"Preprocessed tensor shape: {image_tensor.shape}")

        with torch.no_grad():
            attn_choice = "o" if use_attention_features else "none"
            features_out = model.forward_sequential(
                image_tensor,
                attn_choice=attn_choice,
                use_multi_layer=use_multi_layer,
                layer_indices=layer_indices,
                feature_aggregation=feature_aggregation,
            )

            if isinstance(features_out, dict):
                patch_features = features_out["x_norm_patchtokens"]
                attn_features = features_out.get("x_patchattn", None) if use_attention_features else None
                if verbose:
                    print(f"patch_features shape: {patch_features.shape}")
                    if attn_features is not None:
                        print(f"attn_features shape: {attn_features.shape}")
            else:
                if verbose:
                    print(f"features_out shape: {getattr(features_out, 'shape', 'N/A')}")
                if hasattr(features_out, "dim") and features_out.dim() == 4:
                    features = features_out.mean(dim=0)
                else:
                    features = features_out
                H = W = 518 // stride
                features = features.unsqueeze(0)
                features = torch.nn.functional.interpolate(
                    features, size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(0)
                features = features.permute(1, 2, 0)
                patch_features = features
                attn_features = None

        if patch_features.dim() == 2:
            n_patches = patch_features.shape[0]
            H = W = int(np.sqrt(n_patches))
            patch_features = patch_features.view(H, W, -1)
            if attn_features is not None:
                attn_features = attn_features.view(H, W, -1)
        else:
            H, W = patch_features.shape[:2]

        if attn_features is not None and use_attention_features:
            features_np = np.concatenate([patch_features.cpu().numpy(), attn_features.cpu().numpy()], axis=-1)
            if verbose:
                print(f"Combined features shape: {features_np.shape}")
        else:
            features_np = patch_features.cpu().numpy()
            if verbose:
                print(f"Patch-only features shape: {features_np.shape}")

    return features_np, H, W


def extract_tiled_features(
    image_np: np.ndarray,
    model,
    preprocess,
    stride: int,
    device,
    use_attention_features: bool,
    use_multi_layer: bool,
    layer_indices: tuple,
    feature_aggregation: str,
    tile_size: int,
    tile_overlap: int,
    tile_threshold: int,
    verbose: bool,
) -> tuple[np.ndarray, int, int]:
    """Extract features with tiling."""
    h, w = image_np.shape[:2]
    tile_config = TileConfig(
        tile_size=tile_size,
        overlap=tile_overlap,
        auto_tile_threshold=tile_threshold,
        blend_mode="linear",
    )
    tile_manager = TileManager(tile_config)
    needs_tiling = tile_manager.should_tile(h, w)

    if not needs_tiling:
        return None, None, None  # Signal to caller to use non-tiling path

    tiles = tile_manager.extract_tiles(image_np)

    tile_features_list = []
    tile_coords_list = []

    for i, tile_info in enumerate(tiles):
        if verbose and (i % 10 == 0 or i == len(tiles) - 1):
            print(f"   Processing tile {i+1}/{len(tiles)}...")

        tile_pil = Image.fromarray(tile_info.tile_array)
        tile_tensor = preprocess(tile_pil).to(device)

        with torch.no_grad():
            attn_choice = "o" if use_attention_features else "none"
            features_out = model.forward_sequential(
                tile_tensor,
                attn_choice=attn_choice,
                use_multi_layer=use_multi_layer,
                layer_indices=layer_indices,
                feature_aggregation=feature_aggregation,
            )
            tile_patch_features = features_out["x_norm_patchtokens"]
            tile_attn_features = features_out.get("x_patchattn", None) if use_attention_features else None

        if tile_patch_features.dim() == 2:
            n_patches = tile_patch_features.shape[0]
            tile_H = tile_W = int(np.sqrt(n_patches))
            tile_patch_features = tile_patch_features.view(tile_H, tile_W, -1)
            if tile_attn_features is not None:
                tile_attn_features = tile_attn_features.view(tile_H, tile_W, -1)

        if tile_attn_features is not None and use_attention_features:
            tile_features_combined = np.concatenate(
                [tile_patch_features.cpu().numpy(), tile_attn_features.cpu().numpy()],
                axis=-1,
            )
        else:
            tile_features_combined = tile_patch_features.cpu().numpy()

        tile_features_list.append(tile_features_combined)
        tile_coords_list.append((tile_info.x_start, tile_info.y_start, tile_info.x_end, tile_info.y_end))

    # Stitch features with weighted blending
    if verbose:
        print("   Stitching tile features...")

    first_tile_feat_h, first_tile_feat_w = tile_features_list[0].shape[:2]
    feat_scale = tile_size / first_tile_feat_h

    output_h = int(h / feat_scale)
    output_w = int(w / feat_scale)

    features_np = tile_manager.stitch_features(
        tile_features_list,
        tile_coords_list,
        output_shape=(output_h, output_w),
    )

    H, W = features_np.shape[:2]

    if verbose:
        print(f"   Stitched feature map: {H}×{W}×{features_np.shape[-1]}")

    return features_np, H, W
