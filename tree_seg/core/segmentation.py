"""
Core segmentation functionality for tree segmentation.
"""

import os
import time
import traceback
import numpy as np
import torch
import cv2
from PIL import Image
from .clustering import cluster_features
from .refine import refine_labels, apply_vegetation_filter
from .metrics import compile_metrics

from ..analysis.elbow_method import find_optimal_k_elbow, plot_elbow_analysis
from .features import extract_tiled_features, extract_features
from .kselect import maybe_run_pca, clean_features


def process_image(image_path, model, preprocess, n_clusters, stride, device,
                 auto_k=False, k_range=(3, 10), elbow_threshold=0.035, use_pca=False, pca_dim=None,
                 feature_upsample_factor: int = 1, refine: str | None = None,
                 refine_slic_compactness: float = 10.0, refine_slic_sigma: float = 1.0,
                 collect_metrics: bool = False, model_name=None, output_dir="data/output",
                 verbose: bool = True, pipeline: str = "v1_5",
                 apply_vegetation_filter: bool = False, exg_threshold: float = 0.10,
                 use_tiling: bool = True, tile_size: int = 2048, tile_overlap: int = 256,
                 tile_threshold: int = 2048, downsample_before_tiling: bool = False,
                 clustering_method: str = "kmeans",
                 use_multi_layer: bool = False, layer_indices: tuple = (3, 6, 9, 12),
                 feature_aggregation: str = "concat",
                 use_pyramid: bool = False, pyramid_scales: tuple = (0.5, 1.0, 2.0),
                 pyramid_aggregation: str = "concat",
                 use_soft_refine: bool = False, soft_refine_temperature: float = 1.0,
                 soft_refine_iterations: int = 5, soft_refine_spatial_alpha: float = 0.0,
                 use_attention_features: bool = True):
    """
    Process a single image for tree segmentation.

    Args:
        image_path: Path to the input image OR numpy array (H, W, 3)
        model: Initialized model
        preprocess: Preprocessing pipeline
        n_clusters: Number of clusters (if auto_k=False)
        stride: Model stride
        version: (deprecated) unused
        device: PyTorch device
        auto_k: Whether to use automatic K selection
        k_range: Range for K selection (min_k, max_k)
        elbow_threshold: Threshold for elbow method (as decimal, e.g., 0.035)
        apply_vegetation_filter: Whether to apply ExG-based vegetation filtering
        exg_threshold: ExG threshold for vegetation filtering (default: 0.10)
        verbose: Whether to print detailed processing information

    Returns:
        Tuple of (image_np, labels_resized) or (None, None) on error
    """
    try:
        # Handle both file paths and numpy arrays
        if isinstance(image_path, np.ndarray):
            # Direct numpy array input (for benchmarking)
            image_np = image_path
            image_path_str = "<numpy_array>"
        else:
            # File path input
            image_path_str = str(image_path)
            if verbose:
                print(f"\n--- Processing {image_path_str} ---")
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

        t0 = time.perf_counter()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        h, w = image_np.shape[:2]
        if verbose:
            print(f"Original image size: {w}x{h}")

        # Optional downsampling before tiling
        if downsample_before_tiling and (h > tile_threshold or w > tile_threshold):
            image_np = cv2.resize(image_np, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            h, w = image_np.shape[:2]
            if verbose:
                print(f"Downsampled to: {w}x{h}")

        # Pyramid features require tiling to be disabled
        if use_pyramid:
            if use_tiling and verbose:
                print("⚠️  Pyramid mode: disabling tiling for multi-scale feature extraction")
            use_tiling = False

        # Decide if tiling is needed
        from ..models.tiling import TileManager, TileConfig

        tile_config = TileConfig(
            tile_size=tile_size,
            overlap=tile_overlap,
            auto_tile_threshold=tile_threshold,
            blend_mode="linear"
        )
        tile_manager = TileManager(tile_config)
        needs_tiling = use_tiling and tile_manager.should_tile(h, w)

        t_pre_start = time.perf_counter()

        if needs_tiling and verbose:
            grid_info = tile_manager.get_grid_info(h, w)
            print("🔲 Using tile-based processing:")
            print(f"   Tile size: {tile_size}×{tile_size}, Overlap: {tile_overlap}px")
            print(f"   Grid: {grid_info['n_tiles_y']}×{grid_info['n_tiles_x']} = {grid_info['n_tiles']} tiles")
        elif verbose:
            if use_pyramid:
                print(f"🔺 Using pyramid processing at {len(pyramid_scales)} scales: {pyramid_scales}")
            else:
                print("Using full-image processing (no tiling)")

        t_pre_end = time.perf_counter()

        features_np = None
        H, W = None, None

        if needs_tiling:
            features_np, H, W = extract_tiled_features(
                image_np=image_np,
                model=model,
                preprocess=preprocess,
                stride=stride,
                use_attention_features=use_attention_features,
                use_multi_layer=use_multi_layer,
                layer_indices=layer_indices,
                feature_aggregation=feature_aggregation,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                tile_threshold=tile_threshold,
                verbose=verbose,
                device=device,
            )

        if features_np is None:
            features_np, H, W = extract_features(
                image_np=image_np,
                model=model,
                preprocess=preprocess,
                stride=stride,
                use_attention_features=use_attention_features,
                use_multi_layer=use_multi_layer,
                layer_indices=layer_indices,
                feature_aggregation=feature_aggregation,
                use_pyramid=use_pyramid,
                pyramid_scales=pyramid_scales,
                pyramid_aggregation=pyramid_aggregation,
                verbose=verbose,
                device=device,
            )
            needs_tiling = False

        # Optional upsampling of the feature grid for smoother segmentation
        if isinstance(feature_upsample_factor, int) and feature_upsample_factor > 1:
            up_h, up_w = H * feature_upsample_factor, W * feature_upsample_factor
            features_np = cv2.resize(features_np, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
            H, W = up_h, up_w
            if verbose:
                print(f"Upsampled features to: {H}x{W}")

        if use_pyramid and pyramid_aggregation == "concat" and features_np.shape[-1] > 1536:
            H_pyr, W_pyr, D_pyr = features_np.shape
            features_flat_pca = features_np.reshape(-1, D_pyr)
            from sklearn.decomposition import PCA

            pca = PCA(n_components=1536, random_state=42)
            features_reduced = pca.fit_transform(features_flat_pca)
            features_np = features_reduced.reshape(H_pyr, W_pyr, 1536)
            if verbose:
                explained_var = pca.explained_variance_ratio_.sum() * 100
                print(f"   PCA reduced {D_pyr}D → 1536D ({explained_var:.1f}% variance)")

        features_flat = features_np.reshape(-1, features_np.shape[-1])
        if verbose:
            print(f"Flattened features shape: {features_flat.shape}")

        features_flat = maybe_run_pca(features_flat, use_pca, pca_dim, verbose)

        t_features = time.perf_counter()

        # Automatic K selection using elbow method
        if auto_k:
            if verbose:
                print("\n--- Automatic K Selection using elbow method ---")
            optimal_k, k_scores = find_optimal_k_elbow(features_flat, k_range, elbow_threshold * 100)

            if verbose:
                print(f"Selected optimal K = {optimal_k}")

            # Save K selection analysis plot
            output_prefix = os.path.splitext(os.path.basename(image_path_str))[0]
            # Use png subdirectory to match OutputManager expectations
            png_output_dir = os.path.join(output_dir, "png")
            os.makedirs(png_output_dir, exist_ok=True)
            plot_elbow_analysis(k_scores, png_output_dir, output_prefix, elbow_threshold * 100,
                               model_name, "n/a", stride, optimal_k, auto_k, image_path_str)

            n_clusters = optimal_k
            k_requested = int(optimal_k)
        else:
            if verbose:
                print(f"Using fixed K = {n_clusters}")
            k_requested = int(n_clusters)

        features_flat = clean_features(features_flat, verbose)

        t_kselect = time.perf_counter()

        labels = cluster_features(
            features_flat=features_flat,
            method=clustering_method,
            n_clusters=n_clusters,
            H=H,
            W=W,
            verbose=verbose,
        )

        # V2: Optional soft EM refinement in feature space
        if use_soft_refine:
            if verbose:
                print(f"🔬 Refining with soft EM (τ={soft_refine_temperature}, iter={soft_refine_iterations}, α={soft_refine_spatial_alpha})...")
            from tree_seg.clustering.head_refine import soft_em_refine

            labels = soft_em_refine(
                features=features_flat,
                initial_labels=labels,
                n_clusters=n_clusters,
                temperature=soft_refine_temperature,
                iterations=soft_refine_iterations,
                spatial_blend_alpha=soft_refine_spatial_alpha,
                height=H,
                width=W,
            )

        if verbose:
            print(f"Labels shape after clustering: {labels.shape}")
        labels = labels.reshape(H, W)
        if verbose:
            print(f"Labels shape after reshape: {labels.shape}")

        labels_resized = cv2.resize(
            labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        if verbose:
            print(f"labels_resized shape: {labels_resized.shape}")

        # Optional edge-aware refinement
        t_refine_start = time.perf_counter()
        labels_resized, refine_time = refine_labels(
            image_np=image_np,
            labels_resized=labels_resized,
            method=refine,
            compactness=refine_slic_compactness,
            sigma=refine_slic_sigma,
            verbose=verbose,
        )
        t_refine_end = t_refine_start + refine_time

        # Vegetation filtering (if enabled - works with any pipeline)
        # Automatically enabled for V3 pipeline for backward compatibility
        should_apply_filter = apply_vegetation_filter or (pipeline == "v3")
        
        if should_apply_filter:
            if verbose:
                print("🌳 Applying vegetation filtering (ExG-based cluster selection)...")
            labels_resized = apply_vegetation_filter(
                image_np,
                labels_resized,
                exg_threshold=exg_threshold,
                verbose=verbose
            )

        # Info / Metrics
        metrics = {'k_requested': k_requested}
        if collect_metrics:
            t_end = time.perf_counter()
            peak_vram_mb = None
            if torch.cuda.is_available():
                try:
                    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                except Exception:
                    peak_vram_mb = None
            metrics.update(
                compile_metrics(
                    t_start=t0,
                    t_pre_start=t_pre_start,
                    t_pre_end=t_pre_end,
                    t_features=t_features,
                    t_kselect=t_kselect,
                    t_refine_start=t_refine_start if refine_time > 0 else t_end,
                    t_refine_end=t_refine_end if refine_time > 0 else t_end,
                    auto_k=auto_k,
                    refine_time=refine_time,
                    H=H,
                    W=W,
                    features_flat=features_flat,
                    n_clusters=n_clusters,
                    device=device,
                    peak_vram_mb=peak_vram_mb,
                    needs_tiling=needs_tiling,
                )
            )

        return (image_np, labels_resized, metrics) if collect_metrics else (image_np, labels_resized)

    except Exception as e:
        # Use image_path_str if available, otherwise fallback to image_path
        path_for_error = image_path_str if 'image_path_str' in locals() else str(image_path)
        if verbose:
            print(f"Error processing {path_for_error}: {e}")
            traceback.print_exc()
        return None, None
