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

from ..analysis.elbow_method import find_optimal_k_elbow, plot_elbow_analysis
from .features import extract_tiled_features, extract_features
try:
    from skimage.segmentation import slic
except Exception:
    slic = None


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

        # Optional PCA with configurable dimension
        effective_pca_dim = None
        if pca_dim is not None and pca_dim > 0:
            effective_pca_dim = min(pca_dim, features_flat.shape[-1])
        elif use_pca and features_flat.shape[-1] > 128:
            effective_pca_dim = 128

        if effective_pca_dim is not None and effective_pca_dim < features_flat.shape[-1]:
            if verbose:
                print(f"Running PCA to {effective_pca_dim} dims...")
            features_flat_tensor = torch.tensor(features_flat, dtype=torch.float32)
            mean = features_flat_tensor.mean(dim=0)
            features_flat_centered = features_flat_tensor - mean
            U, S, V = torch.pca_lowrank(features_flat_centered, q=effective_pca_dim, center=False)
            features_flat = (features_flat_centered @ V[:, :effective_pca_dim]).numpy()
            if verbose:
                print(f"PCA-reduced features shape: {features_flat.shape}")
        else:
            if verbose:
                print(f"Using {features_flat.shape[-1]}-D features (no PCA)")

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

        # Clean features for main clustering (same as elbow method)
        if np.isnan(features_flat).any():
            if verbose:
                print("⚠️  Warning: Features contain NaN values for main clustering")
                print("🧹 Cleaning NaN values...")
            features_flat = np.nan_to_num(features_flat, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.all(features_flat == 0):
                if verbose:
                    print("🎲 Adding small random noise to zero features...")
                features_flat += np.random.normal(0, 0.001, features_flat.shape)

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
        # Optional edge-aware refinement
        if refine == "slic":
            # Try OpenCV SLIC first (much faster)
            if hasattr(cv2, 'ximgproc'):
                if verbose:
                    print("🔧 Refining with fast OpenCV SLIC...")
                t_refine_start = time.perf_counter()
                labels_resized = _refine_with_opencv_slic(
                    image_np,
                    labels_resized,
                    compactness=refine_slic_compactness,
                    region_size=48  # Approx matching 48x48 target area
                )
                t_refine_end = time.perf_counter()
            # Fallback to skimage SLIC
            elif slic is not None:
                if verbose:
                    print("🔧 Refining with skimage SLIC (slow)...")
                t_refine_start = time.perf_counter()
                labels_resized = _refine_with_slic(
                    image_np,
                    labels_resized,
                    compactness=refine_slic_compactness,
                    sigma=refine_slic_sigma,
                )
                t_refine_end = time.perf_counter()
            else:
                if verbose:
                    print("⚠️  No SLIC implementation available (install opencv-contrib-python or scikit-image)")
        elif refine in ("slic_skimage", "slic-skimage"):
            if slic is not None:
                if verbose:
                    print("🔧 Refining with skimage SLIC (forced)...")
                t_refine_start = time.perf_counter()
                labels_resized = _refine_with_slic(
                    image_np,
                    labels_resized,
                    compactness=refine_slic_compactness,
                    sigma=refine_slic_sigma,
                )
                t_refine_end = time.perf_counter()
            elif verbose:
                print("⚠️  scikit-image SLIC unavailable; skipping refinement (install scikit-image)")
        elif refine == "bilateral":
            if verbose:
                print("🔧 Refining segmentation with bilateral filter...")
            t_refine_start = time.perf_counter()
            labels_resized = _refine_with_bilateral(
                image_np,
                labels_resized,
            )
            t_refine_end = time.perf_counter()

        # Vegetation filtering (if enabled - works with any pipeline)
        # Automatically enabled for V3 pipeline for backward compatibility
        should_apply_filter = apply_vegetation_filter or (pipeline == "v3")
        
        if should_apply_filter:
            if verbose:
                print("🌳 Applying vegetation filtering (ExG-based cluster selection)...")
            labels_resized = _apply_vegetation_filter(
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
            metrics.update({
                'time_total_s': round(t_end - t0, 3),
                'time_preprocess_s': round(t_pre_end - t_pre_start, 3),
                'time_features_s': round(t_features - t_pre_end, 3),
                'time_kselect_s': round(t_kselect - t_features, 3) if auto_k else 0.0,
                'time_kmeans_s': round((t_refine_start if refine == 'slic' and slic is not None else t_end) - t_kselect, 3),
                'time_refine_s': round((t_refine_end - t_refine_start), 3) if refine == 'slic' and slic is not None else 0.0,
                'grid_H': int(H),
                'grid_W': int(W),
                'n_features': int(features_flat.shape[-1]),
                'n_vectors': int(features_flat.shape[0]),
                'n_clusters': int(n_clusters),
                'device_requested': str(device),
                'device_actual': str(device),
                'peak_vram_mb': round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
                'used_tiling': needs_tiling,
            })

        return (image_np, labels_resized, metrics) if collect_metrics else (image_np, labels_resized)

    except Exception as e:
        # Use image_path_str if available, otherwise fallback to image_path
        path_for_error = image_path_str if 'image_path_str' in locals() else str(image_path)
        if verbose:
            print(f"Error processing {path_for_error}: {e}")
            traceback.print_exc()
        return None, None


def _apply_vegetation_filter(
    image_np: np.ndarray,
    cluster_labels: np.ndarray,
    exg_threshold: float = 0.10,
    verbose: bool = True
) -> np.ndarray:
    """
    Apply vegetation filtering to cluster labels.

    Args:
        image_np: RGB image (H, W, 3)
        cluster_labels: Cluster labels (H, W)
        exg_threshold: ExG threshold for vegetation classification
        verbose: Print progress

    Returns:
        Filtered labels (H, W) with only vegetation clusters (0 = background)
    """
    try:
        from ..vegetation_filter import apply_vegetation_filter

        # Apply vegetation filter
        filtered_labels, filter_info = apply_vegetation_filter(
            image_np,
            cluster_labels,
            exg_threshold=exg_threshold,
            verbose=verbose
        )

        if verbose:
            print(f"  ✓ Filtered to {filter_info['n_vegetation_clusters']} vegetation clusters")
            print(f"  ✓ Vegetation coverage: {filter_info['vegetation_percentage']:.1f}%")

        return filtered_labels

    except Exception as e:
        if verbose:
            print(f"  ⚠️  Vegetation filtering failed: {e}, returning original clusters")
        return cluster_labels


def _refine_with_slic(image_np: np.ndarray, labels_resized: np.ndarray,
                       compactness: float = 10.0, sigma: float = 1.0) -> np.ndarray:
    """Refine cluster labels using SLIC superpixels with majority voting.

    Args:
        image_np: Original RGB image as numpy array (H, W, 3)
        labels_resized: Initial labels (H, W)
        compactness: SLIC compactness parameter
        sigma: SLIC Gaussian smoothing parameter

    Returns:
        Refined labels (H, W)
    """
    h, w = labels_resized.shape[:2]
    # Target ~ one superpixel per ~48x48 area (tunable)
    target_area = 48 * 48
    n_segments = max(100, int((h * w) / target_area))
    
    # Cap at reasonable maximum to prevent hanging on huge images
    # (e.g., 9000x9000 FORTRESS orthomosaics)
    MAX_SEGMENTS = 2000
    n_segments = min(n_segments, MAX_SEGMENTS)

    # Ensure float image in [0,1]
    img_float = image_np.astype(np.float32)
    if img_float.max() > 1.5:
        img_float = img_float / 255.0

    segments = slic(
        img_float,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
    )

    refined = labels_resized.copy()
    # Majority vote within each superpixel
    # Vectorized approach via flattening
    seg_flat = segments.reshape(-1)
    lab_flat = labels_resized.reshape(-1)

    # For each segment id, compute the mode of labels
    seg_ids = np.unique(seg_flat)
    for sid in seg_ids:
        mask = (seg_flat == sid)
        if not np.any(mask):
            continue
        # bincount of labels in this segment
        vals = lab_flat[mask]
        max_label = np.bincount(vals).argmax()
        refined.reshape(-1)[mask] = max_label

    return refined.astype(np.uint8)


def _refine_with_opencv_slic(image_np: np.ndarray, labels_resized: np.ndarray,
                            compactness: float = 10.0, region_size: int = 48) -> np.ndarray:
    """Refine cluster labels using OpenCV's fast SLIC implementation.
    
    Much faster than skimage.segmentation.slic.
    
    Args:
        image_np: Original RGB image (H, W, 3)
        labels_resized: Initial labels (H, W)
        compactness: SLIC compactness (smoothness)
        region_size: Average superpixel size
        
    Returns:
        Refined labels (H, W)
    """
    # Target number of segments to keep runtime reasonable
    # Same logic as skimage implementation
    MAX_SEGMENTS = 2000
    
    h, w = image_np.shape[:2]
    
    # If region_size is default (48), override it based on MAX_SEGMENTS
    # otherwise respect the provided region_size if it results in fewer segments
    if region_size == 48:
        calculated_region_size = int(np.sqrt((h * w) / MAX_SEGMENTS))
        region_size = max(region_size, calculated_region_size)
    
    # Initialize SLIC
    # algorithm: 100 = SLICO (optimization), 101 = SLIC, 102 = MSLIC
    slic = cv2.ximgproc.createSuperpixelSLIC(
        image_np, 
        algorithm=cv2.ximgproc.SLIC,
        region_size=region_size,
        ruler=float(compactness)
    )
    
    # Run SLIC
    slic.iterate(10)  # 10 iterations is standard
    
    # Get labels
    segments = slic.getLabels()
    
    # Fast vectorized majority voting
    # Compute 2D histogram of (segment_id, label)
    # Rows = segments, Cols = labels
    seg_flat = segments.reshape(-1)
    lab_flat = labels_resized.reshape(-1)
    
    n_segments_actual = segments.max() + 1
    n_labels = labels_resized.max() + 1
    
    # histogram2d is fast and avoids the loop
    hist, _, _ = np.histogram2d(
        seg_flat, 
        lab_flat, 
        bins=[n_segments_actual, n_labels],
        range=[[0, n_segments_actual], [0, n_labels]]
    )
    
    # Find mode label for each segment (argmax along label axis)
    segment_modes = np.argmax(hist, axis=1).astype(np.uint8)
    
    # Map back to image
    refined = segment_modes[segments]
        
    return refined.astype(np.uint8)


def _refine_with_bilateral(image_np: np.ndarray, labels_resized: np.ndarray) -> np.ndarray:
    """Refine cluster labels using bilateral filtering.
    
    Fast edge-aware refinement alternative to SLIC. Smooths labels while preserving
    edges using spatial and intensity-based filtering.
    
    Args:
        image_np: Original RGB image as numpy array (H, W, 3)
        labels_resized: Initial labels (H, W)
        
    Returns:
        Refined labels (H, W)
    """
    # Apply bilateral filter to smooth image while preserving edges
    # Parameters tuned for aerial imagery edge preservation
    d = 9  # Neighborhood diameter  
    sigmaColor = 75  # Filter sigma in color space
    sigmaSpace = 75  # Filter sigma in coordinate space
    
    # Filter the original image to get edge map
    # filtered = cv2.bilateralFilter(image_np, d, sigmaColor, sigmaSpace)
    
    # Compute edge strength between filtered and original
    # Strong edges in the filtered image indicate object boundaries
    # edge_strength = np.abs(filtered.astype(np.float32) - image_np.astype(np.float32)).mean(axis=2)
    
    # Create label probability map by filtering the labels
    # Convert labels to float for filtering
    h, w = labels_resized.shape
    n_labels = int(labels_resized.max()) + 1
    
    # Create one-hot encoded label maps
    label_smoothed = np.zeros((h, w), dtype=np.uint8)
    
    for label_id in range(n_labels):
        # Create binary mask for this label
        mask = (labels_resized == label_id).astype(np.float32)
        
        # Apply bilateral filter to the mask
        # This smooths within regions but preserves edges
        smoothed_mask = cv2.bilateralFilter(
            (mask * 255).astype(np.uint8), 
            d, 
            sigmaColor, 
            sigmaSpace
        ).astype(np.float32) / 255.0
        
        # Update labels where this label has highest confidence
        label_smoothed[smoothed_mask > 0.5] = label_id
    
    return label_smoothed
