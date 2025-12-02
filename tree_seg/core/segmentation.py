"""
Core segmentation functionality for tree segmentation.
"""

import os
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from sklearn.cluster import KMeans

from ..analysis.elbow_method import find_optimal_k_elbow, plot_elbow_analysis

try:
    from skimage.segmentation import slic
except Exception:
    slic = None


def process_image(image_path, model, preprocess, n_clusters, stride, version, device,
                 auto_k=False, k_range=(3, 10), elbow_threshold=0.035, use_pca=False, pca_dim=None,
                 feature_upsample_factor: int = 1, refine: str | None = None,
                 refine_slic_compactness: float = 10.0, refine_slic_sigma: float = 1.0,
                 collect_metrics: bool = False, model_name=None, output_dir="data/output",
                 verbose: bool = True, pipeline: str = "v1_5",
                 apply_vegetation_filter: bool = False, exg_threshold: float = 0.10):
    """
    Process a single image for tree segmentation.
    
    Args:
        image_path: Path to the input image
        model: Initialized model
        preprocess: Preprocessing pipeline
        n_clusters: Number of clusters (if auto_k=False)
        stride: Model stride
        version: Model version ("v1" or "v1.5")
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
        if verbose:
            print(f"\n--- Processing {image_path} ---")
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        if verbose:
            print(f"Original image size: {w}x{h}")

        t_pre_start = time.perf_counter()
        image_tensor = preprocess(image).to(device)
        t_pre_end = time.perf_counter()
        if verbose:
            print(f"Preprocessed tensor shape: {image_tensor.shape}")

        with torch.no_grad():
            # DINOv3 always uses attention features for v3 (equivalent to v1.5)
            attn_choice = "none" if version == "v1" else "o"
            features_out = model.forward_sequential(image_tensor, attn_choice=attn_choice)
            if verbose:
                print(f"features_out type: {type(features_out)}")
            
            # DINOv3 adapter returns a dictionary with patch features
            if isinstance(features_out, dict):
                patch_features = features_out["x_norm_patchtokens"]
                attn_features = features_out.get("x_patchattn", None) if version in ["v1.5", "v3"] else None
                if verbose:
                    print(f"patch_features shape: {patch_features.shape}")
                    if attn_features is not None:
                        print(f"attn_features shape: {attn_features.shape}")
                # DINOv3 features are already in spatial format (H, W, D)
                # No need to take mean across batch dimension
            else:
                # Fallback for legacy tensor format
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

        time.perf_counter()
        if patch_features.dim() == 2:
            n_patches = patch_features.shape[0]
            H = W = int(np.sqrt(n_patches))
            if H * W != n_patches:
                raise ValueError(f"Cannot infer H, W from n_patches={n_patches}")
            patch_features = patch_features.view(H, W, -1)
            if attn_features is not None:
                attn_features = attn_features.view(H, W, -1)
        else:
            H, W = patch_features.shape[:2]
        if verbose:
            print(f"patch_features reshaped: {patch_features.shape}")
            if attn_features is not None:
                print(f"attn_features reshaped: {attn_features.shape}")
            # Report actual compute device used by features
            try:
                print(f"🖥️ Compute device: {patch_features.device}")
            except Exception:
                pass

        # Optional upsampling of the feature grid for smoother segmentation
        if isinstance(feature_upsample_factor, int) and feature_upsample_factor > 1:
            up_h, up_w = H * feature_upsample_factor, W * feature_upsample_factor
            # Upsample patch features (H, W, D) -> (up_h, up_w, D)
            pf = patch_features.permute(2, 0, 1).unsqueeze(0)  # 1, D, H, W
            pf_up = F.interpolate(pf, size=(up_h, up_w), mode="bilinear", align_corners=False)
            patch_features = pf_up.squeeze(0).permute(1, 2, 0)
            if attn_features is not None:
                af = attn_features.permute(2, 0, 1).unsqueeze(0)
                af_up = F.interpolate(af, size=(up_h, up_w), mode="bilinear", align_corners=False)
                attn_features = af_up.squeeze(0).permute(1, 2, 0)
            H, W = up_h, up_w
            if verbose:
                print(f"Upsampled features to: {H}x{W}")

        if attn_features is not None and version in ["v1.5", "v3"]:
            features_np = np.concatenate(
                [patch_features.cpu().numpy(), attn_features.cpu().numpy()], axis=-1
            )
            if verbose:
                print(f"Combined features shape: {features_np.shape}")
        else:
            features_np = patch_features.cpu().numpy()
            if verbose:
                print(f"Patch-only features shape: {features_np.shape}")

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
            output_prefix = os.path.splitext(os.path.basename(image_path))[0]
            # Use png subdirectory to match OutputManager expectations
            png_output_dir = os.path.join(output_dir, "png")
            os.makedirs(png_output_dir, exist_ok=True)
            plot_elbow_analysis(k_scores, png_output_dir, output_prefix, elbow_threshold * 100,
                               model_name, version, stride, optimal_k, auto_k, image_path)

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
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(features_flat)
        if verbose:
            print(f"labels shape after kmeans: {labels.shape}")
        labels = labels.reshape(H, W)
        if verbose:
            print(f"Labels shape after reshape: {labels.shape}")

        labels_resized = cv2.resize(
            labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        if verbose:
            print(f"labels_resized shape: {labels_resized.shape}")

        # Optional edge-aware refinement
        if refine == "slic":
            if slic is None:
                if verbose:
                    print("⚠️  skimage not available; skipping SLIC refinement.")
            else:
                if verbose:
                    print("🔧 Refining segmentation with SLIC superpixels...")
                t_refine_start = time.perf_counter()
                labels_resized = _refine_with_slic(
                    image_np,
                    labels_resized,
                    compactness=refine_slic_compactness,
                    sigma=refine_slic_sigma,
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
                'device_actual': str(getattr(patch_features, 'device', device)),
                'peak_vram_mb': round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
            })

        return (image_np, labels_resized, metrics) if collect_metrics else (image_np, labels_resized)

    except Exception as e:
        if verbose:
            print(f"Error processing {image_path}: {e}")
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


def run_processing(
    input_dir="data/input",
    output_dir="data/output",
    n_clusters=5,
    overlay_ratio=5,
    stride=4,
    model_name="dinov2_vits14",
    filename=None,
    version="v1.5"
):
    """
    Run processing on single file or directory (legacy function).
    Note: This function is deprecated. Use tree_seg_with_auto_k for full functionality.
    """
    from ..models import print_gpu_info, initialize_model, get_preprocess
    
    print_gpu_info()
    os.makedirs(output_dir, exist_ok=True)
    overlay_ratio = float(overlay_ratio)
    if overlay_ratio < 1 or overlay_ratio > 10:
        print("overlay_ratio must be between 1 and 10. Using default value 5.")
        overlay_ratio = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = initialize_model(stride, model_name, device)
    preprocess = get_preprocess(image_size=518)

    if filename:
        image_path = os.path.join(input_dir, filename)
        if os.path.exists(image_path) and filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            output_prefix = os.path.splitext(filename)[0]
            print(f"Processing {filename} ...")
            return process_image(image_path, model, preprocess, n_clusters, stride, version, device,
                               auto_k=False, k_range=(3, 10), verbose=True), output_prefix
        else:
            print(f"File {filename} not found or is not a supported image format.")
            return None, None
    else:
        results = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                image_path = os.path.join(input_dir, filename)
                output_prefix = os.path.splitext(filename)[0]
                print(f"Processing {filename} ...")
                result = process_image(image_path, model, preprocess, n_clusters, stride, version, device,
                                      auto_k=False, k_range=(3, 10), verbose=True)
                results.append((result, output_prefix))
        return results 
