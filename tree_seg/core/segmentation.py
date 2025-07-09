"""
Core segmentation functionality for tree segmentation.
"""

import os
import traceback
import numpy as np
import torch
import cv2
from PIL import Image
from sklearn.cluster import KMeans

from ..analysis.elbow_method import find_optimal_k_elbow, plot_elbow_analysis


def process_image(image_path, model, preprocess, n_clusters, stride, version, device,
                 auto_k=False, k_range=(3, 10), elbow_threshold=3.0, use_pca=False):
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
        elbow_threshold: Threshold for elbow method
        
    Returns:
        Tuple of (image_np, labels_resized) or (None, None) on error
    """
    try:
        print(f"\n--- Processing {image_path} ---")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        print(f"Original image size: {w}x{h}")

        image_tensor = preprocess(image).to(device)
        print(f"Preprocessed tensor shape: {image_tensor.shape}")

        with torch.no_grad():
            attn_choice = "none" if version == "v1" else "o"
            features_out = model.forward_sequential(image_tensor, attn_choice=attn_choice)
            print(f"features_out type: {type(features_out)}")
            if isinstance(features_out, dict):
                patch_features = features_out["x_norm_patchtokens"]
                attn_features = features_out.get("x_patchattn", None) if version == "v1.5" else None
                print(f"patch_features shape: {patch_features.shape}")
                if attn_features is not None:
                    print(f"attn_features shape: {attn_features.shape}")
                patch_features = patch_features.mean(dim=0).squeeze()
                if attn_features is not None:
                    attn_features = attn_features.mean(dim=0).squeeze()
            else:
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
            if H * W != n_patches:
                raise ValueError(f"Cannot infer H, W from n_patches={n_patches}")
            patch_features = patch_features.view(H, W, -1)
            if attn_features is not None:
                attn_features = attn_features.view(H, W, -1)
        else:
            H, W = patch_features.shape[:2]
        print(f"patch_features reshaped: {patch_features.shape}")
        if attn_features is not None:
            print(f"attn_features reshaped: {attn_features.shape}")

        if attn_features is not None and version == "v1.5":
            features_np = np.concatenate(
                [patch_features.cpu().numpy(), attn_features.cpu().numpy()], axis=-1
            )
            print(f"Combined features shape: {features_np.shape}")
        else:
            features_np = patch_features.cpu().numpy()
            print(f"Patch-only features shape: {features_np.shape}")

        features_flat = features_np.reshape(-1, features_np.shape[-1])
        print(f"Flattened features shape: {features_flat.shape}")

        if use_pca and features_flat.shape[-1] > 128:
            print("Running PCA on flat features...")
            features_flat_tensor = torch.tensor(features_flat, dtype=torch.float32)
            mean = features_flat_tensor.mean(dim=0)
            features_flat_centered = features_flat_tensor - mean
            U, S, V = torch.pca_lowrank(features_flat_centered, q=128, center=False)
            features_flat = (features_flat_centered @ V[:, :128]).numpy()
            print(f"PCA-reduced features shape: {features_flat.shape}")
        elif features_flat.shape[-1] > 128:
            print(f"Skipping PCA (disabled). Using full {features_flat.shape[-1]} dimensions.")
        else:
            print(f"Features already low-dimensional ({features_flat.shape[-1]} dims), no PCA needed.")

        # Automatic K selection using elbow method
        if auto_k:
            print("\n--- Automatic K Selection using elbow method ---")
            optimal_k, k_scores = find_optimal_k_elbow(features_flat, k_range, elbow_threshold)

            print(f"Selected optimal K = {optimal_k}")

            # Save K selection analysis plot
            output_dir = "/kaggle/working/output"
            output_prefix = os.path.splitext(os.path.basename(image_path))[0]
            plot_elbow_analysis(k_scores, output_dir, output_prefix, elbow_threshold)

            n_clusters = optimal_k
        else:
            print(f"Using fixed K = {n_clusters}")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(features_flat)
        print(f"labels shape after kmeans: {labels.shape}")
        labels = labels.reshape(H, W)
        print(f"Labels shape after reshape: {labels.shape}")

        labels_resized = cv2.resize(
            labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        print(f"labels_resized shape: {labels_resized.shape}")

        return image_np, labels_resized

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        traceback.print_exc()
        return None, None


def run_processing(
    input_dir="input",
    output_dir="output",
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
    preprocess = get_preprocess()

    if filename:
        image_path = os.path.join(input_dir, filename)
        if os.path.exists(image_path) and filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            output_prefix = os.path.splitext(filename)[0]
            print(f"Processing {filename} ...")
            return process_image(image_path, model, preprocess, n_clusters, stride, version, device,
                               auto_k=False, k_range=(3, 10)), output_prefix
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
                                      auto_k=False, k_range=(3, 10))
                results.append((result, output_prefix))
        return results 