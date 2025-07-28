"""
Main tree segmentation function with automatic K selection.
This module brings together all the components from the modular structure.
"""

import os
import numpy as np
import torch

from .models import print_gpu_info, initialize_model, get_preprocess
from .core.segmentation import process_image
from .visualization import generate_outputs


def tree_seg_with_auto_k(
    input_dir="input",
    output_dir="output",
    n_clusters=5,
    overlay_ratio=5,
    stride=2,
    model_name="dinov2_vits14",
    filename=None,
    version="v1.5",
    auto_k=False,
    k_range=(3, 10),
    elbow_threshold=3.0,
    edge_width=2,
    use_hatching=False,
    use_pca=False,
):
    """
    Enhanced tree segmentation with automatic K selection.

    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for results
        n_clusters: Number of clusters (if auto_k=False)
        overlay_ratio: Overlay transparency (1-10, higher = more transparent)
        stride: Model stride (lower = higher resolution, slower)
        model_name: DINO model name
        filename: Specific filename to process (None = all files)
        version: Model version ("v1" or "v1.5")
        auto_k: Whether to use automatic K selection
        k_range: Range for K selection (min_k, max_k)
        elbow_threshold: Sensitivity for elbow detection (lower = more sensitive)
        edge_width: Width of edge lines in pixels for edge overlay visualization
        use_hatching: Whether to add hatch patterns to regions (borders are always shown)
        use_pca: Whether to use PCA dimensionality reduction (default: False)
    """
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
        # Process single file
        image_path = os.path.join(input_dir, filename)
        if os.path.exists(image_path) and filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            output_prefix = os.path.splitext(filename)[0]
            print(f"Processing {filename} ...")

            # Process with automatic K selection parameters
            result = process_image(
                image_path, model, preprocess, n_clusters, stride, version, device,
                auto_k=auto_k, k_range=k_range, elbow_threshold=elbow_threshold, use_pca=use_pca
            )

            if result[0] is not None:
                image_np, labels_resized = result
                # Get the actual number of clusters used (may be different if auto_k=True)
                actual_n_clusters = len(np.unique(labels_resized))

                generate_outputs(
                    image_np, labels_resized, output_prefix, output_dir,
                    actual_n_clusters, overlay_ratio, stride, model_name,
                    image_path, version, edge_width, use_hatching, elbow_threshold
                )

                print(f"✅ Processing completed! Used K = {actual_n_clusters}")
                if auto_k:
                    print("📊 K selection method: elbow")
                    print(f"🎯 Elbow threshold used: {elbow_threshold}")
                    print(f"📈 K selection analysis saved as: {output_prefix}_elbow_analysis.png")
            else:
                print("❌ Processing failed")
        else:
            print(f"File {filename} not found or is not a supported image format.")
    else:
        # Process all images in directory
        print("Processing all images in directory...")
        for fname in os.listdir(input_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                image_path = os.path.join(input_dir, fname)
                output_prefix = os.path.splitext(fname)[0]
                print(f"\nProcessing {fname} ...")

                result = process_image(
                    image_path, model, preprocess, n_clusters, stride, version, device,
                    auto_k=auto_k, k_range=k_range, elbow_threshold=elbow_threshold, use_pca=use_pca
                )

                if result[0] is not None:
                    image_np, labels_resized = result
                    actual_n_clusters = len(np.unique(labels_resized))

                    generate_outputs(
                        image_np, labels_resized, output_prefix, output_dir,
                        actual_n_clusters, overlay_ratio, stride, model_name,
                        image_path, version, edge_width, use_hatching, elbow_threshold
                    )

                    print(f"✅ {fname} completed! Used K = {actual_n_clusters}")
                    if auto_k:
                        print(f"🎯 Elbow threshold used: {elbow_threshold}")

    print("\n🎉 All processing completed!")


# Model configurations for easy access
MODELS = {
    "small": "dinov2_vits14",   # 21M params - Fast, saves credits
    "base": "dinov2_vitb14",    # 86M params - Good balance (recommended)
    "large": "dinov2_vitl14",   # 307M params - Better quality, more credits
    "giant": "dinov2_vitg14"    # 1.1B params - May not fit on T4
}