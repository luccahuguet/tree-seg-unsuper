#!/usr/bin/env python3
"""
Local Tree Segmentation Script
Adapted from the Kaggle notebook for local execution
"""

import os
import sys
import torch
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import traceback

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.upsampler import HighResDV2
    from src.transform import get_shift_transforms, get_flip_transforms, get_rotation_transforms, combine_transforms, iden_partial
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you have the src directory with the required modules")
    sys.exit(1)

def print_gpu_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        gpu_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        total_mem = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {total_mem:.2f} GB")
    else:
        print("No CUDA-compatible GPU found. Using CPU.")

def get_config_text(n_clusters, overlay_ratio, stride, model_name, filename, version):
    """Generate a formatted string of configuration parameters."""
    config_lines = [
        f"Version: {version}",
        f"Clusters: {n_clusters}",
        f"Overlay Ratio: {overlay_ratio}",
        f"Stride: {stride}",
        f"Model: {model_name}",
        f"File: {filename if filename else 'All files in directory'}"
    ]
    return "\n".join(config_lines)

def initialize_model(stride, model_name, device):
    """Initialize the DINOv2 model with transforms."""
    try:
        model = HighResDV2(model_name, stride=stride, dtype=torch.float16).to(device)
        model.eval()
        shift_transforms, shift_inv_transforms = get_shift_transforms(dists=[1], pattern="Moore")
        flip_transforms, flip_inv_transforms = get_flip_transforms()
        rot_transforms, rot_inv_transforms = get_rotation_transforms()
        all_fwd_transforms, all_inv_transforms = combine_transforms(
            shift_transforms, flip_transforms, shift_inv_transforms, flip_inv_transforms
        )
        all_transforms = [t for t in all_fwd_transforms if t != iden_partial]
        all_inv_transforms = [t for t in all_inv_transforms if t != iden_partial]
        model.set_transforms(all_transforms, all_inv_transforms)
        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

def get_preprocess():
    """Get the preprocessing pipeline."""
    return Compose([
        Resize((518, 518)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def process_image(image_path, model, preprocess, n_clusters, stride, version, device):
    """Process a single image for segmentation."""
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

        if features_flat.shape[-1] > 128:
            print("Running PCA on flat features...")
            features_flat_tensor = torch.tensor(features_flat, dtype=torch.float32)
            mean = features_flat_tensor.mean(dim=0)
            features_flat_centered = features_flat_tensor - mean
            U, S, V = torch.pca_lowrank(features_flat_centered, q=128, center=False)
            features_flat = (features_flat_centered @ V[:, :128]).numpy()
            print(f"PCA-reduced features shape: {features_flat.shape}")

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

def generate_outputs(
    image_np,
    labels_resized,
    output_prefix,
    output_dir,
    n_clusters,
    overlay_ratio,
    stride,
    model_name,
    image_path,
    version,
):
    """Generate visualization outputs."""
    if image_np is None or labels_resized is None:
        print(f"Skipping output generation for {image_path} due to processing error.")
        return

    alpha = (10 - overlay_ratio) / 10.0
    filename = os.path.basename(image_path)

    if n_clusters <= 10:
        cmap = plt.get_cmap("tab10", n_clusters)
    elif n_clusters <= 20:
        cmap = plt.get_cmap("tab20", n_clusters)
    else:
        cmap = plt.get_cmap("gist_ncar", n_clusters)

    config_text = get_config_text(n_clusters, overlay_ratio, stride, model_name, filename, version)
    
    # Save segmentation map with legend
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(labels_resized, cmap=cmap, vmin=0, vmax=n_clusters - 1)
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_clusters))
    cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
    ax.axis("off")
    bbox_props = dict(facecolor='white', alpha=0.7, edgecolor='none')
    ax.text(
        0.02, 0.98, config_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=bbox_props
    )
    plt.tight_layout()
    legend_path = os.path.join(output_dir, f"{output_prefix}_segmentation_legend.png")
    plt.savefig(legend_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved segmentation with legend: {legend_path}")

    # Save overlay
    norm = colors.Normalize(vmin=0, vmax=n_clusters - 1)
    segmentation_rgb = cmap(norm(labels_resized))[:, :, :3]
    segmentation_rgb = (segmentation_rgb * 255).astype(np.uint8)
    overlay = (alpha * image_np + (1 - alpha) * segmentation_rgb).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.axis("off")
    ax.text(
        0.02, 0.98, config_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=bbox_props
    )
    plt.tight_layout()
    overlay_path = os.path.join(output_dir, f"{output_prefix}_overlay.png")
    plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved overlay: {overlay_path}")

    # Save side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")
    im = axes[1].imshow(labels_resized, cmap=cmap, vmin=0, vmax=n_clusters - 1)
    axes[1].set_title("Segmentation Map", fontsize=12)
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], ticks=range(n_clusters))
    cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
    axes[1].text(
        0.02, 0.98, config_text,
        transform=axes[1].transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=bbox_props
    )
    plt.tight_layout()
    side_by_side_path = os.path.join(output_dir, f"{output_prefix}_side_by_side.png")
    plt.savefig(side_by_side_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved side-by-side image: {side_by_side_path}")

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
    """Run the main processing pipeline."""
    print_gpu_info()
    os.makedirs(output_dir, exist_ok=True)
    overlay_ratio = float(overlay_ratio)
    if overlay_ratio < 1 or overlay_ratio > 10:
        print("overlay_ratio must be between 1 and 10. Using default value 5.")
        overlay_ratio = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = initialize_model(stride, model_name, device)
    if model is None:
        print("Failed to initialize model. Exiting.")
        return None, None
        
    preprocess = get_preprocess()

    if filename:
        image_path = os.path.join(input_dir, filename)
        if os.path.exists(image_path) and filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            output_prefix = os.path.splitext(filename)[0]
            print(f"Processing {filename} ...")
            return process_image(image_path, model, preprocess, n_clusters, stride, version, device), output_prefix
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
                result = process_image(image_path, model, preprocess, n_clusters, stride, version, device)
                results.append((result, output_prefix))
        return results

def run_visualization(
    input_dir="input",
    output_dir="output",
    n_clusters=5,
    overlay_ratio=5,
    stride=4,
    model_name="dinov2_vits14",
    filename=None,
    version="v1.5"
):
    """Run the complete visualization pipeline."""
    results = run_processing(
        input_dir, output_dir, n_clusters, overlay_ratio, stride, model_name, filename, version
    )
    if results is None:
        print("Processing failed. Exiting.")
        return
        
    if filename:
        if results[0] is not None and results[0][0] is not None:
            (image_np, labels_resized), output_prefix = results
            generate_outputs(
                image_np, labels_resized, output_prefix, output_dir,
                n_clusters, overlay_ratio, stride, model_name,
                os.path.join(input_dir, filename), version
            )
    else:
        if isinstance(results, list):
            for result, output_prefix in results:
                if result is not None and result[0] is not None:
                    image_np, labels_resized = result
                    generate_outputs(
                        image_np, labels_resized, output_prefix, output_dir,
                        n_clusters, overlay_ratio, stride, model_name,
                        os.path.join(input_dir, output_prefix + ".jpg"), version
                    )

def tree_seg(
    input_dir="input",
    output_dir="output",
    n_clusters=5,
    overlay_ratio=5,
    stride=2,
    model_name="dinov2_vits14",
    filename=None,
    version="v1.5"
):
    """Main entry point for tree segmentation."""
    run_visualization(input_dir, output_dir, n_clusters, overlay_ratio, stride, model_name, filename, version)

if __name__ == "__main__":
    # Example usage - modify these paths for your local setup
    config = {
        "input_dir": "input",  # Change this to your input directory
        "output_dir": "output",
        "n_clusters": 6,
        "overlay_ratio": 4,
        "stride": 4,
        "model_name": "dinov2_vits14",
        "filename": None,  # Set to a specific filename or None for all images
        "version": "v1.5"
    }
    
    print("Starting Tree Segmentation...")
    print(f"Input directory: {config['input_dir']}")
    print(f"Output directory: {config['output_dir']}")
    
    # Run the segmentation
    tree_seg(**config)
    
    print("Tree segmentation completed!") 