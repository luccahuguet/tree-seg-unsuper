# %%
# NOTE: This file is a Jupyter notebook exported as a Python file.
# It contains special cell markers (e.g., # %%) to preserve notebook cell boundaries.
# You can open and edit it as a notebook in Jupyter or Kaggle.
# When editing as a .py file, be careful to preserve these cell markers and cell order.
# Avoid removing or reordering cells unless you know what you are doing.
#
# To convert back to a notebook, use Jupyter or VSCode's 'Python: Import Notebook' feature.
#
# This approach is used for easier version control and editing.
#

# %%
# Change to a safe directory first
%cd /kaggle/working

# Remove the old project directory if it exists
!rm -rf /kaggle/working/project

# Clone the repository again
!git clone https://github.com/luccahuguet/tree-seg-unsuper /kaggle/working/project

# Change into the new project directory
%cd /kaggle/working/project

# %%
# Uninstall conflicting packages
%pip uninstall -y torchaudio fastai

# Install dependencies with CUDA 12.4
%pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
%pip install timm
%pip install xformers --index-url https://download.pytorch.org/whl/cu124

# %%
# gpu stuff
import os
import torch

def print_gpu_info():
    if torch.cuda.is_available():
        gpu_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        total_mem = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {total_mem:.2f} GB")
    else:
        print("No CUDA-compatible GPU found.")

def setup_segmentation(output_dir):
    print_gpu_info()
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# %%
# init model
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from src.upsampler import HighResDV2

def init_model_and_preprocess(model_name, stride, device):
    model = HighResDV2(model_name, stride=stride, dtype=torch.float16).to(device)
    model.eval()
    model.set_transforms([lambda x: x], [lambda x: x])  # Identity transforms
    preprocess = Compose(
        [
            Resize((518, 518)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess

# %%
# Setup cell
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from src.upsampler import HighResDV2
from src.transform import get_shift_transforms, get_flip_transforms, get_rotation_transforms, combine_transforms, iden_partial
import traceback

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

def get_preprocess():
    return Compose([
        Resize((518, 518)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# %%
# Processing cell with automatic K selection (Elbow Method Only)

def find_optimal_k_elbow(features_flat, k_range=(3, 10), elbow_threshold=3.0):
    """
    Find optimal K using enhanced elbow method optimized for tree segmentation.

    Args:
        features_flat: Flattened feature array
        k_range: Tuple of (min_k, max_k) - default (3,10) optimized for tree species
        elbow_threshold: Percentage threshold for diminishing returns (lower = more sensitive)

    Returns:
        optimal_k: Best number of clusters
        scores: Dictionary with analysis results
    """
    min_k, max_k = k_range
    k_values = list(range(min_k, max_k + 1))
    wcss = []

    print(f"🔍 Testing K values from {min_k} to {max_k} using elbow method...")

    # Calculate WCSS for each K
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(features_flat)
        wcss.append(kmeans.inertia_)
        print(f"   K={k}: WCSS={wcss[-1]:.2f}")

    # Enhanced elbow detection with multiple methods
    wcss_array = np.array(wcss)

    # Method 1: Second derivative (curvature)
    if len(wcss_array) >= 3:
        first_diff = np.diff(wcss_array)
        second_diff = np.diff(first_diff)
        curvature_idx = np.argmax(np.abs(second_diff)) + 1  # +1 due to diff operations
    else:
        curvature_idx = 0

    # Method 2: Percentage decrease threshold
    pct_decrease = []
    for i in range(1, len(wcss_array)):
        pct = (wcss_array[i-1] - wcss_array[i]) / wcss_array[i-1] * 100
        pct_decrease.append(pct)

    # Find where percentage decrease drops below threshold (diminishing returns)
    threshold_idx = 0
    for i, pct in enumerate(pct_decrease):
        if pct < elbow_threshold:  # Less than threshold% improvement
            threshold_idx = i
            break

    # Choose the more conservative estimate (earlier elbow)
    elbow_idx = min(int(curvature_idx), int(threshold_idx)) if threshold_idx > 0 else int(curvature_idx)

    # Safety bounds
    elbow_idx = max(0, min(int(elbow_idx), len(k_values) - 1))
    optimal_k = k_values[elbow_idx]

    # Validate result
    if optimal_k < 3:
        print(f"⚠️  Optimal K={optimal_k} seems too low for tree species, using K=3")
        optimal_k = 3
        elbow_idx = k_values.index(3) if 3 in k_values else 0
    elif optimal_k > 8:
        print(f"⚠️  Optimal K={optimal_k} seems high for typical tree species, using K=8")
        optimal_k = min(8, max(k_values))
        elbow_idx = k_values.index(optimal_k)

    print(f"📊 Elbow method suggests optimal K = {optimal_k}")

    return optimal_k, {
        'k_values': k_values,
        'wcss': wcss,
        'elbow_idx': elbow_idx,
        'optimal_k': optimal_k,
        'pct_decrease': pct_decrease,
        'method': 'elbow'
    }

def plot_elbow_analysis(scores, output_dir, output_prefix, elbow_threshold=3.0):
    """
    Create enhanced elbow plot with additional analysis information.
    """
    k_values = scores['k_values']
    wcss = scores['wcss']
    elbow_idx = scores['elbow_idx']
    optimal_k = scores['optimal_k']
    pct_decrease = scores.get('pct_decrease', [])

    # Create subplot with elbow curve and percentage decrease
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Main elbow plot
    ax1.plot(k_values, wcss, 'bo-', linewidth=3, markersize=10, alpha=0.7)
    ax1.plot(k_values[elbow_idx], wcss[elbow_idx], 'ro', markersize=15,
             label=f'Optimal K = {optimal_k}', zorder=5)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    ax1.set_title('Tree Species Clustering - Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    # Add annotations
    ax1.annotate(f'Selected K = {optimal_k}',
                xy=(k_values[elbow_idx], wcss[elbow_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Percentage decrease plot
    if pct_decrease:
        ax2.plot(k_values[1:], pct_decrease, 'go-', linewidth=2, markersize=8)
        ax2.axhline(y=elbow_threshold, color='r', linestyle='--', alpha=0.7, label=f'{elbow_threshold}% Threshold')
        ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax2.set_ylabel('WCSS Improvement (%)', fontsize=12)
        ax2.set_title('Diminishing Returns Analysis', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f"{output_prefix}_elbow_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 Saved elbow analysis: {plot_path}")
    return plot_path

def process_image(image_path, model, preprocess, n_clusters, stride, version, device,
                 auto_k=False, k_range=(3, 10), elbow_threshold=3.0):
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

        # Automatic K selection using elbow method
        if auto_k:
            print(f"\n--- Automatic K Selection using elbow method ---")
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

# %%
# Visualization functions (needed before tree_seg_with_auto_k)
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
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(labels_resized, cmap=cmap, vmin=0, vmax=n_clusters - 1)
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_clusters))
    cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
    ax.axis("off")
    ax.text(
        0.02, 0.98, config_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.tight_layout()
    legend_path = os.path.join(output_dir, f"{output_prefix}_segmentation_legend.png")
    plt.savefig(legend_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved segmentation with legend: {legend_path}")

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
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.tight_layout()
    overlay_path = os.path.join(output_dir, f"{output_prefix}_overlay.png")
    plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved overlay: {overlay_path}")

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
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.tight_layout()
    side_by_side_path = os.path.join(output_dir, f"{output_prefix}_side_by_side.png")
    plt.savefig(side_by_side_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved side-by-side image: {side_by_side_path}")

# %%
# MAIN CONFIG - Edit this to change settings
import sys
sys.path.append("/kaggle/working/project/src")

# Available models
MODELS = {
    "small": "dinov2_vits14",   # 21M params - Fast, saves credits
    "base": "dinov2_vitb14",    # 86M params - Good balance (recommended)
    "large": "dinov2_vitl14",   # 307M params - Better quality, more credits
    "giant": "dinov2_vitg14"    # 1.1B params - May not fit on T4
}

# SINGLE CONFIG - Change these settings as needed
config = {
    "input_dir": "/kaggle/input/drone-10-best",
    "output_dir": "/kaggle/working/output",
    "model_name": MODELS["base"],           # Choose: "small", "base", "large", "giant"
    "filename": "DJI_20250127150117_0029_D.JPG",
    "version": "v1.5",
    "auto_k": True,                         # Automatic K selection (recommended)
    "k_range": (3, 10),                     # K range for auto selection
    "elbow_threshold": 3.0,                 # Sensitivity for elbow detection (lower = more sensitive)
    "n_clusters": 6,                        # Only used if auto_k=False
    "overlay_ratio": 4,                     # Transparency: 1=opaque, 10=transparent
    "stride": 4,                            # Lower=higher resolution, slower
}

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
    elbow_threshold=3.0
):
    """Enhanced tree segmentation with automatic K selection."""
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

            # Process with automatic K selection parameters
            result = process_image(
                image_path, model, preprocess, n_clusters, stride, version, device,
                auto_k=auto_k, k_range=k_range, elbow_threshold=elbow_threshold
            )

            if result[0] is not None:
                image_np, labels_resized = result
                # Get the actual number of clusters used (may be different if auto_k=True)
                actual_n_clusters = len(np.unique(labels_resized))

                generate_outputs(
                    image_np, labels_resized, output_prefix, output_dir,
                    actual_n_clusters, overlay_ratio, stride, model_name,
                    image_path, version
                )

                print(f"✅ Processing completed! Used K = {actual_n_clusters}")
                if auto_k:
                    print(f"📊 K selection method: elbow")
                    print(f"📈 K selection analysis saved as: {output_prefix}_elbow_analysis.png")
            else:
                print("❌ Processing failed")
        else:
            print(f"File {filename} not found or is not a supported image format.")
    else:
        print("Processing all images in directory...")
        # Process all images with auto K selection
        for fname in os.listdir(input_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                image_path = os.path.join(input_dir, fname)
                output_prefix = os.path.splitext(fname)[0]
                print(f"\nProcessing {fname} ...")

                result = process_image(
                    image_path, model, preprocess, n_clusters, stride, version, device,
                    auto_k=auto_k, k_range=k_range, elbow_threshold=elbow_threshold
                )

                if result[0] is not None:
                    image_np, labels_resized = result
                    actual_n_clusters = len(np.unique(labels_resized))

                    generate_outputs(
                        image_np, labels_resized, output_prefix, output_dir,
                        actual_n_clusters, overlay_ratio, stride, model_name,
                        image_path, version
                    )

                    print(f"✅ {fname} completed! Used K = {actual_n_clusters}")

# Run the enhanced segmentation
print("🌳 Starting Enhanced Tree Segmentation with Automatic K Selection...")
print(f"📁 Input: {config['input_dir']}")
print(f"📁 Output: {config['output_dir']}")
print(f"🔧 Auto K: {config['auto_k']}")
if config['auto_k']:
    print(f"📊 Method: Elbow (optimized for trees)")
    print(f"📈 K Range: {config['k_range']}")
else:
    print(f"🔢 Fixed K: {config['n_clusters']}")

tree_seg_with_auto_k(**config)


# %%
# Display results
from IPython.display import Image, display

filename = config["filename"]
output_prefix = os.path.splitext(filename)[0]
output_dir = config["output_dir"]

legend_path = os.path.join(output_dir, f"{output_prefix}_segmentation_legend.png")
overlay_path = os.path.join(output_dir, f"{output_prefix}_overlay.png")
side_by_side_path = os.path.join(output_dir, f"{output_prefix}_side_by_side.png")
elbow_path = os.path.join(output_dir, f"{output_prefix}_elbow_analysis.png")

# Display the overlay
if os.path.exists(overlay_path):
    print("🖼️ Overlay Image:")
    display(Image(filename=overlay_path))

# Display the side-by-side comparison
if os.path.exists(side_by_side_path):
    print("📊 Original and Segmentation Side by Side:")
    display(Image(filename=side_by_side_path))

# Display the K selection analysis
if os.path.exists(elbow_path):
    print("📈 K Selection Analysis (Elbow Method):")
    display(Image(filename=elbow_path))


