# %% [code] {"execution":{"iopub.status.busy":"2025-06-11T03:10:53.784722Z","iopub.execute_input":"2025-06-11T03:10:53.785492Z","iopub.status.idle":"2025-06-11T03:10:54.414887Z","shell.execute_reply.started":"2025-06-11T03:10:53.785464Z","shell.execute_reply":"2025-06-11T03:10:54.414193Z"},"jupyter":{"source_hidden":true,"outputs_hidden":true}}
# Change to a safe directory first
%cd /kaggle/working

# Remove the old project directory if it exists
!rm -rf /kaggle/working/project

# Clone the repository again
!git clone https://github.com/luccahuguet/tree-seg-unsuper /kaggle/working/project

# Change into the new project directory
%cd /kaggle/working/project

# %% [code] {"execution":{"iopub.status.busy":"2025-06-11T03:10:54.416319Z","iopub.execute_input":"2025-06-11T03:10:54.416566Z","iopub.status.idle":"2025-06-11T03:11:05.235277Z","shell.execute_reply.started":"2025-06-11T03:10:54.416544Z","shell.execute_reply":"2025-06-11T03:11:05.234297Z"},"jupyter":{"source_hidden":true,"outputs_hidden":true}}
# Uninstall conflicting packages
!pip uninstall -y torchaudio fastai

# Install dependencies with CUDA 12.4
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
!pip install timm
!pip install xformers --index-url https://download.pytorch.org/whl/cu124

# %% [code] {"execution":{"iopub.status.busy":"2025-06-11T03:11:05.236442Z","iopub.execute_input":"2025-06-11T03:11:05.236670Z","iopub.status.idle":"2025-06-11T03:11:05.241387Z","shell.execute_reply.started":"2025-06-11T03:11:05.236648Z","shell.execute_reply":"2025-06-11T03:11:05.240517Z"},"jupyter":{"outputs_hidden":false}}
config = {
    "input_dir": "/kaggle/input/drone-10-best",
    "output_dir": "/kaggle/working/output",
    "n_clusters": 6,
    "overlay_ratio": 4,
    "stride": 4,
    "model_name": "dinov2_vits14",
    "filename": "DJI_20250127150117_0029_D.JPG",
    "version": "v1.5"  # Options: "v1" (patch features only) or "v1.5" (patch + attention features)
}

# %% [code] {"execution":{"iopub.status.busy":"2025-06-11T03:11:05.243509Z","iopub.execute_input":"2025-06-11T03:11:05.244191Z","iopub.status.idle":"2025-06-11T03:11:05.253714Z","shell.execute_reply.started":"2025-06-11T03:11:05.244171Z","shell.execute_reply":"2025-06-11T03:11:05.253131Z"},"jupyter":{"source_hidden":true}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-06-11T03:11:05.254545Z","iopub.execute_input":"2025-06-11T03:11:05.254795Z","iopub.status.idle":"2025-06-11T03:11:05.264708Z","shell.execute_reply.started":"2025-06-11T03:11:05.254774Z","shell.execute_reply":"2025-06-11T03:11:05.264098Z"},"jupyter":{"source_hidden":true}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-06-11T03:11:05.265510Z","iopub.execute_input":"2025-06-11T03:11:05.265977Z","iopub.status.idle":"2025-06-11T03:11:05.280024Z","shell.execute_reply.started":"2025-06-11T03:11:05.265953Z","shell.execute_reply":"2025-06-11T03:11:05.279376Z"},"jupyter":{"source_hidden":true}}
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

def print_gpu_info():
    if torch.cuda.is_available():
        gpu_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        total_mem = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {total_mem:.2f} GB")
    else:
        print("No CUDA-compatible GPU found.")

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

# %% [code] {"execution":{"iopub.status.busy":"2025-06-11T03:14:09.912179Z","iopub.execute_input":"2025-06-11T03:14:09.912453Z","iopub.status.idle":"2025-06-11T03:14:09.929787Z","shell.execute_reply.started":"2025-06-11T03:14:09.912432Z","shell.execute_reply":"2025-06-11T03:14:09.929071Z"}}
# Processing cell
def process_image(image_path, model, preprocess, n_clusters, stride, version, device):
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

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
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
        print("overlay_ratio must be between 1 and 10. Using default value output5.")
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
            return process_image(image_path, model, preprocess, n_clusters, stride, model_name, version, device), output_prefix

        else:
            print(f"File {filename} not supported or is not a supported image format.")
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


# %% [code] {"execution":{"iopub.status.busy":"2025-06-11T03:16:46.006940Z","iopub.execute_input":"2025-06-11T03:16:46.007263Z","iopub.status.idle":"2025-06-11T03:16:46.022289Z","shell.execute_reply.started":"2025-06-11T03:16:46.007239Z","shell.execute_reply":"2025-06-11T03:16:46.021384Z"}}
# Visualization cell
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
    legend_path = os.path.join(output_dir, f"{output_prefix}_segmentation_map.png")
    plt.savefig(legend_path, bbox_inches='tight', pad_inches=0.1, dpi=200)
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
    results = run_processing(
        input_dir, output_dir, n_clusters, overlay_ratio, stride, model_name, filename, version
    )
    if filename:
        if results[0][0] is not None:
            (image_np, labels_resized), output_prefix = results
            generate_outputs(
                image_np, labels_resized, output_prefix, output_dir,
                n_clusters, overlay_ratio, stride, model_name,
                os.path.join(input_dir, filename), version
            )
    else:
        for (image_np, labels_resized), output_prefix in results:
            if image_np is not None:
                generate_outputs(
                    image_np, labels_resized, output_prefix, output_dir,
                    n_clusters, overlay_ratio, stride, model_name,
                    os.path.join(input_dir, output_prefix + ".jpg"), version
                )

# %% [code] {"execution":{"iopub.status.busy":"2025-06-11T03:16:51.633862Z","iopub.execute_input":"2025-06-11T03:16:51.634349Z","iopub.status.idle":"2025-06-11T03:17:27.445727Z","shell.execute_reply.started":"2025-06-11T03:16:51.634327Z","shell.execute_reply":"2025-06-11T03:17:27.444946Z"}}
import sys
sys.path.append("/kaggle/working/project/src")

# Run the segmentation
tree_seg(**config)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-11T03:17:34.122652Z","iopub.execute_input":"2025-06-11T03:17:34.123387Z","execution_failed":"2025-06-11T11:42:42.614Z"}}
from IPython.display import Image, display

# Paths to the output files
filename = config["filename"]
output_prefix = os.path.splitext(filename)[0]
legend_path = os.path.join(config["output_dir"], f"{output_prefix}_segmentation_legend.png")
overlay_path = os.path.join(config["output_dir"], f"{output_prefix}_overlay.png")
side_by_side_path = os.path.join(config["output_dir"], f"{output_prefix}_side_by_side.png")

# Display the segmentation map with legend
if os.path.exists(legend_path):
    print("Segmentation Map with Legend:")
    display(Image(filename=legend_path))

# Display the overlay
if os.path.exists(overlay_path):
    print("Overlay Image:")
    display(Image(filename=overlay_path))

# %% [code] {"execution":{"iopub.status.busy":"2025-06-11T03:18:29.148421Z","iopub.execute_input":"2025-06-11T03:18:29.148626Z","iopub.status.idle":"2025-06-11T03:18:29.299219Z","shell.execute_reply.started":"2025-06-11T03:18:29.148610Z","shell.execute_reply":"2025-06-11T03:18:29.298218Z"}}
# Display the side-by-side image
if os.path.exists(side_by_side_path):
    print("Original and Segmentation Side by Side:")
    display(Image(filename=side_by_side_path))
