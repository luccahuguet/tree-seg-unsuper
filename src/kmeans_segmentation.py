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


def run_kmeans_segmentation(
    input_dir="input",
    output_dir="output",
    n_clusters=5,
    overlay_ratio=5,
    stride=4,
    model_name="dinov2_vits14",
):
    print_gpu_info()
    os.makedirs(output_dir, exist_ok=True)

    overlay_ratio = int(overlay_ratio)
    if overlay_ratio < 1 or overlay_ratio > 10:
        print("overlay_ratio must be between 1 and 10. Using default value 5.")
        overlay_ratio = 5
    alpha = (10 - overlay_ratio) / 10.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HighResDV2(model_name, stride=stride, dtype=torch.float16).to(device)
    model.eval()

    # Use only identity transform for simplicity
    model.set_transforms([lambda x: x], [lambda x: x])

    preprocess = Compose(
        [
            Resize((518, 518)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def process_image_with_legend(image_path, output_prefix):
        try:
            print(f"\n--- Processing {image_path} ---")
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            h, w = image_np.shape[:2]
            print(f"Original image size: {w}x{h}")

            image_tensor = preprocess(image).to(device)
            print(f"Preprocessed tensor shape: {image_tensor.shape}")

            with torch.no_grad():
                # --- Extract both patch and attention features ---
                features_out = model.forward_sequential(image_tensor, attn_choice="o")
                print(f"features_out type: {type(features_out)}")
                if isinstance(features_out, dict):
                    patch_features = features_out["x_norm_patchtokens"]  # (T, H*W, C)
                    attn_features = features_out.get(
                        "x_patchattn", None
                    )  # (T, H*W, C_attn) or None
                    print(f"patch_features shape: {patch_features.shape}")
                    if attn_features is not None:
                        print(f"attn_features shape: {attn_features.shape}")
                    # Average over transforms if present
                    patch_features = patch_features.mean(dim=0).squeeze()  # (H*W, C)
                    if attn_features is not None:
                        attn_features = attn_features.mean(
                            dim=0
                        ).squeeze()  # (H*W, C_attn)
                else:
                    # Fallback: treat as before
                    print(
                        f"features_out shape: {getattr(features_out, 'shape', 'N/A')}"
                    )
                    if hasattr(features_out, "dim") and features_out.dim() == 4:
                        features = features_out.mean(dim=0)  # (C, H, W)
                    else:
                        features = features_out  # (C, H, W)
                    H = W = 518 // stride
                    features = features.unsqueeze(0)
                    features = torch.nn.functional.interpolate(
                        features,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    features = features.permute(1, 2, 0)  # (H, W, C)
                    patch_features = features
                    attn_features = None

            # --- Reshape and concatenate features ---
            if patch_features.dim() == 2:  # (H*W, C)
                n_patches = patch_features.shape[0]
                H = W = int(np.sqrt(n_patches))
                if H * W != n_patches:
                    raise ValueError(f"Cannot infer H, W from n_patches={n_patches}")
                patch_features = patch_features.view(H, W, -1)
                if attn_features is not None and attn_features.dim() == 2:
                    attn_features = attn_features.view(H, W, -1)
            else:
                H, W = patch_features.shape[:2]
            print(f"patch_features reshaped: {patch_features.shape}")
            if attn_features is not None:
                print(f"attn_features reshaped: {attn_features.shape}")

            # --- Combine features ---
            if attn_features is not None:
                features_np = np.concatenate(
                    [patch_features.cpu().numpy(), attn_features.cpu().numpy()],
                    axis=-1,
                )
                print(f"Combined features shape: {features_np.shape}")
            else:
                features_np = patch_features.cpu().numpy()
                print(f"Patch-only features shape: {features_np.shape}")

            # --- Flatten for KMeans/PCA ---
            features_flat = features_np.reshape(-1, features_np.shape[-1])
            print(f"Flattened features shape: {features_flat.shape}")

            # --- PCA on flat features ---
            if features_flat.shape[-1] > 128:
                print("Running PCA on flat features...")
                features_flat_tensor = torch.tensor(features_flat, dtype=torch.float32)
                mean = features_flat_tensor.mean(dim=0)
                features_flat_centered = features_flat_tensor - mean
                U, S, V = torch.pca_lowrank(features_flat_centered, q=128, center=False)
                features_flat = (features_flat_centered @ V[:, :128]).numpy()
                print(f"PCA-reduced features shape: {features_flat.shape}")

            # --- KMeans clustering ---
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            labels = kmeans.fit_predict(features_flat)
            print(f"labels shape after kmeans: {labels.shape}")
            labels = labels.reshape(H, W)
            print(f"Labels shape after reshape: {labels.shape}")

            # --- Resize to original image size ---
            labels_resized = cv2.resize(
                labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
            print(f"labels_resized shape: {labels_resized.shape}")

            # --- Colormap and legend ---
            if n_clusters <= 10:
                cmap = plt.get_cmap("tab10", n_clusters)
            elif n_clusters <= 20:
                cmap = plt.get_cmap("tab20", n_clusters)
            else:
                cmap = plt.get_cmap("gist_ncar", n_clusters)

            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(labels_resized, cmap=cmap, vmin=0, vmax=n_clusters - 1)
            cbar = plt.colorbar(im, ax=ax, ticks=range(n_clusters))
            cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
            ax.axis("off")
            plt.tight_layout()
            legend_path = os.path.join(
                output_dir, f"{output_prefix}_segmentation_legend.png"
            )
            plt.savefig(legend_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
            plt.close()
            print(f"Saved segmentation with legend: {legend_path}")

            # --- Overlay ---
            norm = colors.Normalize(vmin=0, vmax=n_clusters - 1)
            segmentation_rgb = cmap(norm(labels_resized))[:, :, :3]
            segmentation_rgb = (segmentation_rgb * 255).astype(np.uint8)
            overlay = (alpha * image_np + (1 - alpha) * segmentation_rgb).astype(
                np.uint8
            )
            overlay_path = os.path.join(output_dir, f"{output_prefix}_overlay.png")
            Image.fromarray(overlay).save(overlay_path)
            print(f"Saved overlay: {overlay_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            traceback.print_exc()

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            image_path = os.path.join(input_dir, filename)
            output_prefix = os.path.splitext(filename)[0]
            print(f"Processing {filename} ...")
            process_image_with_legend(image_path, output_prefix)
