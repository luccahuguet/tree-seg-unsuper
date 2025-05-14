import os
import torch
import timm
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2
from torchvision import transforms
from matplotlib import colormaps

# --- Paths ---
input_dir = "input"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model ---
model_name = "vit_base_patch14_dinov2.lvd142m"
model = timm.create_model(model_name, pretrained=True)
model.eval()
model = model.to(device)

# --- Preprocessing ---
expected_size = model.patch_embed.img_size
patch_size = model.patch_embed.patch_size[0]
print(f"Model expects input image size: {expected_size}, patch size: {patch_size}")

preprocess = transforms.Compose([
    transforms.Resize(expected_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image_path, output_prefix, n_clusters=5, alpha=0.5):
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        # Preprocess
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        # Extract patch features
        with torch.no_grad():
            output = model.forward_features(image_tensor)
            features = output[:, 1:].squeeze(0).cpu().numpy()  # (num_patches, dim)

        # Patch grid calculation
        num_patches = features.shape[0]
        patch_grid_h = expected_size[0] // patch_size
        patch_grid_w = expected_size[1] // patch_size
        assert patch_grid_h * patch_grid_w == num_patches, \
            f"Patch grid mismatch: {patch_grid_h}x{patch_grid_w} != {num_patches}"

        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        # Remap labels to contiguous range for colormap
        unique_labels, labels_contig = np.unique(labels, return_inverse=True)
        labels_patch_grid = labels_contig.reshape(patch_grid_h, patch_grid_w)

        # Resize labels grid to original image size
        labels_resized = cv2.resize(
            labels_patch_grid.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        # Colormap selection
        if n_clusters <= 10:
            cmap = colormaps['tab10']
        elif n_clusters <= 20:
            cmap = colormaps['tab20']
        else:
            cmap = colormaps['gist_ncar']

        # Map each label to a color
        segmentation_rgb = cmap(labels_resized / (n_clusters - 1))[:, :, :3]
        segmentation_rgb = (segmentation_rgb * 255).astype(np.uint8)

        # Overlay segmentation on the original image
        overlay = (alpha * image_np + (1 - alpha) * segmentation_rgb).astype(np.uint8)

        # Save outputs
        segmentation_path = os.path.join(output_dir, f"{output_prefix}_segmentation.png")
        overlay_path = os.path.join(output_dir, f"{output_prefix}_overlay.png")
        Image.fromarray(segmentation_rgb).save(segmentation_path)
        Image.fromarray(overlay).save(overlay_path)
        print(f"Saved: {segmentation_path} and {overlay_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# --- Batch Processing ---
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
        image_path = os.path.join(input_dir, filename)
        output_prefix = os.path.splitext(filename)[0]
        print(f"Processing {filename} ...")
        process_image(image_path, output_prefix, n_clusters=5, alpha=0.5)
