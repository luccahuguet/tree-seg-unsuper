import os
import torch
import timm
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2
from torchvision import transforms
import matplotlib

# --- Paths ---
input_dir = "input"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Set your input image filename here
input_image_filename = "input/aerial.jpg" 
image_path = os.path.join(input_dir, input_image_filename)

# Load DINOv2 model
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
model.eval()

# --- Get expected image size from the model ---
expected_size = model.patch_embed.img_size
print(f"Model expects input image size: {expected_size}")

# Load image
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize(expected_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_tensor = preprocess(image).unsqueeze(0)

# Extract patch features
with torch.no_grad():
    output = model.forward_features(image_tensor)
    features = output[:, 1:].squeeze(0).numpy()  # Shape: (num_patches, dim)

print(f"Shape of features for clustering: {features.shape}")

# K-Means on patch features
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(features)

# Reshape labels back to the patch grid size
patch_grid_h = image_tensor.shape[-2] // model.patch_embed.patch_size[0]
patch_grid_w = image_tensor.shape[-1] // model.patch_embed.patch_size[1]
labels_patch_grid = labels.reshape(patch_grid_h, patch_grid_w)
print(f"Shape of patch labels grid: {labels_patch_grid.shape}")

# Resize labels grid back to original image size
h, w = image_np.shape[:2]
labels_resized = cv2.resize(
    labels_patch_grid.astype(np.uint8),
    (w, h),
    interpolation=cv2.INTER_NEAREST
)
print(f"Shape of resized labels: {labels_resized.shape}")

# --- Visualization and Saving ---

# Assign a color to each cluster label
num_clusters = np.max(labels_resized) + 1
if num_clusters <= 10:
    colormap = matplotlib.colormaps['tab10']
elif num_clusters <= 20:
    colormap = matplotlib.colormaps['tab20']
else:
    colormap = matplotlib.colormaps['gist_ncar']

# Map each label to a color
segmentation_rgb = colormap(labels_resized / (num_clusters - 1))[:, :, :3]
segmentation_rgb = (segmentation_rgb * 255).astype(np.uint8)

# Overlay segmentation on the original image
alpha = 0.5  # transparency for overlay
overlay = (alpha * image_np + (1 - alpha) * segmentation_rgb).astype(np.uint8)

# Save the segmentation and overlay images
segmentation_path = os.path.join(output_dir, "segmentation.png")
overlay_path = os.path.join(output_dir, "overlay.png")
Image.fromarray(segmentation_rgb).save(segmentation_path)
Image.fromarray(overlay).save(overlay_path)
print(f"Saved {segmentation_path} and {overlay_path}")
