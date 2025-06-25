# %%
# NOTE: This file is a Jupyter notebook exported as a Python file.
# It contains special cell markers (e.g., # %%) to preserve notebook cell boundaries.
# You can open and edit it as a notebook in Jupyter or Kaggle.
# When editing as a .py file, be careful to preserve these cell markers and cell order.
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
# Import the modular tree segmentation package
import sys
import os
import numpy as np
import torch
sys.path.append("/kaggle/working/project")

from tree_seg import tree_seg_with_auto_k, MODELS, print_gpu_info

# %%
# Setup cell - Now using modular imports
print("✅ Using modular tree segmentation package!")
print("📦 Functions are now organized in tree_seg/ modules")
print_gpu_info()


# %%
# All functions are now in the modular structure!
print("📊 Elbow method analysis: tree_seg.analysis.elbow_method")
print("🔧 Model utilities: tree_seg.models")
print("🎨 Visualization: tree_seg.visualization")
print("🏃 Core processing: tree_seg.core")

# All these functions are now available from the modular structure!

# %%
# MAIN CONFIG - Edit this to change settings
import sys
import os
import numpy as np
import torch
sys.path.append("/kaggle/working/project/src")

# Import additional functions needed by the local function
from tree_seg.models import initialize_model, get_preprocess
from tree_seg.core.segmentation import process_image
from tree_seg.visualization import generate_outputs

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
    "edge_width": 6,                        # Width of edge lines in edge overlay visualization
    "min_region_size": 100,                 # Remove regions smaller than this many pixels (0 to disable)
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
    elbow_threshold=3.0,
    edge_width=6,
    min_region_size=100
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
                    image_path, version, edge_width
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
                        image_path, version, edge_width
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
edge_overlay_path = os.path.join(output_dir, f"{output_prefix}_edge_overlay.png")
side_by_side_path = os.path.join(output_dir, f"{output_prefix}_side_by_side.png")
elbow_path = os.path.join(output_dir, f"{output_prefix}_elbow_analysis.png")

# Display the overlay
if os.path.exists(overlay_path):
    print("🖼️ Overlay Image:")
    display(Image(filename=overlay_path))

# Display the NEW edge overlay
if os.path.exists(edge_overlay_path):
    print("🔳 Edge Overlay (Original + Boundaries):")
    display(Image(filename=edge_overlay_path))

# Display the side-by-side comparison
if os.path.exists(side_by_side_path):
    print("📊 Original and Segmentation Side by Side:")
    display(Image(filename=side_by_side_path))

# Display the K selection analysis
if os.path.exists(elbow_path):
    print("📈 K Selection Analysis (Elbow Method):")
    display(Image(filename=elbow_path))


