# %%
# NOTE: This file is a Jupyter notebook exported as a Python file.
# It contains special cell markers (e.g., # %%) to preserve notebook cell boundaries.
# You can open and edit it as a notebook in Jupyter or Kaggle.
# When editing as a .py file, be careful to preserve these cell markers and cell order.
#
# To convert back to a notebook, use Jupyter or VSCode's 'Import Notebook' feature.
#
# This approach is used for easier version control and editing.
#

# %%
# Setup - Change to project directory and install dependencies
%cd /kaggle/working

# Remove the old project directory if it exists
!rm -rf /kaggle/working/project

# Clone the repository
!git clone https://github.com/luccahuguet/tree-seg-unsuper /kaggle/working/project

# Change into the project directory
%cd /kaggle/working/project

# %%
# Install dependencies
%pip uninstall -y torchaudio fastai
%pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
%pip install timm
%pip install xformers --index-url https://download.pytorch.org/whl/cu124

# %%
# Import the modular tree segmentation package
import sys
sys.path.append("/kaggle/working/project")

# Force reload to ensure we get the latest version
import importlib
try:
    import tree_seg
    importlib.reload(tree_seg)
    from tree_seg import tree_seg_with_auto_k, MODELS, print_gpu_info
except ImportError:
    # Fallback if module not found
    print("⚠️ Could not import tree_seg module. Please check the path.")

from tree_seg.utils.notebook_helpers import print_config_summary
from IPython.display import Image, display

# %%
# Setup confirmation
print("✅ Using modular tree segmentation package!")
print("📦 Functions are now organized in tree_seg/ modules")
print_gpu_info()

# %%
# Configuration - Edit these settings as needed
config = {
    "input_dir": "/kaggle/input/drone-10-best",
    "output_dir": "/kaggle/working/output",
    "model_name": MODELS["base"],           # Choose: "small", "base", "large", "giant"
    "filename": "DJI_20250127150117_0029_D.JPG",
    "version": "v1.5",
    "auto_k": True,                         # Automatic K selection (recommended)
    "k_range": (3, 10),                     # K range for auto selection
    "elbow_threshold": 3.0,                 # Elbow detection threshold
    # Tips for elbow_threshold values:
    # - 0.05-0.1: Sensitive detection, finds subtle elbows (more clusters)
    # - 0.1-0.2: Balanced detection, good for most cases (recommended) 
    # - 0.2-0.3: Conservative detection, finds only clear elbows (fewer clusters)
    # - 3.0: Very conservative (current default)
    # Common values: 0.1 (sensitive), 0.15 (balanced), 3.0 (conservative)
    "n_clusters": 6,                        # Only used if auto_k=False
    "overlay_ratio": 4,                     # Transparency: 1=opaque, 10=transparent
    "stride": 4,                            # Lower=higher resolution, slower
    "edge_width": 2,                        # Width of edge lines in edge overlay visualization
    "use_hatching": True,                    # Whether to add hatch patterns to regions (borders always shown) - now with transparent hatching
    "use_pca": False,                        # Whether to use PCA dimensionality reduction (default: False)
}

# %%
# Clear output folder before processing
import shutil
import os

if os.path.exists(config["output_dir"]):
    print(f"🗑️ Clearing output directory: {config['output_dir']}")
    shutil.rmtree(config["output_dir"])
os.makedirs(config["output_dir"], exist_ok=True)
print(f"✅ Output directory ready: {config['output_dir']}")

# %%
# Run segmentation
print_config_summary(config)

# Debug: Check function signature
import inspect
print(f"Function signature: {inspect.signature(tree_seg_with_auto_k)}")
print(f"Config keys: {list(config.keys())}")

tree_seg_with_auto_k(**config)

# %%
# Display results - Split into separate cells for reliability

# Get file paths using new config-based naming
import glob
output_dir = config["output_dir"]

# Find files using glob patterns (works with new config-based naming)
legend_files = glob.glob(os.path.join(output_dir, "*_segmentation_legend.png"))
edge_overlay_files = glob.glob(os.path.join(output_dir, "*_edge_overlay.png"))
side_by_side_files = glob.glob(os.path.join(output_dir, "*_side_by_side.png"))
elbow_files = glob.glob(os.path.join(output_dir, "*_elbow_analysis.png"))

# Use the most recent files (in case of multiple runs)
legend_path = max(legend_files, key=os.path.getmtime) if legend_files else None
edge_overlay_path = max(edge_overlay_files, key=os.path.getmtime) if edge_overlay_files else None
side_by_side_path = max(side_by_side_files, key=os.path.getmtime) if side_by_side_files else None
elbow_path = max(elbow_files, key=os.path.getmtime) if elbow_files else None

print("📁 Generated files:")
all_files = [legend_path, edge_overlay_path, side_by_side_path, elbow_path]
file_types = ["segmentation_legend", "edge_overlay", "side_by_side", "elbow_analysis"]

for path, file_type in zip(all_files, file_types):
    if path and os.path.exists(path):
        print(f"✅ {os.path.basename(path)}")
    else:
        print(f"❌ {file_type}.png - Not found")

# %%
# Display 1: Edge Overlay (Original + Boundaries)
if edge_overlay_path and os.path.exists(edge_overlay_path):
    print("🔳 Edge Overlay (Original + Boundaries):")
    display(Image(filename=edge_overlay_path))
else:
    print("❌ Edge overlay not found")

# %%
# Display 2: Side-by-Side Comparison
if side_by_side_path and os.path.exists(side_by_side_path):
    print("📊 Original and Segmentation Side by Side:")
    display(Image(filename=side_by_side_path))
else:
    print("❌ Side-by-side comparison not found")

# %%
# Display 3: Segmentation Legend
if legend_path and os.path.exists(legend_path):
    print("🎨 Segmentation Map with Legend:")
    display(Image(filename=legend_path))
else:
    print("❌ Segmentation legend not found")

# %%
# Display 4: K Selection Analysis (if auto_k was used)
if elbow_path and os.path.exists(elbow_path):
    print("📈 K Selection Analysis (Elbow Method):")
    display(Image(filename=elbow_path))
else:
    print("ℹ️ K selection analysis not found (auto_k may be disabled)")


