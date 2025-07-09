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

from tree_seg.utils.notebook_helpers import display_segmentation_results, print_config_summary

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
    "elbow_threshold": 3.0,                 # Sensitivity for elbow detection (lower = more sensitive)
    "n_clusters": 6,                        # Only used if auto_k=False
    "overlay_ratio": 4,                     # Transparency: 1=opaque, 10=transparent
    "stride": 4,                            # Lower=higher resolution, slower
    "edge_width": 2,                        # Width of edge lines in edge overlay visualization
    "use_hatching": False,                   # Whether to add hatch patterns to regions (borders always shown) - now default False
    "generate_overlay": False,               # Whether to generate the colored overlay visualization
}

# %%
# Run segmentation
print_config_summary(config)

# Debug: Check function signature
import inspect
print(f"Function signature: {inspect.signature(tree_seg_with_auto_k)}")
print(f"Config keys: {list(config.keys())}")

tree_seg_with_auto_k(**config)

# %%
# Display results
display_segmentation_results(config)


