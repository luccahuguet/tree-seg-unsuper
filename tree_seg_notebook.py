# %%
# Modern Tree Segmentation Notebook
# Clean, simple API with dataclasses and modern Python patterns

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
%pip install Pillow  # Required for web optimization feature

# %%
# Import the modern API
import sys
sys.path.append("/kaggle/working/project")

from tree_seg import TreeSegmentation, segment_trees, Config, print_gpu_info

print("✅ Modern Tree Segmentation API loaded!")
print_gpu_info()

# %%
# Advanced Usage - Full control with Config class
print("🎯 Advanced usage with Config class:")

config = Config(
    # Input/Output
    input_dir="/kaggle/input/drone-10-best",
    output_dir="/kaggle/working/output",
    filename="DJI_20250127150117_0029_D.JPG",
    
    # Model settings  
    model_name="base",      # or "small", "large", "giant"
    version="v1.5",         # Current version
    stride=4,               # Balance of speed vs quality
    
    # Clustering
    auto_k=True,            # Let elbow method choose K
    elbow_threshold=0.04,   # Sensitive threshold (4%)
    k_range=(3, 8),         # Narrower range for trees
    
    # Visualization
    overlay_ratio=4,        # Good transparency
    edge_width=2,           # Clear borders
    use_hatching=True,      # Distinguish regions
    use_pca=False,          # Keep full feature space
    
    # Web optimization (NEW!)
    web_optimize=True,      # Auto-compress 7MB → 1-2MB for GitHub Pages
    web_quality=85,         # JPEG quality (85 = good balance)
    web_max_width=1200      # Max width for web display
)

# Create segmentation instance
segmenter = TreeSegmentation(config)

# Process the image
results, paths = segmenter.process_and_visualize(
    "/kaggle/input/drone-10-best/DJI_20250127150117_0029_D.JPG"
)

print(f"🎯 Advanced processing complete!")
print(f"📊 Stats: {results.processing_stats}")

# %%
# Display results using modern OutputManager
from IPython.display import Image, display
import os

# Find latest outputs automatically
latest_outputs = segmenter.find_latest_outputs()

if latest_outputs:
    print("🖼️ Latest Segmentation Results:")
else:
    print("❌ No output files found")

# %%
# Display edge overlay
if latest_outputs and latest_outputs.edge_overlay:
    print("🔳 Edge Overlay (Original + Boundaries):")
    display(Image(filename=latest_outputs.edge_overlay))

# %%
# Display side-by-side comparison
if latest_outputs and latest_outputs.side_by_side:
    print("📊 Side-by-Side Comparison:")
    display(Image(filename=latest_outputs.side_by_side))

# %%
# Display segmentation legend
if latest_outputs and latest_outputs.segmentation_legend:
    print("🎨 Segmentation Map with Legend:")
    display(Image(filename=latest_outputs.segmentation_legend))

# %%
# Display elbow analysis
if latest_outputs and latest_outputs.elbow_analysis:
    print("📈 K Selection Analysis (Elbow Method):")
    display(Image(filename=latest_outputs.elbow_analysis))

# %%
# Summary
print("🎉 Tree Segmentation Complete!")
print(f"📁 Results saved with config-based naming")
print(f"🌐 Web optimization: {'enabled' if config.web_optimize else 'disabled'}")