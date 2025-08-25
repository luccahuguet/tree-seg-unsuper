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
# Install dependencies for DINOv3
%pip uninstall -y torchaudio fastai
%pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
%pip install omegaconf ftfy regex torchmetrics  # DINOv3 dependencies
%pip install Pillow opencv-python  # Required for processing

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
    filename="DJI_20250127150117_0029_D.JPG",  # Your image filename (or None to process all images)
    
    # Model settings  
    model_name="base",      # "small", "base", "large", "giant", "mega" (DINOv3)
    version="v3",           # DINOv3 version
    stride=4,               # Balance of speed vs quality
    
    # Clustering
    auto_k=True,            # Let elbow method choose K
    elbow_threshold=0.15,   # DINOv3 optimized threshold (0.15 = 15%)
    k_range=(3, 8),         # Narrower range for trees
    
    # Visualization
    overlay_ratio=4,        # Good transparency
    edge_width=2,           # Clear borders
    use_hatching=True,      # Distinguish regions
    use_pca=False,          # Keep full feature space
    
    # Web optimization (enabled by default)
    web_optimize=True,      # Auto-compress 7MB → 1-2MB for GitHub Pages  
    web_quality=85,         # JPEG quality (85 = good balance)
    web_max_width=1200      # Max width for web display
)

# Create segmentation instance
segmenter = TreeSegmentation(config)

# Process the image using config settings
# To use a different image, change the 'filename' in the config above
import os

if config.filename:
    # Process single image
    image_path = os.path.join(config.input_dir, config.filename)
    print(f"🖼️ Processing single image: {image_path}")
    results, paths = segmenter.process_and_visualize(image_path)
    print(f"🎯 Single image processing complete!")
    print(f"📊 Stats: {results.processing_stats}")
else:
    # Process all images in directory
    print(f"🖼️ Processing all images in: {config.input_dir}")
    results_list = segmenter.process_directory()
    print(f"🎯 Processed {len(results_list)} images!")
    if results_list:
        results, paths = results_list[0]  # Use first result for display

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
print(f"📁 PNG results saved to: output/png/")
print(f"🌐 Web-optimized results saved to: output/web/")
print(f"🌐 Web optimization: {'enabled' if config.web_optimize else 'disabled'}")
