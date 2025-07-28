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
    elbow_threshold=0.15,   # Slightly conservative
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
    
    # Display edge overlay
    if latest_outputs.edge_overlay:
        print("\n🔳 Edge Overlay (Original + Boundaries):")
        display(Image(filename=latest_outputs.edge_overlay))
    
    # Display side-by-side
    if latest_outputs.side_by_side:
        print("\n📊 Side-by-Side Comparison:")
        display(Image(filename=latest_outputs.side_by_side))
    
    # Display segmentation legend
    if latest_outputs.segmentation_legend:
        print("\n🎨 Segmentation Map with Legend:")
        display(Image(filename=latest_outputs.segmentation_legend))
    
    # Display elbow analysis if available
    if latest_outputs.elbow_analysis:
        print("\n📈 K Selection Analysis (Elbow Method):")
        display(Image(filename=latest_outputs.elbow_analysis))
        
else:
    print("❌ No output files found")

# %%
# Batch Processing - Process multiple images
print("📁 Batch processing example:")

# Process all images in directory
batch_results = segmenter.process_directory()

print(f"✅ Processed {len(batch_results)} images")
for result, paths in batch_results:
    filename = os.path.basename(result.image_path)
    print(f"  • {filename}: K={result.n_clusters_used}")

# %%
# Cleanup and file management
print("🧹 File management:")

# List all outputs
all_outputs = segmenter.output_manager.list_all_outputs()
print(f"📄 Total output files: {len(all_outputs)}")

# Show latest files with their names (showcasing config-based naming)
if all_outputs:
    print("\n📋 Recent files (showing config-based naming):")
    for file_path in all_outputs[:5]:  # Show latest 5
        filename = os.path.basename(file_path)
        size_mb = os.path.getsize(file_path) / (1024*1024)
        print(f"  • {filename} ({size_mb:.1f}MB)")

# Clean up old files (keep latest 3 of each type)
segmenter.cleanup_old_outputs(keep_latest=3)

# %%
# Configuration validation and tips
print("⚙️ Configuration tips:")

try:
    # This will fail validation
    bad_config = Config(overlay_ratio=15, stride=20)
    bad_config.validate()
except ValueError as e:
    print(f"❌ Invalid config: {e}")

# Show model mapping
config = Config()
print(f"🤖 Model mapping examples:")
for model in ["small", "base", "large", "giant"]:
    config.model_name = model
    print(f"  • {model} → {config.model_display_name}")

print("\n🎉 Modern Tree Segmentation Demo Complete!")
print("💡 Key improvements:")
print("  • Clean Config dataclass with validation")
print("  • Type-safe Results objects")
print("  • Automatic file management")
print("  • Intelligent filename generation")
print("  • Simple API for both quick and advanced usage")
print("  • Automatic web optimization (7MB → 1-2MB)")
print("  • Ready for GitHub Pages deployment")