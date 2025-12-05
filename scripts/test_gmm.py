#!/usr/bin/env python3
"""Test GMM clustering on a single image using numpy array (like benchmark)."""

from pathlib import Path
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from tree_seg.core.types import Config
from tree_seg.api import TreeSegmentation

load_dotenv()

# Create a config with GMM
config = Config(
    version="v1.5",
    clustering_method="gmm",
    refine=None,
    model_name="base",
    stride=4,
    auto_k=True,
    elbow_threshold=5.0,
    use_tiling=True,
    verbose=True,
)

print("Testing GMM clustering...")
print(f"Clustering method: {config.clustering_method}")

# Initialize segmenter
segmenter = TreeSegmentation(config)

# Get CFB003 image (known to work in benchmark)
dataset_path = Path("data/fortress_processed")
image_path = dataset_path / "images" / "CFB003.tif"

print(f"\nLoading image: {image_path.name}")

# Load image as numpy array (like the benchmark does)
# Increase PIL limit to handle large FORTRESS images
Image.MAX_IMAGE_PIXELS = None
img = Image.open(image_path).convert("RGB")
image_np = np.array(img)

print(f"Image shape: {image_np.shape}")
print("Testing segment_image() with numpy array...")

try:
    # Process the numpy array (this is what benchmark does)
    results = segmenter.segment_image(image_np)
    print("\n✅ SUCCESS!")
    print(f"Clusters used: {results.n_clusters_used}")
    print(f"Labels shape: {results.labels_resized.shape}")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
