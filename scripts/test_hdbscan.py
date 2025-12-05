#!/usr/bin/env python3
"""Test HDBSCAN clustering on a single image."""

from pathlib import Path
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from tree_seg.core.types import Config
from tree_seg.api import TreeSegmentation

load_dotenv()

# Create a config with HDBSCAN
config = Config(
    version="v1.5",
    clustering_method="hdbscan",
    refine=None,
    model_name="base",
    stride=4,
    auto_k=True,
    elbow_threshold=5.0,
    use_tiling=True,
    verbose=True,
)

print("Testing HDBSCAN clustering...")
print(f"Clustering method: {config.clustering_method}")

# Initialize segmenter
segmenter = TreeSegmentation(config)

# Get CFB003 image
dataset_path = Path("data/fortress_processed")
image_path = dataset_path / "images" / "CFB003.tif"

print(f"\nLoading image: {image_path.name}")

# Load image as numpy array
Image.MAX_IMAGE_PIXELS = None
img = Image.open(image_path).convert("RGB")
image_np = np.array(img)

print(f"Image shape: {image_np.shape}")
print("Testing segment_image() with numpy array...")

try:
    # Process the numpy array
    results = segmenter.segment_image(image_np)
    print("\n✅ SUCCESS!")
    print(f"Clusters used: {results.n_clusters_used}")
    print(f"Labels shape: {results.labels_resized.shape}")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
