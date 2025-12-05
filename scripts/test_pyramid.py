#!/usr/bin/env python3
"""Test pyramid features on FORTRESS CFB003."""

from pathlib import Path
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from tree_seg.core.types import Config
from tree_seg.api import TreeSegmentation

load_dotenv()

# Create config with pyramid features
config = Config(
    version="v1.5",
    clustering_method="kmeans",
    refine="slic",
    model_name="base",
    stride=4,
    auto_k=True,
    elbow_threshold=5.0,
    use_tiling=False,  # Pyramid disables tiling automatically
    use_pyramid=True,
    pyramid_scales=(0.5, 1.0, 2.0),
    pyramid_aggregation="concat",
    verbose=True,
)

print("Testing pyramid features...")
print(f"Scales: {config.pyramid_scales}")
print(f"Aggregation: {config.pyramid_aggregation}\n")

# Initialize segmenter
segmenter = TreeSegmentation(config)

# Get CFB003 image
dataset_path = Path("data/fortress_processed")
image_path = dataset_path / "images" / "CFB003.tif"

print(f"Loading image: {image_path.name}")

# Load image as numpy array
Image.MAX_IMAGE_PIXELS = None
img = Image.open(image_path).convert("RGB")
image_np = np.array(img)

print(f"Image shape: {image_np.shape}")
print("Processing with pyramid features...\n")

try:
    results = segmenter.segment_image(image_np)
    print("\n✅ SUCCESS!")
    print(f"Clusters used: {results.n_clusters_used}")
    print(f"Labels shape: {results.labels_resized.shape}")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
