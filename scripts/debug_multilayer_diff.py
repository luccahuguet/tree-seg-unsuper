#!/usr/bin/env python3
"""Debug: verify multi-layer features are actually different."""

import numpy as np
from tree_seg.core.types import Config
from tree_seg.api import TreeSegmentation
from tree_seg.evaluation.datasets import FortressDataset

# Load one image
dataset = FortressDataset("data/fortress_processed")
image, mask, image_id = dataset[0]
print(f"Image: {image_id}, shape: {image.shape}")

# Test 1: Baseline
print("\n=== BASELINE (single layer) ===")
config1 = Config(
    use_multi_layer=False,
    verbose=True,
)
seg1 = TreeSegmentation(config1)
result1 = seg1.segment_image(image)
print(f"Baseline clusters: {result1.n_clusters_used}")
print(f"Baseline labels unique: {np.unique(result1.labels_resized)}")

# Test 2: Multi-layer
print("\n=== MULTI-LAYER (concat) ===")
config2 = Config(
    use_multi_layer=True,
    layer_indices=(3, 6, 9, 12),
    feature_aggregation="concat",
    pca_dim=512,
    verbose=True,
)
seg2 = TreeSegmentation(config2)
result2 = seg2.segment_image(image)
print(f"Multi-layer clusters: {result2.n_clusters_used}")
print(f"Multi-layer labels unique: {np.unique(result2.labels_resized)}")

# Compare
print("\n=== COMPARISON ===")
print(f"Labels are identical: {np.array_equal(result1.labels_resized, result2.labels_resized)}")
print(f"Labels difference sum: {np.sum(result1.labels_resized != result2.labels_resized)}")
