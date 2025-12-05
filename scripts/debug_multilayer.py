#!/usr/bin/env python3
"""Simple test to debug multi-layer feature extraction."""

import traceback
from tree_seg.core.types import Config
from tree_seg.api import TreeSegmentation
import numpy as np

# Create a simple test image
test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

print("Testing baseline (single layer)...")
try:
    config = Config(
        use_multi_layer=False,
        verbose=True,
    )
    config.validate()
    segmenter = TreeSegmentation(config)
    result = segmenter.segment_image(test_image)
    print(f"✅ Baseline works! Clusters: {result.n_clusters_used}")
except Exception as e:
    print(f"❌ Baseline failed: {e}")
    traceback.print_exc()

print("\n" + "="*60 + "\n")

print("Testing multi-layer (concat)...")
try:
    config = Config(
        use_multi_layer=True,
        layer_indices=(3, 6, 9, 12),
        feature_aggregation="concat",
        pca_dim=128,
        verbose=True,
    )
    config.validate()
    segmenter = TreeSegmentation(config)
    result = segmenter.segment_image(test_image)
    print(f"✅ Multi-layer works! Clusters: {result.n_clusters_used}")
except Exception as e:
    print(f"❌ Multi-layer failed: {e}")
    traceback.print_exc()
