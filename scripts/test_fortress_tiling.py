#!/usr/bin/env python3
"""
Test tiling implementation on a real FORTRESS image.
"""

import time
from tree_seg import Config, TreeSegmentation

# Configuration with tiling enabled (verbose to see progress)
config = Config(
    model_name="base",
    use_tiling=True,
    tile_size=2048,
    tile_overlap=256,
    auto_k=True,
    elbow_threshold=10.0,
    refine="slic",
    pipeline="v1_5",  # Just clustering, no vegetation filter for this test
    verbose=True,      # Enable verbose output
    metrics=True,      # Collect performance metrics
)

print("=" * 70)
print("Testing Tiling on FORTRESS CFB003 (9372×9372)")
print("=" * 70)
print("\nConfiguration:")
print(f"  Model: {config.model_display_name}")
print(f"  Tiling: {config.use_tiling}")
print(f"  Tile size: {config.tile_size}×{config.tile_size}")
print(f"  Overlap: {config.tile_overlap}px")
print(f"  Auto K: {config.auto_k}")
print()

# Initialize segmenter
segmenter = TreeSegmentation(config)

# Process the image
image_path = "data/fortress_processed/images/CFB003.tif"

print(f"Processing: {image_path}")
print()

start_time = time.time()
results = segmenter.process_single_image(image_path)
elapsed = time.time() - start_time

print()
print("=" * 70)
print("Results")
print("=" * 70)
print(f"Processing time: {elapsed:.1f}s")
print(f"Output shape: {results.labels_resized.shape}")
print(f"Number of clusters: {results.n_clusters_used}")
print(f"Auto-selected K: {results.n_clusters_requested}")
print()

# Show some metrics if available
if results.processing_stats:
    stats = results.processing_stats
    if 'grid_H' in stats and 'grid_W' in stats:
        print(f"Feature grid: {stats['grid_H']}×{stats['grid_W']}")
    if 'n_features' in stats:
        print(f"Feature dimension: {stats['n_features']}")
    if 'time_total_s' in stats:
        print(f"Total time: {stats['time_total_s']:.1f}s")

print()
print("✅ Tiling test completed successfully!")
