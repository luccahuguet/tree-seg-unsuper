#!/usr/bin/env python3
"""
Test V3 Pipeline on Sample Data

Quick smoke test to verify V3 components work together.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from tree_seg.tree_focus.v3_pipeline import V3Pipeline, create_v3_preset


def create_synthetic_test_data():
    """Create synthetic test image and clusters."""
    # Create 512x512 test image
    height, width = 512, 512
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add "trees" as green circular regions
    tree_centers = [
        (150, 150),
        (350, 150),
        (150, 350),
        (350, 350),
        (250, 250),
    ]

    for idx, (cy, cx) in enumerate(tree_centers, start=1):
        # Draw green circle (tree canopy)
        cv2.circle(image, (cx, cy), 40, (50, 150, 50), -1)
        # Add some brown in center (trunk)
        cv2.circle(image, (cx, cy), 5, (80, 60, 40), -1)

    # Add non-tree regions (roads, buildings)
    cv2.rectangle(image, (0, 240), (100, 270), (120, 120, 120), -1)  # Road
    cv2.rectangle(image, (400, 400), (500, 500), (180, 150, 130), -1)  # Building

    # Create simple cluster labels (simulating V1.5 output)
    cluster_labels = np.zeros((height, width), dtype=np.int32)

    for idx, (cy, cx) in enumerate(tree_centers, start=1):
        # Create circular cluster
        y, x = np.ogrid[:height, :width]
        mask = (x - cx)**2 + (y - cy)**2 <= 40**2
        cluster_labels[mask] = idx

    # Add non-tree clusters
    cluster_labels[240:270, 0:100] = 6  # Road
    cluster_labels[400:500, 400:500] = 7  # Building

    return image, cluster_labels


def test_v3_pipeline():
    """Test V3 pipeline on synthetic data."""
    print("=" * 80)
    print("V3 Pipeline Smoke Test")
    print("=" * 80)

    # Create test data
    print("\n1. Creating synthetic test data...")
    image, cluster_labels = create_synthetic_test_data()
    print(f"   Image shape: {image.shape}")
    print(f"   Clusters: {len(np.unique(cluster_labels)) - 1}")

    # Initialize V3 pipeline
    print("\n2. Initializing V3 pipeline...")
    config = create_v3_preset('balanced')
    pipeline = V3Pipeline(config)
    print(f"   Config: {config.vegetation_method}, IoU≥{config.iou_threshold}")

    # Run V3
    print("\n3. Running V3 pipeline...")
    results = pipeline.process(image, cluster_labels)

    print("\n4. Results:")
    print(f"   Vegetation coverage: {results.vegetation_mask.mean():.1%}")
    print(f"   Tree mask coverage: {results.tree_mask.mean():.1%}")
    print(f"   Number of trees detected: {results.num_trees}")
    print(f"   Selected clusters: {len(results.selected_cluster_ids)}/{len(results.cluster_stats)}")

    # Print cluster stats
    print("\n5. Cluster Statistics:")
    for stat in results.cluster_stats:
        status = "✓ TREE" if stat['is_tree'] else "✗ NON-TREE"
        print(f"   Cluster {stat['cluster_id']}: {status} "
              f"(IoU={stat['iou']:.2f}, VegScore={stat['vegetation_score']:.2f}, "
              f"Area={stat['area_m2']:.1f}m²)")

    # Visualize
    print("\n6. Creating visualization...")
    visualize_v3_results(image, cluster_labels, results)

    print("\n" + "=" * 80)
    print("✓ V3 Pipeline test completed successfully!")
    print("=" * 80)

    return results


def visualize_v3_results(image, cluster_labels, results):
    """Visualize V3 results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # Input clusters
    axes[0, 1].imshow(cluster_labels, cmap='tab20')
    axes[0, 1].set_title(f"Input Clusters (V1.5)\n{len(np.unique(cluster_labels))-1} clusters")
    axes[0, 1].axis('off')

    # Vegetation mask
    axes[0, 2].imshow(results.vegetation_mask, cmap='Greens')
    axes[0, 2].set_title(f"Vegetation Mask\n{results.vegetation_mask.mean():.1%} coverage")
    axes[0, 2].axis('off')

    # Tree mask
    axes[1, 0].imshow(results.tree_mask, cmap='Greens')
    axes[1, 0].set_title(f"Tree Mask (Selected Clusters)\n{len(results.selected_cluster_ids)} clusters")
    axes[1, 0].axis('off')

    # Instance segmentation
    axes[1, 1].imshow(results.instance_labels, cmap='tab20')
    axes[1, 1].set_title(f"Tree Instances\n{results.num_trees} trees detected")
    axes[1, 1].axis('off')

    # Overlay
    axes[1, 2].imshow(image)
    axes[1, 2].imshow(results.instance_labels > 0, alpha=0.5, cmap='Greens')
    axes[1, 2].set_title("Overlay")
    axes[1, 2].axis('off')

    plt.suptitle("V3 Pipeline Test Results", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = Path("data/output/v3_test_results.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    results = test_v3_pipeline()
