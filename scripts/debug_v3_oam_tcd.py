#!/usr/bin/env python3
"""
Debug V3 on OAM-TCD Sample

Runs V3 on a single OAM-TCD image with full debug output and visualizations.
"""

import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datasets import load_from_disk
from tree_seg import TreeSegmentation, Config

def debug_v3_single_sample(
    dataset_path: str = "data/oam_tcd",
    sample_idx: int = 0,
    preset: str = "permissive"  # Start with most permissive
):
    """
    Debug V3 on a single OAM-TCD sample.

    Args:
        dataset_path: Path to OAM-TCD dataset
        sample_idx: Sample index to debug
        preset: V3 preset to test
    """
    print("=" * 80)
    print("V3 Debug on Single OAM-TCD Sample")
    print("=" * 80)

    # Load test set
    test_path = Path(dataset_path) / "test"
    print(f"\nLoading test set from {test_path}...")
    test_data = load_from_disk(str(test_path))

    sample = test_data[sample_idx]
    image_id = sample['image_id']

    # Parse COCO annotations
    coco_anns = json.loads(sample['coco_annotations']) if sample['coco_annotations'] else []

    print(f"Sample {sample_idx}: {image_id}")
    print(f"Ground truth trees: {len(coco_anns)}")

    # Get image
    image_np = np.array(sample['image'])
    print(f"Image shape: {image_np.shape}")
    print(f"Image dtype: {image_np.dtype}")
    print(f"Image range: [{image_np.min()}, {image_np.max()}]")

    # Save debug directory
    debug_dir = Path("data/output/v3_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Save original image
    orig_path = debug_dir / f"{image_id}_original.jpg"
    cv2.imwrite(str(orig_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    print(f"\nSaved original to: {orig_path}")

    # Test vegetation index directly
    print("\n" + "=" * 80)
    print("Testing Vegetation Index")
    print("=" * 80)

    from tree_seg.tree_focus.vegetation_indices import (
        excess_green_index,
        create_vegetation_mask
    )

    exg = excess_green_index(image_np)
    print(f"ExG range: [{exg.min():.3f}, {exg.max():.3f}]")
    print(f"ExG mean: {exg.mean():.3f}")
    print(f"ExG std: {exg.std():.3f}")

    # Try multiple thresholds
    for threshold in [0.0, 0.05, 0.1, 0.15, 0.2]:
        veg_mask = create_vegetation_mask(image_np, method="exg", threshold=threshold)
        veg_pixels = veg_mask.sum()
        veg_percent = 100.0 * veg_pixels / veg_mask.size
        print(f"ExG threshold {threshold:.2f}: {veg_pixels:,} pixels ({veg_percent:.1f}%)")

    # Visualize ExG
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    im = axes[1].imshow(exg, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
    axes[1].set_title("ExG Index")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    veg_mask = create_vegetation_mask(image_np, method="exg", threshold=0.1)
    axes[2].imshow(veg_mask, cmap='gray')
    axes[2].set_title(f"Vegetation Mask (threshold=0.1)\n{veg_mask.sum():,} pixels")
    axes[2].axis('off')

    exg_path = debug_dir / f"{image_id}_exg.png"
    plt.tight_layout()
    plt.savefig(exg_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ExG visualization to: {exg_path}")

    # Run V1.5 baseline first
    print("\n" + "=" * 80)
    print("Running V1.5 Baseline")
    print("=" * 80)

    config_v1_5 = Config(
        pipeline="v1_5",
        auto_k=True,
        elbow_threshold=10.0,
        verbose=True
    )
    seg_v1_5 = TreeSegmentation(config_v1_5)

    # Save temp image
    temp_path = debug_dir / f"temp_{image_id}.jpg"
    cv2.imwrite(str(temp_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    results_v1_5 = seg_v1_5.process_single_image(str(temp_path))
    print(f"V1.5 clusters: {results_v1_5.n_clusters_used}")
    print(f"V1.5 labels shape: {results_v1_5.labels_resized.shape}")
    print(f"V1.5 unique labels: {np.unique(results_v1_5.labels_resized)}")

    # Visualize V1.5 clusters
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(results_v1_5.labels_resized, cmap='tab20')
    ax.set_title(f"V1.5 Clusters (K={results_v1_5.n_clusters_used})")
    ax.axis('off')
    v1_5_path = debug_dir / f"{image_id}_v1_5_clusters.png"
    plt.tight_layout()
    plt.savefig(v1_5_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved V1.5 clusters to: {v1_5_path}")

    # Run V3 with verbose output
    print("\n" + "=" * 80)
    print(f"Running V3 (preset: {preset})")
    print("=" * 80)

    config_v3 = Config(
        pipeline="v3",
        v3_preset=preset,
        auto_k=True,
        elbow_threshold=10.0,
        verbose=True
    )
    seg_v3 = TreeSegmentation(config_v3)

    results_v3 = seg_v3.process_single_image(str(temp_path))
    print(f"\nV3 labels shape: {results_v3.labels_resized.shape}")
    print(f"V3 unique labels: {np.unique(results_v3.labels_resized)}")
    print(f"V3 n_clusters_used: {results_v3.n_clusters_used}")

    # Count instances
    unique_labels = np.unique(results_v3.labels_resized)
    n_instances = len(unique_labels[unique_labels > 0])  # Exclude background
    print(f"V3 detected instances: {n_instances}")

    # Visualize V3 results
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if n_instances > 0:
        ax.imshow(results_v3.labels_resized, cmap='tab20')
        ax.set_title(f"V3 Instances (n={n_instances})")
    else:
        ax.imshow(results_v3.labels_resized, cmap='gray')
        ax.set_title("V3 Instances (NONE DETECTED)")
    ax.axis('off')
    v3_path = debug_dir / f"{image_id}_v3_instances.png"
    plt.tight_layout()
    plt.savefig(v3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved V3 instances to: {v3_path}")

    # Clean up temp file
    temp_path.unlink()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Ground truth trees: {len(coco_anns)}")
    print(f"V1.5 clusters: {results_v1_5.n_clusters_used}")
    print(f"V3 instances: {n_instances}")
    print(f"\nDebug files saved to: {debug_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug V3 on OAM-TCD")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/oam_tcd",
        help="Path to OAM-TCD dataset"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index to debug"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="permissive",
        choices=["permissive", "balanced", "strict"],
        help="V3 preset"
    )

    args = parser.parse_args()

    debug_v3_single_sample(
        dataset_path=args.dataset,
        sample_idx=args.sample,
        preset=args.preset
    )
