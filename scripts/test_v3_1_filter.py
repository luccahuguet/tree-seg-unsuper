#!/usr/bin/env python3
"""
Test V3.1 Vegetation Filter

Tests the minimal vegetation filtering module on sample images.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

from tree_seg import TreeSegmentation, Config


def test_v3_1_filter(
    image_path: str,
    k_value: int = 20,
    exg_threshold: float = 0.10,
    output_dir: str = "data/output/v3_1_test"
):
    """
    Test V3.1 vegetation filter on a single image.

    Args:
        image_path: Path to input image
        k_value: Number of clusters
        exg_threshold: ExG threshold for vegetation
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem

    print("=" * 80)
    print(f"Testing V3.1 Vegetation Filter: {image_name}")
    print("=" * 80)
    print()

    # Load image
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Run V1.5 baseline
    print(f"Running V1.5 (K={k_value})...")
    config_v1_5 = Config(
        pipeline="v1_5",
        auto_k=False,
        n_clusters=k_value,
        verbose=True
    )

    seg_v1_5 = TreeSegmentation(config_v1_5)
    results_v1_5 = seg_v1_5.process_single_image(image_path)

    print(f"  V1.5: {results_v1_5.n_clusters_used} clusters")
    print()

    # Run V3.1 with vegetation filter
    print(f"Running V3.1 (ExG threshold={exg_threshold})...")
    config_v3_1 = Config(
        pipeline="v3_1",
        auto_k=False,
        n_clusters=k_value,
        v3_1_exg_threshold=exg_threshold,
        verbose=True
    )

    seg_v3_1 = TreeSegmentation(config_v3_1)
    results_v3_1 = seg_v3_1.process_single_image(image_path)

    print(f"  V3.1: {results_v3_1.n_clusters_used} vegetation clusters")
    print()

    # Create visualization with outlines and hatching
    from skimage import segmentation as skimage_seg

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # 1. Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # 2. V1.5 clusters with boundaries (no hatching)
    axes[0, 1].imshow(image_np)
    # Draw cluster boundaries in white
    boundaries = skimage_seg.find_boundaries(results_v1_5.labels_resized, mode='thick')
    axes[0, 1].contour(boundaries, levels=[0.5], colors='white', linewidths=1.5, alpha=0.8)
    axes[0, 1].set_title(f"V1.5: All Clusters (K={results_v1_5.n_clusters_used})")
    axes[0, 1].axis('off')

    # 3. V3.1 filtered vegetation with boundaries only (no hatching)
    axes[1, 0].imshow(image_np)
    # Draw vegetation boundaries
    veg_boundaries = skimage_seg.find_boundaries(results_v3_1.labels_resized, mode='thick')
    axes[1, 0].contour(veg_boundaries, levels=[0.5], colors='lime', linewidths=2, alpha=0.9)
    axes[1, 0].set_title(f"V3.1: Vegetation Only ({results_v3_1.n_clusters_used} clusters)")
    axes[1, 0].axis('off')

    # 4. Show removed non-vegetation with subtle red tint (no hatching)
    v1_5_mask = results_v1_5.labels_resized > 0
    v3_1_mask = results_v3_1.labels_resized > 0
    removed_mask = v1_5_mask & ~v3_1_mask

    # Create image with removed regions tinted red
    display_image = image_np.copy().astype(float)
    if removed_mask.any():
        # Apply red tint to removed regions
        display_image[removed_mask] = display_image[removed_mask] * 0.5 + np.array([128, 0, 0]) * 0.5

    axes[1, 1].imshow(display_image.astype(np.uint8))

    # Draw boundaries of removed regions
    if removed_mask.any():
        removed_labels = results_v1_5.labels_resized * removed_mask
        removed_boundaries = skimage_seg.find_boundaries(removed_labels.astype(int), mode='thick')
        axes[1, 1].contour(removed_boundaries, levels=[0.5], colors='red',
                          linewidths=2, alpha=0.8, linestyles='--')

    axes[1, 1].set_title(f"Removed Non-Vegetation (Red Tint=Filtered Out)\n"
                        f"Removed: {removed_mask.sum():,} px ({100*removed_mask.sum()/(v1_5_mask.sum()):.1f}%)")
    axes[1, 1].axis('off')

    plt.tight_layout()

    save_path = output_path / f"{image_name}_v3_1_test.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comparison to: {save_path}")
    print()

    # Print statistics
    print("Summary:")
    print(f"  Image: {image_name}")
    print(f"  K value: {k_value}")
    print(f"  ExG threshold: {exg_threshold}")
    print()
    print(f"  V1.5 clusters: {results_v1_5.n_clusters_used}")
    print(f"  V3.1 vegetation clusters: {results_v3_1.n_clusters_used}")
    print(f"  Removed clusters: {results_v1_5.n_clusters_used - results_v3_1.n_clusters_used}")
    print()

    v1_5_pixels = v1_5_mask.sum()
    v3_1_pixels = v3_1_mask.sum()
    removed_pixels = removed_mask.sum()

    print(f"  V1.5 pixels: {v1_5_pixels:,} (100.0%)")
    print(f"  V3.1 vegetation pixels: {v3_1_pixels:,} ({100*v3_1_pixels/v1_5_pixels:.1f}%)")
    print(f"  Removed pixels: {removed_pixels:,} ({100*removed_pixels/v1_5_pixels:.1f}%)")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test V3.1 vegetation filter")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--k", type=int, default=20,
                       help="Number of clusters")
    parser.add_argument("--threshold", type=float, default=0.10,
                       help="ExG threshold for vegetation")
    parser.add_argument("--output", type=str, default="data/output/v3_1_test",
                       help="Output directory")

    args = parser.parse_args()

    test_v3_1_filter(
        image_path=args.image,
        k_value=args.k,
        exg_threshold=args.threshold,
        output_dir=args.output
    )
