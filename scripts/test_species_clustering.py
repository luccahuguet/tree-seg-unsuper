#!/usr/bin/env python3
"""
Test V1.5 Species Clustering

Tests if DINOv3 + K-means with higher K naturally separates vegetation species.
Generates visualizations for different K values to assess cluster quality.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tree_seg import TreeSegmentation, Config


def test_species_clustering(
    image_path: str,
    k_values: list = [5, 10, 15, 20, 25],
    output_dir: str = "data/output/species_clustering"
):
    """
    Test V1.5 clustering at different K values.

    Args:
        image_path: Path to input image
        k_values: List of K values to test
        output_dir: Output directory for visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem

    print("=" * 80)
    print(f"Testing Species Clustering on {image_name}")
    print("=" * 80)
    print()

    results_by_k = {}

    for k in k_values:
        print(f"Testing K={k}...")

        # Run V1.5 with fixed K
        config = Config(
            pipeline="v1_5",
            auto_k=False,
            n_clusters=k,
            verbose=False
        )

        segmenter = TreeSegmentation(config)
        results = segmenter.process_single_image(image_path)

        results_by_k[k] = results

        # Count unique labels
        unique_labels = np.unique(results.labels_resized)
        n_clusters = len(unique_labels)

        print(f"  K={k}: {n_clusters} clusters found")

    # Create comparison visualization
    n_k = len(k_values)
    n_cols = min(3, n_k)
    n_rows = (n_k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_k == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, k in enumerate(k_values):
        results = results_by_k[k]

        axes[idx].imshow(results.labels_resized, cmap='tab20')
        axes[idx].set_title(f"K={k} ({len(np.unique(results.labels_resized))} clusters)")
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(n_k, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"Species Clustering Comparison: {image_name}", fontsize=16, y=0.98)
    plt.tight_layout()

    save_path = output_path / f"{image_name}_k_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print()
    print(f"✓ Saved comparison to: {save_path}")
    print()

    # Create detailed view for highest K
    highest_k = max(k_values)
    results_high_k = results_by_k[highest_k]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original image
    import cv2
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Segmentation with K=highest
    axes[1].imshow(results_high_k.labels_resized, cmap='tab20')
    axes[1].set_title(f"Segmentation (K={highest_k})")
    axes[1].axis('off')

    plt.tight_layout()

    detail_path = output_path / f"{image_name}_k{highest_k}_detail.png"
    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved detail view to: {detail_path}")
    print()

    # Cluster statistics
    print("Cluster Statistics:")
    print(f"  Image: {image_name}")
    print(f"  Tested K values: {k_values}")
    print()

    for k in k_values:
        results = results_by_k[k]
        labels = results.labels_resized
        unique_labels = np.unique(labels)

        # Compute cluster sizes
        sizes = []
        for label in unique_labels:
            size = (labels == label).sum()
            sizes.append(size)

        sizes = np.array(sizes)
        total_pixels = labels.size

        print(f"  K={k}:")
        print(f"    Clusters: {len(unique_labels)}")
        print(f"    Avg size: {sizes.mean():.0f} pixels ({100*sizes.mean()/total_pixels:.1f}%)")
        print(f"    Min size: {sizes.min():.0f} pixels ({100*sizes.min()/total_pixels:.2f}%)")
        print(f"    Max size: {sizes.max():.0f} pixels ({100*sizes.max()/total_pixels:.1f}%)")
        print()


def test_multiple_images(
    image_dir: str = "data/input",
    k_values: list = [5, 10, 15, 20]
):
    """Test species clustering on multiple images."""

    image_paths = list(Path(image_dir).glob("*.jpg"))
    image_paths.extend(Path(image_dir).glob("*.png"))

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images in {image_dir}")
    print()

    for image_path in image_paths[:3]:  # Test first 3 images
        test_species_clustering(str(image_path), k_values=k_values)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test species clustering with different K values")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to single image (default: test multiple from data/input)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 15, 20, 25],
                       help="K values to test")
    parser.add_argument("--output", type=str, default="data/output/species_clustering",
                       help="Output directory")

    args = parser.parse_args()

    if args.image:
        test_species_clustering(args.image, k_values=args.k_values, output_dir=args.output)
    else:
        test_multiple_images(k_values=args.k_values)
