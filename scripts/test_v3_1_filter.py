#!/usr/bin/env python3
"""
Test V3.1 Vegetation Filter

Tests the minimal vegetation filtering module on sample images.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # 1. Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # 2. V1.5 clusters
    axes[0, 1].imshow(results_v1_5.labels_resized, cmap='tab20')
    axes[0, 1].set_title(f"V1.5: All Clusters (K={results_v1_5.n_clusters_used})")
    axes[0, 1].axis('off')

    # 3. V3.1 filtered vegetation
    axes[1, 0].imshow(results_v3_1.labels_resized, cmap='tab20')
    axes[1, 0].set_title(f"V3.1: Vegetation Only ({results_v3_1.n_clusters_used} clusters)")
    axes[1, 0].axis('off')

    # 4. Overlay comparison
    # Create binary masks
    v1_5_mask = results_v1_5.labels_resized > 0
    v3_1_mask = results_v3_1.labels_resized > 0

    # Compute difference
    removed = v1_5_mask & ~v3_1_mask  # Pixels removed by filter
    kept = v3_1_mask  # Pixels kept

    comparison = np.zeros((*image_np.shape[:2], 3), dtype=np.uint8)
    comparison[removed] = [255, 0, 0]  # Red = removed (non-vegetation)
    comparison[kept] = [0, 255, 0]  # Green = kept (vegetation)

    axes[1, 1].imshow(image_np, alpha=0.5)
    axes[1, 1].imshow(comparison, alpha=0.5)
    axes[1, 1].set_title(f"Filtering Effect\n"
                        f"Green=Vegetation ({kept.sum():,} px), "
                        f"Red=Removed ({removed.sum():,} px)")
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
    removed_pixels = removed.sum()

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
