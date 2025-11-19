#!/usr/bin/env python3
"""
Visualize V3.1 semantic clustering with distinct colors per region.

Shows vegetation clusters as distinct colored regions, with non-vegetation
as a separate background cluster.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import numpy as np

from tree_seg import TreeSegmentation, Config


def visualize_v3_1_semantic(
    image_path: str,
    k_value: int = 20,
    exg_threshold: float = 0.10,
    output_dir: str = "data/output/v3_1_semantic"
):
    """
    Visualize V3.1 as semantic segmentation with colored regions.

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
    print(f"V3.1 Semantic Visualization: {image_name}")
    print("=" * 80)
    print()

    # Load image
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Run V3.1
    print(f"Running V3.1 (K={k_value}, ExG threshold={exg_threshold})...")
    config_v3_1 = Config(
        pipeline="v3_1",
        auto_k=False,
        n_clusters=k_value,
        v3_1_exg_threshold=exg_threshold,
        verbose=True
    )

    seg_v3_1 = TreeSegmentation(config_v3_1)
    results_v3_1 = seg_v3_1.process_single_image(image_path)

    # Also run V1.5 to get non-vegetation regions
    print(f"\nRunning V1.5 for comparison (K={k_value})...")
    config_v1_5 = Config(
        pipeline="v1_5",
        auto_k=False,
        n_clusters=k_value,
        verbose=False
    )

    seg_v1_5 = TreeSegmentation(config_v1_5)
    results_v1_5 = seg_v1_5.process_single_image(image_path)

    # Create semantic maps
    v1_5_labels = results_v1_5.labels_resized
    v3_1_labels = results_v3_1.labels_resized

    # Create non-vegetation mask
    v1_5_mask = v1_5_labels > 0
    v3_1_mask = v3_1_labels > 0
    non_veg_mask = v1_5_mask & ~v3_1_mask

    # Create combined semantic map: vegetation clusters + non-veg as cluster 0
    semantic_map = v3_1_labels.copy()
    # Set non-vegetation to label 0 (background)
    semantic_map[non_veg_mask] = 0

    # Get unique labels
    unique_labels = np.unique(semantic_map)
    n_labels = len(unique_labels)

    print("\nSemantic Map:")
    print(f"  Total regions: {n_labels}")
    print(f"  Vegetation clusters: {n_labels - 1}")
    print("  Non-vegetation: 1 background cluster (label 0)")
    print()

    # Generate colormap - gray for background (0), distinct colors for vegetation
    cmap = plt.colormaps.get_cmap('tab20')
    colors = [cmap(i / n_labels) for i in range(n_labels)]
    # Override color 0 to be gray for non-vegetation
    colors[0] = (0.5, 0.5, 0.5, 1.0)  # Gray
    custom_cmap = mcolors.ListedColormap(colors)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # 1. Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. V1.5 semantic map (all clusters)
    im1 = axes[0, 1].imshow(v1_5_labels, cmap='tab20', interpolation='nearest')
    axes[0, 1].set_title(f"V1.5: All Clusters (K={results_v1_5.n_clusters_used})",
                        fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label('Cluster ID', rotation=270, labelpad=15)

    # 3. V3.1 semantic map (vegetation only)
    # Create map without background for cleaner visualization
    veg_only_map = v3_1_labels.copy()
    veg_only_map[~v3_1_mask] = 0
    im2 = axes[1, 0].imshow(veg_only_map, cmap='tab20', interpolation='nearest')
    axes[1, 0].set_title(f"V3.1: Vegetation Clusters Only ({results_v3_1.n_clusters_used})",
                        fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar2.set_label('Cluster ID', rotation=270, labelpad=15)

    # 4. Combined semantic map (vegetation + non-veg background)
    im3 = axes[1, 1].imshow(semantic_map, cmap=custom_cmap, interpolation='nearest',
                           vmin=0, vmax=n_labels-1)

    # Calculate coverage
    non_veg_pct = 100 * non_veg_mask.sum() / semantic_map.size
    veg_pct = 100 * v3_1_mask.sum() / semantic_map.size

    axes[1, 1].set_title(
        f"V3.1: Semantic Map\n"
        f"Gray=Non-Veg ({non_veg_pct:.1f}%), Colors=Vegetation ({veg_pct:.1f}%)",
        fontsize=14, fontweight='bold'
    )
    axes[1, 1].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar3.set_label('Region (0=Non-Veg, 1+=Veg Clusters)', rotation=270, labelpad=20)

    plt.tight_layout()

    # Save
    save_path = output_path / f"{image_name}_semantic.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved semantic visualization to: {save_path}")
    print()

    # Print region statistics
    print("Region Statistics:")
    print(f"  Label 0 (Non-vegetation): {non_veg_mask.sum():,} px ({non_veg_pct:.1f}%)")
    for label in unique_labels[1:]:  # Skip 0
        mask = semantic_map == label
        pct = 100 * mask.sum() / semantic_map.size
        print(f"  Label {label:2d} (Vegetation):     {mask.sum():,} px ({pct:.1f}%)")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize V3.1 semantic clustering")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--k", type=int, default=20,
                       help="Number of clusters")
    parser.add_argument("--threshold", type=float, default=0.10,
                       help="ExG threshold for vegetation")
    parser.add_argument("--output", type=str, default="data/output/v3_1_semantic",
                       help="Output directory")

    args = parser.parse_args()

    visualize_v3_1_semantic(
        image_path=args.image,
        k_value=args.k,
        exg_threshold=args.threshold,
        output_dir=args.output
    )
