#!/usr/bin/env python3
"""
Analyze DINOv3 Feature Separation for Vegetation

Investigates whether DINOv3 features naturally separate vegetation from non-vegetation.
Creates visualizations showing:
1. Feature space clustering (PCA/UMAP)
2. Per-cluster vegetation statistics
3. Correlation between DINOv3 clusters and ExG vegetation mask
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from tree_seg import TreeSegmentation, Config
from tree_seg.tree_focus.vegetation_indices import (
    excess_green_index,
    create_vegetation_mask,
)


def analyze_vegetation_features(
    image_path: str,
    k_value: int = 15,
    output_dir: str = "data/outputs/feature_analysis",
):
    """
    Analyze how well DINOv3 features separate vegetation from non-vegetation.

    Args:
        image_path: Path to input image
        k_value: Number of clusters to test
        output_dir: Output directory for visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem

    print("=" * 80)
    print(f"DINOv3 Feature Analysis: {image_name}")
    print("=" * 80)
    print()

    # Load image
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Run V1.5 to get clusters and features
    print(f"Running V1.5 with K={k_value}...")
    config = Config(pipeline="v1_5", auto_k=False, n_clusters=k_value, verbose=False)

    segmenter = TreeSegmentation(config)
    results = segmenter.process_single_image(image_path)

    # Get cluster labels
    labels = results.labels_resized
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    print(f"  Found {n_clusters} clusters")
    print()

    # Compute vegetation mask (ground truth for comparison)
    print("Computing vegetation indices...")
    exg = excess_green_index(image_np)
    veg_mask = create_vegetation_mask(image_np, method="exg", threshold=0.1)

    veg_percent = 100.0 * veg_mask.sum() / veg_mask.size
    print(f"  ExG vegetation: {veg_percent:.1f}% of pixels")
    print()

    # Analyze each cluster
    print("Cluster Vegetation Analysis:")
    print("-" * 80)

    cluster_stats = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_size = cluster_mask.sum()

        # Vegetation overlap
        veg_in_cluster = np.logical_and(cluster_mask, veg_mask).sum()
        veg_coverage = veg_in_cluster / cluster_size if cluster_size > 0 else 0

        # Mean ExG in cluster
        mean_exg = exg[cluster_mask].mean() if cluster_size > 0 else 0

        # Mean green channel
        green_channel = image_np[:, :, 1] / 255.0
        mean_green = green_channel[cluster_mask].mean() if cluster_size > 0 else 0

        cluster_stats.append(
            {
                "label": label,
                "size": cluster_size,
                "veg_coverage": veg_coverage,
                "mean_exg": mean_exg,
                "mean_green": mean_green,
            }
        )

        # Classify cluster
        is_veg = veg_coverage > 0.5
        category = "VEG" if is_veg else "NON-VEG"

        print(
            f"  Cluster {label:2d}: {category:8s} | "
            f"Size: {cluster_size:7d} px ({100 * cluster_size / labels.size:4.1f}%) | "
            f"Veg: {100 * veg_coverage:5.1f}% | "
            f"ExG: {mean_exg:6.3f} | "
            f"Green: {mean_green:5.3f}"
        )

    print()

    # Separate vegetation vs non-vegetation clusters
    veg_clusters = [s for s in cluster_stats if s["veg_coverage"] > 0.5]
    nonveg_clusters = [s for s in cluster_stats if s["veg_coverage"] <= 0.5]

    print("Summary:")
    print(f"  Vegetation clusters: {len(veg_clusters)}/{n_clusters}")
    print(f"  Non-vegetation clusters: {len(nonveg_clusters)}/{n_clusters}")
    print()

    if len(veg_clusters) > 0:
        veg_labels = [s["label"] for s in veg_clusters]
        total_veg_pixels = sum(s["size"] for s in veg_clusters)
        print(
            f"  Total vegetation pixels (via clusters): {total_veg_pixels:,} "
            f"({100 * total_veg_pixels / labels.size:.1f}%)"
        )
        print(f"  ExG vegetation pixels: {veg_mask.sum():,} ({veg_percent:.1f}%)")
        print()

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # 2. ExG vegetation mask
    axes[0, 1].imshow(veg_mask, cmap="RdYlGn", vmin=0, vmax=1)
    axes[0, 1].set_title(f"ExG Vegetation Mask ({veg_percent:.1f}%)")
    axes[0, 1].axis("off")

    # 3. DINOv3 clusters
    axes[0, 2].imshow(labels, cmap="tab20")
    axes[0, 2].set_title(f"DINOv3 Clusters (K={k_value})")
    axes[0, 2].axis("off")

    # 4. Cluster vegetation coverage heatmap
    veg_coverage_map = np.zeros_like(labels, dtype=float)
    for stat in cluster_stats:
        veg_coverage_map[labels == stat["label"]] = stat["veg_coverage"]

    im = axes[1, 0].imshow(veg_coverage_map, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1, 0].set_title("Cluster Vegetation Coverage")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # 5. Filtered vegetation (clusters with >50% veg)
    if len(veg_clusters) > 0:
        filtered_veg = np.isin(labels, veg_labels)
        axes[1, 1].imshow(filtered_veg, cmap="RdYlGn", vmin=0, vmax=1)
        axes[1, 1].set_title(
            f"Vegetation Filter (K-means + ExG)\n"
            f"{len(veg_clusters)} clusters, "
            f"{100 * total_veg_pixels / labels.size:.1f}% of image"
        )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No vegetation clusters found",
            transform=axes[1, 1].transAxes,
            ha="center",
            va="center",
        )
        axes[1, 1].set_title("Vegetation Filter (none)")
    axes[1, 1].axis("off")

    # 6. Scatter: Cluster ExG vs Vegetation Coverage
    veg_coverages = [s["veg_coverage"] for s in cluster_stats]
    mean_exgs = [s["mean_exg"] for s in cluster_stats]
    sizes = [s["size"] for s in cluster_stats]

    axes[1, 2].scatter(mean_exgs, veg_coverages, s=np.array(sizes) / 1000, alpha=0.6)
    axes[1, 2].axhline(0.5, color="red", linestyle="--", label="Veg threshold (50%)")
    axes[1, 2].axvline(0.1, color="orange", linestyle="--", label="ExG threshold (0.1)")
    axes[1, 2].set_xlabel("Mean ExG in Cluster")
    axes[1, 2].set_ylabel("Vegetation Coverage (via ExG mask)")
    axes[1, 2].set_title("Cluster ExG vs Vegetation Coverage")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = output_path / f"{image_name}_vegetation_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved analysis to: {save_path}")
    print()

    # Print conclusions
    print("=" * 80)
    print("Conclusions:")
    print("=" * 80)

    # Check correlation
    from scipy.stats import pearsonr

    if len(cluster_stats) > 1:
        corr, p_value = pearsonr(mean_exgs, veg_coverages)
        print(
            f"Correlation between ExG and vegetation coverage: {corr:.3f} (p={p_value:.4f})"
        )

        if corr > 0.7:
            print("  ✓ Strong correlation: DINOv3 clusters align with vegetation!")
        elif corr > 0.4:
            print("  ~ Moderate correlation: DINOv3 partially captures vegetation")
        else:
            print("  ✗ Weak correlation: DINOv3 clusters don't match vegetation well")

    # Check separation quality
    if len(veg_clusters) > 0 and len(nonveg_clusters) > 0:
        veg_exg_mean = np.mean([s["mean_exg"] for s in veg_clusters])
        nonveg_exg_mean = np.mean([s["mean_exg"] for s in nonveg_clusters])
        separation = veg_exg_mean - nonveg_exg_mean

        print("\nVegetation vs Non-vegetation separation:")
        print(f"  Veg clusters mean ExG: {veg_exg_mean:.3f}")
        print(f"  Non-veg clusters mean ExG: {nonveg_exg_mean:.3f}")
        print(f"  Separation: {separation:.3f}")

        if separation > 0.2:
            print("  ✓ Good separation: Clusters naturally group by vegetation")
        elif separation > 0.1:
            print("  ~ Moderate separation: Some mixing of vegetation/non-vegetation")
        else:
            print("  ✗ Poor separation: Clusters don't align with vegetation")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze DINOv3 vegetation features")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--k", type=int, default=15, help="Number of clusters")
    parser.add_argument(
        "--output",
        type=str,
        default="data/outputs/feature_analysis",
        help="Output directory",
    )

    args = parser.parse_args()

    analyze_vegetation_features(args.image, k_value=args.k, output_dir=args.output)
