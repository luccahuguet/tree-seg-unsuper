"""
Vegetation Filter Module (V3.1)

Minimal cluster-level vegetation filtering based on ExG indices.
DINOv3 analysis showed 0.95+ correlation between clusters and vegetation,
so we only need simple ExG thresholding per cluster.
"""

import numpy as np
from typing import Tuple, List, Dict
from tree_seg.tree_focus.vegetation_indices import excess_green_index


def compute_cluster_vegetation_scores(
    image: np.ndarray,
    cluster_labels: np.ndarray,
    verbose: bool = False
) -> Dict[int, float]:
    """
    Compute mean ExG vegetation score for each cluster.

    Args:
        image: RGB image (H, W, 3) in [0, 255]
        cluster_labels: Cluster labels (H, W)
        verbose: Print cluster statistics

    Returns:
        Dictionary mapping cluster_id -> mean_exg_score
    """
    # Compute ExG for entire image
    exg = excess_green_index(image)

    # Compute mean ExG per cluster
    unique_labels = np.unique(cluster_labels)
    cluster_scores = {}

    if verbose:
        print("Cluster Vegetation Scores:")
        print("-" * 60)

    for label in unique_labels:
        cluster_mask = cluster_labels == label
        mean_exg = exg[cluster_mask].mean()
        cluster_scores[int(label)] = float(mean_exg)

        if verbose:
            size = cluster_mask.sum()
            size_pct = 100.0 * size / cluster_labels.size
            print(f"  Cluster {label:2d}: ExG={mean_exg:6.3f} | "
                  f"Size={size:7d} px ({size_pct:4.1f}%)")

    if verbose:
        print()

    return cluster_scores


def filter_vegetation_clusters(
    cluster_labels: np.ndarray,
    cluster_scores: Dict[int, float],
    exg_threshold: float = 0.10,
    verbose: bool = False
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Filter clusters to keep only vegetation based on ExG scores.

    Args:
        cluster_labels: Original cluster labels (H, W)
        cluster_scores: Dict mapping cluster_id -> exg_score
        exg_threshold: Minimum ExG to consider vegetation
        verbose: Print filtering results

    Returns:
        filtered_labels: Labels with only vegetation clusters (H, W)
        vegetation_clusters: List of cluster IDs classified as vegetation
        removed_clusters: List of cluster IDs classified as non-vegetation
    """
    # Classify clusters
    vegetation_clusters = []
    removed_clusters = []

    for cluster_id, score in cluster_scores.items():
        if score >= exg_threshold:
            vegetation_clusters.append(cluster_id)
        else:
            removed_clusters.append(cluster_id)

    if verbose:
        print(f"Vegetation Filtering (ExG threshold = {exg_threshold}):")
        print(f"  Vegetation clusters: {len(vegetation_clusters)}")
        print(f"  Non-vegetation clusters: {len(removed_clusters)}")
        print()

    # Create filtered labels (set non-vegetation to 0)
    filtered_labels = cluster_labels.copy()
    for cluster_id in removed_clusters:
        filtered_labels[cluster_labels == cluster_id] = 0

    # Relabel to remove gaps (0, 3, 7, 9 → 0, 1, 2, 3)
    filtered_labels = _relabel_sequential(filtered_labels, vegetation_clusters)

    return filtered_labels, vegetation_clusters, removed_clusters


def _relabel_sequential(
    labels: np.ndarray,
    keep_labels: List[int]
) -> np.ndarray:
    """
    Relabel array to be sequential (0, 1, 2, ...) keeping only specified labels.

    Args:
        labels: Original labels
        keep_labels: Labels to keep (will become 1, 2, 3, ...)

    Returns:
        Relabeled array where 0=background, 1-N=vegetation clusters
    """
    output = np.zeros_like(labels)

    # Sort to ensure consistent ordering
    keep_labels_sorted = sorted(keep_labels)

    # Assign new sequential labels
    for new_id, old_id in enumerate(keep_labels_sorted, start=1):
        output[labels == old_id] = new_id

    return output


def apply_vegetation_filter(
    image: np.ndarray,
    cluster_labels: np.ndarray,
    exg_threshold: float = 0.10,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Complete vegetation filtering pipeline.

    Args:
        image: RGB image (H, W, 3) in [0, 255]
        cluster_labels: Cluster labels from V1.5 (H, W)
        exg_threshold: ExG threshold for vegetation classification
        verbose: Print progress

    Returns:
        filtered_labels: Vegetation-only labels (H, W) with 0=background
        filter_info: Dictionary with filtering statistics
    """
    if verbose:
        print("=" * 60)
        print("V3.1 Vegetation Filter")
        print("=" * 60)
        print()

    # Step 1: Compute cluster scores
    cluster_scores = compute_cluster_vegetation_scores(
        image, cluster_labels, verbose=verbose
    )

    # Step 2: Filter clusters
    filtered_labels, veg_clusters, removed_clusters = filter_vegetation_clusters(
        cluster_labels, cluster_scores, exg_threshold, verbose=verbose
    )

    # Compute statistics
    n_original_clusters = len(cluster_scores)
    n_veg_clusters = len(veg_clusters)
    n_removed_clusters = len(removed_clusters)

    original_pixels = (cluster_labels > 0).sum()
    filtered_pixels = (filtered_labels > 0).sum()
    removed_pixels = original_pixels - filtered_pixels

    if verbose:
        print("Filtering Results:")
        print(f"  Original clusters: {n_original_clusters}")
        print(f"  Vegetation clusters: {n_veg_clusters}")
        print(f"  Removed clusters: {n_removed_clusters}")
        print()
        print(f"  Original pixels: {original_pixels:,} (100.0%)")
        print(f"  Vegetation pixels: {filtered_pixels:,} "
              f"({100.0*filtered_pixels/original_pixels:.1f}%)")
        print(f"  Removed pixels: {removed_pixels:,} "
              f"({100.0*removed_pixels/original_pixels:.1f}%)")
        print()

    # Build info dictionary
    filter_info = {
        'n_original_clusters': n_original_clusters,
        'n_vegetation_clusters': n_veg_clusters,
        'n_removed_clusters': n_removed_clusters,
        'exg_threshold': exg_threshold,
        'cluster_scores': cluster_scores,
        'vegetation_cluster_ids': veg_clusters,
        'removed_cluster_ids': removed_clusters,
        'vegetation_pixels': int(filtered_pixels),
        'removed_pixels': int(removed_pixels),
        'vegetation_percentage': float(100.0 * filtered_pixels / original_pixels) if original_pixels > 0 else 0.0,
    }

    return filtered_labels, filter_info


if __name__ == "__main__":
    # Example usage
    print("Vegetation Filter Module")
    print()
    print("Usage:")
    print("  from tree_seg.vegetation_filter import apply_vegetation_filter")
    print()
    print("  # After running V1.5 to get cluster labels")
    print("  filtered_labels, info = apply_vegetation_filter(")
    print("      image_rgb,")
    print("      cluster_labels,")
    print("      exg_threshold=0.10,")
    print("      verbose=True")
    print("  )")
