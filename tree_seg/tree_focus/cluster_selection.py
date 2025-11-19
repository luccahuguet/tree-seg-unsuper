"""
Cluster Selection for Tree Detection

Selects tree clusters from semantic segmentation based on:
- IoU with vegetation mask
- Green ratio within cluster
- Shape and size filters
"""

import numpy as np
from typing import List, Tuple, Dict
from tree_seg.tree_focus.vegetation_indices import compute_vegetation_score


def compute_cluster_iou(cluster_mask: np.ndarray, vegetation_mask: np.ndarray) -> float:
    """
    Compute IoU between cluster and vegetation mask.

    Args:
        cluster_mask: Binary mask for cluster (H, W)
        vegetation_mask: Binary vegetation mask (H, W)

    Returns:
        IoU score in [0, 1]
    """
    if not cluster_mask.any():
        return 0.0

    intersection = np.logical_and(cluster_mask, vegetation_mask).sum()
    union = np.logical_or(cluster_mask, vegetation_mask).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_cluster_metrics(
    image: np.ndarray,
    cluster_mask: np.ndarray,
    vegetation_mask: np.ndarray,
    gsd_cm: float = 10.0
) -> Dict[str, float]:
    """
    Compute metrics for a cluster.

    Args:
        image: RGB image (H, W, 3)
        cluster_mask: Binary mask for cluster (H, W)
        vegetation_mask: Binary vegetation mask (H, W)
        gsd_cm: Ground Sample Distance in cm/pixel

    Returns:
        Dictionary of metrics
    """
    if not cluster_mask.any():
        return {
            'iou': 0.0,
            'vegetation_score': 0.0,
            'area_pixels': 0,
            'area_m2': 0.0,
            'coverage_ratio': 0.0,
        }

    # IoU with vegetation
    iou = compute_cluster_iou(cluster_mask, vegetation_mask)

    # Vegetation score
    veg_score = compute_vegetation_score(image, cluster_mask)

    # Area metrics
    area_pixels = int(cluster_mask.sum())
    area_m2 = area_pixels * (gsd_cm / 100.0) ** 2

    # Coverage ratio (how much of vegetation mask does this cluster cover)
    if vegetation_mask.any():
        intersection_count = np.logical_and(cluster_mask, vegetation_mask).sum()
        coverage_ratio = float(intersection_count / vegetation_mask.sum())
    else:
        coverage_ratio = 0.0

    return {
        'iou': iou,
        'vegetation_score': veg_score,
        'area_pixels': area_pixels,
        'area_m2': area_m2,
        'coverage_ratio': coverage_ratio,
    }


def select_tree_clusters(
    image: np.ndarray,
    cluster_labels: np.ndarray,
    vegetation_mask: np.ndarray,
    iou_threshold: float = 0.3,
    veg_score_threshold: float = 0.4,
    min_area_m2: float = 1.0,
    max_area_m2: float = 500.0,
    gsd_cm: float = 10.0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Select tree clusters based on vegetation overlap and characteristics.

    Args:
        image: RGB image (H, W, 3)
        cluster_labels: Cluster labels (H, W) with 0=background
        vegetation_mask: Binary vegetation mask (H, W)
        iou_threshold: Minimum IoU with vegetation mask
        veg_score_threshold: Minimum vegetation score
        min_area_m2: Minimum tree area in m²
        max_area_m2: Maximum tree area in m²
        gsd_cm: Ground Sample Distance in cm/pixel

    Returns:
        tree_mask: Binary mask of selected tree clusters (H, W)
        cluster_stats: List of statistics for selected clusters
    """
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels > 0]  # Remove background

    tree_mask = np.zeros_like(cluster_labels, dtype=bool)
    cluster_stats = []

    for label in unique_labels:
        cluster_mask = cluster_labels == label

        # Compute metrics
        metrics = compute_cluster_metrics(
            image, cluster_mask, vegetation_mask, gsd_cm
        )

        # Apply filters
        is_tree = True

        # Filter 1: IoU with vegetation
        if metrics['iou'] < iou_threshold:
            is_tree = False

        # Filter 2: Vegetation score
        if metrics['vegetation_score'] < veg_score_threshold:
            is_tree = False

        # Filter 3: Area constraints (GSD-aware)
        if metrics['area_m2'] < min_area_m2 or metrics['area_m2'] > max_area_m2:
            is_tree = False

        # Add to tree mask if passes all filters
        if is_tree:
            tree_mask |= cluster_mask
            cluster_stats.append({
                'cluster_id': int(label),
                'is_tree': True,
                **metrics
            })
        else:
            cluster_stats.append({
                'cluster_id': int(label),
                'is_tree': False,
                **metrics
            })

    return tree_mask, cluster_stats


def filter_clusters_by_vegetation(
    cluster_labels: np.ndarray,
    vegetation_mask: np.ndarray,
    iou_threshold: float = 0.3
) -> np.ndarray:
    """
    Simple filter: keep only clusters with IoU > threshold.

    Args:
        cluster_labels: Cluster labels (H, W)
        vegetation_mask: Binary vegetation mask (H, W)
        iou_threshold: Minimum IoU to keep cluster

    Returns:
        Filtered cluster labels (H, W)
    """
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels > 0]

    filtered_labels = np.zeros_like(cluster_labels)

    for label in unique_labels:
        cluster_mask = cluster_labels == label
        iou = compute_cluster_iou(cluster_mask, vegetation_mask)

        if iou >= iou_threshold:
            filtered_labels[cluster_mask] = label

    return filtered_labels


if __name__ == "__main__":
    # Example usage
    print("Cluster selection module loaded")
    print("Functions: select_tree_clusters, compute_cluster_iou, filter_clusters_by_vegetation")
