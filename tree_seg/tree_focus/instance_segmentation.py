"""
Instance Segmentation for Individual Trees

Separates merged tree crowns into individual instances using:
- Distance transform + watershed
- Shape and size filters
- Morphological operations
"""

import numpy as np
import cv2
from scipy import ndimage as ndi
from typing import Tuple, List, Dict
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def watershed_segmentation(
    binary_mask: np.ndarray,
    min_distance: int = 10,
    connectivity: int = 2
) -> np.ndarray:
    """
    Separate instances using distance transform + watershed.

    Args:
        binary_mask: Binary mask (H, W) with True=foreground
        min_distance: Minimum distance between peaks (in pixels)
        connectivity: Connectivity for watershed (1=4-connected, 2=8-connected)

    Returns:
        Instance labels (H, W) with unique ID per instance, 0=background
    """
    if not binary_mask.any():
        return np.zeros_like(binary_mask, dtype=np.int32)

    # Compute distance transform
    distance = ndi.distance_transform_edt(binary_mask)

    # Find local maxima (tree centers)
    local_max = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary_mask.astype(int),
        exclude_border=False
    )

    # Create markers from local maxima
    markers = np.zeros_like(binary_mask, dtype=np.int32)
    for idx, (y, x) in enumerate(local_max, start=1):
        markers[y, x] = idx

    # Label markers
    markers = ndi.label(markers > 0)[0]

    # Watershed from markers
    labels = watershed(-distance, markers, mask=binary_mask, connectivity=connectivity)

    return labels.astype(np.int32)


def filter_instances_by_shape(
    instance_labels: np.ndarray,
    min_area: int = 100,
    max_area: int = 50000,
    min_circularity: float = 0.3,
    max_eccentricity: float = 0.95
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Filter instances by shape and size constraints.

    Args:
        instance_labels: Instance labels (H, W)
        min_area: Minimum instance area in pixels
        max_area: Maximum instance area in pixels
        min_circularity: Minimum circularity (4π*area / perimeter²)
        max_eccentricity: Maximum eccentricity (length/width ratio)

    Returns:
        filtered_labels: Filtered instance labels (H, W)
        instance_stats: List of statistics for each instance
    """
    unique_labels = np.unique(instance_labels)
    unique_labels = unique_labels[unique_labels > 0]

    filtered_labels = np.zeros_like(instance_labels)
    instance_stats = []
    new_label = 1

    for label in unique_labels:
        mask = (instance_labels == label).astype(np.uint8)

        # Compute shape metrics
        area = mask.sum()

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)

        # Circularity: 4π*area / perimeter² (1.0 = perfect circle)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0.0

        # Eccentricity from fitted ellipse
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                if major_axis > 0:
                    eccentricity = minor_axis / major_axis
                else:
                    eccentricity = 1.0
            except (cv2.error, ValueError):
                eccentricity = 0.5  # Default if fit fails
        else:
            eccentricity = 0.5

        # Apply filters
        is_valid = True

        if area < min_area or area > max_area:
            is_valid = False

        if circularity < min_circularity:
            is_valid = False

        if eccentricity < (1.0 - max_eccentricity):
            # Eccentricity close to 0 means very elongated
            is_valid = False

        # Record stats
        stats = {
            'original_label': int(label),
            'area': int(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'eccentricity': float(eccentricity),
            'is_valid': is_valid,
        }

        if is_valid:
            filtered_labels[mask > 0] = new_label
            stats['new_label'] = new_label
            new_label += 1
        else:
            stats['new_label'] = 0

        instance_stats.append(stats)

    return filtered_labels, instance_stats


def separate_tree_instances(
    tree_mask: np.ndarray,
    min_distance: int = 10,
    min_area_pixels: int = 100,
    max_area_pixels: int = 50000,
    gsd_cm: float = 10.0
) -> Tuple[np.ndarray, int, List[Dict]]:
    """
    Full pipeline: watershed + shape filtering.

    Args:
        tree_mask: Binary tree mask (H, W)
        min_distance: Minimum distance between tree centers (pixels)
        min_area_pixels: Minimum tree area (pixels)
        max_area_pixels: Maximum tree area (pixels)
        gsd_cm: Ground Sample Distance in cm/pixel

    Returns:
        instance_labels: Instance labels (H, W) with unique ID per tree
        num_trees: Number of detected trees
        instance_stats: Statistics for each instance
    """
    if not tree_mask.any():
        return np.zeros_like(tree_mask, dtype=np.int32), 0, []

    # Step 1: Watershed segmentation
    instances = watershed_segmentation(tree_mask, min_distance=min_distance)

    # Step 2: Shape filtering
    filtered_instances, stats = filter_instances_by_shape(
        instances,
        min_area=min_area_pixels,
        max_area=max_area_pixels,
        min_circularity=0.3,
        max_eccentricity=0.95
    )

    # Add real-world area to stats
    for stat in stats:
        if 'area' in stat:
            stat['area_m2'] = stat['area'] * (gsd_cm / 100.0) ** 2

    num_trees = len(np.unique(filtered_instances)) - 1  # Exclude background

    return filtered_instances, num_trees, stats


def morphological_cleanup(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
    operation: str = 'close'
) -> np.ndarray:
    """
    Apply morphological operations to clean up mask.

    Args:
        binary_mask: Binary mask (H, W)
        kernel_size: Size of morphological kernel
        operation: 'open', 'close', or 'both'

    Returns:
        Cleaned binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    mask_uint8 = binary_mask.astype(np.uint8) * 255

    if operation == 'open':
        cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    elif operation == 'both':
        # Open then close (remove noise, fill holes)
        opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return (cleaned > 0).astype(bool)


if __name__ == "__main__":
    # Example usage
    print("Instance segmentation module loaded")
    print("Functions: separate_tree_instances, watershed_segmentation, filter_instances_by_shape")
