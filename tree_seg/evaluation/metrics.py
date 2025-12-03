"""Evaluation metrics for semantic segmentation benchmarking."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class EvaluationResults:
    """Container for evaluation metrics results."""

    miou: float
    pixel_accuracy: float
    per_class_iou: Dict[int, float]
    per_class_pixel_accuracy: Dict[int, float]
    confusion_matrix: np.ndarray
    cluster_to_class_mapping: Optional[Dict[int, int]] = None
    num_predicted_clusters: Optional[int] = None
    num_ground_truth_classes: Optional[int] = None


def compute_confusion_matrix(
    pred_labels: np.ndarray, gt_labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix between predictions and ground truth.

    Args:
        pred_labels: Predicted label array (H, W) with values 0 to K-1
        gt_labels: Ground truth label array (H, W) with values 0 to C-1
        num_classes: Number of ground truth classes

    Returns:
        Confusion matrix of shape (num_pred_clusters, num_classes)
    """
    # Flatten arrays
    pred_flat = pred_labels.flatten()
    gt_flat = gt_labels.flatten()

    # Get number of predicted clusters
    num_pred_clusters = int(pred_flat.max()) + 1

    # Initialize confusion matrix
    confusion = np.zeros((num_pred_clusters, num_classes), dtype=np.int64)

    # Populate confusion matrix
    for pred_idx in range(num_pred_clusters):
        pred_mask = pred_flat == pred_idx
        for gt_idx in range(num_classes):
            gt_mask = gt_flat == gt_idx
            confusion[pred_idx, gt_idx] = np.sum(pred_mask & gt_mask)

    return confusion


def hungarian_matching(
    pred_labels: np.ndarray, gt_labels: np.ndarray, num_classes: int
) -> Tuple[Dict[int, int], np.ndarray]:
    """
    Find optimal assignment between predicted clusters and ground truth classes
    using the Hungarian algorithm.

    Args:
        pred_labels: Predicted label array (H, W) with values 0 to K-1
        gt_labels: Ground truth label array (H, W) with values 0 to C-1
        num_classes: Number of ground truth classes

    Returns:
        Tuple of:
        - mapping: Dict mapping predicted cluster index to ground truth class
        - confusion_matrix: Confusion matrix used for matching
    """
    # Compute confusion matrix
    confusion = compute_confusion_matrix(pred_labels, gt_labels, num_classes)

    # Hungarian algorithm minimizes cost, so we negate to maximize agreement
    # Add small epsilon to handle empty clusters
    cost_matrix = -confusion.astype(np.float64)

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create mapping dictionary
    mapping = {int(pred_idx): int(gt_idx) for pred_idx, gt_idx in zip(row_ind, col_ind)}

    return mapping, confusion


def apply_cluster_mapping(pred_labels: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """
    Apply cluster-to-class mapping to predicted labels.

    Args:
        pred_labels: Predicted label array (H, W) with cluster indices
        mapping: Dict mapping cluster indices to class indices

    Returns:
        Remapped label array with ground truth class indices
    """
    remapped = np.zeros_like(pred_labels)
    for cluster_idx, class_idx in mapping.items():
        remapped[pred_labels == cluster_idx] = class_idx
    return remapped


def compute_iou_per_class(
    pred_labels: np.ndarray, gt_labels: np.ndarray, num_classes: int, ignore_index: int = -1
) -> Dict[int, float]:
    """
    Compute Intersection over Union (IoU) for each class.

    Args:
        pred_labels: Predicted label array (H, W)
        gt_labels: Ground truth label array (H, W)
        num_classes: Number of classes
        ignore_index: Label value to ignore (e.g., unlabeled pixels)

    Returns:
        Dictionary mapping class index to IoU score
    """
    iou_per_class = {}

    # Flatten arrays
    pred_flat = pred_labels.flatten()
    gt_flat = gt_labels.flatten()

    # Create mask for valid pixels
    if ignore_index is not None:
        valid_mask = gt_flat != ignore_index
        pred_flat = pred_flat[valid_mask]
        gt_flat = gt_flat[valid_mask]

    for class_idx in range(num_classes):
        # Get masks for current class
        pred_mask = pred_flat == class_idx
        gt_mask = gt_flat == class_idx

        # Compute intersection and union
        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask)

        # Compute IoU (handle division by zero)
        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union

        iou_per_class[class_idx] = iou

    return iou_per_class


def compute_miou(
    pred_labels: np.ndarray, gt_labels: np.ndarray, num_classes: int, ignore_index: int = -1
) -> float:
    """
    Compute mean Intersection over Union (mIoU) across all classes.

    Args:
        pred_labels: Predicted label array (H, W)
        gt_labels: Ground truth label array (H, W)
        num_classes: Number of classes
        ignore_index: Label value to ignore (e.g., unlabeled pixels)

    Returns:
        Mean IoU score
    """
    iou_per_class = compute_iou_per_class(pred_labels, gt_labels, num_classes, ignore_index)
    return np.mean(list(iou_per_class.values()))


def compute_pixel_accuracy(
    pred_labels: np.ndarray, gt_labels: np.ndarray, ignore_index: int = -1
) -> float:
    """
    Compute pixel-level accuracy.

    Args:
        pred_labels: Predicted label array (H, W)
        gt_labels: Ground truth label array (H, W)
        ignore_index: Label value to ignore (e.g., unlabeled pixels)

    Returns:
        Pixel accuracy (fraction of correctly classified pixels)
    """
    # Flatten arrays
    pred_flat = pred_labels.flatten()
    gt_flat = gt_labels.flatten()

    # Create mask for valid pixels
    if ignore_index is not None:
        valid_mask = gt_flat != ignore_index
        pred_flat = pred_flat[valid_mask]
        gt_flat = gt_flat[valid_mask]

    # Compute accuracy
    correct = np.sum(pred_flat == gt_flat)
    total = len(gt_flat)

    if total == 0:
        return 0.0

    return correct / total


def compute_pixel_accuracy_per_class(
    pred_labels: np.ndarray, gt_labels: np.ndarray, num_classes: int, ignore_index: int = -1
) -> Dict[int, float]:
    """
    Compute pixel accuracy for each class individually.

    Args:
        pred_labels: Predicted label array (H, W)
        gt_labels: Ground truth label array (H, W)
        num_classes: Number of classes
        ignore_index: Label value to ignore (e.g., unlabeled pixels)

    Returns:
        Dictionary mapping class index to pixel accuracy for that class
    """
    accuracy_per_class = {}

    # Flatten arrays
    pred_flat = pred_labels.flatten()
    gt_flat = gt_labels.flatten()

    # Create mask for valid pixels
    if ignore_index is not None:
        valid_mask = gt_flat != ignore_index
        pred_flat = pred_flat[valid_mask]
        gt_flat = gt_flat[valid_mask]

    for class_idx in range(num_classes):
        # Get mask for current class in ground truth
        gt_mask = gt_flat == class_idx

        # Count pixels belonging to this class
        total_class_pixels = np.sum(gt_mask)

        if total_class_pixels == 0:
            accuracy_per_class[class_idx] = 0.0
            continue

        # Count correctly predicted pixels for this class
        correct_class_pixels = np.sum((pred_flat == class_idx) & gt_mask)

        # Compute accuracy for this class
        accuracy_per_class[class_idx] = correct_class_pixels / total_class_pixels

    return accuracy_per_class


def evaluate_segmentation(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
    use_hungarian: bool = True,
) -> EvaluationResults:
    """
    Complete evaluation of segmentation results.

    Args:
        pred_labels: Predicted label array (H, W) with cluster indices
        gt_labels: Ground truth label array (H, W) with class indices
        num_classes: Number of ground truth classes
        ignore_index: Label value to ignore in ground truth
        use_hungarian: Whether to use Hungarian matching (for unsupervised methods)

    Returns:
        EvaluationResults containing all metrics
    """
    num_pred_clusters = int(pred_labels.max()) + 1

    # Apply Hungarian matching if needed (for unsupervised methods)
    if use_hungarian:
        mapping, confusion = hungarian_matching(pred_labels, gt_labels, num_classes)
        matched_pred_labels = apply_cluster_mapping(pred_labels, mapping)
    else:
        mapping = None
        confusion = compute_confusion_matrix(pred_labels, gt_labels, num_classes)
        matched_pred_labels = pred_labels

    # Compute metrics
    miou = compute_miou(matched_pred_labels, gt_labels, num_classes, ignore_index)
    pixel_acc = compute_pixel_accuracy(matched_pred_labels, gt_labels, ignore_index)
    per_class_iou = compute_iou_per_class(
        matched_pred_labels, gt_labels, num_classes, ignore_index
    )
    per_class_pixel_acc = compute_pixel_accuracy_per_class(
        matched_pred_labels, gt_labels, num_classes, ignore_index
    )

    return EvaluationResults(
        miou=miou,
        pixel_accuracy=pixel_acc,
        per_class_iou=per_class_iou,
        per_class_pixel_accuracy=per_class_pixel_acc,
        confusion_matrix=confusion,
        cluster_to_class_mapping=mapping,
        num_predicted_clusters=num_pred_clusters,
        num_ground_truth_classes=num_classes,
    )
