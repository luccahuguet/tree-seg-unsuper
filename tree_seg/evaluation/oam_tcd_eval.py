"""
OAM-TCD Tree Detection Evaluation

Evaluates V3 tree segmentation against OAM-TCD ground truth.

Metrics:
- Instance-level Precision/Recall/F1
- IoU per tree instance
- Detection rate at various IoU thresholds
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from datasets import load_from_disk
from dataclasses import dataclass
import cv2


@dataclass
class TreeDetectionMetrics:
    """Tree detection evaluation metrics."""

    # Instance-level metrics
    precision: float
    recall: float
    f1_score: float

    # IoU metrics
    mean_iou: float
    median_iou: float

    # Detection metrics at different IoU thresholds
    ap_50: float  # Average Precision @ IoU=0.5
    ap_75: float  # Average Precision @ IoU=0.75

    # Counts
    num_predictions: int
    num_ground_truth: int
    true_positives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mean_iou": self.mean_iou,
            "median_iou": self.median_iou,
            "ap_50": self.ap_50,
            "ap_75": self.ap_75,
            "num_predictions": self.num_predictions,
            "num_ground_truth": self.num_ground_truth,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


def polygon_to_mask(polygon: List[float], height: int, width: int) -> np.ndarray:
    """Convert polygon coordinates to binary mask.

    Args:
        polygon: Flat list of [x1, y1, x2, y2, ...] coordinates
        height: Mask height
        width: Mask width

    Returns:
        Binary mask (H, W) with 1s inside polygon
    """
    # Reshape to (N, 2) points
    points = np.array(polygon).reshape(-1, 2).astype(np.int32)

    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)

    return mask


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks.

    Args:
        mask1: Binary mask (H, W)
        mask2: Binary mask (H, W)

    Returns:
        IoU score in [0, 1]
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def match_predictions_to_ground_truth(
    pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Match predicted instances to ground truth using Hungarian matching.

    Args:
        pred_masks: List of predicted instance masks
        gt_masks: List of ground truth instance masks
        iou_threshold: Minimum IoU to consider a match

    Returns:
        matches: List of (pred_idx, gt_idx, iou) tuples
        unmatched_preds: List of prediction indices without matches
        unmatched_gts: List of ground truth indices without matches
    """
    if len(pred_masks) == 0:
        return [], [], list(range(len(gt_masks)))

    if len(gt_masks) == 0:
        return [], list(range(len(pred_masks))), []

    # Compute IoU matrix (preds x gts)
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))

    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    # Simple greedy matching (for now - can upgrade to Hungarian later)
    matches = []
    matched_preds = set()
    matched_gts = set()

    # Sort by IoU (highest first)
    iou_sorted = []
    for i in range(iou_matrix.shape[0]):
        for j in range(iou_matrix.shape[1]):
            if iou_matrix[i, j] >= iou_threshold:
                iou_sorted.append((i, j, iou_matrix[i, j]))

    iou_sorted.sort(key=lambda x: x[2], reverse=True)

    # Greedy assignment
    for pred_idx, gt_idx, iou in iou_sorted:
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matches.append((pred_idx, gt_idx, iou))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)

    # Find unmatched
    unmatched_preds = [i for i in range(len(pred_masks)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_masks)) if i not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


def evaluate_sample(
    pred_mask: np.ndarray, gt_annotations: List[Dict], iou_threshold: float = 0.5
) -> TreeDetectionMetrics:
    """Evaluate tree detection on a single sample.

    Args:
        pred_mask: Predicted instance mask (H, W) with unique ID per tree
        gt_annotations: List of COCO-format annotations
        iou_threshold: IoU threshold for matching

    Returns:
        TreeDetectionMetrics object
    """
    height, width = pred_mask.shape

    # Extract predicted instances (category 1 = individual trees only)
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0]  # Remove background
    pred_masks = [pred_mask == pid for pid in pred_ids]

    # Extract ground truth trees (category_id == 1)
    gt_trees = [ann for ann in gt_annotations if ann.get("category_id") == 1]
    gt_masks = []

    for ann in gt_trees:
        seg = ann.get("segmentation")
        if seg and isinstance(seg, list) and len(seg) > 0:
            # Convert first polygon to mask
            polygon = seg[0]
            mask = polygon_to_mask(polygon, height, width)
            gt_masks.append(mask)

    # Match predictions to ground truth
    matches, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
        pred_masks, gt_masks, iou_threshold
    )

    # Compute metrics
    tp = len(matches)
    fp = len(unmatched_preds)
    fn = len(unmatched_gts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # IoU statistics
    ious = [iou for _, _, iou in matches]
    mean_iou = np.mean(ious) if ious else 0.0
    median_iou = np.median(ious) if ious else 0.0

    # AP at different thresholds
    matches_50, _, _ = match_predictions_to_ground_truth(pred_masks, gt_masks, 0.5)
    matches_75, _, _ = match_predictions_to_ground_truth(pred_masks, gt_masks, 0.75)

    ap_50 = len(matches_50) / len(pred_masks) if len(pred_masks) > 0 else 0.0
    ap_75 = len(matches_75) / len(pred_masks) if len(pred_masks) > 0 else 0.0

    return TreeDetectionMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        mean_iou=mean_iou,
        median_iou=median_iou,
        ap_50=ap_50,
        ap_75=ap_75,
        num_predictions=len(pred_masks),
        num_ground_truth=len(gt_masks),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


class OAMTCDEvaluator:
    """Evaluator for V3 predictions against OAM-TCD dataset."""

    def __init__(self, dataset_path: str = "data/datasets/oam_tcd"):
        """Initialize evaluator.

        Args:
            dataset_path: Path to OAM-TCD dataset directory
        """
        self.dataset_path = Path(dataset_path)

        # Load test split
        test_path = self.dataset_path / "test"
        if test_path.exists():
            print(f"Loading OAM-TCD test split from {test_path}...")
            self.test_data = load_from_disk(str(test_path))
            print(f"Loaded {len(self.test_data)} test samples")
        else:
            raise FileNotFoundError(f"Test split not found at {test_path}")

    def evaluate(
        self,
        predictions_dir: Path,
        iou_threshold: float = 0.5,
        max_samples: Optional[int] = None,
    ) -> Dict:
        """Evaluate predictions on test set.

        Args:
            predictions_dir: Directory containing prediction masks
                Format: {image_id}_prediction.png with instance IDs
            iou_threshold: IoU threshold for matching
            max_samples: If specified, only evaluate first N samples

        Returns:
            Dictionary with aggregate metrics and per-sample results
        """
        results = []

        num_samples = (
            min(len(self.test_data), max_samples)
            if max_samples
            else len(self.test_data)
        )

        print(f"\nEvaluating {num_samples} samples...")

        for idx in range(num_samples):
            sample = self.test_data[idx]
            image_id = sample["image_id"]

            # Load prediction
            pred_path = predictions_dir / f"{image_id}_prediction.png"

            if not pred_path.exists():
                print(f"Warning: Prediction not found for image {image_id}, skipping")
                continue

            pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)

            # Load ground truth annotations
            gt_annotations = json.loads(sample["coco_annotations"])

            # Evaluate
            metrics = evaluate_sample(pred_mask, gt_annotations, iou_threshold)

            results.append({"image_id": image_id, "metrics": metrics.to_dict()})

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{num_samples} samples")

        # Aggregate metrics
        aggregate = self._aggregate_metrics(results)

        return {
            "aggregate": aggregate,
            "per_sample": results,
            "num_samples": len(results),
        }

    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate per-sample metrics."""
        if not results:
            return {}

        # Sum counts
        total_tp = sum(r["metrics"]["true_positives"] for r in results)
        total_fp = sum(r["metrics"]["false_positives"] for r in results)
        total_fn = sum(r["metrics"]["false_negatives"] for r in results)
        total_preds = sum(r["metrics"]["num_predictions"] for r in results)
        total_gts = sum(r["metrics"]["num_ground_truth"] for r in results)

        # Compute aggregate precision/recall/F1
        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Mean IoU across all samples
        mean_iou = np.mean([r["metrics"]["mean_iou"] for r in results])
        median_iou = np.median([r["metrics"]["median_iou"] for r in results])

        # Mean AP
        mean_ap_50 = np.mean([r["metrics"]["ap_50"] for r in results])
        mean_ap_75 = np.mean([r["metrics"]["ap_75"] for r in results])

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mean_iou": mean_iou,
            "median_iou": median_iou,
            "ap_50": mean_ap_50,
            "ap_75": mean_ap_75,
            "total_predictions": total_preds,
            "total_ground_truth": total_gts,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        }


def print_metrics(metrics: Dict):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 80)
    print("OAM-TCD Tree Detection Evaluation Results")
    print("=" * 80)

    agg = metrics["aggregate"]

    print("\nInstance-Level Metrics:")
    print(f"  Precision:  {agg['precision']:.3f}")
    print(f"  Recall:     {agg['recall']:.3f}")
    print(f"  F1 Score:   {agg['f1_score']:.3f}")

    print("\nIoU Metrics:")
    print(f"  Mean IoU:   {agg['mean_iou']:.3f}")
    print(f"  Median IoU: {agg['median_iou']:.3f}")

    print("\nAverage Precision:")
    print(f"  AP @ 0.5:   {agg['ap_50']:.3f}")
    print(f"  AP @ 0.75:  {agg['ap_75']:.3f}")

    print("\nCounts:")
    print(f"  Predictions:     {agg['total_predictions']}")
    print(f"  Ground Truth:    {agg['total_ground_truth']}")
    print(f"  True Positives:  {agg['true_positives']}")
    print(f"  False Positives: {agg['false_positives']}")
    print(f"  False Negatives: {agg['false_negatives']}")

    print(f"\nSamples Evaluated: {metrics['num_samples']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate V3 predictions on OAM-TCD")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/datasets/oam_tcd",
        help="Path to OAM-TCD dataset",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Directory containing prediction masks",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching (default: 0.5)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file for results"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = OAMTCDEvaluator(args.dataset)
    results = evaluator.evaluate(
        Path(args.predictions),
        iou_threshold=args.iou_threshold,
        max_samples=args.max_samples,
    )

    # Print results
    print_metrics(results)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {output_path}")
