#!/usr/bin/env python3
"""
Visualize Ground Truth vs V3 Predictions

Creates comparison visualizations showing:
- Original image
- Ground truth annotations (COCO format)
- V3 predictions
- Overlay comparison
"""

import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from datasets import load_from_disk


def visualize_sample_comparison(
    dataset_path: str,
    predictions_dir: str,
    sample_idx: int,
    output_dir: str = "data/output/gt_vs_pred"
):
    """
    Create GT vs Prediction comparison for a single sample.

    Args:
        dataset_path: Path to OAM-TCD dataset
        predictions_dir: Directory with V3 prediction PNGs
        sample_idx: Sample index in test set
        output_dir: Output directory for visualizations
    """
    # Load dataset
    test_path = Path(dataset_path) / "test"
    test_data = load_from_disk(str(test_path))
    sample = test_data[sample_idx]

    image_id = sample['image_id']
    image_np = np.array(sample['image'])

    # Parse ground truth
    coco_anns = json.loads(sample['coco_annotations']) if sample['coco_annotations'] else []

    print(f"Sample {sample_idx} (ID: {image_id})")
    print(f"  Ground truth trees: {len(coco_anns)}")

    # Load prediction
    pred_path = Path(predictions_dir) / f"{image_id}_prediction.png"
    if pred_path.exists():
        pred_labels = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        n_pred = len(np.unique(pred_labels)) - 1  # Exclude background
        print(f"  Predicted trees: {n_pred}")
    else:
        pred_labels = None
        n_pred = 0
        print("  No predictions found")

    # Check for black regions (invalid data)
    black_mask = np.all(image_np == 0, axis=-1)
    black_pixels = black_mask.sum()
    black_percent = 100.0 * black_pixels / black_mask.size
    print(f"  Black pixels: {black_pixels:,} ({black_percent:.1f}%)")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # 1. Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title(f"Original Image (ID: {image_id})")
    axes[0, 0].axis('off')

    # Highlight black regions
    if black_percent > 1.0:
        axes[0, 0].contour(black_mask, levels=[0.5], colors='red', linewidths=2)
        axes[0, 0].text(10, 30, f"Black regions: {black_percent:.1f}%",
                       color='red', fontsize=12, weight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Ground truth annotations
    axes[0, 1].imshow(image_np)
    axes[0, 1].set_title(f"Ground Truth ({len(coco_anns)} trees)")
    axes[0, 1].axis('off')

    for ann in coco_anns:
        # COCO segmentation is list of [x1,y1,x2,y2,...] coordinates
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                try:
                    # Skip malformed segmentations
                    if not isinstance(seg, (list, np.ndarray)) or len(seg) < 6:
                        continue
                    # Reshape flat list to Nx2 array of (x,y) points
                    points = np.array(seg).reshape(-1, 2)
                    poly = Polygon(points, fill=False, edgecolor='lime', linewidth=2, alpha=0.8)
                    axes[0, 1].add_patch(poly)
                except (ValueError, TypeError):
                    continue

    # 3. V3 predictions
    if pred_labels is not None:
        axes[1, 0].imshow(pred_labels, cmap='tab20')
        axes[1, 0].set_title(f"V3 Predictions ({n_pred} instances)")
    else:
        axes[1, 0].imshow(image_np)
        axes[1, 0].text(0.5, 0.5, "No predictions",
                       transform=axes[1, 0].transAxes,
                       ha='center', va='center', fontsize=20, color='red')
        axes[1, 0].set_title("V3 Predictions (none)")
    axes[1, 0].axis('off')

    # 4. Overlay: GT + Predictions
    axes[1, 1].imshow(image_np, alpha=0.7)

    # Draw GT in green
    for ann in coco_anns:
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                try:
                    # Skip malformed segmentations
                    if not isinstance(seg, (list, np.ndarray)) or len(seg) < 6:
                        continue
                    points = np.array(seg).reshape(-1, 2)
                    poly = Polygon(points, fill=False, edgecolor='lime', linewidth=2,
                                 alpha=0.9, label='GT' if 'GT' not in axes[1, 1].get_legend_handles_labels()[1] else '')
                    axes[1, 1].add_patch(poly)
                except (ValueError, TypeError):
                    continue

    # Draw predictions in red
    if pred_labels is not None:
        from skimage import measure
        for instance_id in range(1, n_pred + 1):
            mask = pred_labels == instance_id
            if mask.any():
                contours = measure.find_contours(mask.astype(float), 0.5)
                for contour in contours:
                    # Swap columns (find_contours returns row, col)
                    contour = contour[:, [1, 0]]
                    poly = Polygon(contour, fill=False, edgecolor='red', linewidth=1.5,
                                 alpha=0.7, linestyle='--',
                                 label='Pred' if 'Pred' not in axes[1, 1].get_legend_handles_labels()[1] else '')
                    axes[1, 1].add_patch(poly)

    axes[1, 1].set_title("Overlay: GT (green) vs Pred (red)")
    axes[1, 1].axis('off')
    axes[1, 1].legend(loc='upper right')

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_path = output_path / f"{image_id}_comparison.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved comparison to: {save_path}")
    print()


def visualize_multiple_samples(
    dataset_path: str = "data/oam_tcd",
    predictions_dir: str = "data/oam_tcd/v3_predictions_balanced",
    sample_indices: list = None,
    n_samples: int = 5
):
    """Visualize multiple samples."""

    if sample_indices is None:
        # Use first n samples
        sample_indices = list(range(n_samples))

    print("=" * 80)
    print("Ground Truth vs V3 Predictions Comparison")
    print("=" * 80)
    print()

    for idx in sample_indices:
        visualize_sample_comparison(
            dataset_path=dataset_path,
            predictions_dir=predictions_dir,
            sample_idx=idx
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize GT vs V3 predictions")
    parser.add_argument("--dataset", type=str, default="data/oam_tcd",
                       help="Path to OAM-TCD dataset")
    parser.add_argument("--predictions", type=str,
                       default="data/oam_tcd/v3_predictions_balanced",
                       help="Path to V3 predictions directory")
    parser.add_argument("--samples", type=int, nargs="+", default=None,
                       help="Sample indices to visualize (default: first 5)")
    parser.add_argument("--n-samples", type=int, default=5,
                       help="Number of samples to visualize if --samples not specified")

    args = parser.parse_args()

    visualize_multiple_samples(
        dataset_path=args.dataset,
        predictions_dir=args.predictions,
        sample_indices=args.samples,
        n_samples=args.n_samples
    )
