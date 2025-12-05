#!/usr/bin/env python3
"""Debug mIoU calculation for multi-layer vs baseline."""

import numpy as np
from tree_seg.core.types import Config
from tree_seg.api import TreeSegmentation
from tree_seg.evaluation.datasets import FortressDataset
from tree_seg.evaluation.metrics import evaluate_segmentation

# Load one image
dataset = FortressDataset("data/fortress_processed")
image, gt_labels, image_id = dataset[0]
print(f"Image: {image_id}, shape: {image.shape}")
print(f"GT labels shape: {gt_labels.shape}")
print(f"GT unique labels: {np.unique(gt_labels)}")

# Test 1: Baseline
print("\n=== BASELINE (single layer) ===")
config1 = Config(
    use_multi_layer=False,
    verbose=False,
)
seg1 = TreeSegmentation(config1)
result1 = seg1.segment_image(image)
pred1 = result1.labels_resized
print(f"Pred1 unique labels: {np.unique(pred1)}")
print(f"Pred1 shape: {pred1.shape}")

# Compute mIoU manually
eval1 = evaluate_segmentation(
    pred_labels=pred1,
    gt_labels=gt_labels,
    num_classes=dataset.NUM_CLASSES,
    ignore_index=dataset.IGNORE_INDEX,
    use_hungarian=True,
)
print(f"BASELINE mIoU: {eval1.miou:.6f}")
print(f"BASELINE PA: {eval1.pixel_accuracy:.6f}")
print(f"BASELINE per-class IoU: {eval1.per_class_iou}")

# Test 2: Multi-layer concat
print("\n=== MULTI-LAYER CONCAT ===")
config2 = Config(
    use_multi_layer=True,
    layer_indices=(3, 6, 9, 12),
    feature_aggregation="concat",
    pca_dim=512,
    verbose=False,
)
seg2 = TreeSegmentation(config2)
result2 = seg2.segment_image(image)
pred2 = result2.labels_resized
print(f"Pred2 unique labels: {np.unique(pred2)}")
print(f"Pred2 shape: {pred2.shape}")

# Compute mIoU manually
eval2 = evaluate_segmentation(
    pred_labels=pred2,
    gt_labels=gt_labels,
    num_classes=dataset.NUM_CLASSES,
    ignore_index=dataset.IGNORE_INDEX,
    use_hungarian=True,
)
print(f"MULTI_CONCAT mIoU: {eval2.miou:.6f}")
print(f"MULTI_CONCAT PA: {eval2.pixel_accuracy:.6f}")
print(f"MULTI_CONCAT per-class IoU: {eval2.per_class_iou}")

# Compare predictions
print("\n=== COMPARISON ===")
print(f"Labels identical: {np.array_equal(pred1, pred2)}")
diff_count = np.sum(pred1 != pred2)
print(f"Different pixels: {diff_count} ({100*diff_count/pred1.size:.2f}%)")
print(f"mIoU difference: {eval2.miou - eval1.miou:.6f}")
