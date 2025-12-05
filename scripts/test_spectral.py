#!/usr/bin/env python3
"""Quick test of spectral clustering vs K-means baseline."""

from tree_seg.core.types import Config
from tree_seg.api import TreeSegmentation
from tree_seg.evaluation.datasets import FortressDataset
from tree_seg.evaluation.metrics import evaluate_segmentation
import numpy as np

# Load one image
dataset = FortressDataset("data/fortress_processed")
image, gt_labels, image_id = dataset[0]
print(f"Image: {image_id}, shape: {image.shape}")

# Count GT classes
unique_gt = np.unique(gt_labels)
unique_gt = unique_gt[unique_gt != dataset.IGNORE_INDEX]
smart_k = len(unique_gt)
print(f"GT classes: {smart_k}")

configs = [
    ("kmeans_slic", {"clustering_method": "kmeans", "refine": "slic"}),
    ("spectral", {"clustering_method": "spectral", "refine": None}),
    ("spectral_slic", {"clustering_method": "spectral", "refine": "slic"}),
]

results = []

for label, cfg_opts in configs:
    print(f"\n=== Testing: {label} ===")
    
    config = Config(
        n_clusters=smart_k,
        auto_k=False,  # Use smart K
        verbose=False,
        **cfg_opts
    )
    
    seg = TreeSegmentation(config)
    result = seg.segment_image(image)
    pred = result.labels_resized
    
    # Compute mIoU
    eval_result = evaluate_segmentation(
        pred_labels=pred,
        gt_labels=gt_labels,
        num_classes=dataset.NUM_CLASSES,
        ignore_index=dataset.IGNORE_INDEX,
        use_hungarian=True,
    )
    
    results.append({
        "label": label,
        "miou": eval_result.miou,
        "pixel_acc": eval_result.pixel_accuracy,
    })
    
    print(f"mIoU: {eval_result.miou:.4f}, Pixel Acc: {eval_result.pixel_accuracy:.4f}")

# Summary
print("\n" + "="*60)
print("SPECTRAL CLUSTERING RESULTS")
print("="*60)
baseline = results[0]["miou"]
for r in results:
    diff = r["miou"] - baseline
    diff_str = f"({diff:+.4f})" if diff != 0 else ""
    print(f"{r['label']:<15} mIoU: {r['miou']:.4f} {diff_str}  PA: {r['pixel_acc']:.4f}")
