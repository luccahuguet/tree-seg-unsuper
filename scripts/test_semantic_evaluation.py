#!/usr/bin/env python3
"""
Test script to validate benchmark implementation.

This creates synthetic test data and validates that all components work.
"""

import numpy as np
from pathlib import Path

from tree_seg.evaluation.metrics import evaluate_segmentation, hungarian_matching


def test_metrics():
    """Test basic metric computation."""
    print("Testing evaluation metrics...")

    # Create synthetic data
    # Ground truth: 3 classes (0, 1, 2)
    gt_labels = np.array([
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
    ])

    # Prediction: 3 clusters (but permuted indices)
    # Let's say cluster 0 -> class 2, cluster 1 -> class 0, cluster 2 -> class 1
    pred_labels = np.array([
        [1, 1, 2, 2, 0, 0],
        [1, 1, 2, 2, 0, 0],
        [1, 1, 2, 2, 0, 0],
    ])

    # Test Hungarian matching
    mapping, confusion = hungarian_matching(pred_labels, gt_labels, num_classes=3)
    print(f"  Cluster-to-class mapping: {mapping}")
    print(f"  Confusion matrix shape: {confusion.shape}")

    # Test evaluation
    results = evaluate_segmentation(
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        num_classes=3,
        use_hungarian=True,
    )

    print(f"  mIoU: {results.miou:.3f}")
    print(f"  Pixel Accuracy: {results.pixel_accuracy:.3f}")
    print(f"  Per-class IoU: {results.per_class_iou}")

    # With perfect matching, we should get 100% accuracy
    assert results.pixel_accuracy == 1.0, "Expected perfect accuracy with synthetic data"
    assert results.miou == 1.0, "Expected perfect mIoU with synthetic data"

    print("  ✓ Metrics test passed!\n")


def test_dataset_structure():
    """Test dataset directory structure."""
    print("Testing dataset structure...")

    dataset_path = Path("data/isprs_potsdam")

    if not dataset_path.exists():
        print(f"  ⚠ Dataset not found at {dataset_path}")
        print("  This is expected if you haven't downloaded the dataset yet.")
        print("  See data/isprs_potsdam/README.md for download instructions.\n")
        return

    images_path = dataset_path / "images"
    labels_path = dataset_path / "labels"

    if images_path.exists() and labels_path.exists():
        num_images = len(list(images_path.glob("*.tif"))) + len(list(images_path.glob("*.png")))
        num_labels = len(list(labels_path.glob("*.tif"))) + len(list(labels_path.glob("*.png")))

        print(f"  Found {num_images} images")
        print(f"  Found {num_labels} labels")

        if num_images > 0 and num_labels > 0:
            print("  ✓ Dataset structure is valid!\n")
        else:
            print("  ⚠ Dataset directories exist but no files found.\n")
    else:
        print("  ⚠ Missing images/ or labels/ directories\n")


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        import tree_seg.evaluation  # noqa: F401
        print("  ✓ All imports successful!\n")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}\n")
        raise


def main():
    """Run all tests."""
    print("=" * 60)
    print("BENCHMARK IMPLEMENTATION TEST")
    print("=" * 60 + "\n")

    test_imports()
    test_metrics()
    test_dataset_structure()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
