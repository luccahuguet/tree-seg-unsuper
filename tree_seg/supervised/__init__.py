"""Supervised segmentation baseline using DINOv3 features."""

from tree_seg.supervised.sklearn_baseline import (
    evaluate_sklearn_baseline,
    predict_sklearn,
    train_sklearn_classifier,
)

__all__ = [
    "evaluate_sklearn_baseline",
    "train_sklearn_classifier",
    "predict_sklearn",
]
