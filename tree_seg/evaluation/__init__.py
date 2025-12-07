"""Evaluation module for benchmarking segmentation methods."""

from tree_seg.evaluation.benchmark import (
    BenchmarkResults,
    BenchmarkRunner,
    run_benchmark,
)
from tree_seg.evaluation.datasets import ISPRSPotsdamDataset, load_isprs_potsdam
from tree_seg.evaluation.metrics import (
    EvaluationResults,
    compute_miou,
    compute_pixel_accuracy,
    evaluate_segmentation,
    hungarian_matching,
)

__all__ = [
    "compute_miou",
    "compute_pixel_accuracy",
    "hungarian_matching",
    "evaluate_segmentation",
    "EvaluationResults",
    "ISPRSPotsdamDataset",
    "load_isprs_potsdam",
    "BenchmarkRunner",
    "BenchmarkResults",
    "run_benchmark",
]
