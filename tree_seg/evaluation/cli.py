"""Shared CLI utilities for evaluation scripts."""

import argparse
from pathlib import Path


def add_common_eval_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add common evaluation arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser to add arguments to
        
    Returns:
        Modified parser with common arguments added
    """
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset directory",
    )

    # Method configuration
    parser.add_argument(
        "--method",
        type=str,
        default="v1.5",
        choices=["v1", "v1.5", "v2", "v3", "v4"],
        help="Segmentation method version (default: v1.5)",
    )

    parser.add_argument(
        "--clustering",
        type=str,
        default="kmeans",
        choices=["kmeans", "slic", "bilateral"],
        help="Clustering method (default: kmeans)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["small", "base", "large", "mega"],
        help="DINOv3 model size (default: base)",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Feature extraction stride (default: 4)",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Image resize dimension (default: 1024)",
    )

    parser.add_argument(
        "--elbow-threshold",
        type=float,
        default=5.0,
        help="Elbow method threshold for auto K selection (default: 5.0)",
    )

    parser.add_argument(
        "--fixed-k",
        type=int,
        default=None,
        help="Fixed number of clusters (overrides auto K selection)",
    )
    
    # V3-specific arguments
    parser.add_argument(
        "--apply-vegetation-filter",
        action="store_true",
        help="Apply ExG-based vegetation filtering (works with any method)",
    )
    
    parser.add_argument(
        "--exg-threshold",
        type=float,
        default=0.10,
        help="ExG threshold for vegetation filtering (default: 0.10)",
   )

    # Evaluation options
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )

    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting sample index (default: 0)",
    )

    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualization images",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser


def add_comparison_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add comparison/sweep mode arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser to add arguments to
        
    Returns:
        Modified parser with comparison arguments added
    """
    parser.add_argument(
        "--compare-configs",
        action="store_true",
        help="Run comparison across multiple configurations",
    )

    parser.add_argument(
        "--smart-grid",
        action="store_true",
        help="Smart grid search: test best combinations",
    )

    return parser
