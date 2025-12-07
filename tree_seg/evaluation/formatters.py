"""Formatting utilities for benchmark results."""

from typing import List, Dict, Any
from pathlib import Path
import json


def format_comparison_table(results: List[Dict[str, Any]]) -> str:
    """
    Format benchmark results as a comparison table.

    Args:
        results: List of dicts with 'label', 'config', and 'results' keys

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("COMPARISON RESULTS")
    lines.append("=" * 60)
    lines.append(
        f"{'Configuration':<20} {'mIoU':>8} {'Px Acc':>8} {'Per Img':>10} {'Total':>10}"
    )
    lines.append("-" * 60)

    for item in results:
        label = item["label"]
        r = item["results"]
        total_time = r.mean_runtime * r.total_samples
        lines.append(
            f"{label:<20} {r.mean_miou:>8.3f} {r.mean_pixel_accuracy:>8.1%} "
            f"{r.mean_runtime:>9.2f}s {total_time:>9.1f}s"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


def save_comparison_summary(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save comparison results to JSON.

    Args:
        results: List of dicts with 'label', 'config', and 'results' keys
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_dict = {
        "configurations": [
            {
                "label": item["label"],
                "config": item["config"],
                "mean_miou": float(item["results"].mean_miou),
                "mean_pixel_accuracy": float(item["results"].mean_pixel_accuracy),
                "mean_runtime": float(item["results"].mean_runtime),
            }
            for item in results
        ]
    }

    with open(output_path, "w") as f:
        json.dump(comparison_dict, f, indent=2)


def config_to_dict(config) -> Dict[str, Any]:
    """
    Convert Config object to serializable dict.

    Args:
        config: Config instance

    Returns:
        Dict with relevant config fields
    """
    return {
        "clustering": config.clustering_method,
        "refine": config.refine,
        "model": config.model_display_name,
        "stride": config.stride,
        "elbow_threshold": config.elbow_threshold,
        "auto_k": config.auto_k,
        "fixed_k": config.n_clusters if not config.auto_k else None,
        "supervised": config.supervised,
    }
