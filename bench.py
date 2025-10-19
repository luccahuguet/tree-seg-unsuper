#!/usr/bin/env python3
"""
Benchmark execution script for evaluating segmentation methods.

Example usage:
    # Run V1.5 baseline with default settings
    python bench.py --dataset data/isprs_potsdam --method v1.5

    # Run with specific model and settings
    python bench.py \
        --dataset data/isprs_potsdam \
        --method v1.5 \
        --model large \
        --stride 4 \
        --elbow-threshold 10.0 \
        --num-samples 5 \
        --save-viz

    # Compare multiple configurations
    python bench.py \
        --dataset data/isprs_potsdam \
        --method v1.5 \
        --model base \
        --compare-configs
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import run_benchmark

# Load environment variables from .env file
load_dotenv()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on segmentation methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset directory (e.g., data/isprs_potsdam)",
    )

    # Method configuration
    parser.add_argument(
        "--method",
        type=str,
        default="v1.5",
        choices=["v1", "v1.5", "v2", "v3"],
        help="Segmentation method version (default: v1.5)",
    )

    parser.add_argument(
        "--clustering",
        type=str,
        default="kmeans",
        choices=["kmeans", "slic"],
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
        help="Output directory for results (default: results/<method>_<timestamp>)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    # Comparison mode
    parser.add_argument(
        "--compare-configs",
        action="store_true",
        help="Run comparison across multiple configurations",
    )

    parser.add_argument(
        "--smart-grid",
        action="store_true",
        help="Smart grid search: test best combinations (small/base × elbow 10/20 × kmeans/slic = 8 configs)",
    )

    return parser.parse_args()


def create_config(args) -> Config:
    """Create Config object from command-line arguments."""
    # Map method version to version string
    version_map = {"v1": "v1", "v1.5": "v1.5", "v2": "v2", "v3": "v3"}
    version = version_map.get(args.method, "v3")

    # Map clustering to refine parameter
    # "kmeans" = no refinement (None), "slic" = SLIC refinement
    refine = "slic" if args.clustering == "slic" else None

    # Create config
    config = Config(
        version=version,
        refine=refine,
        model_name=args.model,
        stride=args.stride,
        elbow_threshold=args.elbow_threshold,
        n_clusters=args.fixed_k if args.fixed_k else 6,
        auto_k=(args.fixed_k is None),
    )

    return config


def run_single_benchmark(args, config: Config):
    """Run a single benchmark with given configuration."""
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        refine_str = config.refine if config.refine else "kmeans"
        method_str = f"{config.version}_{refine_str}"
        model_str = config.model_display_name.lower().replace(" ", "_")
        output_dir = Path("results") / f"{method_str}_{model_str}_{timestamp}"

    # Run benchmark
    results = run_benchmark(
        config=config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        num_samples=args.num_samples,
        save_visualizations=args.save_viz,
        verbose=not args.quiet,
    )

    # Save results to JSON
    results_dict = {
        "dataset": results.dataset_name,
        "method": results.method_name,
        "config": {
            "version": config.version,
            "refine": config.refine,
            "model": config.model_display_name,
            "stride": config.stride,
            "elbow_threshold": config.elbow_threshold,
            "auto_k": config.auto_k,
            "fixed_k": config.n_clusters if not config.auto_k else None,
        },
        "metrics": {
            "mean_miou": float(results.mean_miou),
            "mean_pixel_accuracy": float(results.mean_pixel_accuracy),
            "mean_runtime": float(results.mean_runtime),
        },
        "samples": [
            {
                "image_id": s.image_id,
                "miou": float(s.miou),
                "pixel_accuracy": float(s.pixel_accuracy),
                "num_clusters": int(s.num_clusters),
                "runtime_seconds": float(s.runtime_seconds),
            }
            for s in results.samples
        ],
        "total_samples": results.total_samples,
    }

    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    if not args.quiet:
        print(f"\nResults saved to: {output_path}")

    return results


def run_comparison_benchmark(args):
    """Run comparison across multiple configurations."""
    # Smart grid search: Test best combinations
    if args.smart_grid:
        configs_to_test = [
            # small × thresholds × refinement (4 configs)
            {"model_name": "small", "elbow_threshold": 10.0, "refine": None, "label": "small_e10_km"},
            {"model_name": "small", "elbow_threshold": 10.0, "refine": "slic", "label": "small_e10_slic"},
            {"model_name": "small", "elbow_threshold": 20.0, "refine": None, "label": "small_e20_km"},
            {"model_name": "small", "elbow_threshold": 20.0, "refine": "slic", "label": "small_e20_slic"},
            # base × thresholds × refinement (4 configs)
            {"model_name": "base", "elbow_threshold": 10.0, "refine": None, "label": "base_e10_km"},
            {"model_name": "base", "elbow_threshold": 10.0, "refine": "slic", "label": "base_e10_slic"},
            {"model_name": "base", "elbow_threshold": 20.0, "refine": None, "label": "base_e20_km"},
            {"model_name": "base", "elbow_threshold": 20.0, "refine": "slic", "label": "base_e20_slic"},
        ]
    else:
        # OFAT: One-factor-at-time exploration
        configs_to_test = [
            # Different elbow thresholds
            {"elbow_threshold": 2.5, "label": "elbow_2.5"},
            {"elbow_threshold": 5.0, "label": "elbow_5.0"},
            {"elbow_threshold": 10.0, "label": "elbow_10.0"},
            {"elbow_threshold": 20.0, "label": "elbow_20.0"},
            # Different models (with default threshold)
            {"model_name": "small", "label": "model_small"},
            {"model_name": "base", "label": "model_base"},
            {"model_name": "large", "label": "model_large"},
            # Different refinement methods
            {"refine": None, "label": "refine_kmeans"},
            {"refine": "slic", "label": "refine_slic"},
        ]

    # Base config from args
    base_config_dict = {
        "version": {"v1": "v1", "v1.5": "v1.5", "v2": "v2", "v3": "v3"}.get(args.method, "v3"),
        "refine": "slic" if args.clustering == "slic" else None,
        "model_name": args.model,
        "stride": args.stride,
        "elbow_threshold": args.elbow_threshold,
        "auto_k": True,
    }

    all_results = []

    print("\n" + "=" * 60)
    print("RUNNING COMPARISON BENCHMARK")
    print("=" * 60 + "\n")

    for i, config_override in enumerate(configs_to_test):
        config_dict = base_config_dict.copy()
        config_dict.update({k: v for k, v in config_override.items() if k != "label"})

        config = Config(**config_dict)
        label = config_override["label"]

        print(f"\n[{i + 1}/{len(configs_to_test)}] Testing: {label}")
        print("-" * 40)

        # Update args for this run
        args_copy = argparse.Namespace(**vars(args))
        args_copy.output_dir = Path("results") / f"comparison_{label}"

        results = run_single_benchmark(args_copy, config)
        all_results.append({"label": label, "config": config_dict, "results": results})

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Configuration':<20} {'mIoU':>8} {'Px Acc':>8} {'Per Img':>10} {'Total':>10}")
    print("-" * 60)

    for item in all_results:
        label = item["label"]
        r = item["results"]
        total_time = r.mean_runtime * r.total_samples
        print(
            f"{label:<20} {r.mean_miou:>8.3f} {r.mean_pixel_accuracy:>8.1%} "
            f"{r.mean_runtime:>9.2f}s {total_time:>9.1f}s"
        )

    print("=" * 60 + "\n")

    # Save comparison summary
    comparison_path = Path("results") / "comparison_summary.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_dict = {
        "configurations": [
            {
                "label": item["label"],
                "config": item["config"],
                "mean_miou": float(item["results"].mean_miou),
                "mean_pixel_accuracy": float(item["results"].mean_pixel_accuracy),
                "mean_runtime": float(item["results"].mean_runtime),
            }
            for item in all_results
        ]
    }

    with open(comparison_path, "w") as f:
        json.dump(comparison_dict, f, indent=2)

    print(f"Comparison summary saved to: {comparison_path}\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Validate dataset path
    if not args.dataset.exists():
        print(f"Error: Dataset path does not exist: {args.dataset}")
        return 1

    # Run comparison or single benchmark
    if args.compare_configs:
        run_comparison_benchmark(args)
    else:
        config = create_config(args)
        run_single_benchmark(args, config)

    return 0


if __name__ == "__main__":
    exit(main())
