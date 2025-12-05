#!/usr/bin/env python3
"""
Evaluate Semantic Segmentation on FORTRESS Dataset

Evaluates segmentation methods on the FORTRESS species-level dataset.
Uses the generic BenchmarkRunner infrastructure with FortressDataset.

Example usage:
    # Run V3 species clustering on FORTRESS
    python scripts/evaluate_fortress.py \
        --dataset data/fortress_processed \
        --method v3 \
        --model base \
        --num-samples 5 \
        --save-viz
    
    # Run V1.5 baseline
    python scripts/evaluate_fortress.py \
        --dataset data/fortress_processed \
        --method v1.5 \
        --elbow-threshold 10.0
"""

import datetime
import json
from pathlib import Path

from dotenv import load_dotenv

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import run_benchmark
from tree_seg.evaluation.cli import add_common_eval_arguments, add_comparison_arguments
from tree_seg.evaluation.datasets import FortressDataset
from tree_seg.evaluation.formatters import config_to_dict, format_comparison_table, save_comparison_summary
from tree_seg.evaluation.grids import get_grid

# Load environment variables from .env file
load_dotenv()


def create_config(args) -> Config:
    """Create Config object from command-line arguments."""
    # Map method version to version string
    version_map = {"v1": "v1", "v1.5": "v1.5", "v2": "v2", "v3": "v3", "v4": "v4"}
    version = version_map.get(args.method, "v3")

    if version == "v4":
        if args.model != "mega":
            print("⚠️  Mask2Former head only supports the ViT-7B backbone; overriding model to 'mega'.")
        image_size = args.image_size or 896
        if image_size == 1024:
            image_size = 896
        return Config(
            version="v4",
            model_name="mega",
            image_size=image_size,
            auto_k=False,
            n_clusters=6,
            refine=None,
            metrics=True,
            verbose=not args.quiet,
        )

    # Map clustering to refine parameter (default to slic for v1.x baselines)
    clustering = args.clustering or "slic"

    # Separate clustering method from refinement method
    clustering_method = "kmeans"  # Default
    if clustering in ["slic", "bilateral"]:
        refine = clustering
    elif clustering == "slic-skimage":
        refine = "slic_skimage"
    elif clustering == "gmm":
        clustering_method = "gmm"
        refine = None
    else:
        refine = None

    # Create config
    config = Config(
        version=version,
        clustering_method=clustering_method,
        refine=refine,
        model_name=args.model,
        stride=args.stride,
        elbow_threshold=args.elbow_threshold,
        n_clusters=args.fixed_k if args.fixed_k else 6,
        auto_k=(args.fixed_k is None),
        image_size=args.image_size,
        apply_vegetation_filter=args.apply_vegetation_filter or (version == "v3"),
        exg_threshold=args.exg_threshold,
        use_tiling=not args.no_tiling,
        viz_two_panel=args.viz_two_panel,
        viz_two_panel_opaque=args.viz_two_panel_opaque,
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
        if args.smart_k:
            method_str += "_smartk"
        model_str = config.model_display_name.lower().replace(" ", "_")
        output_dir = Path("data/output/results") / f"fortress_{method_str}_{model_str}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load FORTRESS dataset
    dataset = FortressDataset(args.dataset)

    # Run benchmark
    results = run_benchmark(
        config=config,
        dataset=dataset,
        output_dir=output_dir,
        num_samples=args.num_samples,
        save_visualizations=args.save_viz,
        verbose=not args.quiet,
        use_smart_k=args.smart_k,
    )

    # Save results to JSON
    results_dict = {
        "dataset": "FORTRESS",
        "method": results.method_name,
        "config": config_to_dict(config),
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
        "num_classes": FortressDataset.NUM_CLASSES,
    }

    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    if not args.quiet:
        print(f"\n✅ Results saved to: {output_path}")

    return results


def run_comparison_benchmark(args):
    """Run comparison across multiple configurations."""
    # Select grid based on args
    if hasattr(args, 'grid') and args.grid:
        grid_name = args.grid
    elif args.smart_grid:
        grid_name = "smart"
    else:
        grid_name = "tiling"  # Default to tiling grid for FORTRESS

    grid = get_grid(grid_name)
    configs_to_test = grid["configs"]

    print(f"\nUsing grid: {grid['name']}")
    print(f"Description: {grid['description']}")
    print(f"Configurations: {len(configs_to_test)}\n")

    # Create shared sweep directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    smartk_suffix = "_smartk" if args.smart_k else ""
    sweep_dir = Path("data/output/results") / f"sweep_{grid_name}{smartk_suffix}_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Sweep directory: {sweep_dir}\n")

    # Base config from args
    base_config_dict = {
        "version": {"v1": "v1", "v1.5": "v1.5", "v2": "v2", "v3": "v3", "v4": "v4"}.get(args.method, "v1.5"),
        "refine": "slic" if args.clustering == "slic" else None,
        "clustering_method": "kmeans",  # Default, can be overridden by grid configs
        "model_name": args.model,
        "stride": args.stride,
        "elbow_threshold": args.elbow_threshold,
        "auto_k": (args.fixed_k is None),
        "n_clusters": args.fixed_k if args.fixed_k else 6,
        "apply_vegetation_filter": args.apply_vegetation_filter or (args.method == "v3"),
        "exg_threshold": args.exg_threshold,
        "use_tiling": not args.no_tiling,
        "viz_two_panel": args.viz_two_panel,
        "viz_two_panel_opaque": args.viz_two_panel_opaque,
    }

    all_results = []

    print("\n" + "=" * 60)
    print("RUNNING COMPARISON BENCHMARK")
    print("=" * 60 + "\n")

    # Load FORTRESS dataset once
    dataset = FortressDataset(args.dataset)

    # Model cache: key = (model_name, stride, image_size), value = (model, preprocess)
    model_cache = {}

    for i, config_override in enumerate(configs_to_test):
        config_dict = base_config_dict.copy()
        config_dict.update({k: v for k, v in config_override.items() if k != "label"})

        config = Config(**config_dict)
        label = config_override["label"]

        print(f"\n[{i + 1}/{len(configs_to_test)}] Testing: {label}")
        print("-" * 40)

        # Check model cache
        model_key = (config.model_display_name, config.stride, config.image_size)

        if model_key in model_cache:
            print(f"♻️  Reusing cached model: {config.model_display_name} (stride={config.stride})")

        # All configs save to the same sweep directory
        results = run_benchmark(
            config=config,
            dataset=dataset,
            output_dir=sweep_dir,
            num_samples=args.num_samples,
            save_visualizations=args.save_viz,
            verbose=not args.quiet,
            use_smart_k=args.smart_k,
            model_cache=model_cache,
        )

        all_results.append({"label": label, "config": config_dict, "results": results})

    # Print comparison table
    print("\n" + format_comparison_table(all_results) + "\n")

    # Save comparison summary in the same sweep directory
    comparison_path = sweep_dir / f"sweep_summary_{grid_name}.json"
    save_comparison_summary(all_results, comparison_path)
    print(f"📊 Sweep summary saved to: {comparison_path}")
    print(f"📁 All results in: {sweep_dir}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate segmentation methods on FORTRESS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add common arguments
    add_common_eval_arguments(parser)
    add_comparison_arguments(parser)

    # Add FORTRESS-specific arguments
    parser.add_argument(
        "--smart-k",
        action="store_true",
        help="Use smart K mode: set K to match the number of classes in each image's ground truth (slight cheat for debugging)",
    )

    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        choices=["ofat", "smart", "full", "tiling", "tiling_refine", "clustering"],
        help="Grid to use for comparison mode (default: tiling for FORTRESS)",
    )

    args = parser.parse_args()

    # Validate dataset path
    if not args.dataset.exists():
        print(f"❌ Error: Dataset path does not exist: {args.dataset}")
        return 1

    # Validate it's a FORTRESS dataset
    if not (args.dataset / "images").exists() or not (args.dataset / "labels").exists():
        print("❌ Error: Not a valid FORTRESS dataset (missing images/ or labels/ directory)")
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
