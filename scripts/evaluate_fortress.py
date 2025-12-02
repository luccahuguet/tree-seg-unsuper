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

import json
from pathlib import Path

from dotenv import load_dotenv

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import run_benchmark
from tree_seg.evaluation.cli import add_common_eval_arguments, add_comparison_arguments
from tree_seg.evaluation.datasets import FortressDataset
from tree_seg.evaluation.formatters import config_to_dict

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

    # Map clustering to refine parameter
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
        image_size=args.image_size,
        v3_exg_threshold=args.exg_threshold if version == "v3" else 0.10,
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
        print("❌ Comparison mode not yet implemented for FORTRESS")
        print("   Use the main evaluate_semantic_segmentation.py script for comparisons")
        return 1
    else:
        config = create_config(args)
        run_single_benchmark(args, config)

    return 0


if __name__ == "__main__":
    exit(main())
