#!/usr/bin/env python3
"""Generate visualizations for tiling vs non-tiling comparison."""

from pathlib import Path
from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import run_benchmark
from tree_seg.evaluation.datasets import FortressDataset

# Load dataset
dataset = FortressDataset(Path("data/fortress_processed"))

print("Generating visualizations for tiling comparison...\n")

# 1. NO TILING (downsampled to 1024×1024)
print("=" * 60)
print("1/2: Processing WITHOUT tiling...")
print("=" * 60)

config_notile = Config(
    version="v1.5",
    model_name="base",
    stride=4,
    refine="slic",
    elbow_threshold=5.0,
    auto_k=True,
    use_tiling=False,  # Explicitly disable
    metrics=True,
)

results_notile = run_benchmark(
    config=config_notile,
    dataset=dataset,
    output_dir=Path("data/output/results/viz_notile"),
    num_samples=1,
    save_visualizations=True,
    verbose=True,
)

print(f"\n✅ No tiling: {results_notile.mean_miou:.3f} mIoU, "
      f"{results_notile.mean_pixel_accuracy:.3f} pixel acc\n")

# 2. WITH TILING (full 9372×9372 resolution)
print("=" * 60)
print("2/2: Processing WITH tiling...")
print("=" * 60)

config_tile = Config(
    version="v1.5",
    model_name="base",
    stride=4,
    refine="slic",
    elbow_threshold=5.0,
    auto_k=True,
    use_tiling=True,  # Explicitly enable
    metrics=True,
)

results_tile = run_benchmark(
    config=config_tile,
    dataset=dataset,
    output_dir=Path("data/output/results/viz_tile"),
    num_samples=1,
    save_visualizations=True,
    verbose=True,
)

print(f"\n✅ With tiling: {results_tile.mean_miou:.3f} mIoU, "
      f"{results_tile.mean_pixel_accuracy:.3f} pixel acc\n")

# Summary
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"No tiling:   {results_notile.mean_miou:.1%} mIoU, "
      f"{results_notile.mean_pixel_accuracy:.1%} pixel acc, "
      f"{results_notile.mean_runtime:.0f}s")
print(f"With tiling: {results_tile.mean_miou:.1%} mIoU, "
      f"{results_tile.mean_pixel_accuracy:.1%} pixel acc, "
      f"{results_tile.mean_runtime:.0f}s")

improvement = (results_tile.mean_pixel_accuracy - results_notile.mean_pixel_accuracy) / results_notile.mean_pixel_accuracy
print(f"\nPixel accuracy improvement: {improvement:+.1%}")

print("\n📁 Visualizations saved:")
print("   - data/output/results/viz_notile/visualizations/CFB003_comparison.png")
print("   - data/output/results/viz_tile/visualizations/CFB003_comparison.png")
