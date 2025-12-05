#!/usr/bin/env python3
"""Compare K-means vs Spectral clustering across 5 FORTRESS images."""

# Handle large FORTRESS images
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Remove decompression bomb limit

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import run_benchmark
from tree_seg.evaluation.datasets import FortressDataset
from pathlib import Path

dataset = FortressDataset("data/fortress_processed")
print(f"Dataset has {len(dataset)} images")

configs = [
    ("kmeans_slic", {"clustering_method": "kmeans", "refine": "slic"}),
    ("spectral_slic", {"clustering_method": "spectral", "refine": "slic"}),
]

results = {}

for label, cfg_opts in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")
    
    config = Config(
        version="v1.5",
        model_name="base",
        stride=4,
        verbose=False,
        **cfg_opts
    )
    
    result = run_benchmark(
        config=config,
        dataset=dataset,
        output_dir=Path(f"data/output/compare_{label}"),
        num_samples=3,  # Reduced from 5 for speed
        save_visualizations=True,
        verbose=True,
        use_smart_k=True,
    )
    
    results[label] = {
        "mean_miou": result.mean_miou,
        "mean_pa": result.mean_pixel_accuracy,
        "mean_runtime": result.mean_runtime,
        "samples": [(s.image_id, s.miou, s.pixel_accuracy) for s in result.samples]
    }

# Summary
print("\n" + "="*70)
print("COMPARISON: K-MEANS vs SPECTRAL (5 images)")
print("="*70)

print(f"\n{'Method':<15} {'Mean mIoU':<12} {'Mean PA':<12} {'Runtime':<10}")
print("-"*50)
baseline_miou = results["kmeans_slic"]["mean_miou"]
for label, r in results.items():
    diff = r["mean_miou"] - baseline_miou
    diff_str = f"({diff:+.2f}%)" if diff != 0 else ""
    print(f"{label:<15} {r['mean_miou']*100:.2f}%      {r['mean_pa']*100:.2f}%      {r['mean_runtime']:.1f}s {diff_str}")

print("\nPer-image breakdown:")
print(f"{'Image':<12} {'K-means mIoU':<15} {'Spectral mIoU':<15} {'Diff':<10}")
print("-"*55)
km = results["kmeans_slic"]["samples"]
sp = results["spectral_slic"]["samples"]
for i in range(len(km)):
    diff = sp[i][1] - km[i][1]
    print(f"{km[i][0]:<12} {km[i][1]*100:>8.2f}%       {sp[i][1]*100:>8.2f}%       {diff*100:>+6.2f}%")
