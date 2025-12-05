#!/usr/bin/env python3
"""
Test multi-layer feature extraction on FORTRESS dataset.

Compares:
1. Single-layer (baseline): layer 12 only
2. Multi-layer concat: layers [3, 6, 9, 12] concatenated
3. Multi-layer average: layers [3, 6, 9, 12] averaged

Expected result: Multi-layer should improve mIoU by +5-10%
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import run_benchmark
from tree_seg.evaluation.datasets import FortressDataset

def test_multi_layer():
    """Test multi-layer feature extraction."""
    
    dataset = FortressDataset("data/fortress_processed")
    
    configs = [
        {
            "label": "baseline_single_layer",
            "use_multi_layer": False,
            "description": "Single layer (12) - baseline"
        },
        {
            "label": "multi_concat",
            "use_multi_layer": True,
            "layer_indices": (3, 6, 9, 12),
            "feature_aggregation": "concat",
            "pca_dim": 512,  # Reduce from 4*768=3072 to 512
            "description": "Multi-layer [3,6,9,12] concatenated + PCA"
        },
        {
            "label": "multi_average",
            "use_multi_layer": True,
            "layer_indices": (3, 6, 9, 12),
            "feature_aggregation": "average",
            "description": "Multi-layer [3,6,9,12] averaged"
        },
    ]
    
    results = []
    
    for i, cfg_dict in enumerate(configs):
        label = cfg_dict.pop("label")
        description = cfg_dict.pop("description")
        
        print(f"\n[{i+1}/{len(configs)}] Testing: {description}")
        print("=" * 60)
        
        # Create config
        config = Config(
            version="v1.5",
            model_name="base",
            stride=4,
            auto_k=True,
            refine="slic",
            verbose=True,  # Enable verbose to see errors
            **cfg_dict
        )
        
        # Run benchmark on 1 sample
        result = run_benchmark(
            config=config,
            dataset=dataset,
            output_dir=Path(f"data/output/multilayer_test_{label}"),
            num_samples=1,
            save_visualizations=True,
            verbose=True,
            use_smart_k=True,  # For fair comparison
        )
        
        results.append({
            "label": label,
            "description": description,
            "miou": result.mean_miou,
            "pixel_acc": result.mean_pixel_accuracy,
            "runtime": result.mean_runtime,
        })
    
    # Print comparison
    print("\n" + "=" * 60)
    print("MULTI-LAYER FEATURE EXTRACTION RESULTS")
    print("=" * 60)
    print(f"{'Config':<25} {'mIoU':<10} {'Pixel Acc':<12} {'Runtime':<10}")
    print("-" * 60)
    
    baseline_miou = results[0]["miou"]
    
    for r in results:
        improvement = ((r["miou"] - baseline_miou) / baseline_miou * 100) if baseline_miou > 0 else 0
        miou_str = f"{r['miou']:.4f}"
        if improvement != 0:
            miou_str += f" ({improvement:+.1f}%)"
        
        print(f"{r['label']:<25} {miou_str:<10} {r['pixel_acc']:.4f}      {r['runtime']:.1f}s")
    
    print("\n✅ Multi-layer feature extraction test complete!")
    
    # Determine if multi-layer helped
    best = max(results, key=lambda x: x["miou"])
    if best["label"] != "baseline_single_layer":
        improvement_pct = ((best["miou"] - baseline_miou) / baseline_miou * 100)
        print(f"\n🎯 Best config: {best['label']} (+{improvement_pct:.1f}% over baseline)")
    else:
        print(f"\n⚠️  Baseline single-layer was best - multi-layer did not improve results")

if __name__ == "__main__":
    test_multi_layer()
