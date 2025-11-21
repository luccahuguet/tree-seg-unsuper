#!/usr/bin/env python3
"""
Evaluate V3.1 vegetation filter on a sample of OAM-TCD images.

Compares V1.5 (all clusters) vs V3.1 (vegetation only) on 10 test samples.
"""

import json
from pathlib import Path
from datasets import load_from_disk
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tree_seg import TreeSegmentation, Config


def evaluate_v3_1_sample(
    n_samples: int = 10,
    k_value: int = 20,
    exg_threshold: float = 0.10,
    output_dir: str = "data/output/v3_1_eval",
    seed: int = 42
):
    """
    Evaluate V3.1 on a sample of OAM-TCD images.

    Args:
        n_samples: Number of samples to evaluate
        k_value: Number of clusters
        exg_threshold: ExG threshold for vegetation
        output_dir: Output directory
        seed: Random seed for sample selection
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"V3.1 Sample Evaluation (n={n_samples})")
    print("=" * 80)
    print()

    # Load test split
    print("Loading OAM-TCD test split...")
    test_data = load_from_disk("data/oam_tcd/test")
    print(f"Loaded {len(test_data)} test samples")
    print()

    # Sample random images
    np.random.seed(seed)
    sample_indices = np.random.choice(len(test_data), size=n_samples, replace=False)
    sample_indices = sorted(sample_indices.tolist())

    print(f"Selected {n_samples} random samples (seed={seed})")
    print(f"Indices: {sample_indices}")
    print()

    # Initialize configs
    config_v1_5 = Config(
        pipeline="v1_5",
        auto_k=False,
        n_clusters=k_value,
        verbose=False
    )

    config_v3_1 = Config(
        pipeline="v3_1",
        auto_k=False,
        n_clusters=k_value,
        v3_1_exg_threshold=exg_threshold,
        verbose=False
    )

    # Results storage
    results = []

    # Process each sample
    for i, idx in enumerate(sample_indices):
        sample = test_data[int(idx)]
        image_id = sample['image_id']

        print(f"[{i+1}/{n_samples}] Processing image {image_id}...")

        # Save temp image
        image = sample['image']
        temp_path = output_path / f"temp_{image_id}.jpg"
        if isinstance(image, Image.Image):
            image.save(temp_path)
        else:
            Image.fromarray(np.array(image)).save(temp_path)

        # Run V1.5
        seg_v1_5 = TreeSegmentation(config_v1_5)
        results_v1_5 = seg_v1_5.process_single_image(str(temp_path))

        # Run V3.1
        seg_v3_1 = TreeSegmentation(config_v3_1)
        results_v3_1 = seg_v3_1.process_single_image(str(temp_path))

        # Compute statistics
        v1_5_mask = results_v1_5.labels_resized > 0
        v3_1_mask = results_v3_1.labels_resized > 0
        removed_mask = v1_5_mask & ~v3_1_mask

        v1_5_pixels = v1_5_mask.sum()
        v3_1_pixels = v3_1_mask.sum()
        removed_pixels = removed_mask.sum()

        total_pixels = results_v1_5.labels_resized.size

        result = {
            'image_id': image_id,
            'v1_5_clusters': results_v1_5.n_clusters_used,
            'v3_1_clusters': results_v3_1.n_clusters_used,
            'removed_clusters': results_v1_5.n_clusters_used - results_v3_1.n_clusters_used,
            'v1_5_pixels': int(v1_5_pixels),
            'v3_1_pixels': int(v3_1_pixels),
            'removed_pixels': int(removed_pixels),
            'total_pixels': int(total_pixels),
            'v1_5_coverage_pct': float(100 * v1_5_pixels / total_pixels),
            'v3_1_coverage_pct': float(100 * v3_1_pixels / total_pixels),
            'removed_pct': float(100 * removed_pixels / v1_5_pixels) if v1_5_pixels > 0 else 0.0,
        }
        results.append(result)

        print(f"  V1.5: {result['v1_5_clusters']} clusters, {result['v1_5_coverage_pct']:.1f}% coverage")
        print(f"  V3.1: {result['v3_1_clusters']} clusters, {result['v3_1_coverage_pct']:.1f}% coverage")
        print(f"  Removed: {result['removed_pct']:.1f}% ({result['removed_pixels']:,} px)")

        # Generate visualization for this sample
        generate_sample_visualization(
            image_np=np.array(image),
            results_v1_5=results_v1_5,
            results_v3_1=results_v3_1,
            image_id=image_id,
            output_path=output_path
        )

        # Clean up temp file
        temp_path.unlink()
        print()

    # Compute aggregate statistics
    aggregate = {
        'n_samples': n_samples,
        'k_value': k_value,
        'exg_threshold': exg_threshold,
        'avg_v1_5_clusters': float(np.mean([r['v1_5_clusters'] for r in results])),
        'avg_v3_1_clusters': float(np.mean([r['v3_1_clusters'] for r in results])),
        'avg_removed_clusters': float(np.mean([r['removed_clusters'] for r in results])),
        'avg_removed_pct': float(np.mean([r['removed_pct'] for r in results])),
        'std_removed_pct': float(np.std([r['removed_pct'] for r in results])),
        'min_removed_pct': float(np.min([r['removed_pct'] for r in results])),
        'max_removed_pct': float(np.max([r['removed_pct'] for r in results])),
    }

    # Save results
    output_json = output_path / "v3_1_sample_evaluation.json"
    with open(output_json, 'w') as f:
        json.dump({
            'aggregate': aggregate,
            'per_sample': results
        }, f, indent=2)

    print("=" * 80)
    print("Aggregate Results")
    print("=" * 80)
    print()
    print(f"Samples evaluated: {n_samples}")
    print(f"K value: {k_value}")
    print(f"ExG threshold: {exg_threshold}")
    print()
    print("Cluster Statistics:")
    print(f"  Avg V1.5 clusters: {aggregate['avg_v1_5_clusters']:.1f}")
    print(f"  Avg V3.1 clusters: {aggregate['avg_v3_1_clusters']:.1f}")
    print(f"  Avg removed: {aggregate['avg_removed_clusters']:.1f}")
    print()
    print("Filtering Statistics:")
    print(f"  Avg removed: {aggregate['avg_removed_pct']:.1f}% ± {aggregate['std_removed_pct']:.1f}%")
    print(f"  Min removed: {aggregate['min_removed_pct']:.1f}%")
    print(f"  Max removed: {aggregate['max_removed_pct']:.1f}%")
    print()
    print(f"Results saved to: {output_json}")
    print()


def generate_sample_visualization(
    image_np: np.ndarray,
    results_v1_5,
    results_v3_1,
    image_id: int,
    output_path: Path
):
    """Generate 4-panel visualization for a single sample."""
    from skimage import segmentation as skimage_seg

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title(f"Image {image_id}")
    axes[0, 0].axis('off')

    # 2. V1.5 all clusters
    axes[0, 1].imshow(image_np)
    boundaries = skimage_seg.find_boundaries(results_v1_5.labels_resized, mode='thick')
    axes[0, 1].contour(boundaries, levels=[0.5], colors='white', linewidths=1, alpha=0.8)
    axes[0, 1].set_title(f"V1.5 ({results_v1_5.n_clusters_used} clusters)")
    axes[0, 1].axis('off')

    # 3. V3.1 vegetation only
    axes[1, 0].imshow(image_np)
    veg_boundaries = skimage_seg.find_boundaries(results_v3_1.labels_resized, mode='thick')
    axes[1, 0].contour(veg_boundaries, levels=[0.5], colors='lime', linewidths=1.5, alpha=0.9)
    axes[1, 0].set_title(f"V3.1 Vegetation ({results_v3_1.n_clusters_used} clusters)")
    axes[1, 0].axis('off')

    # 4. Removed regions with red tint
    v1_5_mask = results_v1_5.labels_resized > 0
    v3_1_mask = results_v3_1.labels_resized > 0
    removed_mask = v1_5_mask & ~v3_1_mask

    display_image = image_np.copy().astype(float)
    if removed_mask.any():
        display_image[removed_mask] = display_image[removed_mask] * 0.5 + np.array([128, 0, 0]) * 0.5

    axes[1, 1].imshow(display_image.astype(np.uint8))

    if removed_mask.any():
        removed_labels = results_v1_5.labels_resized * removed_mask
        removed_boundaries = skimage_seg.find_boundaries(removed_labels.astype(int), mode='thick')
        axes[1, 1].contour(removed_boundaries, levels=[0.5], colors='red',
                          linewidths=1.5, alpha=0.8, linestyles='--')

    removed_pct = 100 * removed_mask.sum() / v1_5_mask.sum() if v1_5_mask.sum() > 0 else 0
    axes[1, 1].set_title(f"Removed ({removed_pct:.1f}%)")
    axes[1, 1].axis('off')

    plt.tight_layout()

    save_path = output_path / f"sample_{image_id}_v3_1.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate V3.1 on OAM-TCD sample")
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--k", type=int, default=20, help="Number of clusters")
    parser.add_argument("--threshold", type=float, default=0.10, help="ExG threshold")
    parser.add_argument("--output", type=str, default="data/output/v3_1_eval",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    evaluate_v3_1_sample(
        n_samples=args.n,
        k_value=args.k,
        exg_threshold=args.threshold,
        output_dir=args.output,
        seed=args.seed
    )
