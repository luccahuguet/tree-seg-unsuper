#!/usr/bin/env python3
"""
Evaluate V3 Vegetation Filter on OAM-TCD

Measures how well the V3 vegetation filter identifies actual trees by:
1. Loading OAM-TCD ground truth tree instance masks
2. Running V3 vegetation filtering
3. Computing overlap: what % of ground truth tree pixels are classified as vegetation?

Metrics:
- Vegetation Recall: % of GT tree pixels classified as vegetation
- Vegetation Precision: % of predicted vegetation pixels that are actual trees
- Per-image statistics and overall dataset metrics
"""

from pathlib import Path
import numpy as np
import tempfile
from tqdm import tqdm
import json
from datasets import load_from_disk

from tree_seg import TreeSegmentation, Config


def evaluate_vegetation_filter(
    dataset_split: str = "test",
    n_samples: int = None,
    exg_threshold: float = 0.10,
    auto_k: bool = True,
    elbow_threshold: float = 5.0,
    output_dir: str = "data/output/veg_filter_eval",
    random_seed: int = 42
):
    """
    Evaluate vegetation filter performance on OAM-TCD.

    Args:
        dataset_split: "train" or "test"
        n_samples: Number of samples to evaluate (None = all)
        exg_threshold: ExG threshold for vegetation filtering
        auto_k: Use auto K selection
        elbow_threshold: Elbow threshold for auto K
        output_dir: Output directory for results
        random_seed: Random seed for sampling
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("V3 Vegetation Filter Evaluation on OAM-TCD")
    print("=" * 80)
    print()

    # Load dataset
    print(f"Loading OAM-TCD {dataset_split} split...")
    dataset = load_from_disk(f"data/oam_tcd/{dataset_split}")
    print(f"  Total images: {len(dataset)}")

    # Sample if needed
    if n_samples and n_samples < len(dataset):
        np.random.seed(random_seed)
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        samples = [dataset[int(i)] for i in indices]
        print(f"  Evaluating on {n_samples} random samples (seed={random_seed})")
    else:
        samples = list(dataset)
        print(f"  Evaluating on all {len(samples)} samples")
    print()

    # Initialize V3
    print(f"Initializing V3 (ExG threshold={exg_threshold})...")
    config = Config(
        pipeline="v3",
        auto_k=auto_k,
        elbow_threshold=elbow_threshold,
        v3_exg_threshold=exg_threshold,
        verbose=False
    )
    seg = TreeSegmentation(config)
    print()

    # Evaluate each sample
    results = []

    for sample in tqdm(samples, desc="Evaluating samples"):
        image_id = sample['image_id']

        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            sample['image'].save(temp_file.name)
            temp_path = temp_file.name

        try:
            # Run V3
            v3_results = seg.process_single_image(temp_path)

            # Get V3 vegetation mask (everything except cluster 0)
            v3_veg_mask = v3_results.labels_resized > 0

            # Get ground truth tree mask from instance annotations
            gt_instance_mask = np.array(sample['annotation'])

            # Convert to grayscale if needed (some masks are RGB)
            if len(gt_instance_mask.shape) == 3:
                gt_instance_mask = gt_instance_mask[:, :, 0]

            # Category 1 = individual trees, Category 2 = canopy groups
            # Parse COCO annotations to filter by category
            coco_annotations = json.loads(sample['coco_annotations'])

            # Create ground truth tree mask (Category 1 only)
            gt_tree_mask = np.zeros_like(gt_instance_mask, dtype=bool)

            # Get unique non-zero mask values (sorted)
            unique_mask_values = sorted([v for v in np.unique(gt_instance_mask) if v > 0])

            # Map mask values to annotations
            for i, mask_value in enumerate(unique_mask_values):
                if i < len(coco_annotations) and coco_annotations[i]['category_id'] == 1:
                    # Individual trees only (Category 1)
                    gt_tree_mask |= (gt_instance_mask == mask_value)

            # Compute metrics
            gt_tree_pixels = gt_tree_mask.sum()
            v3_veg_pixels = v3_veg_mask.sum()

            # True Positives: GT tree pixels classified as vegetation
            tp_pixels = (gt_tree_mask & v3_veg_mask).sum()

            # False Negatives: GT tree pixels classified as non-vegetation
            fn_pixels = (gt_tree_mask & ~v3_veg_mask).sum()

            # False Positives: Non-tree pixels classified as vegetation
            fp_pixels = (~gt_tree_mask & v3_veg_mask).sum()

            # Recall: % of GT trees captured by vegetation filter
            recall = tp_pixels / gt_tree_pixels if gt_tree_pixels > 0 else 0.0

            # Precision: % of predicted vegetation that are actual trees
            precision = tp_pixels / v3_veg_pixels if v3_veg_pixels > 0 else 0.0

            # F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Store results
            results.append({
                'image_id': int(image_id),
                'gt_tree_pixels': int(gt_tree_pixels),
                'v3_veg_pixels': int(v3_veg_pixels),
                'tp_pixels': int(tp_pixels),
                'fn_pixels': int(fn_pixels),
                'fp_pixels': int(fp_pixels),
                'recall': float(recall),
                'precision': float(precision),
                'f1': float(f1),
                'n_gt_trees': int(sum(1 for ann in coco_annotations if ann['category_id'] == 1)),
                'v3_clusters': int(v3_results.n_clusters_used)
            })

        finally:
            # Cleanup temp file
            import os
            os.unlink(temp_path)

    print()
    print("=" * 80)
    print("Results - Vegetation Filter Recall Evaluation")
    print("=" * 80)
    print()
    print("⚠️  NOTE: OAM-TCD has incomplete annotations (many unannotated trees/vegetation)")
    print("    → Precision is MEANINGLESS (will be artificially low)")
    print("    → Recall is VALID (measures % of labeled trees captured)")
    print()

    # Compute aggregate statistics
    total_gt_pixels = sum(r['gt_tree_pixels'] for r in results)
    total_v3_pixels = sum(r['v3_veg_pixels'] for r in results)
    total_tp_pixels = sum(r['tp_pixels'] for r in results)
    total_fn_pixels = sum(r['fn_pixels'] for r in results)

    # Overall metrics
    overall_recall = total_tp_pixels / total_gt_pixels if total_gt_pixels > 0 else 0.0

    # Per-image averages
    avg_recall = np.mean([r['recall'] for r in results])
    median_recall = np.median([r['recall'] for r in results])

    print("📊 RECALL METRICS (Primary Focus):")
    print(f"  Overall Recall: {overall_recall:.1%} ({total_tp_pixels:,} / {total_gt_pixels:,} GT tree pixels)")
    print(f"  Mean Recall:    {avg_recall:.1%} ± {np.std([r['recall'] for r in results]):.1%}")
    print(f"  Median Recall:  {median_recall:.1%}")
    print()

    # Breakdown by recall range
    high_recall = sum(1 for r in results if r['recall'] >= 0.8)
    medium_recall = sum(1 for r in results if 0.3 <= r['recall'] < 0.8)
    low_recall = sum(1 for r in results if 0 < r['recall'] < 0.3)
    zero_recall = sum(1 for r in results if r['recall'] == 0.0)

    print("📈 Recall Distribution:")
    print(f"  High (≥80%):   {high_recall:3d} images ({100*high_recall/len(results):.1f}%)")
    print(f"  Medium (30-80%): {medium_recall:3d} images ({100*medium_recall/len(results):.1f}%)")
    print(f"  Low (<30%):    {low_recall:3d} images ({100*low_recall/len(results):.1f}%)")
    print(f"  Zero (0%):     {zero_recall:3d} images ({100*zero_recall/len(results):.1f}%) ⚠️")
    print()

    if zero_recall > 0:
        print(f"⚠️  {zero_recall} images with 0% recall - trees filtered out as non-vegetation")
        print("    Possible causes: very dark trees, heavy shadows, dead/brown vegetation")
        print()

    # Best and worst cases
    best_recall = max(results, key=lambda r: r['recall'])
    worst_recall = min(results, key=lambda r: r['recall'])

    print("Best Recall:")
    print(f"  Image {best_recall['image_id']}: {best_recall['recall']:.3f} recall ({best_recall['n_gt_trees']} trees)")
    print()

    print("Worst Recall:")
    print(f"  Image {worst_recall['image_id']}: {worst_recall['recall']:.3f} recall ({worst_recall['n_gt_trees']} trees)")
    print()

    # Save results
    output_file = output_path / f"veg_filter_eval_{dataset_split}.json"
    output_data = {
        'config': {
            'dataset_split': dataset_split,
            'n_samples': len(results),
            'exg_threshold': exg_threshold,
            'auto_k': auto_k,
            'elbow_threshold': elbow_threshold,
            'random_seed': random_seed
        },
        'note': 'Precision/F1 are MEANINGLESS - OAM-TCD has incomplete annotations. Only recall is valid.',
        'overall_metrics': {
            'recall': overall_recall,
            'total_gt_tree_pixels': total_gt_pixels,
            'total_v3_veg_pixels': total_v3_pixels,
            'total_tp_pixels': total_tp_pixels,
            'total_fn_pixels': total_fn_pixels
        },
        'per_image_stats': {
            'mean_recall': avg_recall,
            'median_recall': median_recall,
            'std_recall': float(np.std([r['recall'] for r in results])),
            'high_recall_count': high_recall,
            'medium_recall_count': medium_recall,
            'low_recall_count': low_recall,
            'zero_recall_count': zero_recall
        },
        'per_image_results': results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved results to: {output_file}")
    print()

    return output_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate V3 vegetation filter on OAM-TCD")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "test"],
                       help="Dataset split to evaluate")
    parser.add_argument("--n-samples", type=int, default=None,
                       help="Number of samples to evaluate (None = all)")
    parser.add_argument("--threshold", type=float, default=0.10,
                       help="ExG threshold for vegetation")
    parser.add_argument("--no-auto-k", action="store_true",
                       help="Disable auto K selection")
    parser.add_argument("--elbow", type=float, default=5.0,
                       help="Elbow threshold for auto K")
    parser.add_argument("--output", type=str, default="data/output/veg_filter_eval",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")

    args = parser.parse_args()

    evaluate_vegetation_filter(
        dataset_split=args.split,
        n_samples=args.n_samples,
        exg_threshold=args.threshold,
        auto_k=not args.no_auto_k,
        elbow_threshold=args.elbow,
        output_dir=args.output,
        random_seed=args.seed
    )
