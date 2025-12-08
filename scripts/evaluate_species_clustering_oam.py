#!/usr/bin/env python3
"""
Evaluate Species Clustering on OAM-TCD Dataset

Dataset-specific wrapper for evaluating V3 species clustering on OAM-TCD.

NOTE: This script does NOT use the generic BenchmarkRunner because OAM-TCD
evaluation is fundamentally different:

**Why OAM-TCD is separate:**
1. **Different task**: Instance detection (individual trees) vs semantic
   segmentation (pixel-level classes)
2. **Different metrics**: Precision/Recall/F1 on tree instances vs mIoU on pixels
3. **Different format**: HuggingFace datasets format vs standard image/label pairs
4. **Custom evaluation**: IoU-based tree matching vs Hungarian pixel matching

For semantic segmentation datasets (ISPRS, FORTRESS), use:
- scripts/evaluate_fortress.py
- scripts/evaluate_semantic_segmentation.py

These use the generic BenchmarkRunner with SegmentationDataset protocol.
"""

import json
from pathlib import Path
import numpy as np
import cv2
from datasets import load_from_disk
from tree_seg import TreeSegmentation, Config
from tree_seg.evaluation.oam_tcd_eval import OAMTCDEvaluator, print_metrics


def run_v3_on_oam_tcd(
    dataset_path: str = "data/datasets/oam_tcd",
    output_dir: str = "data/datasets/oam_tcd/v3_predictions",
    max_samples: int = None,
    exg_threshold: float = 0.10,
):
    """
    Run V3 species clustering on OAM-TCD test set.

    Args:
        dataset_path: Path to OAM-TCD dataset
        output_dir: Directory to save predictions
        max_samples: Max samples to process (None = all)
        exg_threshold: ExG threshold for vegetation filtering
    """
    # Load test set
    test_path = Path(dataset_path) / "test"
    print(f"Loading test set from {test_path}...")
    test_data = load_from_disk(str(test_path))

    num_samples = min(len(test_data), max_samples) if max_samples else len(test_data)
    print(
        f"Processing {num_samples} test samples with V3 (ExG threshold: {exg_threshold})"
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize V3 pipeline
    config = Config(
        pipeline="v3",
        auto_k=True,
        elbow_threshold=5.0,  # Default
        v3_exg_threshold=exg_threshold,
        verbose=False,  # Quiet for batch processing
    )
    segmenter = TreeSegmentation(config)

    # Process each sample
    print("\nProcessing samples...")
    for idx in range(num_samples):
        sample = test_data[idx]
        image_id = sample["image_id"]

        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{num_samples}] Processing image {image_id}...")

        # Get image and save temporarily
        image_np = np.array(sample["image"])

        # Save temp image
        temp_path = output_path / f"temp_{image_id}.jpg"
        cv2.imwrite(str(temp_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # Run V3
        try:
            results = segmenter.process_single_image(str(temp_path))
            semantic_labels = results.labels_resized

            # Save prediction
            pred_path = output_path / f"{image_id}_prediction.png"
            cv2.imwrite(str(pred_path), semantic_labels.astype(np.uint16))

            # Clean up temp file
            temp_path.unlink()

        except Exception as e:
            print(f"  Error processing {image_id}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            continue

    print(f"\n✓ Predictions saved to {output_path}")
    return output_path


def evaluate_v3_results(
    dataset_path: str = "data/datasets/oam_tcd",
    predictions_dir: str = "data/datasets/oam_tcd/v3_predictions",
    output_json: str = "results/v3_oam_tcd_evaluation.json",
    max_samples: int = None,
):
    """
    Evaluate V3 predictions against OAM-TCD ground truth.

    Args:
        dataset_path: Path to OAM-TCD dataset
        predictions_dir: Directory with predictions
        output_json: Output file for results JSON
        max_samples: Max samples to evaluate
    """
    print("\n" + "=" * 80)
    print("Evaluating V3 on OAM-TCD Test Set")
    print("=" * 80)

    # Run evaluation
    evaluator = OAMTCDEvaluator(dataset_path)
    results = evaluator.evaluate(
        Path(predictions_dir), iou_threshold=0.5, max_samples=max_samples
    )

    # Print metrics
    print_metrics(results)

    # Save results
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate V3 on OAM-TCD")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/datasets/oam_tcd",
        help="Path to OAM-TCD dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to process (default: all)",
    )
    parser.add_argument(
        "--exg-threshold",
        type=float,
        default=0.10,
        help="ExG threshold for vegetation filtering (default: 0.10)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference, only evaluate existing predictions",
    )

    args = parser.parse_args()

    predictions_dir = f"data/datasets/oam_tcd/v3_predictions_exg{args.exg_threshold}"

    # Run inference (unless skipped)
    if not args.skip_inference:
        run_v3_on_oam_tcd(
            dataset_path=args.dataset,
            output_dir=predictions_dir,
            max_samples=args.max_samples,
            exg_threshold=args.exg_threshold,
        )

    # Evaluate
    evaluate_v3_results(
        dataset_path=args.dataset,
        predictions_dir=predictions_dir,
        output_json=f"results/v3_exg{args.exg_threshold}_oam_tcd.json",
        max_samples=args.max_samples,
    )
