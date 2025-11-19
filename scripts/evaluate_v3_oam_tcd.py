#!/usr/bin/env python3
"""
Evaluate V3 on OAM-TCD Test Set

Runs V3 tree detection on OAM-TCD test images and computes metrics.
"""

import json
from pathlib import Path
import numpy as np
import cv2
from datasets import load_from_disk
from tree_seg import TreeSegmentation, Config
from tree_seg.evaluation.oam_tcd_eval import OAMTCDEvaluator, print_metrics


def run_v3_on_oam_tcd(
    dataset_path: str = "data/oam_tcd",
    output_dir: str = "data/oam_tcd/v3_predictions",
    max_samples: int = None,
    preset: str = "balanced"
):
    """
    Run V3 on OAM-TCD test set.

    Args:
        dataset_path: Path to OAM-TCD dataset
        output_dir: Directory to save predictions
        max_samples: Max samples to process (None = all)
        preset: V3 preset ("permissive", "balanced", "strict")
    """
    # Load test set
    test_path = Path(dataset_path) / "test"
    print(f"Loading test set from {test_path}...")
    test_data = load_from_disk(str(test_path))

    num_samples = min(len(test_data), max_samples) if max_samples else len(test_data)
    print(f"Processing {num_samples} test samples with V3 (preset: {preset})")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize V3 pipeline
    config = Config(
        pipeline="v3",
        v3_preset=preset,
        auto_k=True,
        elbow_threshold=10.0,  # Conservative
        verbose=False  # Quiet for batch processing
    )
    segmenter = TreeSegmentation(config)

    # Process each sample
    print("\nProcessing samples...")
    for idx in range(num_samples):
        sample = test_data[idx]
        image_id = sample['image_id']

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{num_samples}] Processing image {image_id}...")

        # Get image and save temporarily
        image_np = np.array(sample['image'])

        # Save temp image
        temp_path = output_path / f"temp_{image_id}.jpg"
        cv2.imwrite(str(temp_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # Run V3
        try:
            results = segmenter.process_single_image(str(temp_path))
            instance_labels = results.labels_resized

            # Save prediction
            pred_path = output_path / f"{image_id}_prediction.png"
            cv2.imwrite(str(pred_path), instance_labels.astype(np.uint16))

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
    dataset_path: str = "data/oam_tcd",
    predictions_dir: str = "data/oam_tcd/v3_predictions",
    output_json: str = "results/v3_oam_tcd_evaluation.json",
    max_samples: int = None
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
        Path(predictions_dir),
        iou_threshold=0.5,
        max_samples=max_samples
    )

    # Print metrics
    print_metrics(results)

    # Save results
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate V3 on OAM-TCD")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/oam_tcd",
        help="Path to OAM-TCD dataset"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to process (default: all)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=["permissive", "balanced", "strict"],
        help="V3 preset"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference, only evaluate existing predictions"
    )

    args = parser.parse_args()

    predictions_dir = f"data/oam_tcd/v3_predictions_{args.preset}"

    # Run inference (unless skipped)
    if not args.skip_inference:
        run_v3_on_oam_tcd(
            dataset_path=args.dataset,
            output_dir=predictions_dir,
            max_samples=args.max_samples,
            preset=args.preset
        )

    # Evaluate
    evaluate_v3_results(
        dataset_path=args.dataset,
        predictions_dir=predictions_dir,
        output_json=f"results/v3_{args.preset}_oam_tcd.json",
        max_samples=args.max_samples
    )
