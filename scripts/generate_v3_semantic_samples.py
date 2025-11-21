#!/usr/bin/env python3
"""
Generate V3 semantic visualizations for multiple OAM-TCD samples.
"""

from pathlib import Path
from datasets import load_from_disk
import numpy as np
from PIL import Image

from visualize_v3_semantic import visualize_v3_semantic


def generate_semantic_samples(
    n_samples: int = 10,
    elbow_threshold: float = 5.0,
    exg_threshold: float = 0.10,
    output_dir: str = "data/output/v3_semantic",
    seed: int = 42
):
    """
    Generate semantic visualizations for multiple samples.

    Args:
        n_samples: Number of samples to visualize
        elbow_threshold: Elbow threshold for auto K selection (default: 5.0)
        exg_threshold: ExG threshold for vegetation
        output_dir: Output directory
        seed: Random seed for sample selection
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Generating V3 Semantic Visualizations (n={n_samples})")
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

    # Process each sample
    for i, idx in enumerate(sample_indices):
        sample = test_data[int(idx)]
        image_id = sample['image_id']

        print(f"[{i+1}/{n_samples}] Generating semantic visualization for image {image_id}...")

        # Save temp image
        image = sample['image']
        temp_path = output_path / f"temp_{image_id}.jpg"
        if isinstance(image, Image.Image):
            image.save(temp_path)
        else:
            Image.fromarray(np.array(image)).save(temp_path)

        # Generate visualization
        try:
            visualize_v3_semantic(
                image_path=str(temp_path),
                elbow_threshold=elbow_threshold,
                exg_threshold=exg_threshold,
                output_dir=output_dir
            )

            # Rename output to include image_id
            old_name = output_path / f"temp_{image_id}_semantic.png"
            new_name = output_path / f"semantic_{image_id}.png"
            if old_name.exists():
                old_name.rename(new_name)

        except Exception as e:
            print(f"  ⚠️  Error processing {image_id}: {e}")

        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

        print()

    print("=" * 80)
    print(f"✓ Generated {n_samples} semantic visualizations")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate V3 semantic visualizations")
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--elbow-threshold", type=float, default=5.0,
                       help="Elbow threshold for auto K selection (default: 5.0)")
    parser.add_argument("--threshold", type=float, default=0.10, help="ExG threshold")
    parser.add_argument("--output", type=str, default="data/output/v3_semantic",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_semantic_samples(
        n_samples=args.n,
        elbow_threshold=args.elbow_threshold,
        exg_threshold=args.threshold,
        output_dir=args.output,
        seed=args.seed
    )
