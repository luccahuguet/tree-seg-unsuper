#!/usr/bin/env python3
"""Download and inspect OAM-TCD dataset from HuggingFace."""

import json
from pathlib import Path
from datasets import load_dataset

def download_oam_tcd(output_dir: str = "data/oam_tcd", subset_size: int = None):
    """
    Download OAM-TCD dataset from HuggingFace.

    Args:
        output_dir: Directory to save the dataset
        subset_size: If specified, only download this many samples (for testing)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading OAM-TCD dataset from HuggingFace...")
    print("Note: This may take a while for the full dataset (5,072 images)")

    # Load the dataset
    # The dataset is at restor/tcd
    dataset = load_dataset("restor/tcd", trust_remote_code=True)

    print("\nDataset loaded successfully!")
    print(f"Splits available: {list(dataset.keys())}")

    for split_name, split_data in dataset.items():
        print(f"\n{split_name.upper()} split:")
        print(f"  - Number of samples: {len(split_data)}")
        print(f"  - Features: {split_data.features}")

        # Show first example
        if len(split_data) > 0:
            example = split_data[0]
            print(f"\n  First example keys: {list(example.keys())}")

            # Save a subset or full dataset
            num_samples = subset_size if subset_size else len(split_data)
            subset = split_data.select(range(num_samples))

            # Save to disk
            split_path = output_path / split_name
            split_path.mkdir(exist_ok=True)

            print(f"\n  Saving {num_samples} samples to {split_path}...")

            # Save as parquet for efficient loading
            subset.save_to_disk(str(split_path))

            # Also export metadata as JSON
            metadata = {
                "split": split_name,
                "num_samples": num_samples,
                "features": str(split_data.features),
                "total_samples": len(split_data),
            }

            with open(split_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  ✓ Saved {num_samples} {split_name} samples")

    # Save dataset info
    info_path = output_path / "dataset_info.json"
    dataset_info = {
        "name": "OAM-TCD",
        "source": "HuggingFace: restor/tcd",
        "description": "OpenAerialMap Tree Cover Dataset",
        "splits": {k: len(v) for k, v in dataset.items()},
        "license": "Predominantly CC BY 4.0",
        "resolution_cm": 10,
    }

    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("\n✓ Dataset download complete!")
    print(f"  Location: {output_path}")
    print(f"  Info saved to: {info_path}")

    return dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download OAM-TCD dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/oam_tcd",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Download only N samples per split (for testing)"
    )

    args = parser.parse_args()

    download_oam_tcd(args.output_dir, args.subset)
