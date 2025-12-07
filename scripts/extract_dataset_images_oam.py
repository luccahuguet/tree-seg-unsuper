#!/usr/bin/env python3
"""
Extract OAM-TCD images from Arrow format to individual files.
"""

from pathlib import Path
from datasets import load_from_disk
import numpy as np
from PIL import Image


def extract_images(
    split: str = "test", output_dir: str = "data/oam_tcd_images", max_images: int = None
):
    """
    Extract OAM-TCD images to individual files.

    Args:
        split: Dataset split ('test' or 'train')
        output_dir: Output directory for extracted images
        max_images: Maximum number of images to extract (None = all)
    """
    output_path = Path(output_dir) / split
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading OAM-TCD {split} split...")
    data = load_from_disk(f"data/oam_tcd/{split}")

    n_images = len(data)
    if max_images:
        n_images = min(n_images, max_images)

    print(f"Extracting {n_images} images to {output_path}...")
    print()

    for i in range(n_images):
        sample = data[i]
        image_id = sample["image_id"]
        image = sample["image"]

        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        # Save image
        output_file = output_path / f"{image_id:05d}.jpg"
        image.save(output_file, quality=95)

        if (i + 1) % 50 == 0 or i + 1 == n_images:
            print(f"  Extracted {i + 1}/{n_images} images...")

    print()
    print(f"✓ Extracted {n_images} images to {output_path}")
    print(f"📁 Browse images: ls {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract OAM-TCD images")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Dataset split to extract",
    )
    parser.add_argument(
        "--output", type=str, default="data/oam_tcd_images", help="Output directory"
    )
    parser.add_argument(
        "--max", type=int, default=None, help="Maximum number of images (default: all)"
    )

    args = parser.parse_args()

    extract_images(split=args.split, output_dir=args.output, max_images=args.max)
