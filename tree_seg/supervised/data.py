"""Dataset loading utilities for supervised training."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_dataset(
    dataset_path: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """
    Load images and ground truth masks from dataset.

    Expected structure:
        dataset_path/
        ├── images/
        │   ├── img001.png
        │   └── ...
        └── masks/  (or labels/, annotations/, gt/)
            ├── img001.png
            └── ...

    Returns:
        images: List of numpy arrays (H, W, 3)
        masks: List of numpy arrays (H, W) with integer class labels
        class_names: List of class names (from classes.txt if exists)
    """
    dataset_path = Path(dataset_path)

    # Find images directory
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        images_dir = dataset_path  # Images in root

    # Find masks directory
    masks_dir = None
    for name in ["masks", "labels", "annotations", "gt"]:
        candidate = dataset_path / name
        if candidate.exists():
            masks_dir = candidate
            break

    if masks_dir is None:
        raise ValueError(f"No masks directory found in {dataset_path}")

    # Load images and masks
    images = []
    masks = []

    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))

    for img_path in image_files:
        # Find corresponding mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            # Try with .png extension
            mask_path = masks_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            continue

        # Load image and mask
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        images.append(img)
        masks.append(mask)

    if not images:
        raise ValueError(f"No matching image/mask pairs found in {dataset_path}")

    # Load class names if available
    class_names = []
    classes_file = dataset_path / "classes.txt"
    if classes_file.exists():
        class_names = classes_file.read_text().strip().split("\n")

    return images, masks, class_names


def resize_masks_to_features(
    masks: list[np.ndarray],
    target_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize ground truth masks to match feature resolution using nearest neighbor.

    Args:
        masks: List of masks, each (H, W)
        target_size: Target (height, width)

    Returns:
        Stacked resized masks (N, H, W)
    """
    resized = []
    for mask in masks:
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        # PIL uses (width, height)
        mask_resized = mask_pil.resize(
            (target_size[1], target_size[0]),
            resample=Image.NEAREST,
        )
        resized.append(np.array(mask_resized))
    return np.stack(resized)
