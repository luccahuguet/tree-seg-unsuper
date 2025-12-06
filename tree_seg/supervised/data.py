"""Dataset loading utilities for supervised training."""

from pathlib import Path

import numpy as np
from PIL import Image

# Allow very large ortho mosaics (FORTRESS tiles)
Image.MAX_IMAGE_PIXELS = None


def _read_downsampled(path: Path, max_side: int, is_mask: bool) -> np.ndarray:
    """
    Read an image with downsampling to limit RAM. Prefers rasterio for TIFFs.
    """
    try:
        import rasterio
        from rasterio.enums import Resampling

        with rasterio.open(path) as src:
            scale = min(1.0, max_side / max(src.height, src.width))
            out_h = max(1, int(src.height * scale))
            out_w = max(1, int(src.width * scale))
            data = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=Resampling.nearest if is_mask else Resampling.bilinear,
            )
            data = np.moveaxis(data, 0, -1)  # C,H,W -> H,W,C
            if not is_mask and data.shape[-1] > 3:
                data = data[..., :3]
            if is_mask and data.ndim == 3:
                data = data[..., 0]
            return data
    except Exception:
        # Fallback to PIL; still resized to cap memory usage
        img = Image.open(path).convert("RGB" if not is_mask else "L")
        img.thumbnail((max_side, max_side), Image.BILINEAR)
        arr = np.array(img)
        if not is_mask and arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]
        return arr


def load_dataset(
    dataset_path: Path,
    max_side: int | None = None,
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

    image_files = (
        sorted(images_dir.glob("*.png"))
        + sorted(images_dir.glob("*.jpg"))
        + sorted(images_dir.glob("*.jpeg"))
        + sorted(images_dir.glob("*.tif"))
        + sorted(images_dir.glob("*.tiff"))
    )

    for img_path in image_files:
        # Find corresponding mask (common patterns: same name; name with _label suffix)
        candidates = [
            masks_dir / img_path.name,
            masks_dir / (img_path.stem + ".png"),
            masks_dir / (img_path.stem + ".tif"),
            masks_dir / (img_path.stem + ".tiff"),
            masks_dir / (img_path.stem + "_label.png"),
            masks_dir / (img_path.stem + "_label.tif"),
            masks_dir / (img_path.stem + "_label.tiff"),
        ]
        mask_path = next((c for c in candidates if c.exists()), None)
        if mask_path is None:
            continue

        # Load image and mask
        if max_side:
            img = _read_downsampled(img_path, max_side=max_side, is_mask=False)
            mask = _read_downsampled(mask_path, max_side=max_side, is_mask=True)
        else:
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
