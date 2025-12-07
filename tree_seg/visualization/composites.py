"""Composite rendering helpers for segmentation outputs."""

from __future__ import annotations

import numpy as np
from PIL import Image


def overlay_labels(
    image_np: np.ndarray, labels: np.ndarray, alpha: float = 0.5
) -> Image.Image:
    """Overlay segmentation labels on an image with given alpha."""
    image = Image.fromarray(image_np)
    labels_img = Image.fromarray(labels.astype(np.uint8))
    labels_resized = labels_img.resize(image.size, resample=Image.NEAREST)

    labels_rgb = np.array(labels_resized.convert("RGB"))
    blended = (alpha * labels_rgb + (1 - alpha) * image_np).astype(np.uint8)
    return Image.fromarray(blended)
