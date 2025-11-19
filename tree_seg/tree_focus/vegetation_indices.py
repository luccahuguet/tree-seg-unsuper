"""
Vegetation Indices for Tree Detection

Implements RGB-based vegetation indices for tree prefiltering:
- ExG (Excess Green Index)
- CIVE (Color Index of Vegetation Extraction)
- Green ratio
"""

import numpy as np
from typing import Tuple


def normalize_rgb(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize RGB channels to [0, 1] range.

    Args:
        image: RGB image (H, W, 3) in [0, 255] uint8 format

    Returns:
        Tuple of (R, G, B) normalized to [0, 1]
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.max() > 1.0:
        image = image.astype(np.float32) / 255.0

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    return r, g, b


def excess_green_index(image: np.ndarray) -> np.ndarray:
    """
    Compute Excess Green Index (ExG).

    ExG = 2*G - R - B

    Highlights green vegetation (positive values indicate vegetation).
    Range: [-1, 1] for normalized RGB, but typically [-0.5, 0.5]

    Args:
        image: RGB image (H, W, 3)

    Returns:
        ExG index map (H, W)

    Reference:
        Woebbecke et al. (1995) "Color indices for weed identification under
        various soil, residue, and lighting conditions"
    """
    r, g, b = normalize_rgb(image)

    exg = 2.0 * g - r - b

    return exg


def color_index_vegetation_extraction(image: np.ndarray) -> np.ndarray:
    """
    Compute Color Index of Vegetation Extraction (CIVE).

    CIVE = 0.441*R - 0.881*G + 0.385*B + 18.787

    Developed for plant segmentation in agricultural imagery.
    Negative values typically indicate vegetation.

    Args:
        image: RGB image (H, W, 3)

    Returns:
        CIVE index map (H, W)

    Reference:
        Kataoka et al. (2003) "Crop growth estimation system using machine vision"
    """
    r, g, b = normalize_rgb(image)

    cive = 0.441 * r - 0.881 * g + 0.385 * b + 18.787

    return cive


def green_ratio(image: np.ndarray) -> np.ndarray:
    """
    Compute green channel ratio.

    Green Ratio = G / (R + G + B + eps)

    Simple but effective vegetation indicator.
    Trees typically have ratios > 0.35-0.40.

    Args:
        image: RGB image (H, W, 3)

    Returns:
        Green ratio map (H, W) in [0, 1]
    """
    r, g, b = normalize_rgb(image)

    # Add small epsilon to avoid division by zero
    eps = 1e-8
    total = r + g + b + eps

    g_ratio = g / total

    return g_ratio


def create_vegetation_mask(
    image: np.ndarray,
    method: str = 'exg',
    threshold: float = None
) -> np.ndarray:
    """
    Create binary vegetation mask from RGB image.

    Args:
        image: RGB image (H, W, 3)
        method: Vegetation index method ('exg', 'cive', 'green_ratio', 'combined')
        threshold: Threshold value (if None, uses method-specific default)

    Returns:
        Binary vegetation mask (H, W) with True=vegetation
    """
    if method == 'exg':
        index = excess_green_index(image)
        # Default: ExG > 0 indicates vegetation
        thresh = threshold if threshold is not None else 0.0
        mask = index > thresh

    elif method == 'cive':
        index = color_index_vegetation_extraction(image)
        # Default: CIVE < 0 indicates vegetation
        thresh = threshold if threshold is not None else 0.0
        mask = index < thresh

    elif method == 'green_ratio':
        index = green_ratio(image)
        # Default: Green ratio > 0.36 indicates vegetation
        thresh = threshold if threshold is not None else 0.36
        mask = index > thresh

    elif method == 'combined':
        # Combine multiple indices for robustness
        exg = excess_green_index(image)
        g_ratio = green_ratio(image)

        # Vegetation if BOTH ExG > 0 AND green ratio > 0.35
        mask = (exg > 0.0) & (g_ratio > 0.35)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'exg', 'cive', 'green_ratio', or 'combined'")

    return mask.astype(bool)


def compute_vegetation_score(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute vegetation score for a masked region.

    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W) defining region

    Returns:
        Vegetation score in [0, 1] (higher = more vegetation-like)
    """
    if not mask.any():
        return 0.0

    # Compute ExG only within mask
    exg = excess_green_index(image)
    g_ratio = green_ratio(image)

    # Average vegetation indicators within mask
    exg_mean = exg[mask].mean()
    g_ratio_mean = g_ratio[mask].mean()

    # Normalize ExG to [0, 1] (assume typical range [-0.5, 0.5])
    exg_norm = np.clip((exg_mean + 0.5), 0, 1)

    # Combine scores (weighted average)
    score = 0.5 * exg_norm + 0.5 * g_ratio_mean

    return float(score)


# Recommended thresholds for different scenarios
THRESHOLDS = {
    'permissive': {  # Catch more vegetation (higher recall)
        'exg': -0.05,
        'cive': 5.0,
        'green_ratio': 0.33,
    },
    'balanced': {  # Default balanced thresholds
        'exg': 0.0,
        'cive': 0.0,
        'green_ratio': 0.36,
    },
    'strict': {  # Only strong vegetation (higher precision)
        'exg': 0.05,
        'cive': -5.0,
        'green_ratio': 0.40,
    },
}


if __name__ == "__main__":
    # Example usage
    import cv2
    from pathlib import Path

    # Load sample image
    img_path = "data/input/sample.jpg"
    if Path(img_path).exists():
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Compute indices
        exg = excess_green_index(image)
        cive = color_index_vegetation_extraction(image)
        g_ratio = green_ratio(image)

        # Create masks with different thresholds
        mask_permissive = create_vegetation_mask(image, 'exg', THRESHOLDS['permissive']['exg'])
        mask_balanced = create_vegetation_mask(image, 'exg', THRESHOLDS['balanced']['exg'])
        mask_strict = create_vegetation_mask(image, 'exg', THRESHOLDS['strict']['exg'])

        print("Vegetation coverage:")
        print(f"  Permissive: {mask_permissive.mean():.1%}")
        print(f"  Balanced:   {mask_balanced.mean():.1%}")
        print(f"  Strict:     {mask_strict.mean():.1%}")
    else:
        print(f"Sample image not found at {img_path}")
