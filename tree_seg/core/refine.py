"""Refinement and post-processing helpers for segmentation."""

from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np

try:
    from skimage.segmentation import slic as skimage_slic
except Exception:
    skimage_slic = None


def refine_labels(
    image_np: np.ndarray,
    labels_resized: np.ndarray,
    method: str | None,
    compactness: float = 10.0,
    sigma: float = 1.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """Run optional post-clustering refinement and report elapsed seconds."""
    if not method or method in ("none", "None"):
        return labels_resized, 0.0

    method_norm = method.lower()
    start = time.perf_counter()

    if method_norm in ("slic", "slic-opencv", "opencv-slic"):
        labels_refined = _refine_with_opencv_slic(
            image_np,
            labels_resized,
            compactness=compactness,
            region_size=48,
            verbose=verbose,
        )
    elif method_norm in ("slic_skimage", "slic-skimage", "slic-ski"):
        labels_refined = _refine_with_slic(
            image_np,
            labels_resized,
            compactness=compactness,
            sigma=sigma,
            force_skimage=True,
            verbose=verbose,
        )
    elif method_norm == "bilateral":
        labels_refined = _refine_with_bilateral(
            image_np=image_np,
            labels_resized=labels_resized,
            verbose=verbose,
        )
    else:
        if verbose:
            print(f"⚠️  Unknown refine method '{method}', skipping refinement")
        return labels_resized, 0.0

    elapsed = time.perf_counter() - start
    return labels_refined, elapsed


def apply_vegetation_filter(
    image_np: np.ndarray,
    cluster_labels: np.ndarray,
    exg_threshold: float = 0.10,
    verbose: bool = True,
) -> np.ndarray:
    """Run ExG vegetation filtering (delegates to vegetation_filter module)."""
    try:
        from ..vegetation_filter import apply_vegetation_filter as _apply
        filtered_labels, filter_info = _apply(
            image_np=image_np,
            cluster_labels=cluster_labels,
            exg_threshold=exg_threshold,
            verbose=verbose,
        )
        if verbose:
            print(f"  ✓ Filtered to {filter_info['n_vegetation_clusters']} vegetation clusters")
            print(f"  ✓ Vegetation coverage: {filter_info['vegetation_percentage']:.1f}%")
        return filtered_labels
    except Exception as exc:
        if verbose:
            print(f"  ⚠️  Vegetation filtering failed: {exc}, returning original clusters")
        return cluster_labels


def _refine_with_slic(
    image_np: np.ndarray,
    labels_resized: np.ndarray,
    compactness: float = 10.0,
    sigma: float = 1.0,
    force_skimage: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Refine cluster labels using SLIC superpixels with majority voting."""
    if not force_skimage and hasattr(cv2, "ximgproc"):
        return _refine_with_opencv_slic(
            image_np, labels_resized, compactness=compactness, region_size=48, verbose=verbose
        )

    if skimage_slic is None:
        if verbose:
            print("⚠️  scikit-image SLIC unavailable; skipping refinement (install scikit-image)")
        return labels_resized

    h, w = labels_resized.shape[:2]
    target_area = 48 * 48
    n_segments = max(100, int((h * w) / target_area))
    n_segments = min(n_segments, 2000)

    img_float = image_np.astype(np.float32)
    if img_float.max() > 1.5:
        img_float = img_float / 255.0

    segments = skimage_slic(
        img_float,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
    )

    refined = labels_resized.copy()
    seg_flat = segments.reshape(-1)
    lab_flat = labels_resized.reshape(-1)

    seg_ids = np.unique(seg_flat)
    for sid in seg_ids:
        mask = seg_flat == sid
        if not np.any(mask):
            continue
        vals = lab_flat[mask]
        max_label = np.bincount(vals).argmax()
        refined.reshape(-1)[mask] = max_label

    return refined.astype(np.uint8)


def _refine_with_opencv_slic(
    image_np: np.ndarray,
    labels_resized: np.ndarray,
    compactness: float = 10.0,
    region_size: int = 48,
    verbose: bool = True,
) -> np.ndarray:
    """Refine cluster labels using OpenCV's fast SLIC implementation."""
    if not hasattr(cv2, "ximgproc"):
        if verbose:
            print("⚠️  OpenCV ximgproc unavailable; skipping refinement (install opencv-contrib-python)")
        return labels_resized

    max_segments = 2000
    h, w = image_np.shape[:2]
    if region_size == 48:
        calculated_region_size = int(np.sqrt((h * w) / max_segments))
        region_size = max(region_size, calculated_region_size)

    slic = cv2.ximgproc.createSuperpixelSLIC(
        image_np,
        algorithm=cv2.ximgproc.SLIC,
        region_size=region_size,
        ruler=float(compactness),
    )
    slic.iterate(10)
    segments = slic.getLabels()

    seg_flat = segments.reshape(-1)
    lab_flat = labels_resized.reshape(-1)
    n_segments_actual = segments.max() + 1
    n_labels = labels_resized.max() + 1

    hist, _, _ = np.histogram2d(
        seg_flat,
        lab_flat,
        bins=[n_segments_actual, n_labels],
        range=[[0, n_segments_actual], [0, n_labels]],
    )
    segment_modes = np.argmax(hist, axis=1).astype(np.uint8)
    refined = segment_modes[segments]
    return refined.astype(np.uint8)


def _refine_with_bilateral(
    image_np: np.ndarray,
    labels_resized: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Refine cluster labels using bilateral filtering."""
    d = 9
    sigma_color = 75
    sigma_space = 75

    h, w = labels_resized.shape
    n_labels = int(labels_resized.max()) + 1
    label_smoothed = np.zeros((h, w), dtype=np.uint8)

    for label_id in range(n_labels):
        mask = (labels_resized == label_id).astype(np.float32)
        smoothed_mask = cv2.bilateralFilter(
            (mask * 255).astype(np.uint8),
            d,
            sigma_color,
            sigma_space,
        ).astype(np.float32) / 255.0
        label_smoothed[smoothed_mask > 0.5] = label_id

    if verbose:
        print("🔧 Refining segmentation with bilateral filter...")
    return label_smoothed
