"""Metrics assembly helpers for segmentation runs."""

from __future__ import annotations

from typing import Optional


def compile_metrics(
    *,
    t_start: float,
    t_pre_start: float,
    t_pre_end: float,
    t_features: float,
    t_kselect: float,
    t_refine_start: float,
    t_refine_end: float,
    auto_k: bool,
    refine_time: float,
    H: int,
    W: int,
    features_flat,
    n_clusters: int,
    device,
    peak_vram_mb: Optional[float],
    needs_tiling: bool,
) -> dict:
    """Assemble metrics dict for a segmentation run."""
    t_end = t_refine_end if refine_time > 0 else t_features if auto_k else t_features

    return {
        "time_total_s": round(t_end - t_start, 3),
        "time_preprocess_s": round(t_pre_end - t_pre_start, 3),
        "time_features_s": round(t_features - t_pre_end, 3),
        "time_kselect_s": round(t_kselect - t_features, 3) if auto_k else 0.0,
        "time_kmeans_s": round(
            (t_refine_start if refine_time > 0 else t_end) - t_kselect, 3
        ),
        "time_refine_s": round(refine_time, 3) if refine_time > 0 else 0.0,
        "grid_H": int(H),
        "grid_W": int(W),
        "n_features": int(features_flat.shape[-1]),
        "n_vectors": int(features_flat.shape[0]),
        "n_clusters": int(n_clusters),
        "device_requested": str(device),
        "device_actual": str(device),
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
        "used_tiling": needs_tiling,
    }
