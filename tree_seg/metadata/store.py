"""Metadata storage for benchmark runs."""

import csv
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from typing import TYPE_CHECKING

from tree_seg.core.types import Config, SegmentationResults
from tree_seg.evaluation.formatters import config_to_dict

if TYPE_CHECKING:  # Avoid circular import at runtime
    from tree_seg.evaluation.benchmark import BenchmarkResults

# Keys that affect runtime/outputs; must be present (with defaults) in hashes.
HASH_KEYS = [
    "dataset",
    "model",
    "use_attention_features",
    "clustering",
    "k",
    "smart_k",
    "elbow_threshold",
    "refine",
    "soft_refine",
    "soft_refine_temperature",
    "soft_refine_iterations",
    "vegetation_filter",
    "supervised",
    "stride",
    "tiling",
    "tile_overlap",
    "image_size",
    "pyramid",
    "pyramid_scales",
    "pyramid_aggregation",
    "grid_label",
]

# Defaults used when a field is absent in the raw config.
HASH_DEFAULTS: Dict[str, object] = {
    "tiling": False,
    "tile_overlap": 0,
    "pyramid": False,
    "pyramid_scales": [1],
    "pyramid_aggregation": "mean",
    "soft_refine": False,
    "soft_refine_temperature": 1.0,
    "soft_refine_iterations": 5,
    "vegetation_filter": False,
    "supervised": False,
    "smart_k": False,
    "use_attention_features": True,
}

# GPU tier buckets used for ETA scaling (rough buckets).
GPU_TIERS: Dict[str, List[str]] = {
    "extreme": ["A100", "H100", "RTX 4090", "RTX 3090"],
    "high": ["RTX 4080", "RTX 3080", "A6000", "V100"],
    "mid": ["RTX 4070", "RTX 3070", "RTX 2080", "T4"],
    "low": ["RTX 3060", "GTX 1080", "P100", "CPU"],
}


def _now_iso() -> str:
    """UTC ISO timestamp with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _detect_gpu_tier(name: str) -> str:
    """Map GPU name to tier bucket."""
    if not name:
        return "low"
    name_upper = name.upper()
    for tier, matches in GPU_TIERS.items():
        if any(m.upper() in name_upper for m in matches):
            return tier
    return "mid"


def _hardware_info() -> Dict[str, object]:
    """Collect lightweight hardware info."""
    gpu_name = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = None

    cpu_name = platform.processor() or platform.machine() or "unknown-cpu"
    threads = os.cpu_count() or 1
    return {
        "gpu": gpu_name or "CPU",
        "gpu_tier": _detect_gpu_tier(gpu_name or "CPU"),
        "cpu": cpu_name,
        "cpu_threads": threads,
        "ram_gb": None,  # Can be filled later if needed
    }


def normalize_config(raw_config: Dict[str, object]) -> Dict[str, object]:
    """Normalize config for consistent hashing."""
    normalized: Dict[str, object] = {}
    for key in HASH_KEYS:
        value = raw_config.get(key, HASH_DEFAULTS.get(key))
        if value is None:
            continue
        # Freeze mutable types for hashing
        if isinstance(value, list):
            value = tuple(value)
        normalized[key] = value
    return normalized


def derive_tags(config: Dict[str, object]) -> List[str]:
    """Generate auto-tags from normalized config."""
    tags: List[str] = []
    if config.get("dataset"):
        tags.append(str(config["dataset"]))
    if config.get("clustering"):
        tags.append(str(config["clustering"]))
    if "k" in config and config["k"] is not None:
        tags.append(f"k{config['k']}")
    if config.get("refine"):
        tags.append(str(config["refine"]))
    if config.get("model"):
        tags.append(str(config["model"]))
    if config.get("stride") is not None:
        tags.append(f"stride-{config['stride']}")
    if config.get("vegetation_filter"):
        tags.append("veg-filter")
    if config.get("smart_k"):
        tags.append("smart-k")
    if config.get("supervised"):
        tags.append("supervised")
    if config.get("pyramid"):
        tags.append("pyramid")
    return tags


def config_hash(normalized_config: Dict[str, object]) -> str:
    """Generate stable hash for normalized config."""
    serializable = {
        k: list(v) if isinstance(v, tuple) else v for k, v in normalized_config.items()
    }
    canonical = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
    return hashlib_sha256(canonical)[:10]


def hashlib_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _append_to_metrics_csv(
    base_dir: Path,
    hash_id: str,
    timestamp: str,
    dataset: str,
    config: Dict[str, object],
    metrics: Dict[str, float],
    timing: Dict[str, float],
    hardware: Dict[str, object],
    num_samples: int,
    smart_k: bool,
    grid_label: Optional[str],
) -> None:
    """Upsert a row to the git-tracked metrics CSV (update if hash exists, append if new)."""
    csv_path = base_dir / "metrics.csv"

    # Prepare row data
    row = {
        "timestamp": timestamp,
        "hash": hash_id,
        "dataset": dataset,
        "model": config.get("model", ""),
        "clustering": config.get("clustering", ""),
        "refine": config.get("refine", "none"),
        "stride": config.get("stride", ""),
        "tiling": config.get("tiling", False),
        "image_size": config.get("image_size", ""),
        "smart_k": smart_k,
        "grid_label": grid_label or "",
        "n_samples": num_samples,
        "mean_miou": f"{metrics.get('mean_miou', 0):.6f}",
        "mean_pixel_acc": f"{metrics.get('mean_pixel_accuracy', 0):.6f}",
        "mean_runtime_s": f"{timing.get('mean_runtime_s', 0):.2f}",
        "total_runtime_s": f"{timing.get('total_runtime_s', 0):.2f}",
        "gpu": hardware.get("gpu", "CPU"),
        "gpu_tier": hardware.get("gpu_tier", "low"),
    }

    # Column order
    fieldnames = [
        "timestamp",
        "hash",
        "dataset",
        "model",
        "clustering",
        "refine",
        "stride",
        "tiling",
        "image_size",
        "smart_k",
        "grid_label",
        "n_samples",
        "mean_miou",
        "mean_pixel_acc",
        "mean_runtime_s",
        "total_runtime_s",
        "gpu",
        "gpu_tier",
    ]

    # Read existing rows if file exists, update matching hash or append new
    existing_rows = []
    updated = False
    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for existing_row in reader:
                if existing_row["hash"] == hash_id:
                    # Update existing row with new data
                    existing_rows.append(row)
                    updated = True
                else:
                    existing_rows.append(existing_row)

    # Append new row if not updated
    if not updated:
        existing_rows.append(row)

    # Write all rows back
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def _config_to_hash_config(
    config: Config,
    dataset_id: str,
    smart_k: bool,
    grid_label: Optional[str],
) -> Dict[str, object]:
    """Build the minimal config dict used for hashing and tags."""
    image_size = config.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    hash_config: Dict[str, object] = {
        "dataset": dataset_id,
        "model": config.model_name,
        "use_attention_features": config.use_attention_features,
        "clustering": config.clustering_method,
        "k": config.n_clusters,
        "smart_k": smart_k,
        "elbow_threshold": config.elbow_threshold,
        "refine": config.refine or "none",
        "soft_refine": config.use_soft_refine,
        "soft_refine_temperature": config.soft_refine_temperature,
        "soft_refine_iterations": config.soft_refine_iterations,
        "vegetation_filter": config.apply_vegetation_filter,
        "supervised": config.supervised,
        "stride": config.stride,
        "tiling": config.use_tiling,
        "tile_overlap": getattr(config, "tile_overlap", 0),
        "image_size": image_size,
        "pyramid": config.use_pyramid,
        "pyramid_scales": config.pyramid_scales,
        "pyramid_aggregation": config.pyramid_aggregation,
        "grid_label": grid_label,
    }
    return hash_config


def store_run(
    results: "BenchmarkResults",
    config: Config,
    dataset_path: Path,
    smart_k: bool = False,
    user_tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    base_dir: Path | str = "results",
    grid_label: Optional[str] = None,
    artifacts: Optional[Dict[str, str]] = None,
) -> str:
    """
    Persist benchmark results to the metadata bank.

    Returns:
        hash_id: The hash of the normalized config (used as directory name).
    """
    base_dir = Path(base_dir)
    by_hash_dir = base_dir / "by-hash"
    _ensure_dir(by_hash_dir)

    dataset_id = dataset_path.name
    hash_config = _config_to_hash_config(config, dataset_id, smart_k, grid_label)
    normalized = normalize_config(hash_config)
    hash_id = config_hash(normalized)

    run_dir = by_hash_dir / hash_id
    _ensure_dir(run_dir)

    created_at = _now_iso()
    auto_tags = derive_tags(normalized)
    all_tags = {
        "auto": auto_tags,
        "user": user_tags or [],
    }

    # Metrics and timing
    total_runtime = sum(s.runtime_seconds for s in results.samples)
    per_sample_stats = [
        {
            "image_id": s.image_id,
            "miou": float(s.miou),
            "pixel_accuracy": float(s.pixel_accuracy),
            "num_clusters": int(s.num_clusters),
            "runtime_seconds": float(s.runtime_seconds),
            "image_shape": list(s.image_shape),
        }
        for s in results.samples
    ]

    hardware = _hardware_info()
    config_full = config_to_dict(config)
    config_full["smart_k"] = smart_k
    config_full["grid_label"] = grid_label

    meta = {
        "hash": hash_id,
        "created_at": created_at,
        "git_sha": os.getenv("GIT_SHA"),
        "dataset": dataset_id,
        "dataset_root": str(dataset_path),
        "config": normalized,
        "config_full": config_full,
        "tags": all_tags,
        "samples": {
            "num_samples": results.total_samples,
            "per_sample_stats": per_sample_stats,
        },
        "hardware": hardware,
        "timing": {
            "mean_runtime_s": float(results.mean_runtime),
            "total_runtime_s": float(total_runtime),
        },
        "metrics": {
            "mean_miou": float(results.mean_miou),
            "mean_pixel_accuracy": float(results.mean_pixel_accuracy),
        },
        "artifacts": artifacts or {},
        "notes": notes,
        "metadata_version": 1,
    }

    # Write meta.json
    meta_path = run_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    # Append to index
    index_path = base_dir / "index.jsonl"
    index_entry = {
        "hash": hash_id,
        "dataset": dataset_id,
        "tags": auto_tags + (user_tags or []),
        "mIoU": float(results.mean_miou),
        "pixel_accuracy": float(results.mean_pixel_accuracy),
        "total_s": float(total_runtime),
        "created_at": created_at,
        "config": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in normalized.items()
        },
    }
    with index_path.open("a") as f:
        f.write(json.dumps(index_entry) + "\n")

    # Append to git-tracked CSV for easy performance tracking
    _append_to_metrics_csv(
        base_dir=base_dir,
        hash_id=hash_id,
        timestamp=created_at,
        dataset=dataset_id,
        config=normalized,
        metrics=meta["metrics"],
        timing=meta["timing"],
        hardware=hardware,
        num_samples=results.total_samples,
        smart_k=smart_k,
        grid_label=grid_label,
    )

    return hash_id


def store_segment_run(
    config: Config,
    input_path: Path,
    outputs: List[tuple[SegmentationResults, object]],
    base_dir: Path | str = "results",
    user_tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Persist unlabeled segment runs (no GT metrics), capturing timings and artifacts.

    Args:
        config: Config used for segmentation
        input_path: Path to processed file or directory
        outputs: List of (SegmentationResults, OutputPaths)
    """
    base_dir = Path(base_dir)
    by_hash_dir = base_dir / "by-hash"
    _ensure_dir(by_hash_dir)

    dataset_id = input_path.name if input_path.is_dir() else input_path.parent.name
    hash_config = _config_to_hash_config(
        config, dataset_id, smart_k=False, grid_label=None
    )
    normalized = normalize_config(hash_config)
    hash_id = config_hash(normalized)

    run_dir = by_hash_dir / hash_id
    _ensure_dir(run_dir)

    created_at = _now_iso()
    auto_tags = derive_tags(normalized)
    all_tags = {"auto": auto_tags, "user": user_tags or []}

    # Sample stats
    sample_entries = []
    total_runtime = 0.0
    for seg_result, path_obj in outputs:
        stats = getattr(seg_result, "processing_stats", {}) or {}
        runtime = float(stats.get("time_total_s") or 0.0)
        total_runtime += runtime
        sample_entries.append(
            {
                "image_id": Path(seg_result.image_path).stem
                if seg_result.image_path
                else None,
                "n_clusters_used": seg_result.n_clusters_used,
                "runtime_seconds": runtime,
                "stats": stats,
                "outputs": path_obj.__dict__ if hasattr(path_obj, "__dict__") else {},
            }
        )

    hardware = _hardware_info()
    config_full = config_to_dict(config)

    meta = {
        "hash": hash_id,
        "created_at": created_at,
        "git_sha": os.getenv("GIT_SHA"),
        "dataset": dataset_id,
        "dataset_root": str(input_path),
        "config": normalized,
        "config_full": config_full,
        "tags": all_tags,
        "samples": {
            "num_samples": len(outputs),
            "per_sample_stats": sample_entries,
        },
        "hardware": hardware,
        "timing": {
            "total_runtime_s": total_runtime,
        },
        "metrics": {},
        "artifacts": {},
        "notes": notes,
        "metadata_version": 1,
        "type": "segment",
    }

    meta_path = run_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    index_path = base_dir / "index.jsonl"
    index_entry = {
        "hash": hash_id,
        "dataset": dataset_id,
        "tags": auto_tags + (user_tags or []),
        "total_s": float(total_runtime),
        "created_at": created_at,
        "config": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in normalized.items()
        },
        "type": "segment",
    }
    with index_path.open("a") as f:
        f.write(json.dumps(index_entry) + "\n")

    return hash_id
