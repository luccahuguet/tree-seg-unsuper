"""Shared helpers for running evaluation benchmarks (single and cache-aware)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import run_benchmark
from tree_seg.evaluation.formatters import config_to_dict, save_comparison_summary
from tree_seg.metadata.store import (
    _config_to_hash_config,
    config_hash,
    normalize_config,
    store_run,
)


def create_config(
    *,
    clustering: str,
    refine: Optional[str],
    model: str,
    stride: int,
    image_size: int,
    elbow_threshold: float,
    fixed_k: Optional[int],
    apply_vegetation_filter: bool,
    exg_threshold: float,
    no_tiling: bool,
    viz_two_panel: bool,
    viz_two_panel_opaque: bool,
    use_pyramid: bool,
    pyramid_scales: str,
    pyramid_aggregation: str,
    use_supervised: bool,
    quiet: bool,
    use_attention_features: bool,
) -> Config:
    """Create Config object from CLI parameters."""
    if use_supervised:
        version = "v4"
    elif refine == "soft-em" or refine == "soft-em+slic":
        version = "v2"
    elif apply_vegetation_filter:
        version = "v3"
    else:
        version = "v1.5"

    if use_supervised:
        if model != "mega":
            model = "mega"
        if image_size == 1024:
            image_size = 896
        return Config(
            version="v4",
            model_name="mega",
            image_size=image_size,
            auto_k=False,
            n_clusters=6,
            refine=None,
            metrics=True,
            verbose=not quiet,
            use_attention_features=use_attention_features,
        )

    clustering_method = clustering
    refine_method = None
    use_soft_refine = False

    if refine and refine != "none":
        if refine == "soft-em":
            use_soft_refine = True
            refine_method = None
        elif refine == "slic":
            refine_method = "slic"
        elif refine == "bilateral":
            refine_method = "bilateral"
        elif refine == "soft-em+slic":
            use_soft_refine = True
            refine_method = "slic"

    scales = tuple(float(s.strip()) for s in pyramid_scales.split(",")) if pyramid_scales else (0.5, 1.0, 2.0)

    return Config(
        version=version,
        clustering_method=clustering_method,
        refine=refine_method,
        model_name=model,
        stride=stride,
        elbow_threshold=elbow_threshold,
        n_clusters=fixed_k if fixed_k else 6,
        auto_k=fixed_k is None,
        image_size=image_size,
        apply_vegetation_filter=apply_vegetation_filter,
        exg_threshold=exg_threshold,
        use_tiling=not no_tiling,
        viz_two_panel=viz_two_panel,
        viz_two_panel_opaque=viz_two_panel_opaque,
        use_pyramid=use_pyramid,
        pyramid_scales=scales,
        pyramid_aggregation=pyramid_aggregation,
        use_soft_refine=use_soft_refine,
        soft_refine_temperature=1.0,
        soft_refine_iterations=5,
        soft_refine_spatial_alpha=0.0,
        use_attention_features=use_attention_features,
        metrics=True,
        verbose=not quiet,
    )


def resolve_output_dir(
    *,
    config: Config,
    dataset_type: str,
    smart_k: bool,
    output_dir: Optional[Path],
) -> Path:
    """Choose output directory for a single benchmark run."""
    if output_dir:
        return output_dir

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    refine_str = config.refine if config.refine else config.clustering_method
    method_str = f"{config.version}_{refine_str}"
    if smart_k:
        method_str += "_smartk"
    model_str = config.model_display_name.lower().replace(" ", "_")
    return Path("data/output/results") / f"{dataset_type}_{method_str}_{model_str}_{timestamp}"


def try_cached_results(
    *,
    config: Config,
    dataset_path: Path,
    dataset_type: str,
    smart_k: bool,
    console: Console,
) -> bool:
    """Best-effort cache lookup; returns True if we reused existing results."""
    hash_config = _config_to_hash_config(config, dataset_path.name, smart_k, grid_label=None)
    normalized = normalize_config(hash_config)
    hash_id = config_hash(normalized)
    meta_path = Path("results") / "by-hash" / hash_id / "meta.json"
    results_json = Path("results") / "by-hash" / hash_id / "results.json"
    if not meta_path.exists():
        return False

    try:
        with meta_path.open() as f:
            meta = json.load(f)
        artifacts = meta.get("artifacts", {})
        rjson = artifacts.get("results_json")
        rpath = Path(rjson) if rjson else results_json
        if not rpath.exists():
            return False
        console.print(f"[green]♻️  Cache hit for {hash_id}; reusing existing results.[/green]")
        with rpath.open() as f:
            cached = json.load(f)
        table = Table(title="Cached Benchmark Results", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        metrics = cached.get("metrics", {})
        table.add_row("Dataset", cached.get("dataset", dataset_type.upper()))
        table.add_row("Method", cached.get("method", ""))
        table.add_row("Mean mIoU", f"{metrics.get('mean_miou', 0):.4f}")
        table.add_row("Mean Pixel Accuracy", f"{metrics.get('mean_pixel_accuracy', 0):.4f}")
        table.add_row("Mean Runtime", f"{metrics.get('mean_runtime', 0):.2f}s")
        table.add_row("Total Samples", str(cached.get("total_samples", 0)))
        console.print()
        console.print(table)
        console.print(f"[dim]Source: {rpath}[/dim]")
        return True
    except Exception:
        return False


def run_single_benchmark(
    *,
    dataset_path: Path,
    dataset_type: str,
    config: Config,
    output_dir: Path,
    num_samples: Optional[int],
    save_viz: bool,
    save_labels: bool,
    quiet: bool,
    smart_k: bool,
):
    """Run a single benchmark configuration and persist metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(
        dataset_path,
        dataset_type,
        config=config,
        output_dir=output_dir,
        num_samples=num_samples,
        save_viz=save_viz,
        save_labels=save_labels,
        silent=quiet,
    )

    summary_info = save_comparison_summary([config_to_dict(config)], results, output_dir, smart_k=smart_k)
    try:
        store_run(summary_info)
    except Exception:
        pass

    return results
