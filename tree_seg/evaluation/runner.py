"""Shared helpers for running evaluation benchmarks (single and cache-aware)."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import run_benchmark
from tree_seg.evaluation.datasets import FortressDataset, ISPRSPotsdamDataset
from tree_seg.metadata.store import (
    _config_to_hash_config,
    config_hash,
    normalize_config,
    store_run,
)


def detect_dataset_type(dataset_path: Path) -> str:
    """Best-effort dataset type detection."""
    if (dataset_path / "images").exists() and (dataset_path / "labels").exists():
        return "fortress"
    if (dataset_path / "2_Ortho_RGB").exists():
        return "isprs"
    return "generic"


def load_dataset(dataset_path: Path, dataset_type: Optional[str] = None):
    """Instantiate a dataset based on type or auto-detection."""
    dtype = dataset_type or detect_dataset_type(dataset_path)
    if dtype == "fortress":
        return FortressDataset(dataset_path), dtype
    if dtype == "isprs":
        return ISPRSPotsdamDataset(dataset_path), dtype
    # Fallback to ISPRS interface for now
    return ISPRSPotsdamDataset(dataset_path), dtype


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
        if model != "mega":
            model = "mega"
        if image_size == 1024:
            image_size = 896
        return Config(
            model_name="mega",
            image_size=image_size,
            auto_k=False,
            n_clusters=6,
            refine=None,
            metrics=True,
            verbose=not quiet,
            use_attention_features=use_attention_features,
            supervised=True,
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

    scales = (
        tuple(float(s.strip()) for s in pyramid_scales.split(","))
        if pyramid_scales
        else (0.5, 1.0, 2.0)
    )

    return Config(
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
    method_str = f"{config.clustering_method}_{refine_str}"
    if smart_k:
        method_str += "_smartk"
    model_str = config.model_display_name.lower().replace(" ", "_")
    return (
        Path("data/outputs/results")
        / f"{dataset_type}_{method_str}_{model_str}_{timestamp}"
    )


def _hash_and_run_dir(
    config: Config, dataset_path: Path, smart_k: bool, grid_label: Optional[str]
) -> tuple[str, Path]:
    """Compute hash + canonical run directory for a config/dataset."""
    hash_config = _config_to_hash_config(
        config, dataset_path.name, smart_k, grid_label=grid_label
    )
    normalized = normalize_config(hash_config)
    hash_id = config_hash(normalized)
    run_dir = Path("results") / "by-hash" / hash_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return hash_id, run_dir


def _symlink_artifacts(
    src_dir: Path, dest_dir: Path, *, label: Optional[str], ensure_label: bool
) -> None:
    """Create flat symlinks for all files in src_dir into dest_dir."""
    if not src_dir.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_dir.glob("*"):
        dest_name = src_file.name
        if ensure_label and label:
            stem, suffix = src_file.stem, src_file.suffix
            if label not in stem:
                dest_name = f"{stem}_{label}{suffix}"
        dest_file = dest_dir / dest_name
        if dest_file.exists() or dest_file.is_symlink():
            try:
                if dest_file.is_symlink() and dest_file.resolve() == src_file.resolve():
                    continue
                dest_file.unlink()
            except Exception:
                continue
        try:
            relative = os.path.relpath(src_file, dest_file.parent)
            dest_file.symlink_to(relative)
        except FileExistsError:
            continue


def _link_run_into_sweep(sweep_dir: Path, label: str, run_dir: Path) -> None:
    """Symlink canonical artifacts into sweep folder for browsing."""
    viz_src = run_dir / "visualizations"
    labels_src = run_dir / "labels"

    if viz_src.exists():
        _symlink_artifacts(
            viz_src,
            sweep_dir / "visualizations",
            label=label,
            ensure_label=True,
        )
    if labels_src.exists():
        _symlink_artifacts(
            labels_src,
            sweep_dir / "labels",
            label=label,
            ensure_label=True,
        )


def try_cached_results(
    *,
    config: Config,
    dataset_path: Path,
    smart_k: bool,
    console: Console,
    force: bool = False,
) -> bool:
    """Best-effort cache lookup; returns True if we reused existing results."""
    if force:
        return False

    hash_config = _config_to_hash_config(
        config, dataset_path.name, smart_k, grid_label=None
    )
    normalized = normalize_config(hash_config)
    hash_id = config_hash(normalized)
    meta_path = Path("results") / "by-hash" / hash_id / "meta.json"
    if not meta_path.exists():
        return False

    try:
        with meta_path.open() as f:
            meta = json.load(f)

        console.print(
            f"[green]♻️  Cache hit for {hash_id}; reusing existing results.[/green]"
        )

        table = Table(
            title="Cached Benchmark Results", show_header=True, header_style="bold cyan"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        # Extract from meta.json structure
        metrics = meta.get("metrics", {})
        timing = meta.get("timing", {})
        samples = meta.get("samples", {})
        config_full = meta.get("config_full", {})

        # Build method name
        method = config_full.get("clustering", "unknown")
        refine = config_full.get("refine")
        if refine:
            method = f"{method}_{refine}"

        table.add_row("Dataset", meta.get("dataset", ""))
        table.add_row("Method", method)
        table.add_row("Mean mIoU", f"{metrics.get('mean_miou', 0):.4f}")
        table.add_row(
            "Mean Pixel Accuracy", f"{metrics.get('mean_pixel_accuracy', 0):.4f}"
        )
        table.add_row("Mean Runtime", f"{timing.get('mean_runtime_s', 0):.2f}s")
        table.add_row("Total Samples", str(samples.get("num_samples", 0)))
        console.print()
        console.print(table)
        console.print(f"[dim]Source: {meta_path}[/dim]")
        return True
    except Exception as e:
        console.print(f"[yellow]⚠️  Cache lookup failed: {e}[/yellow]")
        return False


def run_single_benchmark(
    *,
    dataset_path: Path,
    dataset_type: Optional[str],
    config: Config,
    output_dir: Path,
    num_samples: Optional[int],
    save_viz: bool,
    save_labels: bool,
    quiet: bool,
    smart_k: bool,
    use_cache: bool,
):
    """Run a single benchmark configuration and persist metadata."""
    # Canonical storage keyed by hash
    hash_id, run_dir = _hash_and_run_dir(
        config=config, dataset_path=dataset_path, smart_k=smart_k, grid_label=None
    )

    # User-facing directory (for browsing) gets symlinks; canonical files live under results/by-hash
    browse_dir = output_dir
    browse_dir.mkdir(parents=True, exist_ok=True)

    dataset, _dtype = load_dataset(dataset_path, dataset_type)

    results = run_benchmark(
        config=config,
        dataset=dataset,
        output_dir=run_dir,
        num_samples=num_samples,
        save_visualizations=save_viz,
        save_labels=save_labels,
        verbose=not quiet,
        use_smart_k=smart_k,
    )

    # Mirror artifacts into the requested output directory via symlinks
    if browse_dir != run_dir:
        _symlink_artifacts(
            run_dir / "visualizations",
            browse_dir / "visualizations",
            label=None,
            ensure_label=False,
        )
        _symlink_artifacts(
            run_dir / "labels",
            browse_dir / "labels",
            label=None,
            ensure_label=False,
        )

    # Store results in metadata bank
    try:
        hash_id = store_run(
            results=results,
            config=config,
            dataset_path=dataset_path,
            smart_k=smart_k,
        )
        print(f"\n📦 Metadata stored: results/by-hash/{hash_id}/")
    except Exception as e:
        print(f"⚠️  Warning: Failed to store metadata: {e}")

    return results


def run_sweep(
    *,
    dataset_path: Path,
    dataset_type: Optional[str],
    grid_name: str,
    configs_to_test: list[dict],
    base_config_params: dict,
    num_samples: Optional[int],
    save_viz: bool,
    save_labels: bool,
    quiet: bool,
    smart_k: bool,
    console: Console,
    use_cache: bool = True,
):
    """Run a grid sweep and return results + sweep_dir."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    smartk_suffix = "_smartk" if smart_k else ""
    sweep_dir = (
        Path("data/outputs/results") / f"sweep_{grid_name}{smartk_suffix}_{timestamp}"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[green]📁 Sweep directory: {sweep_dir}[/green]\n")

    all_results = []
    model_cache = {}
    dataset, dtype_resolved = load_dataset(dataset_path, dataset_type)

    for i, config_override in enumerate(configs_to_test):
        config_dict = base_config_params.copy()
        config_dict.update({k: v for k, v in config_override.items() if k != "label"})

        config = Config(**config_dict)
        label = config_override["label"]

        console.print(
            f"\n[bold][{i + 1}/{len(configs_to_test)}] Testing: {label}[/bold]"
        )
        console.print("-" * 40)

        hash_id, run_dir = _hash_and_run_dir(
            config=config, dataset_path=dataset_path, smart_k=smart_k, grid_label=label
        )

        if use_cache:
            meta_path = Path("results") / "by-hash" / hash_id / "meta.json"
            if meta_path.exists():
                console.print(
                    f"[green]♻️  Cache hit for {label} ({hash_id}); skipping.[/green]"
                )
                _link_run_into_sweep(sweep_dir, label, run_dir)
                continue

        model_key = (config.model_display_name, config.stride, config.image_size)
        if model_key in model_cache:
            console.print(
                f"[dim]♻️  Reusing cached model: {config.model_display_name} (stride={config.stride})[/dim]"
            )

        results = run_benchmark(
            config=config,
            dataset=dataset,
            output_dir=run_dir,
            num_samples=num_samples,
            save_visualizations=save_viz,
            save_labels=save_labels,
            verbose=not quiet,
            use_smart_k=smart_k,
            model_cache=model_cache,
            config_label=label,
        )

        all_results.append({"label": label, "config": config_dict, "results": results})

        _link_run_into_sweep(sweep_dir, label, run_dir)

        try:
            artifacts = {
                "sweep_dir": str(sweep_dir),
                "labels_dir": str(run_dir / "labels"),
                "visualizations_dir": str(run_dir / "visualizations"),
            }
            store_run(
                results=results,
                config=config,
                dataset_path=dataset_path,
                smart_k=smart_k,
                user_tags=[grid_name, label],
                grid_label=label,
                artifacts=artifacts,
            )
        except Exception:
            pass

    return all_results, sweep_dir
