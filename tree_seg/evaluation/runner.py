"""Shared helpers for running evaluation benchmarks (single and cache-aware)."""

from __future__ import annotations

import json
import os
import datetime
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console
from rich.table import Table

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import (
    BenchmarkResults,
    BenchmarkSample,
    run_benchmark,
)
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


def _filter_dataset_samples(dataset, filter_ids: Optional[list[str]]) -> None:
    """Filter dataset.samples in place by image_id."""
    if not filter_ids:
        return
    id_set = set(filter_ids)
    samples_before = len(dataset.samples)
    dataset.samples = [s for s in dataset.samples if s.image_id in id_set]
    if not dataset.samples:
        raise ValueError(
            f"No samples matched filter ids: {', '.join(filter_ids)} "
            f"(available: {samples_before})"
        )


def load_dataset(
    dataset_path: Path,
    dataset_type: Optional[str] = None,
    filter_ids: Optional[list[str]] = None,
):
    """Instantiate a dataset based on type or auto-detection."""
    dtype = dataset_type or detect_dataset_type(dataset_path)
    if dtype == "fortress":
        ds = FortressDataset(dataset_path)
    elif dtype == "isprs":
        ds = ISPRSPotsdamDataset(dataset_path)
    else:
        ds = ISPRSPotsdamDataset(dataset_path)

    _filter_dataset_samples(ds, filter_ids)
    return ds, dtype


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
    tiling: bool,
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
        use_tiling=tiling,
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


def _benchmark_results_from_samples(
    *, config: Config, dataset_name: str, samples: list[BenchmarkSample]
) -> BenchmarkResults:
    """Build BenchmarkResults from a list of per-sample results."""
    if samples:
        mean_miou = float(np.mean([s.miou for s in samples]))
        mean_pixel_acc = float(np.mean([s.pixel_accuracy for s in samples]))
        mean_runtime = float(np.mean([s.runtime_seconds for s in samples]))
    else:
        mean_miou = 0.0
        mean_pixel_acc = 0.0
        mean_runtime = 0.0

    refine_str = (
        "mask2former"
        if config.supervised
        else (config.refine if config.refine else config.clustering_method)
    )
    method_name = f"{config.clustering_method}_{refine_str}"

    return BenchmarkResults(
        dataset_name=dataset_name,
        method_name=method_name,
        config=config,
        samples=samples,
        mean_miou=mean_miou,
        mean_pixel_accuracy=mean_pixel_acc,
        mean_runtime=mean_runtime,
        total_samples=len(samples),
    )


def _load_cached_samples(meta_path: Path) -> dict[str, BenchmarkSample]:
    """Parse cached per-sample stats into BenchmarkSample objects."""
    if not meta_path.exists():
        return {}

    try:
        with meta_path.open() as f:
            meta = json.load(f)
    except Exception:
        return {}

    per_sample_stats = meta.get("samples", {}).get("per_sample_stats", []) or []
    cached: dict[str, BenchmarkSample] = {}
    for entry in per_sample_stats:
        image_id = entry.get("image_id")
        if not image_id:
            continue
        cached[image_id] = BenchmarkSample(
            image_id=image_id,
            miou=float(entry.get("miou", 0.0)),
            pixel_accuracy=float(entry.get("pixel_accuracy", 0.0)),
            per_class_iou={
                k: float(v) for k, v in (entry.get("per_class_iou") or {}).items()
            },
            num_clusters=int(entry.get("num_clusters", 0) or 0),
            runtime_seconds=float(entry.get("runtime_seconds", 0.0)),
            image_shape=tuple(entry.get("image_shape") or (0, 0, 0)),
        )
    return cached


def _target_image_ids(dataset, num_samples: Optional[int]) -> list[str]:
    """Return the ordered list of image_ids that will be evaluated."""
    target_ids = [s.image_id for s in dataset.samples]
    if num_samples is not None:
        target_ids = target_ids[:num_samples]
    return target_ids


def _combine_samples(
    target_ids: list[str],
    cached_samples: dict[str, BenchmarkSample],
    new_samples: list[BenchmarkSample] | None,
) -> list[BenchmarkSample]:
    """Merge cached + newly computed samples preserving dataset order."""
    new_map = {s.image_id: s for s in (new_samples or [])}
    combined: list[BenchmarkSample] = []
    for image_id in target_ids:
        if image_id in cached_samples:
            combined.append(cached_samples[image_id])
        elif image_id in new_map:
            combined.append(new_map[image_id])
    return combined


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


def _apply_spectral_guard(config: Config, console: Optional[Console] = None) -> Config:
    """Downsize spectral runs to avoid OOM on large images."""
    if config.clustering_method != "spectral":
        return config

    changed = []
    if config.image_size > 768:
        config.image_size = 768
        changed.append("image_size->768")

    if changed:
        msg = "[yellow]⚠️  Spectral guard applied: {}[/yellow]".format(
            ", ".join(changed)
        )
        if console:
            console.print(msg)
        else:
            print(msg)

    return config


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
    filter_ids: Optional[list[str]],
    save_viz: bool,
    save_labels: bool,
    quiet: bool,
    smart_k: bool,
    use_cache: bool,
):
    """Run a single benchmark configuration and persist metadata."""
    config = _apply_spectral_guard(config)
    # Canonical storage keyed by hash
    hash_id, run_dir = _hash_and_run_dir(
        config=config, dataset_path=dataset_path, smart_k=smart_k, grid_label=None
    )

    # User-facing directory (for browsing) gets symlinks; canonical files live under results/by-hash
    browse_dir = output_dir
    browse_dir.mkdir(parents=True, exist_ok=True)

    dataset, _dtype = load_dataset(dataset_path, dataset_type, filter_ids=filter_ids)
    target_ids = _target_image_ids(dataset, num_samples)
    meta_path = run_dir / "meta.json"
    cached_samples = _load_cached_samples(meta_path) if use_cache else {}
    missing_ids = (
        [iid for iid in target_ids if iid not in cached_samples]
        if use_cache
        else target_ids
    )

    run_num_samples = num_samples
    if use_cache:
        if missing_ids and len(missing_ids) < len(target_ids):
            print(
                f"♻️  Cache hit for {len(target_ids) - len(missing_ids)}/{len(target_ids)} sample(s); "
                f"running {len(missing_ids)} new sample(s)."
            )
        elif not missing_ids and cached_samples:
            print(
                f"[green]♻️  Cache hit for all {len(target_ids)} sample(s); reusing cached results.[/green]"
            )
        run_num_samples = len(missing_ids)
        if missing_ids:
            missing_set = set(missing_ids)
            dataset.samples = [s for s in dataset.samples if s.image_id in missing_set]
        else:
            dataset.samples = []

    results = None
    if run_num_samples:
        results = run_benchmark(
            config=config,
            dataset=dataset,
            output_dir=run_dir,
            num_samples=run_num_samples,
            save_visualizations=save_viz,
            save_labels=save_labels,
            verbose=not quiet,
            use_smart_k=smart_k,
        )

    combined_results: BenchmarkResults
    if use_cache and cached_samples:
        combined_samples = _combine_samples(
            target_ids, cached_samples, results.samples if results else []
        )
        combined_results = _benchmark_results_from_samples(
            config=config,
            dataset_name=dataset_path.name,
            samples=combined_samples,
        )
    else:
        combined_results = results  # type: ignore[assignment]

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
    should_store = (
        results is not None
        or not use_cache
        or not cached_samples
        or not meta_path.exists()
    )
    if should_store and combined_results is not None:
        try:
            hash_id = store_run(
                results=combined_results,
                config=config,
                dataset_path=dataset_path,
                smart_k=smart_k,
            )
            print(f"\n📦 Metadata stored: results/by-hash/{hash_id}/")
        except Exception as e:
            print(f"⚠️  Warning: Failed to store metadata: {e}")

    return combined_results


def run_sweep(
    *,
    dataset_path: Path,
    dataset_type: Optional[str],
    grid_name: str,
    configs_to_test: list[dict],
    base_config_params: dict,
    num_samples: Optional[int],
    filter_ids: Optional[list[str]],
    save_viz: bool,
    save_labels: bool,
    quiet: bool,
    smart_k: bool,
    console: Console,
    use_cache: bool = True,
):
    """Run a grid sweep and return results + sweep_dir."""
    sweep_dir = Path("data/outputs/results") / f"sweep_{grid_name}"
    if sweep_dir.exists():
        shutil.rmtree(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[green]📁 Sweep directory: {sweep_dir}[/green]\n")

    all_results = []
    model_cache = {}
    dataset, dtype_resolved = load_dataset(
        dataset_path, dataset_type, filter_ids=filter_ids
    )
    base_samples = list(dataset.samples)

    for i, config_override in enumerate(configs_to_test):
        dataset.samples = list(base_samples)
        config_dict = base_config_params.copy()
        config_dict.update({k: v for k, v in config_override.items() if k != "label"})

        config = Config(**config_dict)
        config = _apply_spectral_guard(config, console=console)
        # Keep dict in sync for reporting/hashing consistency
        config_dict["image_size"] = config.image_size
        config_dict["stride"] = config.stride
        config_dict["use_tiling"] = config.use_tiling
        config_dict["tile_threshold"] = config.tile_threshold
        label = config_override["label"]

        console.print(
            f"\n[bold][{i + 1}/{len(configs_to_test)}] Testing: {label}[/bold]"
        )
        console.print("-" * 40)

        hash_id, run_dir = _hash_and_run_dir(
            config=config, dataset_path=dataset_path, smart_k=smart_k, grid_label=label
        )

        target_ids = _target_image_ids(dataset, num_samples)
        meta_path = run_dir / "meta.json"
        cached_samples = _load_cached_samples(meta_path) if use_cache else {}
        missing_ids = (
            [iid for iid in target_ids if iid not in cached_samples]
            if use_cache
            else target_ids
        )

        if use_cache:
            if missing_ids and len(missing_ids) < len(target_ids):
                console.print(
                    f"[green]♻️  Cache hit for {len(target_ids) - len(missing_ids)}/{len(target_ids)} sample(s); running {len(missing_ids)} new sample(s).[/green]"
                )
            elif not missing_ids and cached_samples:
                console.print(
                    f"[green]♻️  Cache hit for all {len(target_ids)} sample(s); skipping compute.[/green]"
                )

        if use_cache and missing_ids:
            missing_set = set(missing_ids)
            dataset.samples = [s for s in dataset.samples if s.image_id in missing_set]
            run_num_samples = len(missing_ids)
        elif use_cache and not missing_ids and cached_samples:
            dataset.samples = []
            run_num_samples = 0
        else:
            run_num_samples = num_samples

        model_key = (config.model_display_name, config.stride, config.image_size)
        if model_key in model_cache:
            console.print(
                f"[dim]♻️  Reusing cached model: {config.model_display_name} (stride={config.stride})[/dim]"
            )

        results = None
        if run_num_samples:
            results = run_benchmark(
                config=config,
                dataset=dataset,
                output_dir=run_dir,
                num_samples=run_num_samples,
                save_visualizations=save_viz,
                save_labels=save_labels,
                verbose=not quiet,
                use_smart_k=smart_k,
                model_cache=model_cache,
                config_label=label,
            )

        combined_results: BenchmarkResults
        if use_cache and cached_samples:
            combined_samples = _combine_samples(
                target_ids, cached_samples, results.samples if results else []
            )
            combined_results = _benchmark_results_from_samples(
                config=config,
                dataset_name=dataset_path.name,
                samples=combined_samples,
            )
        else:
            combined_results = results  # type: ignore[assignment]

        if combined_results is not None:
            all_results.append(
                {"label": label, "config": config_dict, "results": combined_results}
            )

        _link_run_into_sweep(sweep_dir, label, run_dir)

        try:
            artifacts = {
                "sweep_dir": str(sweep_dir),
                "labels_dir": str(run_dir / "labels"),
                "visualizations_dir": str(run_dir / "visualizations"),
            }
            should_store = (
                results is not None
                or not use_cache
                or not cached_samples
                or not meta_path.exists()
            )
            if should_store and combined_results is not None:
                store_run(
                    results=combined_results,
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
