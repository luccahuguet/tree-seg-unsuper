"""Multiplicative sweep runner for parameter exploration."""

from __future__ import annotations

import gc
import itertools
import shutil
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import run_benchmark
from tree_seg.evaluation.formatters import (
    format_comparison_table,
    save_comparison_summary,
)
from tree_seg.evaluation.runner import (
    load_dataset,
    _hash_and_run_dir,
    _link_run_into_sweep,
    _apply_spectral_guard,
    _benchmark_results_from_samples,
    _combine_samples,
    _load_cached_samples,
    _target_image_ids,
)
from tree_seg.metadata.store import store_run


def generate_sweep_configs(
    base_params: dict,
    sweep_params: dict[str, list],
) -> list[dict]:
    """
    Generate all combinations (multiplicative/factorial sweep).

    Args:
        base_params: Fixed parameters for all configs
        sweep_params: Parameters to sweep (each value is a list)

    Returns:
        List of config dicts with labels

    Example:
        >>> base = {"stride": 4, "model_name": "base"}
        >>> sweep = {"clustering_method": ["kmeans", "gmm"], "refine": ["slic", None]}
        >>> generate_sweep_configs(base, sweep)
        [
            {"stride": 4, "model_name": "base", "clustering_method": "kmeans", "refine": "slic", "label": "kmeans_slic"},
            {"stride": 4, "model_name": "base", "clustering_method": "kmeans", "refine": None, "label": "kmeans_none"},
            {"stride": 4, "model_name": "base", "clustering_method": "gmm", "refine": "slic", "label": "gmm_slic"},
            {"stride": 4, "model_name": "base", "clustering_method": "gmm", "refine": None, "label": "gmm_none"},
        ]
    """
    if not sweep_params:
        return [{"label": "baseline", **base_params}]

    # Generate all combinations
    keys = list(sweep_params.keys())
    values_lists = [sweep_params[k] for k in keys]
    combinations = list(itertools.product(*values_lists))

    configs = []
    for combo in combinations:
        config = base_params.copy()
        label_parts = []

        for key, value in zip(keys, combo):
            config[key] = value

            # Build label from key-value pairs
            if key == "clustering_method":
                label_parts.append(str(value) if value else "none")
            elif key == "refine":
                label_parts.append(str(value) if value else "none")
            elif key == "model_name":
                label_parts.append(str(value))
            elif key == "use_tiling":
                label_parts.append("tile" if value else "notile")
            elif key == "stride":
                label_parts.append(f"s{value}")
            elif key == "elbow_threshold":
                label_parts.append(f"e{value}".replace(".", "-"))
            else:
                # Generic: use key_value
                label_parts.append(f"{key[:3]}{value}")

        config["label"] = "_".join(label_parts)
        configs.append(config)

    return configs


def run_multiplicative_sweep(
    *,
    dataset_path: Path,
    dataset_type: Optional[str],
    base_config_params: dict,
    sweep_params: dict[str, list],
    sweep_name: str,
    num_samples: Optional[int],
    filter_ids: Optional[list[str]],
    save_viz: bool,
    save_labels: bool,
    quiet: bool,
    smart_k: bool,
    console: Console,
    use_cache: bool = True,
) -> tuple[list[dict], Path]:
    """
    Run multiplicative sweep across all parameter combinations.

    Args:
        dataset_path: Path to dataset directory
        dataset_type: Dataset type (fortress, isprs, generic)
        base_config_params: Fixed parameters
        sweep_params: Parameters to sweep (dict of param -> list of values)
        sweep_name: Name for this sweep
        num_samples: Number of samples to evaluate
        save_viz: Save visualizations
        save_labels: Save predicted labels
        quiet: Suppress progress
        smart_k: Use smart K mode
        console: Rich console for output
        use_cache: Reuse cached results

    Returns:
        (all_results, sweep_dir)
    """
    # Generate all config combinations
    configs_to_test = generate_sweep_configs(base_config_params, sweep_params)

    sweep_dir = Path("data/outputs/results") / f"sweep_{sweep_name}"
    if sweep_dir.exists():
        shutil.rmtree(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[bold cyan]🔄 Running multiplicative sweep: {sweep_name}[/bold cyan]"
    )
    console.print(f"[dim]Total configurations: {len(configs_to_test)}[/dim]")
    console.print(f"[green]📁 Sweep directory: {sweep_dir}[/green]\n")

    # Show preview of configs
    preview_table = Table(
        title="Sweep Preview", show_header=True, header_style="bold cyan"
    )
    preview_table.add_column("#", style="dim")
    preview_table.add_column("Label", style="cyan")
    preview_table.add_column("Config", style="dim")

    for i, cfg in enumerate(configs_to_test, 1):
        label = cfg.get("label", f"config_{i}")
        # Show only swept params
        swept_items = {
            k: v for k, v in cfg.items() if k in sweep_params or k == "label"
        }
        preview_table.add_row(
            str(i),
            label,
            ", ".join(f"{k}={v}" for k, v in swept_items.items() if k != "label"),
        )

    console.print(preview_table)
    console.print()

    all_results = []
    model_cache = {}
    dataset, dtype_resolved = load_dataset(
        dataset_path, dataset_type, filter_ids=filter_ids
    )
    base_samples = list(dataset.samples)

    for i, config_dict in enumerate(configs_to_test):
        dataset.samples = list(base_samples)
        label = config_dict.pop("label")
        config = Config(**config_dict)
        config = _apply_spectral_guard(config, console=console)
        config_dict["image_size"] = config.image_size
        config_dict["stride"] = config.stride
        config_dict["use_tiling"] = config.use_tiling
        config_dict["tile_threshold"] = config.tile_threshold

        console.print()
        console.print(
            Panel(
                f"[cyan]{label}[/cyan]",
                title=f"[bold white]Config {i + 1}/{len(configs_to_test)}[/bold white]",
                border_style="cyan",
                padding=(0, 1),
            )
        )

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
                    f"[dim]♻️  Cache hit for {len(target_ids) - len(missing_ids)}/{len(target_ids)} sample(s); running {len(missing_ids)} new sample(s).[/dim]"
                )
            elif not missing_ids and cached_samples:
                console.print(
                    f"[dim]♻️  Cache hit for all {len(target_ids)} sample(s); skipping compute.[/dim]"
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

        # Model reuse optimization
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
                suppress_logs=True,  # Suppress repetitive init messages during sweeps
            )

        if use_cache and cached_samples:
            combined_samples = _combine_samples(
                target_ids, cached_samples, results.samples if results else []
            )
            combined_results = _benchmark_results_from_samples(
                config=config, dataset_name=dataset_path.name, samples=combined_samples
            )
        else:
            combined_results = results  # type: ignore[assignment]

        if combined_results is not None:
            all_results.append(
                {"label": label, "config": config_dict, "results": combined_results}
            )

        _link_run_into_sweep(sweep_dir, label, run_dir)

        # Show cached results summary with same format as fresh runs
        if not results and cached_samples and combined_results is not None:
            results_table = Table(
                show_header=False,
                box=None,
                padding=(0, 2),
                title="[bold cyan]BENCHMARK RESULTS[/bold cyan]",
                title_style="bold cyan",
            )
            results_table.add_column("Metric", style="cyan", justify="right")
            results_table.add_column("Value", style="bold white")

            results_table.add_row("Mean mIoU:", f"{combined_results.mean_miou:.3f}")
            results_table.add_row(
                "Mean Pixel Accuracy:",
                f"{combined_results.mean_pixel_accuracy:.3f}",
            )
            results_table.add_row(
                "Mean Runtime:", f"{combined_results.mean_runtime:.2f}s"
            )
            results_table.add_row("Total Samples:", str(combined_results.total_samples))

            console.print()
            console.print(results_table)
            console.print()

        # Store metadata
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
                    user_tags=[sweep_name, label],
                    grid_label=label,
                    artifacts=artifacts,
                )
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to store metadata: {e}[/yellow]")
            if not quiet:
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")

        # Free memory after each config
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # Summary
    if all_results:
        console.print("\n" + format_comparison_table(all_results) + "\n")

        comparison_path = sweep_dir / f"sweep_summary_{sweep_name}.json"
        save_comparison_summary(all_results, comparison_path)
        console.print(f"[green]📊 Sweep summary saved to: {comparison_path}[/green]")
        console.print(f"[green]📁 All results in: {sweep_dir}[/green]\n")
    else:
        console.print(
            "\n[yellow]⚠️  All configs were cached; no new results.[/yellow]\n"
        )

    return all_results, sweep_dir
