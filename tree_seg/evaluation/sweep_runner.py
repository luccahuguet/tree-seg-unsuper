"""Multiplicative sweep runner for parameter exploration."""

from __future__ import annotations

import datetime
import itertools
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import (
    run_benchmark,
    BenchmarkResults,
    BenchmarkSample,
)
from tree_seg.evaluation.formatters import (
    format_comparison_table,
    save_comparison_summary,
)
from tree_seg.evaluation.runner import load_dataset, _hash_and_run_dir, _link_run_into_sweep
from tree_seg.metadata.store import store_run
from tree_seg.metadata.load import lookup


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

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    smartk_suffix = "_smartk" if smart_k else ""
    sweep_dir = (
        Path("data/outputs/results") / f"sweep_{sweep_name}{smartk_suffix}_{timestamp}"
    )
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
    dataset, dtype_resolved = load_dataset(dataset_path, dataset_type)

    for i, config_dict in enumerate(configs_to_test):
        label = config_dict.pop("label")
        config = Config(**config_dict)

        console.print(
            f"\n[bold][{i + 1}/{len(configs_to_test)}] Running: {label}[/bold]"
        )
        console.print("-" * 60)

        hash_id, run_dir = _hash_and_run_dir(
            config=config, dataset_path=dataset_path, smart_k=smart_k, grid_label=label
        )

        # Check cache
        if use_cache:
            meta_path = Path("results") / "by-hash" / hash_id / "meta.json"
            if meta_path.exists():
                meta = lookup(hash_id)
                console.print(
                    f"[green]♻️  Cache hit for {label} ({hash_id}); skipping.[/green]"
                )
                if meta:
                    metrics = meta.get("metrics", {})
                    timing = meta.get("timing", {})
                    samples_meta = meta.get("samples", {})
                    tags = meta.get("tags", {})
                    auto_tags = tags.get("auto", [])
                    user_tags = tags.get("user", [])

                    console.print(
                        f"[dim]   📊 mIoU={metrics.get('mean_miou', 0):.4f}, "
                        f"PA={metrics.get('mean_pixel_accuracy', 0):.4f}, "
                        f"samples={samples_meta.get('num_samples', 0)}, "
                        f"time={timing.get('mean_runtime_s', 0):.1f}s[/dim]"
                    )
                    if auto_tags or user_tags:
                        all_tags = ", ".join(auto_tags + user_tags)
                        console.print(f"[dim]   🏷️  Tags: {all_tags}[/dim]")

                    per_sample_stats = samples_meta.get("per_sample_stats", [])
                    benchmark_samples = []
                    for sample in per_sample_stats:
                        benchmark_samples.append(
                            BenchmarkSample(
                                image_id=sample.get("image_id", ""),
                                miou=sample.get("miou", 0.0),
                                pixel_accuracy=sample.get("pixel_accuracy", 0.0),
                                per_class_iou=sample.get("per_class_iou", {}),
                                num_clusters=sample.get("num_clusters", 0),
                                runtime_seconds=sample.get("runtime_seconds", 0.0),
                                image_shape=tuple(sample.get("image_shape", [0, 0, 0])),
                            )
                        )

                    cached_results = BenchmarkResults(
                        dataset_name=meta.get("dataset", ""),
                        method_name=f"{config.clustering_method}+{config.refine or 'none'}",
                        config=config,
                        samples=benchmark_samples,
                        mean_miou=metrics.get("mean_miou", 0.0),
                        mean_pixel_accuracy=metrics.get("mean_pixel_accuracy", 0.0),
                        mean_runtime=timing.get("mean_runtime_s", 0.0),
                        total_samples=samples_meta.get("num_samples", 0),
                    )

                    all_results.append(
                        {
                            "label": label,
                            "config": config_dict,
                            "results": cached_results,
                        }
                    )
                _link_run_into_sweep(sweep_dir, label, run_dir)
                continue

        # Model reuse optimization
        model_key = (config.model_display_name, config.stride, config.image_size)
        if model_key in model_cache:
            console.print(
                f"[dim]♻️  Reusing cached model: {config.model_display_name} (stride={config.stride})[/dim]"
            )

        # Run benchmark
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

        # Store metadata
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
                user_tags=[sweep_name, label],
                grid_label=label,
                artifacts=artifacts,
            )
        except Exception:
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
