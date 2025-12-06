"""Evaluate command for benchmarking segmentation methods."""

import datetime
import json
from pathlib import Path
from typing import Literal, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import run_benchmark
from tree_seg.evaluation.datasets import FortressDataset
from tree_seg.evaluation.formatters import config_to_dict, format_comparison_table, save_comparison_summary
from tree_seg.evaluation.grids import get_grid

# Load environment variables
load_dotenv()

console = Console()

# Removed METHOD_TO_VERSION - using explicit clustering/refine flags instead


def _detect_dataset_type(dataset_path: Path) -> str:
    """Auto-detect dataset type from directory structure."""
    # Check for FORTRESS structure (images/ and labels/ directories)
    if (dataset_path / "images").exists() and (dataset_path / "labels").exists():
        return "fortress"
    # Check for ISPRS Potsdam structure
    elif (dataset_path / "2_Ortho_RGB").exists():
        return "isprs"
    else:
        # Generic structure - assume images and labels in root
        return "generic"


def _create_config(
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
) -> Config:
    """Create Config object from parameters."""
    # Determine version based on configuration
    if use_supervised:
        version = "v4"
    elif refine == "soft-em" or refine == "soft-em+slic":
        version = "v2"
    elif apply_vegetation_filter:
        version = "v3"
    else:
        version = "v1.5"

    # Handle V4 special case (supervised)
    if use_supervised:
        if model != "mega":
            console.print("[yellow]⚠️  Mask2Former head only supports the ViT-7B backbone; overriding model to 'mega'.[/yellow]")
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
        )

    # Map clustering algorithm
    clustering_method = clustering  # kmeans, gmm, spectral, hdbscan

    # Map refinement methods
    refine_method = None
    use_soft_refine = False

    if refine and refine != "none":
        if refine == "soft-em":
            use_soft_refine = True
            refine_method = None  # Soft EM is separate from image-space refinement
        elif refine == "slic":
            refine_method = "slic"
        elif refine == "bilateral":
            refine_method = "bilateral"
        elif refine == "soft-em+slic":
            use_soft_refine = True
            refine_method = "slic"

    # Parse pyramid scales
    scales = tuple(float(s.strip()) for s in pyramid_scales.split(",")) if pyramid_scales else (0.5, 1.0, 2.0)

    return Config(
        version=version,
        clustering_method=clustering_method,
        refine=refine_method,
        model_name=model,
        stride=stride,
        elbow_threshold=elbow_threshold,
        n_clusters=fixed_k if fixed_k else 6,
        auto_k=(fixed_k is None),
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
        metrics=True,
        verbose=not quiet,
    )


def _run_single_benchmark(
    dataset_path: Path,
    dataset_type: str,
    config: Config,
    output_dir: Optional[Path],
    num_samples: Optional[int],
    save_viz: bool,
    quiet: bool,
    smart_k: bool,
):
    """Run a single benchmark configuration."""
    # Determine output directory
    if output_dir:
        out_dir = output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        refine_str = config.refine if config.refine else config.clustering_method
        method_str = f"{config.version}_{refine_str}"
        if smart_k:
            method_str += "_smartk"
        model_str = config.model_display_name.lower().replace(" ", "_")
        out_dir = Path("data/output/results") / f"{dataset_type}_{method_str}_{model_str}_{timestamp}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if dataset_type == "fortress":
        dataset = FortressDataset(dataset_path)
    else:
        # Use generic dataset loader (would need to be implemented)
        console.print(f"[yellow]⚠️  Dataset type '{dataset_type}' not fully implemented yet[/yellow]")
        raise typer.Exit(code=1)

    # Run benchmark
    console.print(f"\n[bold cyan]🚀 Running benchmark on {dataset_type.upper()} dataset[/bold cyan]")
    console.print(f"[dim]Config: {config.version} | {config.model_display_name} | stride={config.stride}[/dim]\n")

    results = run_benchmark(
        config=config,
        dataset=dataset,
        output_dir=out_dir,
        num_samples=num_samples,
        save_visualizations=save_viz,
        verbose=not quiet,
        use_smart_k=smart_k,
    )

    # Save results
    results_dict = {
        "dataset": dataset_type.upper(),
        "method": results.method_name,
        "config": config_to_dict(config),
        "metrics": {
            "mean_miou": float(results.mean_miou),
            "mean_pixel_accuracy": float(results.mean_pixel_accuracy),
            "mean_runtime": float(results.mean_runtime),
        },
        "samples": [
            {
                "image_id": s.image_id,
                "miou": float(s.miou),
                "pixel_accuracy": float(s.pixel_accuracy),
                "num_clusters": int(s.num_clusters),
                "runtime_seconds": float(s.runtime_seconds),
            }
            for s in results.samples
        ],
        "total_samples": results.total_samples,
    }

    if dataset_type == "fortress":
        results_dict["num_classes"] = FortressDataset.NUM_CLASSES

    output_path = out_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Print results summary
    table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Dataset", dataset_type.upper())
    table.add_row("Method", results.method_name)
    table.add_row("Mean mIoU", f"{results.mean_miou:.4f}")
    table.add_row("Mean Pixel Accuracy", f"{results.mean_pixel_accuracy:.4f}")
    table.add_row("Mean Runtime", f"{results.mean_runtime:.2f}s")
    table.add_row("Total Samples", str(results.total_samples))

    console.print()
    console.print(table)
    console.print(f"\n[green]✅ Results saved to: {output_path}[/green]")

    return results


def _run_comparison_benchmark(
    dataset_path: Path,
    dataset_type: str,
    grid_name: str,
    base_config_params: dict,
    num_samples: Optional[int],
    save_viz: bool,
    quiet: bool,
    smart_k: bool,
):
    """Run comparison across multiple configurations."""
    grid = get_grid(grid_name)
    configs_to_test = grid["configs"]

    console.print("\n[bold cyan]🔄 Running comparison benchmark[/bold cyan]")
    console.print(f"Grid: [bold]{grid['name']}[/bold]")
    console.print(f"Description: {grid['description']}")
    console.print(f"Configurations: {len(configs_to_test)}\n")

    # Create sweep directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    smartk_suffix = "_smartk" if smart_k else ""
    sweep_dir = Path("data/output/results") / f"sweep_{grid_name}{smartk_suffix}_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[green]📁 Sweep directory: {sweep_dir}[/green]\n")

    # Load dataset once
    if dataset_type == "fortress":
        dataset = FortressDataset(dataset_path)
    else:
        console.print(f"[yellow]⚠️  Dataset type '{dataset_type}' not fully implemented yet[/yellow]")
        raise typer.Exit(code=1)

    all_results = []
    model_cache = {}

    console.print("[bold]" + "=" * 60 + "[/bold]")
    console.print("[bold cyan]RUNNING COMPARISON BENCHMARK[/bold cyan]")
    console.print("[bold]" + "=" * 60 + "[/bold]\n")

    for i, config_override in enumerate(configs_to_test):
        config_dict = base_config_params.copy()
        config_dict.update({k: v for k, v in config_override.items() if k != "label"})

        config = Config(**config_dict)
        label = config_override["label"]

        console.print(f"\n[bold][{i + 1}/{len(configs_to_test)}] Testing: {label}[/bold]")
        console.print("-" * 40)

        # Check model cache
        model_key = (config.model_display_name, config.stride, config.image_size)
        if model_key in model_cache:
            console.print(f"[dim]♻️  Reusing cached model: {config.model_display_name} (stride={config.stride})[/dim]")

        results = run_benchmark(
            config=config,
            dataset=dataset,
            output_dir=sweep_dir,
            num_samples=num_samples,
            save_visualizations=save_viz,
            verbose=not quiet,
            use_smart_k=smart_k,
            model_cache=model_cache,
            config_label=label,
        )

        all_results.append({"label": label, "config": config_dict, "results": results})

    # Print comparison table
    console.print("\n" + format_comparison_table(all_results) + "\n")

    # Save summary
    comparison_path = sweep_dir / f"sweep_summary_{grid_name}.json"
    save_comparison_summary(all_results, comparison_path)
    console.print(f"[green]📊 Sweep summary saved to: {comparison_path}[/green]")
    console.print(f"[green]📁 All results in: {sweep_dir}[/green]\n")


def evaluate_command(
    dataset: Path = typer.Argument(
        ...,
        help="Path to dataset directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    clustering: Literal["kmeans", "gmm", "spectral", "hdbscan"] = typer.Option(
        "kmeans",
        "--clustering",
        "-c",
        help="Clustering algorithm: kmeans (default), gmm, spectral, hdbscan",
    ),
    refine: Optional[Literal["none", "slic", "soft-em", "bilateral", "soft-em+slic"]] = typer.Option(
        "slic",
        "--refine",
        "-r",
        help="Refinement method: none, slic (default), soft-em (V2), bilateral, soft-em+slic (combine both)",
    ),
    supervised: bool = typer.Option(
        False,
        "--supervised",
        help="Use supervised Mask2Former model (V4) instead of unsupervised clustering",
    ),
    model: Literal["small", "base", "large", "mega"] = typer.Option(
        "base",
        "--model",
        "-m",
        help="DINOv3 model size",
    ),
    dataset_type: Optional[Literal["fortress", "isprs", "generic"]] = typer.Option(
        None,
        "--dataset-type",
        "-t",
        help="Dataset type (auto-detected if not specified)",
    ),
    stride: int = typer.Option(
        4,
        "--stride",
        help="Feature extraction stride",
    ),
    image_size: int = typer.Option(
        1024,
        "--image-size",
        "-s",
        help="Image resize dimension",
    ),
    elbow_threshold: float = typer.Option(
        5.0,
        "--elbow-threshold",
        "-e",
        help="Elbow method threshold for auto K selection",
    ),
    fixed_k: Optional[int] = typer.Option(
        None,
        "--fixed-k",
        "-k",
        help="Fixed number of clusters (overrides auto K)",
    ),
    num_samples: Optional[int] = typer.Option(
        None,
        "--num-samples",
        "-n",
        help="Number of samples to evaluate (default: all)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for results (auto-generated if not specified)",
    ),
    save_viz: bool = typer.Option(
        False,
        "--save-viz",
        help="Save visualization images",
    ),
    apply_vegetation_filter: bool = typer.Option(
        False,
        "--vegetation-filter",
        help="Apply vegetation filtering for species-level segmentation (V3 task)",
    ),
    exg_threshold: float = typer.Option(
        0.1,
        "--exg-threshold",
        help="ExG threshold for vegetation filtering",
    ),
    no_tiling: bool = typer.Option(
        False,
        "--no-tiling",
        help="Disable tile-based processing for high-res images",
    ),
    viz_two_panel: bool = typer.Option(
        False,
        "--viz-two-panel",
        help="Use two-panel visualization layout",
    ),
    viz_two_panel_opaque: bool = typer.Option(
        False,
        "--viz-two-panel-opaque",
        help="Use opaque overlay in two-panel visualization",
    ),
    use_pyramid: bool = typer.Option(
        False,
        "--use-pyramid",
        help="Use pyramid feature aggregation",
    ),
    pyramid_scales: str = typer.Option(
        "0.5,1.0,2.0",
        "--pyramid-scales",
        help="Pyramid scales (comma-separated)",
    ),
    pyramid_aggregation: Literal["concat", "mean"] = typer.Option(
        "concat",
        "--pyramid-aggregation",
        help="Pyramid aggregation method",
    ),
    smart_k: bool = typer.Option(
        False,
        "--smart-k",
        help="Use smart K mode (match GT class count - debug only)",
    ),
    compare_configs: bool = typer.Option(
        False,
        "--compare-configs",
        help="Run comparison across multiple configurations",
    ),
    smart_grid: bool = typer.Option(
        False,
        "--smart-grid",
        help="Use smart grid search (8 configs)",
    ),
    grid: Optional[Literal["ofat", "smart", "full", "tiling", "tiling_refine", "clustering", "slic_params"]] = typer.Option(
        None,
        "--grid",
        "-g",
        help="Grid to use for comparison mode",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress output",
    ),
):
    """
    Evaluate segmentation methods on labeled datasets.

    Computes mIoU, pixel accuracy using Hungarian matching.
    Supports FORTRESS, ISPRS Potsdam, and generic datasets.

    Examples:

        # V1.5 baseline: K-means + SLIC
        tree-seg eval data/fortress

        # V2: K-means + soft EM refinement
        tree-seg eval data/fortress --refine soft-em

        # V2 + SLIC: K-means + soft EM + SLIC (combine both refinements)
        tree-seg eval data/fortress --refine soft-em+slic

        # V3 task: Species segmentation with vegetation filter
        tree-seg eval data/fortress --vegetation-filter

        # V2 + V3: Soft EM + vegetation filter
        tree-seg eval data/fortress --refine soft-em --vegetation-filter

        # Experiment: GMM clustering + soft EM
        tree-seg eval data/fortress --clustering gmm --refine soft-em

        # No refinement: just clustering
        tree-seg eval data/fortress --refine none

        # V4: Supervised Mask2Former
        tree-seg eval data/fortress --supervised

        # Run comparison across multiple configs
        tree-seg eval data/fortress --compare-configs --grid tiling
    """
    # Auto-detect dataset type if not specified
    if not dataset_type:
        dataset_type = _detect_dataset_type(dataset)
        console.print(f"[dim]Auto-detected dataset type: {dataset_type}[/dim]")

    # Create config
    config = _create_config(
        clustering=clustering,
        refine=refine,
        model=model,
        stride=stride,
        image_size=image_size,
        elbow_threshold=elbow_threshold,
        fixed_k=fixed_k,
        apply_vegetation_filter=apply_vegetation_filter,
        exg_threshold=exg_threshold,
        no_tiling=no_tiling,
        viz_two_panel=viz_two_panel,
        viz_two_panel_opaque=viz_two_panel_opaque,
        use_pyramid=use_pyramid,
        pyramid_scales=pyramid_scales,
        pyramid_aggregation=pyramid_aggregation,
        use_supervised=supervised,
        quiet=quiet,
    )

    # Run comparison or single benchmark
    if compare_configs:
        # Determine grid
        if grid:
            grid_name = grid
        elif smart_grid:
            grid_name = "smart"
        elif dataset_type == "fortress":
            grid_name = "tiling"  # Default for FORTRESS
        else:
            grid_name = "ofat"  # Default for others

        # Create base config params
        base_config_params = {
            "version": config.version,
            "refine": config.refine,
            "clustering_method": config.clustering_method,
            "model_name": model,
            "stride": stride,
            "elbow_threshold": elbow_threshold,
            "auto_k": (fixed_k is None),
            "n_clusters": fixed_k if fixed_k else 6,
            "apply_vegetation_filter": apply_vegetation_filter,
            "exg_threshold": exg_threshold,
            "use_tiling": not no_tiling,
            "viz_two_panel": viz_two_panel,
            "viz_two_panel_opaque": viz_two_panel_opaque,
        }

        _run_comparison_benchmark(
            dataset_path=dataset,
            dataset_type=dataset_type,
            grid_name=grid_name,
            base_config_params=base_config_params,
            num_samples=num_samples,
            save_viz=save_viz,
            quiet=quiet,
            smart_k=smart_k,
        )
    else:
        _run_single_benchmark(
            dataset_path=dataset,
            dataset_type=dataset_type,
            config=config,
            output_dir=output_dir,
            num_samples=num_samples,
            save_viz=save_viz,
            quiet=quiet,
            smart_k=smart_k,
        )
