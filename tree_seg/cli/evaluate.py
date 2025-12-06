"""Evaluate command for benchmarking segmentation methods."""

import datetime
from pathlib import Path
from typing import Literal, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from tree_seg.core.types import Config
from tree_seg.evaluation.datasets import FortressDataset
from tree_seg.evaluation.formatters import format_comparison_table, save_comparison_summary
from tree_seg.evaluation.grids import get_grid
from tree_seg.evaluation.runner import (
    create_config,
    resolve_output_dir,
    run_single_benchmark,
    run_sweep,
    try_cached_results,
)
from tree_seg.metadata.store import store_run

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
    use_attention_features: bool,
) -> Config:
    """Create Config object from parameters."""
    return create_config(
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
        use_supervised=use_supervised,
        quiet=quiet,
        use_attention_features=use_attention_features,
    )


def _run_single_benchmark(
    dataset_path: Path,
    dataset_type: str,
    config: Config,
    output_dir: Optional[Path],
    num_samples: Optional[int],
    save_viz: bool,
    save_labels: bool,
    quiet: bool,
    smart_k: bool,
    use_cache: bool,
):
    """Run a single benchmark configuration."""
    out_dir = resolve_output_dir(
        config=config,
        dataset_type=dataset_type,
        smart_k=smart_k,
        output_dir=output_dir,
    )

    if use_cache and try_cached_results(
        config=config,
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        smart_k=smart_k,
        console=console,
    ):
        return None

    console.print(f"\n[bold cyan]🚀 Running benchmark on {dataset_type.upper()} dataset[/bold cyan]")
    console.print(f"[dim]Config: {config.version} | {config.model_display_name} | stride={config.stride}[/dim]\n")

    results = run_single_benchmark(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        config=config,
        output_dir=out_dir,
        num_samples=num_samples,
        save_viz=save_viz,
        save_labels=save_labels,
        quiet=quiet,
        smart_k=smart_k,
    )

    if results is None:
        return None

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
    console.print(f"\n[green]✅ Results saved to: {out_dir / 'results.json'}[/green]")

    return results


def _run_comparison_benchmark(
    dataset_path: Path,
    dataset_type: str,
    grid_name: str,
    base_config_params: dict,
    num_samples: Optional[int],
    save_viz: bool,
    save_labels: bool,
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

    all_results, sweep_dir = run_sweep(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        grid_name=grid_name,
        configs_to_test=configs_to_test,
        base_config_params=base_config_params,
        num_samples=num_samples,
        save_viz=save_viz,
        save_labels=save_labels,
        quiet=quiet,
        smart_k=smart_k,
        console=console,
    )

    console.print("\n" + format_comparison_table(all_results) + "\n")

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
    clustering: Literal["kmeans", "gmm", "spectral", "hdbscan", "spherical", "dpmeans", "potts"] = typer.Option(
        "kmeans",
        "--clustering",
        "-c",
        help="Clustering algorithm: kmeans (default), gmm, spectral, hdbscan, spherical (cosine k-means), dpmeans (auto-K), potts (regularized k-means)",
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
    use_cache: bool = typer.Option(
        True,
        "--use-cache/--no-cache",
        help="Reuse cached results if the same config/dataset hash exists",
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
    grid: Optional[Literal["ofat", "smart", "full", "tiling", "tiling_refine", "clustering", "slic_params", "tile_overlap"]] = typer.Option(
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
    save_labels: bool = typer.Option(
        True,
        "--save-labels/--no-save-labels",
        help="Save predicted labels (NPZ) for metadata/viz regeneration",
    ),
    use_attention_features: bool = typer.Option(
        True,
        "--use-attn/--no-use-attn",
        help="Include attention tokens in features (disable for legacy v1 behavior)",
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
        use_attention_features=use_attention_features,
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
            "model_name": config.model_name,
            "stride": stride,
            "image_size": config.image_size,
            "elbow_threshold": elbow_threshold,
            "auto_k": (fixed_k is None),
            "n_clusters": fixed_k if fixed_k else 6,
            "apply_vegetation_filter": apply_vegetation_filter,
            "exg_threshold": exg_threshold,
            "use_tiling": not no_tiling,
            "viz_two_panel": viz_two_panel,
            "viz_two_panel_opaque": viz_two_panel_opaque,
            "use_pyramid": config.use_pyramid,
            "pyramid_scales": config.pyramid_scales,
            "pyramid_aggregation": config.pyramid_aggregation,
            "use_soft_refine": config.use_soft_refine,
            "soft_refine_temperature": config.soft_refine_temperature,
            "soft_refine_iterations": config.soft_refine_iterations,
            "soft_refine_spatial_alpha": config.soft_refine_spatial_alpha,
            "use_attention_features": config.use_attention_features,
            "metrics": True,
        }

        _run_comparison_benchmark(
            dataset_path=dataset,
            dataset_type=dataset_type,
            grid_name=grid_name,
            base_config_params=base_config_params,
            num_samples=num_samples,
            save_viz=save_viz,
            save_labels=save_labels,
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
            save_labels=save_labels,
            quiet=quiet,
            smart_k=smart_k,
            use_cache=use_cache,
        )
