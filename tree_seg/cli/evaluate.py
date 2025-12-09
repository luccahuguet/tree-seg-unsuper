"""Evaluate command for benchmarking segmentation methods."""

from pathlib import Path
from typing import Literal, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from tree_seg.core.types import Config
from tree_seg.evaluation.runner import (
    create_config,
    detect_dataset_type,
    resolve_output_dir,
    run_single_benchmark,
)

# Load environment variables
load_dotenv()

console = Console()

# Removed METHOD_TO_VERSION - using explicit clustering/refine flags instead


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
        tiling=tiling,
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
    filter_ids: Optional[list[str]],
    save_viz: bool,
    save_labels: bool,
    quiet: bool,
    smart_k: bool,
    use_cache: bool,
    force: bool,
):
    """Run a single benchmark configuration."""
    out_dir = resolve_output_dir(
        config=config,
        dataset_type=dataset_type,
        smart_k=smart_k,
        output_dir=output_dir,
    )

    console.print(
        f"\n[bold cyan]🚀 Running benchmark on {dataset_type.upper()} dataset[/bold cyan]"
    )
    console.print(
        f"[dim]Config: {config.clustering_method}/{config.refine} | {config.model_display_name} | stride={config.stride}[/dim]\n"
    )

    effective_use_cache = use_cache and not force

    results = run_single_benchmark(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        config=config,
        output_dir=out_dir,
        num_samples=num_samples,
        filter_ids=filter_ids,
        save_viz=save_viz,
        save_labels=save_labels,
        quiet=quiet,
        smart_k=smart_k,
        use_cache=effective_use_cache,
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


def evaluate_command(
    dataset: Path = typer.Argument(
        ...,
        help="Path to dataset directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    clustering: Literal[
        "kmeans", "gmm", "spectral", "hdbscan", "spherical", "dpmeans", "potts"
    ] = typer.Option(
        "kmeans",
        "--clustering",
        "-c",
        help="Clustering algorithm: kmeans (default), gmm, spectral, hdbscan, spherical (cosine k-means), dpmeans (auto-K), potts (regularized k-means)",
    ),
    refine: Optional[
        Literal["none", "slic", "soft-em", "bilateral", "soft-em+slic"]
    ] = typer.Option(
        "slic",
        "--refine",
        "-r",
        help="Refinement method: none, slic (default), soft-em (V2), bilateral, soft-em+slic (combine both)",
    ),
    supervised: bool = typer.Option(
        False,
        "--supervised",
        help="Use supervised baseline (sklearn or linear head on DINOv3 features)",
    ),
    supervised_head: Literal["linear", "sklearn", "mlp"] = typer.Option(
        "linear",
        "--supervised-head",
        help="Supervised head to use: linear (PyTorch), sklearn logistic, or sklearn MLP",
    ),
    supervised_epochs: int = typer.Option(
        100,
        "--supervised-epochs",
        help="Epochs for the linear supervised head (only used when --supervised-head linear)",
    ),
    supervised_max_patches: int = typer.Option(
        1_000_000,
        "--supervised-max-patches",
        help="Max patches for training the supervised head (linear)",
    ),
    supervised_val_split: float = typer.Option(
        0.1,
        "--supervised-val-split",
        help="Validation split for early stopping (linear head). 0 disables.",
    ),
    supervised_patience: int = typer.Option(
        5,
        "--supervised-patience",
        help="Patience for early stopping on val loss (linear head). 0 disables.",
    ),
    supervised_lr: float = typer.Option(
        1e-3,
        "--supervised-lr",
        help="Learning rate for the linear supervised head",
    ),
    supervised_ignore_index: Optional[int] = typer.Option(
        None,
        "--supervised-ignore-index",
        help="Ignore index for supervised training/eval (set to 255 if your masks use 255 as unlabeled; default=None keeps all labels)",
    ),
    supervised_hidden_dim: int = typer.Option(
        1024,
        "--supervised-hidden-dim",
        help="Hidden dimension for the linear head MLP",
    ),
    supervised_dropout: float = typer.Option(
        0.1,
        "--supervised-dropout",
        help="Dropout for the linear head MLP",
    ),
    supervised_use_xy: bool = typer.Option(
        False,
        "--supervised-use-xy",
        help="Append normalized XY coords to patch features (torch linear head)",
    ),
    supervised_mlp_use_xy: bool = typer.Option(
        False,
        "--supervised-mlp-use-xy",
        help="Append normalized XY coords to patch features for sklearn MLP head",
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
    supervised_train_ratio: float = typer.Option(
        1.0,
        "--supervised-train-ratio",
        help="Fraction of samples to use for supervised training (rest used for validation/holdout)",
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
    tiling: bool = typer.Option(
        False,
        "--tiling/--no-tiling",
        help="Enable tile-based processing for high-res images (default: off)",
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
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-run even if a cache entry exists for this config/dataset hash",
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
    image_ids: Optional[str] = typer.Option(
        None,
        "--image-ids",
        help="Comma-separated image IDs to evaluate (subset of dataset)",
    ),
):
    """
    Evaluate segmentation methods on labeled datasets.

    Computes mIoU, pixel accuracy using Hungarian matching.
    Supports FORTRESS, ISPRS Potsdam, and generic datasets.

    Examples:

        # V1.5 baseline: K-means + SLIC
        tree-seg eval data/datasets/fortress

        # V2: K-means + soft EM refinement
        tree-seg eval data/datasets/fortress --refine soft-em

        # V2 + SLIC: K-means + soft EM + SLIC (combine both refinements)
        tree-seg eval data/datasets/fortress --refine soft-em+slic

        # V3 task: Species segmentation with vegetation filter
        tree-seg eval data/datasets/fortress --vegetation-filter

        # V2 + V3: Soft EM + vegetation filter
        tree-seg eval data/datasets/fortress --refine soft-em --vegetation-filter

        # Experiment: GMM clustering + soft EM
        tree-seg eval data/datasets/fortress --clustering gmm --refine soft-em

        # No refinement: just clustering
        tree-seg eval data/datasets/fortress --refine none

        # Supervised sklearn baseline
        tree-seg eval data/datasets/fortress --supervised

        # Supervised PyTorch linear head
        tree-seg eval data/datasets/fortress --supervised --supervised-head linear

        # For parameter sweeps, use the 'sweep' command instead
        tree-seg sweep data/datasets/fortress --clustering kmeans gmm --refine slic none
    """
    if not dataset_type:
        dataset_type = detect_dataset_type(dataset)
        console.print(f"[dim]Auto-detected dataset type: {dataset_type}[/dim]")

    # Handle supervised baseline
    if supervised:
        from tree_seg.supervised.sklearn_baseline import (
            evaluate_mlp_baseline,
            evaluate_linear_head,
            evaluate_sklearn_baseline,
        )

        console.print("\n[bold cyan]🎓 Running supervised baseline[/bold cyan]\n")

        if supervised_head == "sklearn":
            results = evaluate_sklearn_baseline(
                dataset_path=dataset,
                model_name=model,
                stride=stride,
                verbose=not quiet,
                num_samples=num_samples,
                ignore_index=supervised_ignore_index,
                train_ratio=supervised_train_ratio,
            )
        elif supervised_head == "mlp":
            results = evaluate_mlp_baseline(
                dataset_path=dataset,
                model_name=model,
                stride=stride,
                verbose=not quiet,
                num_samples=num_samples,
                ignore_index=supervised_ignore_index,
                max_samples=supervised_max_patches,
                max_iter=supervised_epochs,
                lr=supervised_lr,
                hidden_dim=supervised_hidden_dim,
                use_xy=supervised_mlp_use_xy,
                val_split=supervised_val_split,
                patience=supervised_patience,
                train_ratio=supervised_train_ratio,
            )
        else:
            results = evaluate_linear_head(
                dataset_path=dataset,
                model_name=model,
                stride=stride,
                verbose=not quiet,
                num_samples=num_samples,
                epochs=supervised_epochs,
                max_patches=supervised_max_patches,
                val_split=supervised_val_split,
                patience=supervised_patience,
                lr=supervised_lr,
                hidden_dim=supervised_hidden_dim,
                dropout=supervised_dropout,
                ignore_index=supervised_ignore_index,
                use_xy=supervised_use_xy,
                train_ratio=supervised_train_ratio,
            )

        # Print summary
        table = Table(
            title="Supervised Baseline Results",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Dataset", results.dataset_name)
        table.add_row("Method", results.method_name)
        table.add_row("Mean mIoU", f"{results.mean_miou:.4f}")
        table.add_row("Mean Pixel Accuracy", f"{results.mean_pixel_accuracy:.4f}")
        table.add_row("Mean Runtime", f"{results.mean_runtime:.2f}s")
        table.add_row("Total Samples", str(results.total_samples))

        console.print()
        console.print(table)
        console.print("\n[green]✅ Supervised baseline evaluation complete[/green]\n")

        return

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
        tiling=tiling,
        viz_two_panel=viz_two_panel,
        viz_two_panel_opaque=viz_two_panel_opaque,
        use_pyramid=use_pyramid,
        pyramid_scales=pyramid_scales,
        pyramid_aggregation=pyramid_aggregation,
        use_supervised=supervised,
        quiet=quiet,
        use_attention_features=use_attention_features,
    )

    # Run single benchmark
    filter_ids = (
        [s.strip() for s in image_ids.split(",") if s.strip()] if image_ids else None
    )
    _run_single_benchmark(
        dataset_path=dataset,
        dataset_type=dataset_type,
        config=config,
        output_dir=output_dir,
        num_samples=num_samples,
        filter_ids=filter_ids,
        save_viz=save_viz,
        save_labels=save_labels,
        quiet=quiet,
        smart_k=smart_k,
        use_cache=use_cache,
        force=force,
    )
