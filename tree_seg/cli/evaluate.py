"""Evaluate command - thin wrapper around sweep for single-config runs."""

from pathlib import Path
from typing import Literal, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from tree_seg.evaluation.runner import detect_dataset_type
from tree_seg.evaluation.sweep_runner import run_multiplicative_sweep

# Load environment variables
load_dotenv()

console = Console()


def evaluate_command(
    dataset: Path = typer.Argument(
        ...,
        help="Path to dataset directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    clustering: Literal[
        "kmeans", "gmm", "spectral", "spherical", "dpmeans", "potts"
    ] = typer.Option(
        "kmeans",
        "--clustering",
        "-c",
        help="Clustering algorithm",
    ),
    refine: Optional[
        Literal["none", "slic", "soft-em", "bilateral", "soft-em+slic"]
    ] = typer.Option(
        "slic",
        "--refine",
        "-r",
        help="Refinement method",
    ),
    supervised: bool = typer.Option(
        False,
        "--supervised",
        help="Use supervised baseline (sklearn or linear head on DINOv3 features)",
    ),
    supervised_head: Literal["linear", "sklearn", "mlp"] = typer.Option(
        "linear",
        "--supervised-head",
        help="Supervised head to use",
    ),
    supervised_epochs: int = typer.Option(100, "--supervised-epochs"),
    supervised_max_patches: int = typer.Option(1_000_000, "--supervised-max-patches"),
    supervised_val_split: float = typer.Option(0.1, "--supervised-val-split"),
    supervised_patience: int = typer.Option(5, "--supervised-patience"),
    supervised_lr: float = typer.Option(1e-3, "--supervised-lr"),
    supervised_ignore_index: Optional[int] = typer.Option(
        None, "--supervised-ignore-index"
    ),
    supervised_hidden_dim: int = typer.Option(1024, "--supervised-hidden-dim"),
    supervised_dropout: float = typer.Option(0.1, "--supervised-dropout"),
    supervised_use_xy: bool = typer.Option(False, "--supervised-use-xy"),
    supervised_mlp_use_xy: bool = typer.Option(False, "--supervised-mlp-use-xy"),
    supervised_train_ratio: float = typer.Option(1.0, "--supervised-train-ratio"),
    model: Literal["small", "base", "large", "mega"] = typer.Option(
        "base", "--model", "-m"
    ),
    dataset_type: Optional[Literal["fortress", "isprs", "generic"]] = typer.Option(
        None, "--dataset-type", "-t"
    ),
    stride: int = typer.Option(4, "--stride"),
    image_size: int = typer.Option(1024, "--image-size", "-s"),
    elbow_threshold: float = typer.Option(5.0, "--elbow-threshold", "-e"),
    fixed_k: Optional[int] = typer.Option(None, "--fixed-k", "-k"),
    num_samples: Optional[int] = typer.Option(None, "--num-samples", "-n"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    save_viz: bool = typer.Option(False, "--save-viz"),
    apply_vegetation_filter: bool = typer.Option(False, "--vegetation-filter"),
    exg_threshold: float = typer.Option(0.1, "--exg-threshold"),
    tiling: bool = typer.Option(False, "--tiling/--no-tiling"),
    viz_two_panel: bool = typer.Option(False, "--viz-two-panel"),
    viz_two_panel_opaque: bool = typer.Option(False, "--viz-two-panel-opaque"),
    use_pyramid: bool = typer.Option(False, "--use-pyramid"),
    pyramid_scales: str = typer.Option("0.5,1.0,2.0", "--pyramid-scales"),
    pyramid_aggregation: Literal["concat", "mean"] = typer.Option(
        "concat", "--pyramid-aggregation"
    ),
    smart_k: bool = typer.Option(False, "--smart-k"),
    use_cache: bool = typer.Option(True, "--use-cache/--no-cache"),
    force: bool = typer.Option(False, "--force", "-f"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    save_labels: bool = typer.Option(True, "--save-labels/--no-save-labels"),
    use_attention_features: bool = typer.Option(True, "--use-attn/--no-use-attn"),
    image_ids: Optional[str] = typer.Option(None, "--image-ids"),
):
    """
    Evaluate a single configuration (thin wrapper around sweep).

    For parameter sweeps, use 'tree-seg sweep' instead.

    Examples:
        tree-seg eval data/datasets/fortress
        tree-seg eval data/datasets/fortress --refine soft-em
        tree-seg eval data/datasets/fortress --clustering gmm --refine none
        tree-seg eval data/datasets/fortress --supervised
    """
    if not dataset_type:
        dataset_type = detect_dataset_type(dataset)
        console.print(f"[dim]Auto-detected dataset type: {dataset_type}[/dim]")

    # Handle supervised baseline (separate path, not part of sweep)
    if supervised:
        from tree_seg.supervised.sklearn_baseline import (
            evaluate_linear_head,
            evaluate_mlp_baseline,
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

    # Unsupervised: use sweep with single-item lists
    console.print(
        f"\n[bold cyan]🚀 Running benchmark on {dataset_type.upper()} dataset[/bold cyan]"
    )
    console.print(
        f"[dim]Config: {clustering}/{refine or 'none'} | {model} | stride={stride}[/dim]\n"
    )

    # Build single-config sweep parameters
    base_params = {
        "model": model,
        "stride": stride,
        "image_size": image_size,
        "elbow_threshold": elbow_threshold,
        "fixed_k": fixed_k,
        "apply_vegetation_filter": apply_vegetation_filter,
        "exg_threshold": exg_threshold,
        "tiling": tiling,
        "viz_two_panel": viz_two_panel,
        "viz_two_panel_opaque": viz_two_panel_opaque,
        "use_pyramid": use_pyramid,
        "pyramid_scales": pyramid_scales,
        "pyramid_aggregation": pyramid_aggregation,
        "use_attention_features": use_attention_features,
    }

    sweep_params = {
        "clustering": [clustering],
        "refine": [refine] if refine and refine != "none" else [None],
    }

    filter_ids = (
        [s.strip() for s in image_ids.split(",") if s.strip()] if image_ids else None
    )

    # Run as a single-config sweep
    all_results, sweep_dir = run_multiplicative_sweep(
        dataset_path=dataset,
        dataset_type=dataset_type,
        base_config_params=base_params,
        sweep_params=sweep_params,
        sweep_name=f"{clustering}_{refine or 'none'}_eval",
        num_samples=num_samples,
        filter_ids=filter_ids,
        save_viz=save_viz,
        save_labels=save_labels,
        quiet=quiet,
        smart_k=smart_k,
        use_cache=use_cache and not force,
    )

    if not all_results:
        console.print("[yellow]No results generated[/yellow]")
        return

    # Extract the single result
    result_dict = all_results[0]
    results = result_dict["results"]

    # Print summary table
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
    console.print(f"\n[green]✅ Results saved to: {sweep_dir}[/green]")
