"""Segment command for tree segmentation CLI."""

import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from tree_seg.constants import PROFILE_DEFAULTS
from tree_seg.evaluation.segment_runner import run_single_segment

# Load environment variables
load_dotenv()

console = Console()


def segment_command(
    image_path: Path = typer.Argument(
        Path("data/input"),
        help="Path to image or directory to process",
        exists=True,
    ),
    model: str = typer.Argument(
        "base",
        help="Model size: small/base/large/giant/mega or full name",
    ),
    output_dir: Path = typer.Option(
        Path("data/output"),
        "--output-dir",
        "-o",
        help="Output directory for results",
    ),
    image_size: int = typer.Option(
        1024,
        "--image-size",
        "-s",
        help="Preprocess resize (square)",
    ),
    feature_upsample: int = typer.Option(
        2,
        "--feature-upsample",
        "-u",
        help="Upsample feature grid before K-Means",
    ),
    pca_dim: Optional[int] = typer.Option(
        None,
        "--pca-dim",
        help="Optional PCA target dimension (e.g., 128)",
    ),
    refine: Literal["none", "slic"] = typer.Option(
        "slic",
        "--refine",
        "-r",
        help="Edge-aware refinement mode",
    ),
    refine_slic_compactness: float = typer.Option(
        10.0,
        "--slic-compactness",
        help="SLIC compactness (higher=smoother, lower=edges)",
    ),
    refine_slic_sigma: float = typer.Option(
        1.0,
        "--slic-sigma",
        help="SLIC Gaussian smoothing sigma",
    ),
    profile: Optional[Literal["quality", "balanced", "speed"]] = typer.Option(
        "balanced",
        "--profile",
        "-p",
        help="Preset quality/speed profile",
    ),
    elbow_threshold: Optional[float] = typer.Option(
        None,
        "--elbow-threshold",
        "-e",
        help="Elbow method percentage threshold (e.g., 5.0)",
    ),
    sweep: Optional[Path] = typer.Option(
        None,
        "--sweep",
        help="Path to JSON/YAML file with config overrides to run in sequence",
        exists=True,
    ),
    sweep_prefix: str = typer.Option(
        "sweeps",
        "--sweep-prefix",
        help="Subfolder under output dir for sweep runs",
    ),
    clean_output: bool = typer.Option(
        False,
        "--clean-output",
        help="Clear output directory before writing results",
    ),
    metrics: bool = typer.Option(
        False,
        "--metrics",
        "-m",
        help="Collect and print timing/VRAM metrics",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress detailed processing output",
    ),
    save_labels: bool = typer.Option(
        True,
        "--save-labels/--no-save-labels",
        help="Save predicted labels (NPZ) for each image",
    ),
    save_metadata: bool = typer.Option(
        True,
        "--save-metadata/--no-save-metadata",
        help="Store run metadata in results index",
    ),
    use_cache: bool = typer.Option(
        True,
        "--use-cache/--no-cache",
        help="Reuse cached results if a matching input/config hash exists",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-run even if cached results exist for this input/config",
    ),
    use_attention_features: bool = typer.Option(
        True,
        "--use-attn/--no-use-attn",
        help="Include attention tokens in features (disable for legacy v1 behavior)",
    ),
    metadata_dir: Path = typer.Option(
        Path("results"),
        "--metadata-dir",
        help="Base directory for metadata storage",
    ),
    metadata_tags: List[str] = typer.Option(
        [],
        "--tag",
        "-t",
        help="User tags to attach to metadata entries",
    ),
):
    """
    Segment trees in aerial imagery using DINOv3 features.

    Process a single image or batch process all images in a directory.
    Supports quality presets and parameter sweeps.

    Examples:

        # Process single image with balanced preset
        tree-seg segment image.jpg base

        # Batch process directory with quality preset
        tree-seg segment data/images/ large --profile quality

        # Run parameter sweep
        tree-seg segment data/images/ base --sweep configs/sweep.json
    """
    verbose = not quiet

    # Apply profile defaults
    config_kwargs = {
        "image_size": image_size,
        "feature_upsample_factor": feature_upsample,
        "pca_dim": pca_dim,
        "refine": None if refine == "none" else refine,
        "refine_slic_compactness": refine_slic_compactness,
        "refine_slic_sigma": refine_slic_sigma,
        "metrics": metrics,
        "verbose": verbose,
        "use_attention_features": use_attention_features,
    }

    # Apply profile defaults if not explicitly overridden
    if profile:
        defaults = PROFILE_DEFAULTS.get(profile, {})
        for key, value in defaults.items():
            # Only apply if the user didn't explicitly set it
            # This is a simplification - in practice we'd check sys.argv
            if key not in ["metrics", "verbose"]:
                config_kwargs.setdefault(key, value)

    if elbow_threshold is not None:
        config_kwargs["elbow_threshold"] = elbow_threshold

    # Sweep mode
    if sweep:
        console.print(f"[bold cyan]🔄 Running parameter sweep from {sweep}[/bold cyan]\n")

        # Load sweep config
        cfg_list = None
        try:
            if sweep.suffix == ".json":
                with open(sweep) as f:
                    cfg_list = json.load(f)
            else:
                import yaml
                with open(sweep) as f:
                    cfg_list = yaml.safe_load(f)
        except Exception as e:
            console.print(f"[red]❌ Failed to parse sweep file: {e}[/red]")
            raise typer.Exit(code=1)

        if not isinstance(cfg_list, list):
            console.print("[red]❌ Sweep file must contain a list of configurations[/red]")
            raise typer.Exit(code=1)

        for idx, item in enumerate(cfg_list):
            if not isinstance(item, dict):
                console.print(f"[yellow]⚠️  Skipping non-dict sweep item at idx {idx}[/yellow]")
                continue

            name = item.get("name") or f"cfg{idx:02d}"
            mdl = item.get("model") or model
            out_dir = output_dir / sweep_prefix / name

            console.print(f"\n[bold]=== Sweep '{name}' (model={mdl}) ===[/bold]")

            overrides = {k: v for k, v in item.items() if k not in {"name", "model", "profile"}}
            prof = item.get("profile")
            if prof and prof in PROFILE_DEFAULTS:
                for key, value in PROFILE_DEFAULTS[prof].items():
                    overrides.setdefault(key, value)

            cfg = config_kwargs.copy()
            cfg.update({k: v for k, v in overrides.items() if v is not None})
            run_single_segment(
                img_path=image_path,
                out_dir=out_dir,
                mdl=mdl,
                cfg=cfg,
                metrics=metrics,
                save_labels=save_labels,
                save_metadata=save_metadata,
                sweep_label=name,
                console=console,
                use_cache=use_cache,
                force=force,
            )

        console.print("\n[bold green]✨ Sweep completed![/bold green]")
        return

    # Single run mode
    if clean_output and output_dir.exists():
        existing_files = list(output_dir.rglob("*.png")) + list(output_dir.rglob("*.jpg"))
        if existing_files:
            console.print(f"[yellow]🗂️  Found {len(existing_files)} existing output file(s)[/yellow]")
            console.print(f"[yellow]🧹 Clearing output directory: {output_dir}[/yellow]")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]📁 Output directory ready: {output_dir}[/green]")
    console.print()

    cfg = config_kwargs.copy()
    run_single_segment(
        img_path=image_path,
        out_dir=output_dir,
        mdl=model,
        cfg=cfg,
        metrics=metrics,
        save_labels=save_labels,
        save_metadata=save_metadata,
        sweep_label=None,
        console=console,
        use_cache=use_cache,
        force=force,
    )
