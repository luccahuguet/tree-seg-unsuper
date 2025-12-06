"""Segment command for tree segmentation CLI."""

import json
import shutil
from pathlib import Path
from typing import Literal, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from tree_seg import segment_trees
from tree_seg.constants import PROFILE_DEFAULTS, SUPPORTED_IMAGE_EXTS

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

    def _run_case(
        img_path: Path,
        mdl: str,
        out_dir: Path,
        overrides: Optional[dict] = None,
    ):
        """Run segmentation for a single case."""
        # Clear output directory
        if out_dir.exists():
            existing_files = list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.jpg"))
            if existing_files:
                console.print(f"[yellow]🗂️  Found {len(existing_files)} existing output file(s) in {out_dir}[/yellow]")
                console.print(f"[yellow]🧹 Clearing output directory: {out_dir}[/yellow]")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]📁 Output directory ready: {out_dir}[/green]")
        console.print()

        cfg = config_kwargs.copy()
        if overrides:
            cfg.update({k: v for k, v in overrides.items() if v is not None})
        if cfg.get("refine") == "none":
            cfg["refine"] = None
        if cfg.get("elbow_threshold") is None:
            cfg.pop("elbow_threshold", None)

        # Process directory or single image
        if img_path.is_dir():
            image_files = []
            for ext in SUPPORTED_IMAGE_EXTS:
                image_files.extend(img_path.glob(f"*{ext}"))
                image_files.extend(img_path.glob(f"*{ext.upper()}"))

            if not image_files:
                console.print(f"[red]❌ No image files found in {img_path}[/red]")
                return

            console.print(f"[cyan]🖼️  Found {len(image_files)} image(s) in {img_path}[/cyan]")

            for p in sorted(image_files):
                console.print(f"\n[bold]🚀 Processing: {p.name}[/bold]")
                try:
                    results = segment_trees(
                        str(p),
                        model=mdl,
                        auto_k=True,
                        output_dir=str(out_dir),
                        **cfg,
                    )
                    if metrics and isinstance(results, list) and results:
                        res, _paths = results[0]
                        stats = getattr(res, "processing_stats", {})
                        console.print(
                            f"[dim]⏱️  total={stats.get('time_total_s')}s, "
                            f"features={stats.get('time_features_s')}s, "
                            f"kselect={stats.get('time_kselect_s')}s, "
                            f"kmeans={stats.get('time_kmeans_s')}s, "
                            f"refine={stats.get('time_refine_s')}s, "
                            f"peak_vram={stats.get('peak_vram_mb')}MB[/dim]"
                        )
                    console.print(f"[green]✅ Completed: {p.name}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Failed: {p.name} - {e}[/red]")
        else:
            console.print(f"[bold]🚀 Processing: {img_path.name}[/bold]")
            try:
                results = segment_trees(
                    str(img_path),
                    model=mdl,
                    auto_k=True,
                    output_dir=str(out_dir),
                    **cfg,
                )
                if metrics and isinstance(results, list) and results:
                    res, _paths = results[0]
                    stats = getattr(res, "processing_stats", {})
                    console.print(
                        f"[dim]⏱️  total={stats.get('time_total_s')}s, "
                        f"features={stats.get('time_features_s')}s, "
                        f"kselect={stats.get('time_kselect_s')}s, "
                        f"kmeans={stats.get('time_kmeans_s')}s, "
                        f"refine={stats.get('time_refine_s')}s, "
                        f"peak_vram={stats.get('peak_vram_mb')}MB[/dim]"
                    )
                console.print("[green]✅ Tree segmentation completed![/green]")
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
                raise typer.Exit(code=1)

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

            _run_case(image_path, mdl, out_dir, overrides)

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

    _run_case(image_path, model, output_dir)
