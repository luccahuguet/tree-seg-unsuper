"""Sweep command for multiplicative parameter exploration."""

from pathlib import Path
from typing import List, Literal, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from tree_seg.evaluation.runner import detect_dataset_type
from tree_seg.evaluation.sweep_runner import run_multiplicative_sweep

# Load environment variables
load_dotenv()

console = Console()


# Define "all" expansions for each parameter
ALL_OPTIONS = {
    "clustering": [
        "kmeans",
        "gmm",
        "spectral",
        "hdbscan",
        "spherical",
        "dpmeans",
        "potts",
    ],
    "refine": ["none", "slic", "soft-em", "bilateral", "soft-em+slic"],
    "model": ["small", "base", "large", "mega"],
    "stride": ["2", "4", "8"],
    "tiling": ["on", "off"],
    "elbow_threshold": ["2.5", "5.0", "10.0", "20.0", "50.0"],
}


def _load_preset(preset_name: str) -> dict:
    """Load preset configuration from presets.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # fallback for Python < 3.11

    presets_file = Path(__file__).parent.parent / "evaluation" / "presets.toml"
    if not presets_file.exists():
        raise FileNotFoundError(f"Presets file not found: {presets_file}")

    with open(presets_file, "rb") as f:
        data = tomllib.load(f)

    if preset_name not in data.get("presets", {}):
        available = ", ".join(data.get("presets", {}).keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )

    preset = data["presets"][preset_name]
    description = preset.pop("description", "")
    console.print(f"[dim]Using preset: {preset_name} - {description}[/dim]")

    return preset


def _load_toml_config(config_path: Path) -> dict:
    """Load sweep configuration from TOML file."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    # Extract sweep params (everything under [sweep] or [params])
    sweep_params = data.get("sweep", data.get("params", {}))
    return sweep_params


def _parse_list_param(
    value: Optional[str], param_name: Optional[str] = None
) -> List[str]:
    """
    Parse comma-separated string into list, with "all" expansion.

    Args:
        value: String like "kmeans,gmm" or "all" or None
        param_name: Parameter name for "all" expansion (clustering, refine, model, etc.)

    Returns:
        List of values, or empty list if None
    """
    if not value:
        return []

    # Handle "all" expansion
    if value.lower() == "all":
        if param_name and param_name in ALL_OPTIONS:
            return ALL_OPTIONS[param_name]
        else:
            raise ValueError(f"'all' not supported for parameter: {param_name}")

    # Comma-separated values
    return [v.strip() for v in value.split(",") if v.strip()]


def _normalize_sweep_param(key: str, values: List[str]) -> list:
    """
    Normalize sweep parameter values.

    Args:
        key: Parameter name
        values: List of string values from CLI

    Returns:
        List of normalized values
    """
    if not values:
        return []

    normalized = []
    for val in values:
        if key == "refine":
            # Map "none" string to None
            normalized.append(None if val == "none" else val)
        elif key == "tiling":
            # Map on/off to boolean
            if val in ("on", "true", "1"):
                normalized.append(True)
            elif val in ("off", "false", "0"):
                normalized.append(False)
            else:
                raise ValueError(f"Invalid tiling value: {val} (use 'on' or 'off')")
        elif key in ("stride", "elbow_threshold"):
            # Convert to numeric
            normalized.append(float(val) if "." in str(val) else int(val))
        else:
            normalized.append(val)

    return normalized


def sweep_command(
    dataset: Path = typer.Argument(
        ...,
        help="Path to dataset directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    # Sweep parameters (comma-separated or space-separated lists)
    clustering: Optional[str] = typer.Option(
        None,
        "--clustering",
        "-c",
        help="Clustering methods to sweep: kmeans,gmm or 'all' for all methods",
    ),
    refine: Optional[str] = typer.Option(
        None,
        "--refine",
        "-r",
        help="Refinement methods to sweep: slic,none or 'all' for all methods",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="DINOv3 model sizes to sweep: base,large or 'all' for all sizes",
    ),
    stride: Optional[str] = typer.Option(
        None,
        "--stride",
        help="Feature extraction strides to sweep: 4,2 or 'all' for common strides",
    ),
    tiling: Optional[str] = typer.Option(
        None,
        "--tiling",
        help="Tiling modes to sweep: on,off or 'all' for both",
    ),
    elbow_threshold: Optional[str] = typer.Option(
        None,
        "--elbow-threshold",
        "-e",
        help="Elbow thresholds to sweep: 5.0,10.0 or 'all' for common thresholds",
    ),
    # Configuration sources
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-p",
        help="Use a curated preset (quick, clustering, refine, tiling, models, elbow, paper, full, stride)",
    ),
    from_file: Optional[Path] = typer.Option(
        None,
        "--from",
        help="Load sweep configuration from TOML file",
        exists=True,
    ),
    # Fixed parameters
    dataset_type: Optional[Literal["fortress", "isprs", "generic"]] = typer.Option(
        None,
        "--dataset-type",
        "-t",
        help="Dataset type (auto-detected if not specified)",
    ),
    image_size: int = typer.Option(
        1024,
        "--image-size",
        "-s",
        help="Image resize dimension",
    ),
    fixed_k: Optional[int] = typer.Option(
        None,
        "--fixed-k",
        "-k",
        help="Fixed number of clusters (overrides auto K for all configs)",
    ),
    num_samples: Optional[int] = typer.Option(
        None,
        "--num-samples",
        "-n",
        help="Number of samples to evaluate (default: all)",
    ),
    save_viz: bool = typer.Option(
        False,
        "--save-viz",
        help="Save visualization images",
    ),
    save_labels: bool = typer.Option(
        True,
        "--save-labels/--no-save-labels",
        help="Save predicted labels (NPZ) for metadata/viz regeneration",
    ),
    image_ids: Optional[str] = typer.Option(
        None,
        "--image-ids",
        help="Comma-separated image IDs to evaluate (subset of dataset)",
    ),
    apply_vegetation_filter: bool = typer.Option(
        False,
        "--vegetation-filter",
        help="Apply vegetation filtering for species-level segmentation",
    ),
    exg_threshold: float = typer.Option(
        0.1,
        "--exg-threshold",
        help="ExG threshold for vegetation filtering",
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
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress output",
    ),
    use_attention_features: bool = typer.Option(
        True,
        "--use-attn/--no-use-attn",
        help="Include attention tokens in features",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Custom name for this sweep (default: auto-generated)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview configurations without running",
    ),
):
    """
    Run multiplicative parameter sweep across multiple configurations.

    This command generates all combinations of specified parameters (factorial design).
    For example, sweeping 2 clustering methods × 2 refinements = 4 total configs.

    Examples:

        # Basic sweep: 2 clustering × 2 refine = 4 configs
        tree-seg sweep data/datasets/fortress -c kmeans,gmm -r slic,none

        # Use "all" for convenience: 7 clustering × 2 refine = 14 configs
        tree-seg sweep data/datasets/fortress -c all -r slic,none

        # All refinement methods with K-means: 5 configs
        tree-seg sweep data/datasets/fortress -c kmeans -r all

        # Multi-parameter: 4 models × 3 strides = 12 configs
        tree-seg sweep data/datasets/fortress --model all --stride all

        # Use a curated preset
        tree-seg sweep data/datasets/fortress --preset clustering

        # From TOML file
        tree-seg sweep data/datasets/fortress --from sweeps/my_study.toml

        # Preview without running
        tree-seg sweep data/datasets/fortress -c all -r slic,none --dry-run

        # Limit samples and save visualizations
        tree-seg sweep data/datasets/fortress -c kmeans,gmm -r slic -n 5 --save-viz
    """
    if not dataset_type:
        dataset_type = detect_dataset_type(dataset)
        console.print(f"[dim]Auto-detected dataset type: {dataset_type}[/dim]")

    # Determine sweep parameters
    sweep_params = {}

    # Priority: CLI flags > preset > from_file
    if from_file:
        sweep_params = _load_toml_config(from_file)
        sweep_name = name or from_file.stem
    elif preset:
        sweep_params = _load_preset(preset)
        sweep_name = name or preset
        # Normalize preset values (same as CLI flags)
        for key in list(sweep_params.keys()):
            if key in ("refine", "stride", "tiling", "elbow_threshold"):
                sweep_params[key] = _normalize_sweep_param(key, sweep_params[key])
    else:
        # Build from CLI flags
        if clustering:
            sweep_params["clustering_method"] = _parse_list_param(
                clustering, "clustering"
            )
        if refine:
            sweep_params["refine"] = _normalize_sweep_param(
                "refine", _parse_list_param(refine, "refine")
            )
        if model:
            sweep_params["model_name"] = _parse_list_param(model, "model")
        if stride:
            sweep_params["stride"] = _normalize_sweep_param(
                "stride", _parse_list_param(stride, "stride")
            )
        if tiling:
            sweep_params["use_tiling"] = _normalize_sweep_param(
                "tiling", _parse_list_param(tiling, "tiling")
            )
        if elbow_threshold:
            sweep_params["elbow_threshold"] = _normalize_sweep_param(
                "elbow_threshold", _parse_list_param(elbow_threshold, "elbow_threshold")
            )

        sweep_name = name or "custom"

    if not sweep_params:
        console.print(
            "[red]❌ No sweep parameters specified. Use --clustering, --refine, --model, etc., or --preset/--from[/red]"
        )
        raise typer.Exit(code=1)

    # Build base config (fixed params applied to all sweep configs)
    base_config_params = {
        "image_size": image_size,
        "auto_k": fixed_k is None,
        "n_clusters": fixed_k if fixed_k else 6,
        "apply_vegetation_filter": apply_vegetation_filter,
        "exg_threshold": exg_threshold,
        "use_pyramid": use_pyramid,
        "pyramid_scales": tuple(float(s.strip()) for s in pyramid_scales.split(",")),
        "pyramid_aggregation": pyramid_aggregation,
        "use_attention_features": use_attention_features,
        "metrics": True,
        "verbose": not quiet,
    }

    # Apply defaults for non-swept params
    if "clustering_method" not in sweep_params:
        base_config_params["clustering_method"] = "kmeans"
    if "refine" not in sweep_params:
        base_config_params["refine"] = "slic"
    if "model_name" not in sweep_params:
        base_config_params["model_name"] = "base"
    if "stride" not in sweep_params:
        base_config_params["stride"] = 4
    if "use_tiling" not in sweep_params:
        base_config_params["use_tiling"] = False  # Disabled by default
    if "elbow_threshold" not in sweep_params:
        base_config_params["elbow_threshold"] = 5.0

    # Dry run: preview configs
    if dry_run:
        from tree_seg.evaluation.sweep_runner import generate_sweep_configs

        configs = generate_sweep_configs(base_config_params, sweep_params)
        console.print(f"\n[bold cyan]Sweep Preview: {sweep_name}[/bold cyan]")
        console.print(f"[dim]Would run {len(configs)} configuration(s):[/dim]\n")

        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Label", style="cyan")
        table.add_column("Parameters", style="dim")

        for i, cfg in enumerate(configs, 1):
            label = cfg.get("label", f"config_{i}")
            # Show swept params only
            swept = {k: v for k, v in cfg.items() if k in sweep_params}
            table.add_row(
                str(i),
                label,
                ", ".join(f"{k}={v}" for k, v in swept.items()),
            )

        console.print(table)
        console.print(
            f"\n[dim]Run without --dry-run to execute this sweep on {dataset}[/dim]\n"
        )
        return

    filter_ids = (
        [s.strip() for s in image_ids.split(",") if s.strip()] if image_ids else None
    )

    # Run sweep
    run_multiplicative_sweep(
        dataset_path=dataset,
        dataset_type=dataset_type,
        base_config_params=base_config_params,
        sweep_params=sweep_params,
        sweep_name=sweep_name,
        num_samples=num_samples,
        filter_ids=filter_ids,
        save_viz=save_viz,
        save_labels=save_labels,
        quiet=quiet,
        smart_k=smart_k,
        console=console,
        use_cache=use_cache,
    )
