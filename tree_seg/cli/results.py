"""Results/metadata command for tree-seg CLI."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tree_seg.metadata.load import lookup, lookup_nearest
from tree_seg.metadata.query import (
    query as query_index,
    compact as compact_index,
    prune_older_than,
    export_to_csv,
)
from tree_seg.visualization.plotting import generate_visualizations
from tree_seg.core.types import Config, OutputPaths, SegmentationResults
import numpy as np
from PIL import Image

console = Console()


def _parse_tags(tags: Optional[str]) -> list[str]:
    if not tags:
        return []
    return [t.strip() for t in tags.split(",") if t.strip()]


def _print_detail(meta: dict, show_config: bool) -> None:
    table = Table(title=f"Run {meta.get('hash')}", show_header=False, box=None)
    table.add_row("Dataset", meta.get("dataset", "unknown"))
    table.add_row("Created", meta.get("created_at", ""))
    table.add_row("Git SHA", meta.get("git_sha") or "n/a")

    metrics = meta.get("metrics", {})
    timing = meta.get("timing", {})
    samples = meta.get("samples", {})
    tags = meta.get("tags", {})
    artifacts = meta.get("artifacts", {})

    table.add_row("mIoU", f"{metrics.get('mean_miou', 0):.4f}")
    table.add_row("Pixel Acc", f"{metrics.get('mean_pixel_accuracy', 0):.4f}")
    table.add_row("Mean Runtime (s)", f"{timing.get('mean_runtime_s', 0):.2f}")
    table.add_row("Total Runtime (s)", f"{timing.get('total_runtime_s', 0):.2f}")
    table.add_row("Samples", str(samples.get("num_samples", 0)))
    table.add_row("Tags", ", ".join((tags.get("auto") or []) + (tags.get("user") or [])))
    if artifacts:
        art_lines = [f"{k}: {v}" for k, v in artifacts.items()]
        table.add_row("Artifacts", "\n".join(art_lines))
    console.print(table)

    if show_config:
        console.print("\n[bold]Config (normalized)[/bold]")
        console.print_json(data=meta.get("config", {}))
        console.print("\n[bold]Config (full)[/bold]")
        console.print_json(data=meta.get("config_full", {}))


def _find_image_path(dataset_root: Path, image_id: str) -> Optional[Path]:
    """Best-effort find image file for given image_id under dataset root."""
    candidates = []
    # Common structure: dataset_root/images/<id>.<ext>
    images_dir = dataset_root / "images"
    if images_dir.exists():
        candidates.extend(images_dir.glob(f"{image_id}.*"))
    # Fallback to root
    candidates.extend(dataset_root.glob(f"{image_id}.*"))
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    for path in candidates:
        if path.suffix.lower() in exts:
            return path
    return None


def _render_visualizations(meta: dict, base_dir: Path) -> None:
    """Regenerate visualizations from stored labels."""
    hash_id = meta.get("hash")
    artifacts = meta.get("artifacts", {})
    labels_dir = artifacts.get("labels_dir")
    dataset_root = meta.get("dataset_root")
    config_full = meta.get("config_full") or meta.get("config")

    if not labels_dir or not dataset_root or not config_full:
        console.print("[yellow]⚠️  Missing labels, dataset root, or config; cannot render.[/yellow]")
        return

    labels_dir = Path(labels_dir)
    dataset_root = Path(dataset_root)
    if not labels_dir.exists():
        console.print(f"[yellow]⚠️  Labels directory not found: {labels_dir}[/yellow]")
        return

    run_dir = base_dir / "by-hash" / hash_id
    viz_dir = Path(artifacts.get("visualizations_dir") or run_dir / "visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(**config_full)
    cfg.output_dir = str(run_dir)

    npz_files = sorted(labels_dir.glob("*.npz"))
    if not npz_files:
        console.print(f"[yellow]⚠️  No label files found in {labels_dir}[/yellow]")
        return

    regenerated = 0
    for npz_path in npz_files:
        image_id = npz_path.stem
        data = np.load(npz_path)
        if "labels" not in data:
            continue
        labels = data["labels"]
        img_path = _find_image_path(dataset_root, image_id)
        if img_path is None:
            console.print(f"[yellow]⚠️  Could not find image for {image_id} under {dataset_root}[/yellow]")
            continue
        image_np = np.array(Image.open(img_path).convert("RGB"))
        n_clusters = int(labels.max()) + 1
        seg_result = SegmentationResults(
            image_np=image_np,
            labels_resized=labels,
            n_clusters_used=n_clusters,
            image_path=str(img_path),
            processing_stats={},
            n_clusters_requested=None,
        )
        prefix = f"{image_id}_regen"
        output_paths = OutputPaths(
            segmentation_legend=str(viz_dir / f"{prefix}_segmentation_legend.png"),
            edge_overlay=str(viz_dir / f"{prefix}_edge_overlay.png"),
            side_by_side=str(viz_dir / f"{prefix}_side_by_side.png"),
            elbow_analysis=None,
        )
        generate_visualizations(seg_result, cfg, output_paths)
        regenerated += 1

    console.print(f"[green]✅ Regenerated visualizations for {regenerated}/{len(npz_files)} sample(s)[/green]")


def results_command(
    hash_id: Optional[str] = typer.Option(
        None,
        "--hash",
        "-h",
        help="Show details for a specific run hash",
    ),
    render: bool = typer.Option(
        False,
        "--render",
        help="Attempt to regenerate visualizations if labels are available",
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Filter by dataset name",
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        "-t",
        help="Comma-separated tags to filter by",
    ),
    sort_by: Optional[str] = typer.Option(
        "mIoU",
        "--sort",
        help="Sort key for listing (e.g., mIoU, total_s, created_at)",
    ),
    top: Optional[int] = typer.Option(
        10,
        "--top",
        "-k",
        help="Limit number of entries to show (default: 10)",
    ),
    show_config: bool = typer.Option(
        False,
        "--show-config",
        help="Print configs in detail for hash lookup",
    ),
    base_dir: Path = typer.Option(
        Path("results"),
        "--results-dir",
        help="Base directory for metadata storage",
    ),
    compact: bool = typer.Option(
        False,
        "--compact",
        help="Remove index entries whose meta.json is missing",
    ),
    prune_days: Optional[int] = typer.Option(
        None,
        "--prune-older-than",
        help="Prune entries older than N days (removes meta dirs and rewrites index)",
    ),
    nearest: Optional[str] = typer.Option(
        None,
        "--nearest",
        help="JSON string of config fields to estimate ETA/runtime (uses nearest match)",
    ),
    export_csv: Optional[Path] = typer.Option(
        None,
        "--export-csv",
        help="Path to write queried results as CSV",
    ),
):
    """
    Query stored experiment metadata or show details for a specific run.

    Examples:
        tree-seg results --dataset fortress --tags kmeans,slic --sort mIoU --top 5
        tree-seg results --hash abc123def0 --show-config
        tree-seg results --compact
        tree-seg results --prune-older-than 30
    """
    tag_list = _parse_tags(tags)

    if compact:
        removed = compact_index(base_dir=base_dir)
        console.print(f"[green]🧹 Compacted index, removed {removed} stale entr{'y' if removed==1 else 'ies'}[/green]")
        if not hash_id and prune_days is None:
            return

    if prune_days is not None:
        removed = prune_older_than(prune_days, base_dir=base_dir)
        console.print(f"[green]🗑️  Pruned {removed} entr{'y' if removed==1 else 'ies'} older than {prune_days}d[/green]")
        if not hash_id:
            return

    if hash_id:
        meta = lookup(hash_id, base_dir=base_dir)
        if not meta:
            console.print(f"[red]❌ No metadata found for hash {hash_id}[/red]")
            raise typer.Exit(code=1)
        _print_detail(meta, show_config=show_config)
        if render:
            _render_visualizations(meta, base_dir=base_dir)
        return

    if nearest:
        try:
            nearest_config = json.loads(nearest)
        except json.JSONDecodeError:
            console.print("[red]❌ --nearest must be valid JSON[/red]")
            raise typer.Exit(code=1)
        match = lookup_nearest(nearest_config, base_dir=base_dir)
        if match:
            console.print("[bold green]Closest match for ETA/runtime:[/bold green]")
            console.print_json(data=match)
        else:
            console.print("[yellow]No runs in index to match against.[/yellow]")
        return

    entries = query_index(
        dataset=dataset,
        tags=tag_list if tag_list else None,
        sort_by=sort_by,
        limit=top,
        base_dir=base_dir,
    )

    if not entries:
        console.print("[yellow]No matching results found.[/yellow]")
        return

    table = Table(title="Results Index", show_header=True, header_style="bold cyan")
    table.add_column("Hash", style="magenta")
    table.add_column("Dataset", style="cyan")
    table.add_column("Tags", style="green")
    table.add_column("mIoU", justify="right")
    table.add_column("PA", justify="right")
    table.add_column("Total s", justify="right")
    table.add_column("Created", justify="right")

    for entry in entries:
        table.add_row(
            entry.get("hash", ""),
            entry.get("dataset", ""),
            ",".join(entry.get("tags", [])),
            f"{entry.get('mIoU', 0):.4f}",
            f"{entry.get('pixel_accuracy', 0):.4f}",
            f"{entry.get('total_s', 0):.2f}",
            entry.get("created_at", ""),
        )
    console.print(table)
    if export_csv:
        rows = export_to_csv(entries, export_csv)
        console.print(f"[green]💾 Exported {rows} row(s) to {export_csv}[/green]")
