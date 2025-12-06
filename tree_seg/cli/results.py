"""Results/metadata command for tree-seg CLI."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tree_seg.metadata.load import lookup
from tree_seg.metadata.query import query as query_index

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
):
    """
    Query stored experiment metadata or show details for a specific run.

    Examples:
        tree-seg results --dataset fortress --tags kmeans,slic --sort mIoU --top 5
        tree-seg results --hash abc123def0 --show-config
    """
    tag_list = _parse_tags(tags)

    if hash_id:
        meta = lookup(hash_id, base_dir=base_dir)
        if not meta:
            console.print(f"[red]❌ No metadata found for hash {hash_id}[/red]")
            raise typer.Exit(code=1)
        _print_detail(meta, show_config=show_config)
        if render:
            artifacts = meta.get("artifacts", {})
            viz_dir = artifacts.get("visualizations_dir")
            labels_dir = artifacts.get("labels_dir")
            if viz_dir:
                console.print(f"[green]✔ Visualizations already exist at {viz_dir}[/green]")
            elif labels_dir:
                console.print("[yellow]⚠️ Labels present but visualization regeneration is not implemented yet.[/yellow]")
            else:
                console.print("[yellow]⚠️ No labels or visualizations found; nothing to render.[/yellow]")
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
