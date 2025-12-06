"""Shared helpers for the segment CLI (config building, single run)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console

from tree_seg import segment_trees
from tree_seg.constants import SUPPORTED_IMAGE_EXTS
from tree_seg.core.types import Config
from tree_seg.metadata.store import store_segment_run


def build_config_from_kwargs(mdl: str, out_dir: Path, cfg: dict) -> Config:
    """Build Config object for a segment run using provided kwargs."""
    cfg_clean = cfg.copy()
    if cfg_clean.get("refine") == "none":
        cfg_clean["refine"] = None
    if cfg_clean.get("elbow_threshold") is None:
        cfg_clean.pop("elbow_threshold", None)

    return Config(
        output_dir=str(out_dir),
        model_name=mdl,
        auto_k=True,
        **cfg_clean,
    )


def clean_output_dir(out_dir: Path, console: Console) -> None:
    """Clear existing outputs in a directory if present."""
    if out_dir.exists():
        existing_files = list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.jpg"))
        if existing_files:
            console.print(f"[yellow]🗂️  Found {len(existing_files)} existing output file(s) in {out_dir}[/yellow]")
            console.print(f"[yellow]🧹 Clearing output directory: {out_dir}[/yellow]")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def collect_image_files(img_path: Path) -> list[Path]:
    """Return supported image files under a directory."""
    image_files: list[Path] = []
    for ext in SUPPORTED_IMAGE_EXTS:
        image_files.extend(img_path.glob(f"*{ext}"))
        image_files.extend(img_path.glob(f"*{ext.upper()}"))
    return sorted(image_files)


def run_single_segment(
    *,
    img_path: Path,
    out_dir: Path,
    mdl: str,
    cfg: dict,
    metrics: bool,
    save_labels: bool,
    save_metadata: bool,
    sweep_label: Optional[str],
    console: Console,
):
    """Run segmentation for a single case (file or directory)."""
    clean_output_dir(out_dir, console)
    console.print(f"[green]📁 Output directory ready: {out_dir}[/green]")
    console.print()

    meta_config = build_config_from_kwargs(mdl, out_dir, cfg)

    collected_outputs: list = []

    if img_path.is_dir():
        image_files = collect_image_files(img_path)
        if not image_files:
            console.print(f"[red]❌ No image files found in {img_path}[/red]")
            return

        from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn

        try:
            progress = Progress(
                TextColumn("[bold cyan]Segment"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            task_id = progress.add_task("segment", total=len(image_files))
            progress.start()
        except Exception:
            progress = None
            task_id = None

        for p in image_files:
            if progress:
                progress.console.print(f"\n[bold]🚀 Processing: {p.name}[/bold]", highlight=False)
            else:
                console.print(f"\n[bold]🚀 Processing: {p.name}[/bold]")
            try:
                results = segment_trees(
                    str(p),
                    model=mdl,
                    auto_k=True,
                    output_dir=str(out_dir),
                    **cfg,
                )
                if isinstance(results, list):
                    collected_outputs.extend(results)
                if metrics and isinstance(results, list) and results:
                    res, _paths = results[0]
                    stats = getattr(res, "processing_stats", {})
                    line = (
                        f"[dim]⏱️  total={stats.get('time_total_s')}s, "
                        f"features={stats.get('time_features_s')}s, "
                        f"kselect={stats.get('time_kselect_s')}s, "
                        f"kmeans={stats.get('time_kmeans_s')}s, "
                        f"refine={stats.get('time_refine_s')}s, "
                        f"peak_vram={stats.get('peak_vram_mb')}MB[/dim]"
                    )
                    (progress.console if progress else console).print(line, highlight=False)
                if save_labels and isinstance(results, list):
                    for res, _paths in results:
                        _save_labels_npz(out_dir, res)
                if progress:
                    progress.console.print(f"[green]✅ Completed: {p.name}[/green]", highlight=False)
                else:
                    console.print(f"[green]✅ Completed: {p.name}[/green]")
            except Exception as exc:
                if progress:
                    progress.console.print(f"[red]❌ Failed: {p.name} - {exc}[/red]", highlight=False)
                else:
                    console.print(f"[red]❌ Failed: {p.name} - {exc}")
            finally:
                if progress:
                    progress.advance(task_id)

        if progress:
            progress.stop()
    else:
        console.print(f"\n[bold]🚀 Processing: {img_path.name}[/bold]")
        results = segment_trees(
            str(img_path),
            model=mdl,
            auto_k=True,
            output_dir=str(out_dir),
            **cfg,
        )
        if isinstance(results, list):
            collected_outputs.extend(results)
        if save_labels and isinstance(results, list):
            for res, _paths in results:
                _save_labels_npz(out_dir, res)

    if save_metadata and collected_outputs:
        try:
            labeled_paths = []
            for _res, paths in collected_outputs:
                labeled_paths.extend(paths)
            hash_id = store_segment_run(meta_config, out_dir, labeled_paths, sweep_label=sweep_label)
            console.print(f"[dim]📝 Metadata stored at results/by-hash/{hash_id}[/dim]")
        except Exception:
            pass


def _save_labels_npz(out_dir: Path, res) -> None:
    """Persist labels to NPZ for regeneration."""
    if not hasattr(res, "labels"):
        return
    labels_np = getattr(res, "labels")
    label_path = out_dir / f"{getattr(res, 'image_stem', 'segmentation')}_labels.npz"
    np.savez_compressed(label_path, labels=labels_np)
