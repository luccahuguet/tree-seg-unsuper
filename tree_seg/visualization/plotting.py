"""
Modern visualization module using the new architecture.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from ..constants import DPI_SEGMENTATION
from ..core.types import Config, OutputPaths, SegmentationResults
from ..utils.config import get_config_text
from .composites import overlay_labels
from .overlay import generate_edge_overlay, generate_side_by_side


def generate_visualizations(
    results: SegmentationResults, config: Config, output_paths: OutputPaths
) -> None:
    """Generate all visualizations using modern architecture."""
    verbose = getattr(config, "verbose", True)
    if not results.success:
        if verbose:
            print("⚠️ Skipping visualization - segmentation failed")
        return

    filename = os.path.basename(results.image_path)
    method_name = "supervised" if config.supervised else config.clustering_method
    config_text = get_config_text(
        results.n_clusters_used,
        config.overlay_ratio,
        config.stride,
        config.model_display_name,
        filename,
        method_name,
        config.edge_width,
        config.elbow_threshold if config.auto_k else None,
        n_clusters_requested=results.n_clusters_requested,
    )

    n_clusters = results.n_clusters_used
    if n_clusters <= 10:
        cmap = plt.get_cmap("Set1", n_clusters)
    elif n_clusters <= 20:
        cmap = plt.get_cmap("Dark2", n_clusters)
    else:
        cmap = plt.get_cmap("viridis", n_clusters)

    _generate_segmentation_legend(
        results, config, output_paths, cmap, config_text, verbose
    )
    generate_edge_overlay(results, config, output_paths, cmap, config_text, verbose)
    generate_side_by_side(results, config, output_paths, cmap, config_text, verbose)
    _generate_overlay(results, config, output_paths, verbose)

    if verbose:
        print(f"✅ Generated visualizations for {filename}")


def _generate_overlay(
    results: SegmentationResults,
    config: Config,
    output_paths: OutputPaths,
    verbose: bool = True,
) -> None:
    """Generate simple overlay image via composites helper."""
    if config.overlay_ratio <= 0:
        return
    overlay_image = overlay_labels(
        results.image_np,
        results.labels_resized,
        alpha=config.overlay_ratio,
    )
    if output_paths.overlay_path:
        overlay_image.save(output_paths.overlay_path)
    if verbose and output_paths.overlay_path:
        print(f"🔳 Saved overlay: {os.path.basename(output_paths.overlay_path)}")


def _generate_segmentation_legend(
    results: SegmentationResults,
    config: Config,
    output_paths: OutputPaths,
    cmap,
    config_text: str,
    verbose: bool = True,
) -> None:
    """Generate segmentation map with legend."""
    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(
        results.labels_resized, cmap=cmap, vmin=0, vmax=results.n_clusters_used - 1
    )
    cbar = plt.colorbar(
        im, ax=ax, ticks=range(results.n_clusters_used), shrink=0.3, aspect=15
    )
    cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(results.n_clusters_used)])
    cbar.ax.tick_params(labelsize=6)

    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        config_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    plt.tight_layout()
    plt.savefig(
        output_paths.segmentation_legend,
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=DPI_SEGMENTATION,
    )
    plt.close()

    if verbose:
        print(f"📊 Saved: {os.path.basename(output_paths.segmentation_legend)}")
