"""Overlay rendering helpers (edge overlays, side-by-side, eval grids)."""

from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt

from ..constants import (
    HATCH_PATTERNS,
    DPI_EDGE_OVERLAY,
    DPI_SIDE_BY_SIDE,
)


def generate_edge_overlay(
    results, config, output_paths, cmap, config_text: str, verbose: bool = True
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(results.image_np)

    hatch_patterns = HATCH_PATTERNS

    def get_cluster_color(cluster_id, n_clusters, cmap_local):
        bright_colors = [
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.5, 0.0),
            (0.5, 0.0, 1.0),
            (1.0, 0.0, 0.5),
            (0.0, 1.0, 0.0),
            (1.0, 0.3, 0.7),
        ]
        if cluster_id < len(bright_colors):
            return bright_colors[cluster_id]
        base_color = cmap_local(cluster_id / (n_clusters - 1))[:3]
        r, g, b = base_color
        scale = 1.3
        r, g, b = min(1.0, r * scale), min(1.0, g * scale), min(1.0, b * scale)
        if max(r, g, b) < 0.7:
            dominant = max(r, g, b)
            if r == dominant:
                r = 1.0
            elif g == dominant:
                g = 1.0
            else:
                b = 1.0
        return (r, g, b)

    for cluster_id in range(results.n_clusters_used):
        cluster_mask = results.labels_resized == cluster_id
        if not cluster_mask.any():
            continue
        cluster_color = get_cluster_color(cluster_id, results.n_clusters_used, cmap)
        ax.contour(
            cluster_mask.astype(int),
            levels=[0.5],
            colors=[cluster_color],
            linewidths=config.edge_width,
            alpha=0.9,
        )
        ax.contour(
            cluster_mask.astype(int),
            levels=[0.5],
            colors="white",
            linewidths=0.5,
            alpha=0.3,
        )
        if config.use_hatching:
            hatch_pattern = hatch_patterns[cluster_id % len(hatch_patterns)]
            cs = ax.contourf(
                cluster_mask.astype(int),
                levels=[0.5, 1.5],
                colors="none",
                hatches=[hatch_pattern],
            )
            for collection in getattr(cs, "collections", []):
                collection.set_facecolor("none")
                collection.set_edgecolor(cluster_color)
                collection.set_alpha(0.7)
                collection.set_linewidth(0.0)

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

    legend_elements = []
    for cluster_id in range(results.n_clusters_used):
        cluster_color = get_cluster_color(cluster_id, results.n_clusters_used, cmap)
        if config.use_hatching:
            hatch_pattern = hatch_patterns[cluster_id % len(hatch_patterns)]
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=cluster_color,
                    lw=config.edge_width,
                    label=f"Cluster {cluster_id} {hatch_pattern}",
                )
            )
        else:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=cluster_color,
                    lw=config.edge_width,
                    label=f"Cluster {cluster_id}",
                )
            )
    legend = ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=6,
        framealpha=0.7,
        fancybox=True,
        shadow=True,
        ncol=1 if results.n_clusters_used <= 6 else 2,
    )
    legend.get_frame().set_facecolor("white")

    plt.tight_layout()
    plt.savefig(
        output_paths.edge_overlay,
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=DPI_EDGE_OVERLAY,
    )
    plt.close()

    hatching_text = "with hatching" if config.use_hatching else "borders only"
    if verbose:
        print(
            f"🔳 Saved edge overlay ({hatching_text}): {os.path.basename(output_paths.edge_overlay)}"
        )


def generate_side_by_side(
    results, config, output_paths, cmap, config_text: str, verbose: bool = True
) -> None:
    """Generate side-by-side comparison."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(results.image_np)
        axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        im = axes[1].imshow(
            results.labels_resized, cmap=cmap, vmin=0, vmax=results.n_clusters_used - 1
        )
        axes[1].set_title("Segmentation Map", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        cbar = fig.colorbar(
            im, ax=axes[1], ticks=range(results.n_clusters_used), shrink=0.4, aspect=20
        )
        cbar.ax.set_yticklabels(
            [f"Cluster {i}" for i in range(results.n_clusters_used)]
        )
        cbar.ax.tick_params(labelsize=8)
        axes[1].text(
            0.02,
            0.98,
            config_text,
            transform=axes[1].transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="black",
                boxstyle="round,pad=0.3",
            ),
        )
        plt.tight_layout(pad=2.0)
        plt.savefig(
            output_paths.side_by_side,
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=DPI_SIDE_BY_SIDE,
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        if verbose:
            print(
                f"📊 Saved side-by-side: {os.path.basename(output_paths.side_by_side)}"
            )
    except Exception as exc:
        if verbose:
            print(f"⚠️ Could not generate side-by-side image: {exc}")
        try:
            plt.close("all")
        except Exception:
            pass
