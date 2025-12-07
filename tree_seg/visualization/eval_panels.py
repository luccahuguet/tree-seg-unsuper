"""Evaluation comparison panels and overlays."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import segmentation
from PIL import Image

from ..constants import HATCH_PATTERNS
from ..core.types import Config


def plot_evaluation_comparison(
    image: np.ndarray,
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    eval_results,  # EvaluationResults type
    dataset_class_names: dict | None = None,
    ignore_index: int = 255,
    output_path: str | None = None,
    image_id: str = "",
    config: Config | None = None,
    two_panel: bool = False,
    two_panel_opaque: bool = False,
    runtime_seconds: float | None = None,
) -> None:
    """
    Generate 2×2 comparison grid with overlay for laptop screen optimization.

    Layout:
    ┌──────────────┬──────────────┐
    │   Original   │ Ground Truth │
    ├──────────────┼──────────────┤
    │  Prediction  │ Edge Overlay │
    └──────────────┴──────────────┘
    + Config card and class legend below
    """
    two_panel = two_panel or (
        config is not None and getattr(config, "viz_two_panel", False)
    )
    two_panel_opaque = two_panel_opaque or (
        config is not None and getattr(config, "viz_two_panel_opaque", False)
    )

    if two_panel or two_panel_opaque:
        fig = plt.figure(figsize=(18, 9), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[0.32, 1.0, 1.0], wspace=0.05)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_gt = fig.add_subplot(gs[0, 1])
        ax_overlay = fig.add_subplot(gs[0, 2])
        ax_original = None
        ax_pred = None
    else:
        fig = plt.figure(figsize=(17, 10), constrained_layout=True)
        gs = fig.add_gridspec(
            2,
            3,
            width_ratios=[0.45, 1.1, 1.1],
            height_ratios=[1, 1],
            wspace=0.06,
            hspace=0.1,
        )
        ax_left = fig.add_subplot(gs[:, 0])
        ax_original = fig.add_subplot(gs[0, 1])
        ax_gt = fig.add_subplot(gs[0, 2])
        ax_pred = fig.add_subplot(gs[1, 1])
        ax_overlay = fig.add_subplot(gs[1, 2])

    max_vis_dim = 2600

    mapped_pred = pred_labels.copy()
    if eval_results.cluster_to_class_mapping:
        mapped_pred = np.zeros_like(pred_labels)
        for cluster_id, class_id in eval_results.cluster_to_class_mapping.items():
            mapped_pred[pred_labels == cluster_id] = class_id

    max_side = max(image.shape[:2])
    if max_side > max_vis_dim:
        scale = max_vis_dim / max_side
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image_vis = np.array(Image.fromarray(image).resize(new_size, Image.BILINEAR))
        mapped_pred_vis = np.array(
            Image.fromarray(mapped_pred.astype(np.uint8)).resize(
                new_size, Image.NEAREST
            )
        )
        gt_labels_vis = np.array(
            Image.fromarray(gt_labels.astype(np.uint8)).resize(new_size, Image.NEAREST)
        )
    else:
        image_vis = image
        mapped_pred_vis = mapped_pred
        gt_labels_vis = gt_labels

    gt_vis = gt_labels_vis.astype(float)
    gt_vis[gt_labels_vis == ignore_index] = np.nan

    cmap = plt.get_cmap("tab20").copy()
    cmap.set_bad(color="black")

    vmin = np.nanmin(gt_vis) if np.any(~np.isnan(gt_vis)) else 0
    vmax = np.nanmax(gt_vis) if np.any(~np.isnan(gt_vis)) else 20

    def class_color(class_id: int):
        norm_val = (class_id - vmin) / (vmax - vmin) if vmax > vmin else 0
        return cmap(norm_val)

    if ax_original is not None:
        ax_original.imshow(image_vis)
        ax_original.set_title("Original Image", fontsize=12, fontweight="bold")
        ax_original.axis("off")

    if config:
        method = (
            "Supervised"
            if config.supervised
            else f"{config.clustering_method}/{config.refine or 'none'}"
        )
        meta_lines = [
            f"Model: {config.model_display_name}",
            f"Method: {method}",
            f"Stride: {config.stride}",
            f"Tiling: {'Yes' if config.use_tiling else 'No'}",
        ]
        if config.refine:
            meta_lines.append(f"Refine: {config.refine}")
        k_used = eval_results.num_predicted_clusters
        unique_gt = np.unique(gt_labels_vis)
        unique_gt = unique_gt[unique_gt != ignore_index]
        num_gt_classes = len(unique_gt)
        if not config.auto_k:
            meta_lines.append(f"K: {k_used} (Fixed) | GT: {num_gt_classes} classes")
        elif config.auto_k:
            meta_lines.append(f"K: {k_used} (Auto) | GT: {num_gt_classes} classes")
        else:
            meta_lines.append(f"K: {k_used} | GT: {num_gt_classes} classes")
        if runtime_seconds is not None:
            minutes = int(runtime_seconds // 60)
            seconds_rem = runtime_seconds % 60
            meta_lines.append(f"Runtime: {minutes:d}m {seconds_rem:04.1f}s")
        meta_text = "\n".join(meta_lines)
    else:
        meta_text = ""

    if ax_pred is not None:
        ax_pred.imshow(
            mapped_pred_vis, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax
        )
        ax_pred.set_title(
            f"Prediction (K={eval_results.num_predicted_clusters})",
            fontsize=12,
            fontweight="bold",
        )
        ax_pred.axis("off")

    ax_gt.imshow(gt_vis, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax_gt.set_title("Ground Truth", fontsize=12, fontweight="bold")
    ax_gt.axis("off")

    if two_panel_opaque:
        ax_overlay.imshow(
            mapped_pred_vis, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax
        )
    else:
        ax_overlay.imshow(image_vis)

    unique_pred_classes = np.unique(mapped_pred_vis)
    unique_pred_classes = unique_pred_classes[unique_pred_classes != ignore_index]
    edge_width = getattr(config, "edge_width", 1.5) if config else 1.5

    if not getattr(config, "use_hatching", False) and not two_panel_opaque:
        overlay_rgba = np.zeros((*mapped_pred_vis.shape, 4), dtype=np.float32)
        for class_id in unique_pred_classes:
            mask = mapped_pred_vis == class_id
            if not mask.any():
                continue
            color = class_color(class_id)
            overlay_rgba[mask] = (*color[:3], 0.32)
        ax_overlay.imshow(overlay_rgba)

    for class_id in unique_pred_classes:
        mask = mapped_pred_vis == class_id
        if not mask.any():
            continue
        if not two_panel_opaque:
            color = class_color(class_id)[:3]
            ax_overlay.contour(
                mask.astype(int),
                levels=[0.5],
                colors=[color],
                linewidths=edge_width * 0.5,
                alpha=0.9,
            )
            if getattr(config, "use_hatching", False):
                hatch_pattern = HATCH_PATTERNS[class_id % len(HATCH_PATTERNS)]
                cs = ax_overlay.contourf(
                    mask.astype(int),
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=[hatch_pattern],
                )
                for collection in getattr(cs, "collections", []):
                    collection.set_facecolor("none")
                    collection.set_edgecolor(color)
                    collection.set_alpha(0.7)
                    collection.set_linewidth(0.0)

    if not two_panel_opaque:
        boundary_mask = segmentation.find_boundaries(mapped_pred_vis, mode="thick")
        boundary_img = np.zeros((*boundary_mask.shape, 4), dtype=np.float32)
        boundary_img[boundary_mask] = (1.0, 1.0, 1.0, 0.8)
        ax_overlay.imshow(boundary_img)

    ax_overlay.set_title(
        "Prediction Overlay" if not two_panel_opaque else "Prediction",
        fontsize=12,
        fontweight="bold",
    )
    ax_overlay.axis("off")

    ax_left.axis("off")
    cursor_y = 1.0

    # Metrics table
    metric_lines = [
        f"mIoU: {eval_results.miou:.2%}",
        f"Pixel Acc: {eval_results.pixel_accuracy:.2%}",
        f"Clusters: {eval_results.num_predicted_clusters} | Classes: {eval_results.num_classes}",
    ]
    metrics_text = "Evaluation Metrics\n" + "\n".join(metric_lines)
    ax_left.text(
        0.0,
        cursor_y,
        metrics_text,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            facecolor="white", alpha=0.9, edgecolor="black", boxstyle="round,pad=0.6"
        ),
        transform=ax_left.transAxes,
    )
    cursor_y -= 0.25

    # Metadata block
    if meta_text:
        ax_left.text(
            0.0,
            cursor_y,
            meta_text,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(
                facecolor="wheat",
                alpha=0.9,
                edgecolor="black",
                boxstyle="round,pad=1",
                linewidth=2,
            ),
            transform=ax_left.transAxes,
        )
        cursor_y -= 0.4

    if dataset_class_names:
        unique_classes = np.unique(gt_labels_vis)
        unique_classes = unique_classes[unique_classes != ignore_index]
        legend_patches = []
        for class_id in unique_classes:
            class_name = dataset_class_names.get(class_id, f"Class {class_id}")
            color = class_color(class_id)
            if (
                hasattr(eval_results, "per_class_pixel_accuracy")
                and class_id in eval_results.per_class_pixel_accuracy
            ):
                accuracy = eval_results.per_class_pixel_accuracy[class_id]
                label = f"{class_id}: {class_name} ({accuracy:.1%})"
            else:
                label = f"{class_id}: {class_name}"
            patch = mpatches.Patch(color=color, label=label)
            legend_patches.append(patch)
        legend = ax_left.legend(
            handles=legend_patches,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            ncol=1 if len(legend_patches) <= 6 else 2,
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=True,
            title="Ground Truth Classes (Pixel Acc)",
            title_fontsize=10,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.9)

    if output_path:
        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=150,
            facecolor="white",
        )
        plt.close()
    else:
        plt.show()
