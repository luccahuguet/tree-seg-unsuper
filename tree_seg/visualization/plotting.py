"""
Modern visualization module using the new architecture.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from ..core.types import Config, SegmentationResults, OutputPaths
from ..utils.config import get_config_text
from ..constants import (
    HATCH_PATTERNS,
    DPI_SEGMENTATION,
    DPI_EDGE_OVERLAY,
    DPI_SIDE_BY_SIDE,
)
# Removed circular import - we'll reimplement detect_segmentation_edges if needed


def generate_visualizations(results: SegmentationResults, config: Config, output_paths: OutputPaths) -> None:
    """
    Generate all visualizations using modern architecture.
    
    Args:
        results: Segmentation results
        config: Configuration object
        output_paths: Output file paths
    """
    verbose = getattr(config, 'verbose', True)
    if not results.success:
        if verbose:
            print("⚠️ Skipping visualization - segmentation failed")
        return
    
    # Get config text for overlays
    filename = os.path.basename(results.image_path)
    config_text = get_config_text(
        results.n_clusters_used,
        config.overlay_ratio, 
        config.stride, 
        config.model_display_name, 
        filename, 
        config.version, 
        config.edge_width,
        config.elbow_threshold if config.auto_k else None,
        n_clusters_requested=results.n_clusters_requested
    )
    
    # Choose colormap based on number of clusters
    n_clusters = results.n_clusters_used
    if n_clusters <= 10:
        cmap = plt.get_cmap("Set1", n_clusters)
    elif n_clusters <= 20:
        cmap = plt.get_cmap("Dark2", n_clusters)
    else:
        cmap = plt.get_cmap("viridis", n_clusters)
    
    # Generate each visualization
    _generate_segmentation_legend(results, config, output_paths, cmap, config_text, verbose)
    _generate_edge_overlay(results, config, output_paths, cmap, config_text, verbose)
    _generate_side_by_side(results, config, output_paths, cmap, config_text, verbose)
    _generate_overlay(results, config, output_paths, verbose)
    
    if verbose:
        print(f"✅ Generated visualizations for {filename}")


def _generate_overlay(results: SegmentationResults, config: Config, output_paths: OutputPaths, verbose: bool = True) -> None:
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


def _generate_segmentation_legend(results: SegmentationResults, config: Config, 
                                 output_paths: OutputPaths, cmap, config_text: str, verbose: bool = True) -> None:
    """Generate segmentation map with legend."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(results.labels_resized, cmap=cmap, vmin=0, vmax=results.n_clusters_used - 1)
    cbar = plt.colorbar(im, ax=ax, ticks=range(results.n_clusters_used), shrink=0.3, aspect=15)
    cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(results.n_clusters_used)])
    cbar.ax.tick_params(labelsize=6)
    
    ax.axis("off")
    ax.text(
        0.02, 0.98, config_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    plt.tight_layout()
    plt.savefig(output_paths.segmentation_legend, bbox_inches="tight", pad_inches=0.1, dpi=DPI_SEGMENTATION)
    plt.close()
    
    if verbose:
        print(f"📊 Saved: {os.path.basename(output_paths.segmentation_legend)}")


def _generate_edge_overlay(results: SegmentationResults, config: Config,
                          output_paths: OutputPaths, cmap, config_text: str, verbose: bool = True) -> None:
    """Generate edge overlay visualization."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(results.image_np)
    
    # Define hatch patterns
    hatch_patterns = HATCH_PATTERNS
    
    # Get cluster colors (reuse from original plotting.py)
    def get_cluster_color(cluster_id, n_clusters, cmap):
        bright_colors = [
            (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (1.0, 0.0, 0.5),
            (0.0, 1.0, 0.0), (1.0, 0.3, 0.7)
        ]
        
        if cluster_id < len(bright_colors):
            return bright_colors[cluster_id]
        else:
            base_color = cmap(cluster_id / (n_clusters - 1))[:3]
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
    
    # Create colored borders with hatching
    for cluster_id in range(results.n_clusters_used):
        cluster_mask = (results.labels_resized == cluster_id)
        
        if not cluster_mask.any():
            continue
        
        cluster_color = get_cluster_color(cluster_id, results.n_clusters_used, cmap)
        
        # Draw borders
        ax.contour(cluster_mask.astype(int), levels=[0.5], colors=[cluster_color],
                  linewidths=config.edge_width, alpha=0.9)
        ax.contour(cluster_mask.astype(int), levels=[0.5], colors=[cluster_color],
                  linewidths=config.edge_width + 1, alpha=0.6)
        ax.contour(cluster_mask.astype(int), levels=[0.5], colors='white',
                  linewidths=1, alpha=0.3)
        
        # Add hatching if enabled
        if config.use_hatching:
            hatch_pattern = hatch_patterns[cluster_id % len(hatch_patterns)]
            cs = ax.contourf(cluster_mask.astype(int), levels=[0.5, 1.5], colors='none',
                           hatches=[hatch_pattern])
            
            # Handle both old and new matplotlib API
            collections = getattr(cs, 'collections', [])
            for collection in collections:
                collection.set_facecolor('none')
                collection.set_edgecolor(cluster_color)
                collection.set_alpha(0.7)
                collection.set_linewidth(0.)
    
    ax.axis("off")
    ax.text(
        0.02, 0.98, config_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    # Add legend
    legend_elements = []
    for cluster_id in range(results.n_clusters_used):
        cluster_color = get_cluster_color(cluster_id, results.n_clusters_used, cmap)
        if config.use_hatching:
            hatch_pattern = hatch_patterns[cluster_id % len(hatch_patterns)]
            legend_elements.append(plt.Line2D([0], [0], color=cluster_color, lw=config.edge_width,
                                           label=f'Cluster {cluster_id} {hatch_pattern}'))
        else:
            legend_elements.append(plt.Line2D([0], [0], color=cluster_color, lw=config.edge_width,
                                           label=f'Cluster {cluster_id}'))
    
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=6,
                      framealpha=0.7, fancybox=True, shadow=True, 
                      ncol=1 if results.n_clusters_used <= 6 else 2)
    legend.get_frame().set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_paths.edge_overlay, bbox_inches="tight", pad_inches=0.1, dpi=DPI_EDGE_OVERLAY)
    plt.close()
    
    hatching_text = "with hatching" if config.use_hatching else "borders only"
    if verbose:
        print(f"🔳 Saved edge overlay ({hatching_text}): {os.path.basename(output_paths.edge_overlay)}")


def _generate_side_by_side(results: SegmentationResults, config: Config,
                          output_paths: OutputPaths, cmap, config_text: str, verbose: bool = True) -> None:
    """Generate side-by-side comparison."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        axes[0].imshow(results.image_np)
        axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0].axis("off")
        
        # Segmentation map
        im = axes[1].imshow(results.labels_resized, cmap=cmap, vmin=0, vmax=results.n_clusters_used - 1)
        axes[1].set_title("Segmentation Map", fontsize=12, fontweight='bold')
        axes[1].axis("off")
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes[1], ticks=range(results.n_clusters_used), shrink=0.4, aspect=20)
        cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(results.n_clusters_used)])
        cbar.ax.tick_params(labelsize=8)
        
        # Add config text
        axes[1].text(
            0.02, 0.98, config_text,
            transform=axes[1].transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3')
        )
        
        plt.tight_layout(pad=2.0)
        plt.savefig(output_paths.side_by_side, bbox_inches="tight", pad_inches=0.2, dpi=DPI_SIDE_BY_SIDE,
                   facecolor='white', edgecolor='none')
        plt.close()
        
        if verbose:
            print(f"📊 Saved side-by-side: {os.path.basename(output_paths.side_by_side)}")
        
    except Exception as e:
        if verbose:
            print(f"⚠️ Could not generate side-by-side image: {e}")
        try:
            plt.close('all')
        except Exception:
            pass


def plot_evaluation_comparison(
    image: np.ndarray,
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    eval_results,  # EvaluationResults type
    dataset_class_names: dict = None,
    ignore_index: int = 255,
    output_path: str = None,
    image_id: str = "",
    config: Config = None,
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

    Args:
        image: Original RGB image
        pred_labels: Predicted segmentation mask
        gt_labels: Ground truth segmentation mask
        eval_results: EvaluationResults object with metrics and mapping
        dataset_class_names: Dictionary mapping class IDs to names
        ignore_index: Value to ignore in ground truth
        output_path: Path to save the figure
        image_id: Identifier for the image
        config: Configuration object for metadata legend
        two_panel: If True, show only GT and overlay to maximize image size
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from skimage import segmentation
    from PIL import Image

    two_panel = two_panel or (config is not None and getattr(config, "viz_two_panel", False))
    two_panel_opaque = two_panel_opaque or (config is not None and getattr(config, "viz_two_panel_opaque", False))

    if two_panel or two_panel_opaque:
        # Compact layout: left column for text/legend, GT and overlay side-by-side
        fig = plt.figure(figsize=(18, 9), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[0.32, 1.0, 1.0], wspace=0.05)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_gt = fig.add_subplot(gs[0, 1])
        ax_overlay = fig.add_subplot(gs[0, 2])
        ax_original = None
        ax_pred = None
    else:
        # Default layout: left column for text/legend, right two columns for 2×2 images
        fig = plt.figure(figsize=(17, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, width_ratios=[0.45, 1.1, 1.1], height_ratios=[1, 1], wspace=0.06, hspace=0.1)
        ax_left = fig.add_subplot(gs[:, 0])
        ax_original = fig.add_subplot(gs[0, 1])
        ax_gt = fig.add_subplot(gs[0, 2])
        ax_pred = fig.add_subplot(gs[1, 1])
        ax_overlay = fig.add_subplot(gs[1, 2])

    # To keep memory manageable on large images, downscale visuals for plotting
    max_vis_dim = 2600

    # Map predictions to GT classes (Hungarian) before resizing
    mapped_pred = pred_labels.copy()
    if eval_results.cluster_to_class_mapping:
        mapped_pred = np.zeros_like(pred_labels)
        for cluster_id, class_id in eval_results.cluster_to_class_mapping.items():
            mapped_pred[pred_labels == cluster_id] = class_id

    # Downscale for visualization if needed (bilinear for image, nearest for labels)
    max_side = max(image.shape[:2])
    if max_side > max_vis_dim:
        scale = max_vis_dim / max_side
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image_vis = np.array(Image.fromarray(image).resize(new_size, Image.BILINEAR))
        mapped_pred_vis = np.array(
            Image.fromarray(mapped_pred.astype(np.uint8)).resize(new_size, Image.NEAREST)
        )
        gt_labels_vis = np.array(
            Image.fromarray(gt_labels.astype(np.uint8)).resize(new_size, Image.NEAREST)
        )
    else:
        image_vis = image
        mapped_pred_vis = mapped_pred
        gt_labels_vis = gt_labels

    # Prepare ground truth first to get color range
    gt_vis = gt_labels_vis.astype(float)
    gt_vis[gt_labels_vis == ignore_index] = np.nan
    
    # Create colormap with black background for NaNs
    cmap = plt.get_cmap("tab20").copy()
    cmap.set_bad(color="black")
    
    # Calculate vmin/vmax from ground truth for consistent coloring
    vmin = np.nanmin(gt_vis) if np.any(~np.isnan(gt_vis)) else 0
    vmax = np.nanmax(gt_vis) if np.any(~np.isnan(gt_vis)) else 20

    # Helper for consistent colors
    def class_color(class_id: int):
        norm_val = (class_id - vmin) / (vmax - vmin) if vmax > vmin else 0
        return cmap(norm_val)

    # Original Image with Metadata
    if ax_original is not None:
        ax_original.imshow(image_vis)
        ax_original.set_title("Original Image", fontsize=12, fontweight="bold")
        ax_original.axis("off")
    
    # Add metadata legend if config is provided
    if config:
        # Construct metadata text
        meta_lines = [
            f"Model: {config.model_display_name}",
            f"Method: {config.version}",
            f"Stride: {config.stride}",
            f"Tiling: {'Yes' if config.use_tiling else 'No'}",
        ]

        if config.refine:
            meta_lines.append(f"Refine: {config.refine}")

        # Add K info with GT comparison
        k_used = eval_results.num_predicted_clusters
        # Count GT classes (excluding ignore index) - use downscaled version for efficiency
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

        # Position metadata in column with GT legend
        pass  # Metadata card is rendered in bottom row; skip overlay next to original

    # Predicted segmentation
    if ax_pred is not None:
        ax_pred.imshow(mapped_pred_vis, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"Prediction (K={eval_results.num_predicted_clusters})", fontsize=12, fontweight="bold")
        ax_pred.axis("off")

    # Ground truth
    ax_gt.imshow(gt_vis, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax_gt.set_title("Ground Truth", fontsize=12, fontweight="bold")
    ax_gt.axis("off")

    # Edge overlay or opaque prediction view
    if two_panel_opaque:
        ax_overlay.imshow(mapped_pred_vis, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    else:
        ax_overlay.imshow(image_vis)

    unique_pred_classes = np.unique(mapped_pred_vis)
    unique_pred_classes = unique_pred_classes[unique_pred_classes != ignore_index]

    edge_width = getattr(config, "edge_width", 1.5)

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

    ax_overlay.set_title("Prediction Overlay" if not two_panel_opaque else "Prediction", fontsize=12, fontweight="bold")
    ax_overlay.axis("off")
    
    # Add legend for Ground Truth classes
    # Left column content: title, metrics, config, and class legend
    ax_left.axis("off")
    cursor_y = 1.0
    title_text = f"{image_id}\nmIoU: {eval_results.miou:.3f}\nPixel Acc: {eval_results.pixel_accuracy:.3f}"
    ax_left.text(
        0.0, cursor_y, title_text,
        ha="left", va="top", fontsize=14, fontweight="bold", transform=ax_left.transAxes,
    )
    cursor_y -= 0.2
    if config:
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

            # Add per-class pixel accuracy if available
            if hasattr(eval_results, 'per_class_pixel_accuracy') and class_id in eval_results.per_class_pixel_accuracy:
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
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150, facecolor="white")
        plt.close()
    else:
        plt.show()
