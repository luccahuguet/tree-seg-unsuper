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
    
    if verbose:
        print(f"✅ Generated visualizations for {filename}")


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
) -> None:
    """
    Generate side-by-side comparison of prediction vs ground truth with metrics.
    
    Args:
        image: Original RGB image
        pred_labels: Predicted segmentation mask
        gt_labels: Ground truth segmentation mask
        eval_results: EvaluationResults object with metrics and mapping
        dataset_class_names: Dictionary mapping class IDs to names
        ignore_index: Value to ignore in ground truth
        output_path: Path to save the figure
        image_id: Identifier for the image
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Prepare ground truth first to get color range
    # Mask out ignored pixels
    gt_vis = gt_labels.copy().astype(float)
    gt_vis[gt_labels == ignore_index] = np.nan
    
    # Create colormap with black background for NaNs
    cmap = plt.get_cmap("tab20").copy()
    cmap.set_bad(color="black")
    
    # Calculate vmin/vmax from ground truth for consistent coloring
    vmin = np.nanmin(gt_vis) if np.any(~np.isnan(gt_vis)) else 0
    vmax = np.nanmax(gt_vis) if np.any(~np.isnan(gt_vis)) else 20

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Predicted segmentation
    # Recolor predictions to match ground truth classes using Hungarian matching
    if eval_results.cluster_to_class_mapping:
        # Create a mapped prediction array
        mapped_pred = np.zeros_like(pred_labels)
        for cluster_id, class_id in eval_results.cluster_to_class_mapping.items():
            mapped_pred[pred_labels == cluster_id] = class_id
        
        axes[1].imshow(mapped_pred, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Prediction (Matched to GT, K={eval_results.num_predicted_clusters})")
    else:
        # Fallback if no mapping available
        axes[1].imshow(pred_labels, cmap="tab20")
        axes[1].set_title(f"Prediction (K={eval_results.num_predicted_clusters})")
    axes[1].axis("off")

    # Ground truth
    axes[2].imshow(gt_vis, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")
    
    # Add legend for Ground Truth classes
    if dataset_class_names:
        # Get unique classes present in the GT (excluding ignore)
        unique_classes = np.unique(gt_labels)
        unique_classes = unique_classes[unique_classes != ignore_index]
        
        legend_patches = []
        for class_id in unique_classes:
            class_name = dataset_class_names.get(class_id, f"Class {class_id}")
            
            # Get color from colormap
            # Normalize class_id to [0, 1] for cmap
            norm_val = (class_id - vmin) / (vmax - vmin) if vmax > vmin else 0
            color = cmap(norm_val)
            
            patch = mpatches.Patch(color=color, label=f"{class_id}: {class_name}")
            legend_patches.append(patch)
        
        # Add legend to the right of the plot
        axes[2].legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    # Add metrics as title
    fig.suptitle(
        f"{image_id} | mIoU: {eval_results.miou:.3f} | Pixel Acc: {eval_results.pixel_accuracy:.3f}",
        fontsize=14,
        y=0.95,
    )

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.show()
