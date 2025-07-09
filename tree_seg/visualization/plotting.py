"""
Visualization and plotting utilities for tree segmentation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Polygon
from scipy import ndimage

# Removed skimage dependency - using matplotlib's built-in contour functions instead

from ..utils.config import get_config_text


def detect_segmentation_edges(labels, edge_width=2):
    """
    Detect edges between different segmentation regions.

    Args:
        labels: 2D array of segmentation labels
        edge_width: Width of the edge lines in pixels

    Returns:
        edges: Binary mask where edges are True
    """
    # Use Sobel filter to detect edges between different regions
    edges_x = ndimage.sobel(labels.astype(float), axis=0)
    edges_y = ndimage.sobel(labels.astype(float), axis=1)
    edges = np.sqrt(edges_x**2 + edges_y**2) > 0

    # Dilate edges to make them more visible
    if edge_width > 1:
        structure = ndimage.generate_binary_structure(2, 2)
        edges = ndimage.binary_dilation(edges, structure=structure, iterations=edge_width-1)

    return edges


def generate_outputs(
    image_np,
    labels_resized,
    output_prefix,
    output_dir,
    n_clusters,
    overlay_ratio,
    stride,
    model_name,
    image_path,
    version,
    edge_width=2,
    use_hatching=False,
):
    """
    Generate visualization outputs for segmentation results.

    Args:
        image_np: Original image as numpy array
        labels_resized: Segmentation labels resized to original image size
        output_prefix: Prefix for output filenames
        output_dir: Output directory
        n_clusters: Number of clusters
        overlay_ratio: Overlay transparency ratio (1-10)
        stride: Model stride
        model_name: Model name
        image_path: Path to original image
        version: Model version
        edge_width: Width of edge lines in pixels for edge overlay visualization
        use_hatching: Whether to add hatch patterns to regions (borders are always shown)
    """
    if image_np is None or labels_resized is None:
        print(f"Skipping output generation for {image_path} due to processing error.")
        return

    alpha = (10 - overlay_ratio) / 10.0
    filename = os.path.basename(image_path)

    # Use labels directly without filtering
    labels_to_plot = np.copy(labels_resized)

    # Choose colormap based on number of clusters - avoid green-heavy colormaps
    if n_clusters <= 10:
        # Use Set1 colormap which has good contrast and less green
        cmap = plt.get_cmap("Set1", n_clusters)
    elif n_clusters <= 20:
        # Use Dark2 which has better contrast than tab20
        cmap = plt.get_cmap("Dark2", n_clusters)
    else:
        # Use viridis but we'll filter out green colors below
        cmap = plt.get_cmap("viridis", n_clusters)

    config_text = get_config_text(n_clusters, overlay_ratio, stride, model_name, filename, version, edge_width)

    # Generate segmentation legend visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(labels_to_plot, cmap=cmap, vmin=0, vmax=n_clusters - 1)
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_clusters), shrink=0.3, aspect=15)
    cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
    cbar.ax.tick_params(labelsize=6)
    ax.axis("off")
    ax.text(
        0.02, 0.98, config_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.tight_layout()
    legend_path = os.path.join(output_dir, f"{output_prefix}_segmentation_legend.png")
    plt.savefig(legend_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved segmentation with legend: {legend_path}")

    # Note: Old overlay visualization removed - keeping only edge overlay and side-by-side comparison

    # Generate NEW edge overlay visualization with colored borders and hatch patterns
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)

    # Define even more spaced hatch patterns - single characters for maximum spacing
    # Removed 'o' as they don't work properly as hatch patterns
    hatch_patterns = ['/', '\\', '|', '.', 'x', '-']

    # Create a custom color function to prioritize bright colors with better visibility
    def get_cluster_color(cluster_id, n_clusters, cmap):
        """Get a bright color for cluster, prioritizing high contrast colors."""
        # Define a set of bright, high-contrast colors optimized for border visibility
        # These colors are chosen to be highly visible against natural backgrounds
        bright_colors = [
            (1.0, 0.0, 0.0),    # Bright Red - excellent visibility
            (0.0, 0.0, 1.0),    # Bright Blue - good contrast
            (1.0, 1.0, 0.0),    # Bright Yellow - very visible
            (1.0, 0.0, 1.0),    # Bright Magenta - high contrast
            (0.0, 1.0, 1.0),    # Bright Cyan - good visibility
            (1.0, 0.5, 0.0),    # Bright Orange - excellent visibility
            (0.5, 0.0, 1.0),    # Bright Purple - good contrast
            (1.0, 0.0, 0.5),    # Bright Pink - high visibility
            (0.0, 1.0, 0.0),    # Bright Green - good for borders
            (1.0, 0.3, 0.7),    # Bright Rose - excellent visibility
        ]

        # Use bright colors first, then fall back to colormap
        if cluster_id < len(bright_colors):
            return bright_colors[cluster_id]
        else:
            # For additional clusters, boost the brightness of colormap colors
            base_color = cmap(cluster_id / (n_clusters - 1))[:3]
            r, g, b = base_color

            # Boost saturation and brightness
            # Convert to HSV-like adjustment
            max_val = max(r, g, b)
            min_val = min(r, g, b)

            if max_val > 0:
                # Increase contrast and brightness
                scale = 1.3  # Brightness boost
                r = min(1.0, r * scale)
                g = min(1.0, g * scale)
                b = min(1.0, b * scale)

                # If still too dim, force it to be brighter
                if max(r, g, b) < 0.7:
                    dominant = max(r, g, b)
                    if r == dominant:
                        r = 1.0
                    elif g == dominant:
                        g = 1.0
                    else:
                        b = 1.0

            return (r, g, b)

    # Create colored borders with improved visibility
    for cluster_id in range(n_clusters):
        # Create mask for this cluster from the filtered labels
        cluster_mask = (labels_to_plot == cluster_id)

        if not cluster_mask.any():
            continue

        # Get cluster color, prioritizing bright colors
        cluster_color = get_cluster_color(cluster_id, n_clusters, cmap)
        
        # Create a more visible border by using multiple contour levels
        # This ensures borders are visible even with thin regions
        ax.contour(cluster_mask.astype(int), levels=[0.5], colors=[cluster_color],
                  linewidths=edge_width, alpha=0.9)
        
        # Add a second, slightly thicker border for better visibility
        ax.contour(cluster_mask.astype(int), levels=[0.5], colors=[cluster_color],
                  linewidths=edge_width + 1, alpha=0.6)
        
        # Add a white outline for maximum visibility (very thin)
        ax.contour(cluster_mask.astype(int), levels=[0.5], colors='white',
                  linewidths=1, alpha=0.3)

        # Add hatch pattern if enabled
        if use_hatching:
            hatch_pattern = hatch_patterns[cluster_id % len(hatch_patterns)]
            cs = ax.contourf(cluster_mask.astype(int), levels=[0.5, 1.5], colors='none',
                       hatches=[hatch_pattern])

            # Set the hatch color and alpha to match the contour
            for collection in cs.collections: # type: ignore
                collection.set_facecolor('none')
                collection.set_edgecolor(cluster_color)
                collection.set_alpha(0.25)  # Much more transparent hatching
                # Do not draw the patch border, only the hatch
                collection.set_linewidth(0.)

    ax.axis("off")

    # Add config text
    ax.text(
        0.02, 0.98, config_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # Add small legend
    legend_elements = []
    for cluster_id in range(n_clusters):
        cluster_color = get_cluster_color(cluster_id, n_clusters, cmap)
        if use_hatching:
            hatch_pattern = hatch_patterns[cluster_id % len(hatch_patterns)]
            legend_elements.append(plt.Line2D([0], [0], color=cluster_color, lw=edge_width,
                                             label=f'Cluster {cluster_id} {hatch_pattern}'))
        else:
            legend_elements.append(plt.Line2D([0], [0], color=cluster_color, lw=edge_width,
                                             label=f'Cluster {cluster_id}'))

    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=6,
                      framealpha=0.7, fancybox=True, shadow=True, ncol=1 if n_clusters <= 6 else 2)
    legend.get_frame().set_facecolor('white')

    plt.tight_layout()
    edge_overlay_path = os.path.join(output_dir, f"{output_prefix}_edge_overlay.png")
    plt.savefig(edge_overlay_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    hatching_text = "with hatch patterns" if use_hatching else "with borders only"
    print(f"Saved colored edge overlay {hatching_text}: {edge_overlay_path}")

    # Generate side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")
    im = axes[1].imshow(labels_to_plot, cmap=cmap, vmin=0, vmax=n_clusters - 1)
    axes[1].set_title("Segmentation Map", fontsize=12)
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], ticks=range(n_clusters), shrink=0.3, aspect=15)
    cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
    cbar.ax.tick_params(labelsize=6)
    axes[1].text(
        0.02, 0.98, config_text,
        transform=axes[1].transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.tight_layout()
    side_by_side_path = os.path.join(output_dir, f"{output_prefix}_side_by_side.png")
    plt.savefig(side_by_side_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved side-by-side image: {side_by_side_path}")