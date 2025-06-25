"""
Visualization and plotting utilities for tree segmentation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Polygon
from scipy import ndimage

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

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
    """
    if image_np is None or labels_resized is None:
        print(f"Skipping output generation for {image_path} due to processing error.")
        return

    alpha = (10 - overlay_ratio) / 10.0
    filename = os.path.basename(image_path)

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
    im = ax.imshow(labels_resized, cmap=cmap, vmin=0, vmax=n_clusters - 1)
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

    # Generate overlay visualization
    norm = colors.Normalize(vmin=0, vmax=n_clusters - 1)
    segmentation_rgb = cmap(norm(labels_resized))[:, :, :3]
    segmentation_rgb = (segmentation_rgb * 255).astype(np.uint8)
    overlay = (alpha * image_np + (1 - alpha) * segmentation_rgb).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.axis("off")
    ax.text(
        0.02, 0.98, config_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.tight_layout()
    overlay_path = os.path.join(output_dir, f"{output_prefix}_overlay.png")
    plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved overlay: {overlay_path}")

    # Generate NEW edge overlay visualization with colored borders and hatch patterns
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)

    # Define even more spaced hatch patterns - single characters for maximum spacing
    # Removed '-' and 'o' as they don't work properly as hatch patterns
    hatch_patterns = ['/', '\\', '|', '.', '~', '*', 'x']

    # Create a custom color function to prioritize bright colors
    def get_cluster_color(cluster_id, n_clusters, cmap):
        """Get a bright color for cluster, prioritizing high contrast colors."""
        # Define a set of bright, high-contrast colors that work well against foliage
        bright_colors = [
            (1.0, 0.0, 0.0),    # Bright Red
            (0.0, 0.0, 1.0),    # Bright Blue
            (1.0, 0.5, 0.0),    # Bright Orange
            (1.0, 0.0, 1.0),    # Bright Magenta
            (0.0, 1.0, 1.0),    # Bright Cyan
            (1.0, 1.0, 0.0),    # Bright Yellow
            (0.5, 0.0, 1.0),    # Bright Purple
            (1.0, 0.0, 0.5),    # Bright Pink
            (0.0, 0.5, 1.0),    # Bright Sky Blue
            (1.0, 0.3, 0.7),    # Bright Rose
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

    # Keep track of successfully drawn clusters for fallback
    successfully_drawn = set()

    # Create colored borders and hatch patterns for each cluster
    for cluster_id in range(n_clusters):
        # Create mask for this cluster
        cluster_mask = (labels_resized == cluster_id)

        if not cluster_mask.any():
            continue

        # Get cluster color, prioritizing bright colors
        cluster_color = get_cluster_color(cluster_id, n_clusters, cmap)

        # Create contour for this cluster with hatch pattern
        if HAS_SKIMAGE:
            try:
                # Find contours for this cluster
                contours = measure.find_contours(cluster_mask.astype(float), 0.5)

                # Smart contour selection to avoid overlap while preserving important regions
                if contours:
                    # Sort contours by length (larger contours first)
                    sorted_contours = sorted(contours, key=len, reverse=True)

                    drawn_contours = []
                    contours_drawn = 0

                    for contour in sorted_contours:
                        if len(contour) > 15:  # Only consider substantial contours
                            # Check if this contour is too close to already drawn ones
                            too_close = False
                            contour_center = np.mean(contour, axis=0)

                            for drawn_contour in drawn_contours:
                                drawn_center = np.mean(drawn_contour, axis=0)
                                distance = np.sqrt(np.sum((contour_center - drawn_center)**2))

                                # If centers are too close, skip this contour
                                min_distance = max(50, len(contour) * 0.3)  # Adaptive minimum distance
                                if distance < min_distance:
                                    too_close = True
                                    break

                            # Only draw if not too close to existing contours
                            if not too_close:
                                # Flip coordinates (find_contours returns row, col)
                                contour_flipped = np.fliplr(contour)

                                                                # Create polygon patch with hatch pattern
                                hatch_pattern = hatch_patterns[cluster_id % len(hatch_patterns)]
                                polygon = Polygon(contour_flipped, closed=True,
                                                fill=False,
                                                edgecolor=cluster_color,
                                                linewidth=edge_width,  # Use configurable edge width
                                                hatch=hatch_pattern,
                                                alpha=0.6)  # Lower alpha to reduce visual clash
                                ax.add_patch(polygon)
                                drawn_contours.append(contour)
                                contours_drawn += 1

                                # Limit number of contours per cluster to avoid clutter
                                if contours_drawn >= 3:  # Max 3 contours per cluster
                                    break

                    # Mark as successfully drawn if we drew any contours
                    if contours_drawn > 0:
                        successfully_drawn.add(cluster_id)

            except:
                pass

        # Fallback for clusters that couldn't be drawn with contours
        if cluster_id not in successfully_drawn:
            cluster_edges = detect_segmentation_edges(cluster_mask.astype(int), edge_width=edge_width)
            border_y, border_x = np.where(cluster_edges)
            if len(border_y) > 0:
                ax.scatter(border_x, border_y, c=[cluster_color], s=6, alpha=0.9)

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
        hatch_pattern = hatch_patterns[cluster_id % len(hatch_patterns)]
        legend_elements.append(plt.Line2D([0], [0], color=cluster_color, lw=edge_width,
                                         label=f'Cluster {cluster_id} {hatch_pattern}'))

    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=6,
                      framealpha=0.7, fancybox=True, shadow=True, ncol=1 if n_clusters <= 6 else 2)
    legend.get_frame().set_facecolor('white')

    plt.tight_layout()
    edge_overlay_path = os.path.join(output_dir, f"{output_prefix}_edge_overlay.png")
    plt.savefig(edge_overlay_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved colored edge overlay with hatch patterns: {edge_overlay_path}")

    # Generate side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")
    im = axes[1].imshow(labels_resized, cmap=cmap, vmin=0, vmax=n_clusters - 1)
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