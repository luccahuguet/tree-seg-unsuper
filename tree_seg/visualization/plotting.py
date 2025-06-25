"""
Visualization and plotting utilities for tree segmentation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import ndimage

from ..utils.config import get_config_text


def detect_segmentation_edges(labels, edge_width=8):
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
    edge_width=8,
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

    # Choose colormap based on number of clusters
    if n_clusters <= 10:
        cmap = plt.get_cmap("tab10", n_clusters)
    elif n_clusters <= 20:
        cmap = plt.get_cmap("tab20", n_clusters)
    else:
        cmap = plt.get_cmap("gist_ncar", n_clusters)

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
    
    # Create colored borders and hatch patterns for each cluster
    for cluster_id in range(n_clusters):
        # Create mask for this cluster
        cluster_mask = (labels_resized == cluster_id)
        
        if not cluster_mask.any():
            continue
            
        # Get cluster color from colormap
        cluster_color = cmap(cluster_id / (n_clusters - 1))[:3]  # RGB only
        
        # Find edges for this specific cluster
        cluster_edges = detect_segmentation_edges(cluster_mask.astype(int), edge_width=edge_width)
        
        # Create hatch pattern within the region (but not too dense)
        # Sample points for hatching to avoid filling everything
        y_coords, x_coords = np.where(cluster_mask)
        if len(y_coords) > 100:  # Only add hatching if region is large enough
            # Create sparse grid for hatching
            step = max(1, len(y_coords) // 50)  # Limit to ~50 hatch marks per region
            hatch_y = y_coords[::step]
            hatch_x = x_coords[::step]
            
            # Add hatch marks every few pixels
            for i in range(0, len(hatch_y), 8):  # Every 8th point
                if i < len(hatch_y):
                    y, x = hatch_y[i], hatch_x[i]
                    # Draw small diagonal lines for hatching
                    ax.plot([x-2, x+2], [y-2, y+2], color=cluster_color, linewidth=1, alpha=0.6)
        
        # Draw colored borders
        border_y, border_x = np.where(cluster_edges)
        if len(border_y) > 0:
            ax.scatter(border_x, border_y, c=[cluster_color], s=4, alpha=0.8)
    
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
        cluster_color = cmap(cluster_id / (n_clusters - 1))[:3]
        legend_elements.append(plt.Line2D([0], [0], color=cluster_color, lw=2, label=f'Cluster {cluster_id}'))
    
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