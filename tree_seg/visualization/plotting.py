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

    # Generate NEW edge overlay visualization with colored regions and hatching
    edges = detect_segmentation_edges(labels_resized, edge_width=edge_width)
    
    # Create the enhanced edge overlay with colored regions and hatching
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)  # Show original image as background
    
    # Create a masked overlay for each cluster with hatching
    unique_labels = np.unique(labels_resized)
    hatch_patterns = ['///', '\\\\\\', '|||', '---', '+++', '...', 'ooo', 'OOO', '***', 'xxx']
    
    for i, label in enumerate(unique_labels):
        if label == -1:  # Skip noise/background if present
            continue
            
        # Create mask for this cluster
        cluster_mask = (labels_resized == label)
        
        # Get color for this cluster
        color = cmap(label / (n_clusters - 1))
        
        # Create a hatched overlay for this region
        hatch_pattern = hatch_patterns[i % len(hatch_patterns)]
        
        # Use contourf to create hatched regions
        mask_for_contour = cluster_mask.astype(float)
        ax.contourf(mask_for_contour, levels=[0.5, 1.5], colors=[color], alpha=0.3, hatches=[hatch_pattern])
    
    # Add white edges on top
    edge_overlay_for_edges = image_np.copy()
    edge_overlay_for_edges[edges] = [255, 255, 255]
    
    # Create a mask for just the edges and overlay them
    edge_mask = np.zeros_like(edges, dtype=float)
    edge_mask[edges] = 1.0
    ax.contour(edge_mask, levels=[0.5], colors='white', linewidths=1, alpha=0.9)
    
    ax.axis("off")
    
    # Add config text
    ax.text(
        0.02, 0.98, config_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    # Add small legend for colors and patterns
    legend_elements = []
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        color = cmap(label / (n_clusters - 1))
        hatch_pattern = hatch_patterns[i % len(hatch_patterns)]
        
        # Create legend patch with color and hatching
        from matplotlib.patches import Rectangle
        legend_patch = Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, 
                               hatch=hatch_pattern, edgecolor='black', linewidth=0.5)
        legend_elements.append((legend_patch, f'Cluster {label}'))
    
    # Add the legend - small and positioned in bottom right
    legend_patches = [elem[0] for elem in legend_elements]
    legend_labels = [elem[1] for elem in legend_elements]
    
    legend = ax.legend(legend_patches, legend_labels, 
                      loc='lower right', fontsize=6, framealpha=0.8,
                      bbox_to_anchor=(0.98, 0.02), ncol=1)
    legend.get_frame().set_linewidth(0.5)
    
    plt.tight_layout()
    edge_overlay_path = os.path.join(output_dir, f"{output_prefix}_edge_overlay.png")
    plt.savefig(edge_overlay_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved enhanced edge overlay with colored regions and legend: {edge_overlay_path}")

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