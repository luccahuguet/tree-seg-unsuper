"""
Visualization and plotting utilities for tree segmentation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import colors
from scipy import ndimage
from skimage import segmentation

from ..utils.config import get_config_text


def detect_segmentation_edges(labels, edge_width=6):
    """
    Detect clean edges between different segmentation regions.
    
    Args:
        labels: 2D array of segmentation labels
        edge_width: Width of the edge lines in pixels
        
    Returns:
        edges: Binary mask where edges are True
    """
    try:
        # Try the advanced method with cv2 and skimage
        # Convert labels to proper format
        labels = labels.astype(np.int32)
        
        # Use skimage's find_boundaries for clean region boundaries
        boundaries = segmentation.find_boundaries(labels, mode='thick')
        
        # Make edges thicker using morphological dilation
        if edge_width > 1:
            # Create a disk-shaped structuring element for round edges
            kernel_size = max(3, edge_width)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            # Convert to uint8 for OpenCV
            boundaries_uint8 = (boundaries * 255).astype(np.uint8)
            # Dilate to make edges thicker
            thick_boundaries = cv2.dilate(boundaries_uint8, kernel, iterations=1)
            boundaries = thick_boundaries > 0
        
        return boundaries
    
    except (ImportError, AttributeError):
        # Fallback method using only scipy and numpy
        print("Using fallback edge detection method (cv2/skimage not available)")
        
        # Create edge detection using gradient approach
        labels = labels.astype(float)
        
        # Use gradient magnitude to find edges between regions
        grad_x = np.abs(np.gradient(labels, axis=1))
        grad_y = np.abs(np.gradient(labels, axis=0))
        edges = (grad_x + grad_y) > 0.1
        
        # Make edges thicker using binary dilation
        if edge_width > 1:
            from scipy import ndimage
            # Create circular structuring element
            y, x = np.ogrid[-edge_width:edge_width+1, -edge_width:edge_width+1]
            mask = x*x + y*y <= edge_width*edge_width
            edges = ndimage.binary_dilation(edges, structure=mask)
        
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
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_clusters))
    cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
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

    # Generate NEW edge overlay visualization
    edges = detect_segmentation_edges(labels_resized, edge_width=edge_width)
    edge_overlay = image_np.copy()
    
    # Use bright yellow for high contrast against both light and dark backgrounds
    # Yellow (255, 255, 0) is more visible than white on many natural images
    edge_overlay[edges] = [255, 255, 0]  # Bright yellow edges
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(edge_overlay)
    ax.axis("off")
    ax.set_title("Edge Overlay - Segmentation Boundaries", fontsize=14, pad=20)
    ax.text(
        0.02, 0.98, config_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=3)
    )
    plt.tight_layout()
    edge_overlay_path = os.path.join(output_dir, f"{output_prefix}_edge_overlay.png")
    plt.savefig(edge_overlay_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved edge overlay: {edge_overlay_path}")

    # Generate side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")
    im = axes[1].imshow(labels_resized, cmap=cmap, vmin=0, vmax=n_clusters - 1)
    axes[1].set_title("Segmentation Map", fontsize=12)
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], ticks=range(n_clusters))
    cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
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