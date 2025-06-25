"""
Visualization and plotting utilities for tree segmentation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import ndimage
# scikit-image import for morphological operations
try:
    from skimage import morphology  # type: ignore
except ImportError:
    # Fallback if scikit-image is not available
    morphology = None

from ..utils.config import get_config_text


def detect_segmentation_edges_enhanced(labels, edge_width=4):
    """
    Detect and enhance edges between different segmentation regions.
    Creates smooth, continuous colored edges with better visual appeal.
    
    Args:
        labels: 2D array of segmentation labels
        edge_width: Width of the edge lines in pixels
        
    Returns:
        edge_map: 2D array where each pixel contains the edge color index (0=no edge)
        n_edge_types: Number of different edge types found
    """
    # Create edge map by finding boundaries between different labels
    edge_map = np.zeros_like(labels, dtype=np.uint8)
    
    # Get unique labels for color assignment
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    print(f"🎨 Creating continuous colored edges between {n_clusters} regions...")
    
    # For each unique pair of adjacent labels, create boundaries
    edge_color = 1
    
    # Method 1: Scan horizontally for label changes
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1] - 1):
            if labels[y, x] != labels[y, x + 1]:
                # Found vertical boundary - paint edge pixels around it
                for dy in range(-edge_width//2, edge_width//2 + 1):
                    for dx in range(-edge_width//2, edge_width//2 + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < labels.shape[0] and 0 <= nx < labels.shape[1]:
                            if edge_map[ny, nx] == 0:  # Don't overwrite existing edges
                                # Assign color based on the adjacent labels
                                label1, label2 = sorted([labels[y, x], labels[y, x + 1]])
                                color_idx = ((label1 * 7 + label2 * 3) % 8) + 1
                                edge_map[ny, nx] = color_idx
    
    # Method 2: Scan vertically for label changes
    for x in range(labels.shape[1]):
        for y in range(labels.shape[0] - 1):
            if labels[y, x] != labels[y + 1, x]:
                # Found horizontal boundary - paint edge pixels around it
                for dy in range(-edge_width//2, edge_width//2 + 1):
                    for dx in range(-edge_width//2, edge_width//2 + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < labels.shape[0] and 0 <= nx < labels.shape[1]:
                            if edge_map[ny, nx] == 0:  # Don't overwrite existing edges
                                # Assign color based on the adjacent labels
                                label1, label2 = sorted([labels[y, x], labels[y + 1, x]])
                                color_idx = ((label1 * 7 + label2 * 3) % 8) + 1
                                edge_map[ny, nx] = color_idx
    
    # Apply morphological operations to smooth and connect edges (if available)
    if morphology is not None:
        # Fill small gaps and smooth the edges
        for color_idx in range(1, 9):
            color_mask = (edge_map == color_idx)
            if np.any(color_mask):
                # Connect nearby edge pixels of the same color
                color_mask = morphology.binary_closing(color_mask, morphology.disk(1))
                # Smooth the boundaries
                color_mask = morphology.binary_opening(color_mask, morphology.disk(1))
                # Update the edge map
                edge_map[color_mask] = color_idx
    
    # Ensure edges are thick enough by dilating
    final_edge_map = np.zeros_like(edge_map)
    for color_idx in range(1, 9):
        color_mask = (edge_map == color_idx)
        if np.any(color_mask):
            # Dilate to make thicker
            if edge_width > 2:
                structure = ndimage.generate_binary_structure(2, 2)
                color_mask = ndimage.binary_dilation(color_mask, structure=structure, iterations=edge_width//3)
            final_edge_map[color_mask] = color_idx
    
    num_colors_used = len(np.unique(final_edge_map[final_edge_map > 0]))
    print(f"✅ Created edges using {num_colors_used} different colors")
    
    return final_edge_map, min(8, n_clusters)


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
    edge_width=6,
    min_region_size=100,
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

    config_text = get_config_text(n_clusters, overlay_ratio, stride, model_name, filename, version, edge_width, min_region_size)

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

    # Generate ENHANCED edge overlay visualization with colors and legend
    edge_map, n_edge_types = detect_segmentation_edges_enhanced(labels_resized, edge_width=edge_width)
    edge_overlay = image_np.copy()
    
    # Define vibrant edge colors
    edge_colors = [
        [255, 255, 255],  # 0: No edge (white background)
        [255, 0, 0],      # 1: Red
        [0, 255, 0],      # 2: Green  
        [0, 0, 255],      # 3: Blue
        [255, 255, 0],    # 4: Yellow
        [255, 0, 255],    # 5: Magenta
        [0, 255, 255],    # 6: Cyan
        [255, 128, 0],    # 7: Orange
        [128, 0, 255],    # 8: Purple
    ]
    
    # Apply colored edges to the image
    for edge_type in range(1, min(len(edge_colors), n_edge_types + 1)):
        mask = edge_map == edge_type
        if np.any(mask):
            edge_overlay[mask] = edge_colors[edge_type]
    
    # Create figure with legend
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(edge_overlay)
    ax.axis("off")
    
    # Add configuration text
    ax.text(
        0.02, 0.98, config_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )
    
    # Add edge legend
    legend_elements = []
    used_edge_types = np.unique(edge_map[edge_map > 0])
    for i, edge_type in enumerate(used_edge_types[:8]):  # Limit to 8 colors
        color = np.array(edge_colors[edge_type]) / 255.0
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=f'Boundary Type {edge_type}'))
    
    if legend_elements:
        legend = ax.legend(handles=legend_elements, loc='upper right', 
                          bbox_to_anchor=(0.98, 0.98), fontsize=8,
                          facecolor='white', framealpha=0.8)
        legend.set_title('Edge Types')
        legend.get_title().set_fontsize(9)
        legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    edge_overlay_path = os.path.join(output_dir, f"{output_prefix}_edge_overlay.png")
    plt.savefig(edge_overlay_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()
    print(f"Saved enhanced edge overlay: {edge_overlay_path}")

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