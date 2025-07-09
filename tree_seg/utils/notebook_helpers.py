"""
Utility functions for Jupyter notebook display and interaction.
"""

import os
from IPython.display import Image, display


def display_segmentation_results(config, verbose=True):
    """
    Display segmentation results from output directory.
    
    Args:
        config: Configuration dictionary with filename and output_dir
        verbose: Whether to print descriptive text
    """
    filename = config["filename"]
    output_prefix = os.path.splitext(filename)[0]
    output_dir = config["output_dir"]
    
    # Define paths
    legend_path = os.path.join(output_dir, f"{output_prefix}_segmentation_legend.png")
    edge_overlay_path = os.path.join(output_dir, f"{output_prefix}_edge_overlay.png")
    side_by_side_path = os.path.join(output_dir, f"{output_prefix}_side_by_side.png")
    elbow_path = os.path.join(output_dir, f"{output_prefix}_elbow_analysis.png")
    
    # Display the edge overlay
    if os.path.exists(edge_overlay_path):
        if verbose:
            print("🔳 Edge Overlay (Original + Boundaries):")
        display(Image(filename=edge_overlay_path))
    
    # Display the side-by-side comparison
    if os.path.exists(side_by_side_path):
        if verbose:
            print("📊 Original and Segmentation Side by Side:")
        display(Image(filename=side_by_side_path))
    
    # Display the K selection analysis
    if os.path.exists(elbow_path):
        if verbose:
            print("📈 K Selection Analysis (Elbow Method):")
        display(Image(filename=elbow_path))


def print_config_summary(config):
    """Print a summary of the configuration settings."""
    print("🌳 Starting Enhanced Tree Segmentation with Automatic K Selection...")
    print(f"📁 Input: {config['input_dir']}")
    print(f"📁 Output: {config['output_dir']}")
    print(f"🔧 Auto K: {config['auto_k']}")
    if config['auto_k']:
        print(f"📊 Method: Elbow (optimized for trees)")
        print(f"📈 K Range: {config['k_range']}")
    else:
        print(f"🔢 Fixed K: {config['n_clusters']}") 