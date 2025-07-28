"""
Utility functions for Jupyter notebook display and interaction.
"""

import os
import glob
from IPython.display import Image, display


def display_segmentation_results(config, verbose=True):
    """
    Display segmentation results from output directory.
    
    Args:
        config: Configuration dictionary with filename and output_dir
        verbose: Whether to print descriptive text
    """
    output_dir = config["output_dir"]
    
    # Find files using glob patterns (works with new config-based naming)
    legend_files = glob.glob(os.path.join(output_dir, "*_segmentation_legend.png"))
    edge_overlay_files = glob.glob(os.path.join(output_dir, "*_edge_overlay.png"))
    side_by_side_files = glob.glob(os.path.join(output_dir, "*_side_by_side.png"))
    elbow_files = glob.glob(os.path.join(output_dir, "*_elbow_analysis.png"))
    
    # Use the most recent files (in case of multiple runs)
    legend_path = max(legend_files, key=os.path.getmtime) if legend_files else None
    edge_overlay_path = max(edge_overlay_files, key=os.path.getmtime) if edge_overlay_files else None
    side_by_side_path = max(side_by_side_files, key=os.path.getmtime) if side_by_side_files else None
    elbow_path = max(elbow_files, key=os.path.getmtime) if elbow_files else None
    
    # Display the edge overlay
    if edge_overlay_path and os.path.exists(edge_overlay_path):
        if verbose:
            print("🔳 Edge Overlay (Original + Boundaries):")
        display(Image(filename=edge_overlay_path))
    
    # Display the side-by-side comparison
    if side_by_side_path and os.path.exists(side_by_side_path):
        if verbose:
            print("📊 Original and Segmentation Side by Side:")
        display(Image(filename=side_by_side_path))
    
    # Display the K selection analysis
    if elbow_path and os.path.exists(elbow_path):
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
        print(f"🎯 Elbow Threshold: {config['elbow_threshold']}")
    else:
        print(f"🔢 Fixed K: {config['n_clusters']}") 