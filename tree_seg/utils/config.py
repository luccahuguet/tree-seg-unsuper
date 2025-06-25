"""
Configuration utilities for tree segmentation.
"""


def get_config_text(n_clusters, overlay_ratio, stride, model_name, filename, version, edge_width=4, min_region_size=100):
    """Generate a formatted string of configuration parameters."""
    config_lines = [
        f"Version: {version}",
        f"Clusters: {n_clusters}",
        f"Overlay Ratio: {overlay_ratio}",
        f"Stride: {stride}",
        f"Model: {model_name}",
        f"Edge Width: {edge_width}",
        f"Min Region Size: {min_region_size}",
        f"File: {filename if filename else 'All files in directory'}"
    ]
    return "\n".join(config_lines) 