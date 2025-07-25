"""
Configuration utilities for tree segmentation.
"""

def get_config_text(n_clusters, overlay_ratio, stride, model_name, filename, version, edge_width):
    """
    Generate a configuration text block for plots.
    """
    config_text = (
        f"Model: {model_name} (v{version})\n"
        f"Image: {filename}\n"
        f"Clusters (k): {n_clusters}\n"
        f"Stride: {stride}\n"
        f"Overlay Ratio: {overlay_ratio}\n"
        f"Edge Width: {edge_width}"
    )
    return config_text 