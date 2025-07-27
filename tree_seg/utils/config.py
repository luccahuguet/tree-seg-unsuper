"""
Configuration utilities for tree segmentation.
"""

def parse_model_info(model_name):
    """
    Parse model name to extract base name, nickname, and version.
    
    Args:
        model_name: String like "dinov2_vits14", "dinov2_vitb14", etc.
        
    Returns:
        tuple: (base_name, nickname, version)
    """
    # Extract version info (v1, v1.5, v2, etc.)
    if "dinov2" in model_name.lower():
        base_name = "DINOv2"
        version = "v2"
    elif "dino" in model_name.lower():
        base_name = "DINO"
        version = "v1"
    else:
        base_name = model_name
        version = "v1"
    
    # Extract size nickname
    if "vits" in model_name.lower():
        nickname = "Small"
    elif "vitb" in model_name.lower():
        nickname = "Base"
    elif "vitl" in model_name.lower():
        nickname = "Large"
    elif "vitg" in model_name.lower():
        nickname = "Giant"
    else:
        nickname = "Unknown"
    
    return base_name, nickname, version


def get_config_text(n_clusters, overlay_ratio, stride, model_name, filename, version, edge_width, elbow_threshold=None):
    """
    Generate a configuration text block for plots.
    """
    base_name, nickname, model_version = parse_model_info(model_name)
    
    config_text = (
        f"Model: {base_name} {nickname}\n"
        f"Version: {model_version}\n"
        f"Image: {filename}\n"
        f"Clusters (k): {n_clusters}\n"
        f"Stride: {stride}\n"
        f"Overlay Ratio: {overlay_ratio}\n"
        f"Edge Width: {edge_width}"
    )
    
    if elbow_threshold is not None:
        config_text += f"\nElbow Threshold: {elbow_threshold}"
    return config_text 