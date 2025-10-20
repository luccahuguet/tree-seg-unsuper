"""Quality presets for tree segmentation.

These presets balance quality, speed, and resource usage for production use.
Each preset is a dictionary of Config parameters.

Usage:
    from tree_seg.presets import PRESETS
    from tree_seg import Config, TreeSegmentation

    config = Config(**PRESETS['balanced'])
    segmenter = TreeSegmentation(config)
"""

from typing import Dict, Any

# Production quality presets
PRESETS: Dict[str, Dict[str, Any]] = {
    "quality": {
        "image_size": 1280,
        "feature_upsample_factor": 2,
        "pca_dim": None,
        "refine": "slic",
        "refine_slic_compactness": 12.0,
        "refine_slic_sigma": 1.5,
    },
    "balanced": {
        "image_size": 1024,
        "feature_upsample_factor": 2,
        "pca_dim": None,
        "refine": "slic",
        "refine_slic_compactness": 10.0,
        "refine_slic_sigma": 1.0,
    },
    "speed": {
        "image_size": 896,
        "feature_upsample_factor": 1,
        "pca_dim": 128,
        "refine": "slic",
        "refine_slic_compactness": 20.0,
        "refine_slic_sigma": 1.0,
    },
}


def get_preset(preset_name: str) -> Dict[str, Any]:
    """Get preset configuration by name.

    Args:
        preset_name: Name of preset ('quality', 'balanced', 'speed')

    Returns:
        Dictionary of Config parameters

    Raises:
        ValueError: If preset_name is not recognized
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )
    return PRESETS[preset_name].copy()
