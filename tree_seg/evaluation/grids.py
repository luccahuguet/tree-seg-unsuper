"""Grid search configurations for benchmark sweeps."""

from typing import Dict, List


def _generate_full_factorial() -> List[Dict]:
    """Generate full factorial design programmatically."""
    models = ["small", "base"]
    thresholds = [2.5, 5.0, 10.0, 20.0]
    refinements = [None, "slic"]

    configs = []
    for model in models:
        for threshold in thresholds:
            for refine in refinements:
                refine_str = "slic" if refine else "km"
                label = f"{model}_e{threshold:.1f}_{refine_str}".replace(".", "-")
                configs.append({
                    "model_name": model,
                    "elbow_threshold": threshold,
                    "refine": refine,
                    "label": label,
                })
    return configs


# Grid definitions for parameter sweeps
GRIDS = {
    "ofat": {
        "name": "One-Factor-At-Time",
        "description": "Systematic exploration of individual parameters",
        "configs": [
            # Elbow threshold sweep (base model, no refinement)
            {"elbow_threshold": 2.5, "label": "elbow_2.5"},
            {"elbow_threshold": 5.0, "label": "elbow_5.0"},
            {"elbow_threshold": 10.0, "label": "elbow_10.0"},
            {"elbow_threshold": 20.0, "label": "elbow_20.0"},
            # Model size sweep (threshold 5.0, no refinement)
            {"model_name": "small", "label": "model_small"},
            {"model_name": "base", "label": "model_base"},
            {"model_name": "large", "label": "model_large"},
            # Refinement sweep (base model, threshold 5.0)
            {"refine": None, "label": "refine_kmeans"},
            {"refine": "slic", "label": "refine_slic"},
        ],
    },
    "smart": {
        "name": "Smart Grid",
        "description": "Focused exploration of best-performing combinations",
        "configs": [
            # small model × top thresholds × both refinements
            {"model_name": "small", "elbow_threshold": 10.0, "refine": None, "label": "small_e10_km"},
            {"model_name": "small", "elbow_threshold": 10.0, "refine": "slic", "label": "small_e10_slic"},
            {"model_name": "small", "elbow_threshold": 20.0, "refine": None, "label": "small_e20_km"},
            {"model_name": "small", "elbow_threshold": 20.0, "refine": "slic", "label": "small_e20_slic"},
            # base model × top thresholds × both refinements
            {"model_name": "base", "elbow_threshold": 10.0, "refine": None, "label": "base_e10_km"},
            {"model_name": "base", "elbow_threshold": 10.0, "refine": "slic", "label": "base_e10_slic"},
            {"model_name": "base", "elbow_threshold": 20.0, "refine": None, "label": "base_e20_km"},
            {"model_name": "base", "elbow_threshold": 20.0, "refine": "slic", "label": "base_e20_slic"},
        ],
    },
    "full": {
        "name": "Full Factorial",
        "description": "Complete exploration of all parameter combinations",
        "configs": _generate_full_factorial(),
    },
}


def get_grid(grid_name: str) -> Dict:
    """
    Get grid configuration by name.

    Args:
        grid_name: Name of grid (ofat, smart, full)

    Returns:
        Dict with 'name', 'description', and 'configs'

    Raises:
        ValueError: If grid_name not found
    """
    if grid_name not in GRIDS:
        available = ", ".join(GRIDS.keys())
        raise ValueError(f"Unknown grid '{grid_name}'. Available: {available}")
    return GRIDS[grid_name]


def list_grids() -> List[str]:
    """Get list of available grid names."""
    return list(GRIDS.keys())
