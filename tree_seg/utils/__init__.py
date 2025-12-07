# Utility and helper modules

from .config import get_config_text

# Notebook helpers (optional import - only available in Jupyter environments)
try:
    from .notebook_helpers import display_segmentation_results, print_config_summary

    NOTEBOOK_HELPERS_AVAILABLE = True
    _notebook_helpers = [display_segmentation_results, print_config_summary]
except ImportError:
    # IPython not available - running outside Jupyter environment
    NOTEBOOK_HELPERS_AVAILABLE = False
    _notebook_helpers = []

__all__ = ["get_config_text", "NOTEBOOK_HELPERS_AVAILABLE"]

# Add notebook helpers to __all__ if available
if NOTEBOOK_HELPERS_AVAILABLE:
    __all__.extend(["display_segmentation_results", "print_config_summary"])
