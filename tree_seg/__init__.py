"""
Tree Segmentation Package

A modular tree segmentation system using DINO features and automatic K selection.
"""

from .tree_segmentation import tree_seg_with_auto_k, MODELS
from .models import print_gpu_info, setup_segmentation
from .core import process_image
from .analysis import find_optimal_k_elbow, plot_elbow_analysis
from .visualization import generate_outputs

__version__ = "2.0.0"
__author__ = "Tree Segmentation Team"

__all__ = [
    # Main functions
    'tree_seg_with_auto_k',
    'MODELS',
    
    # Utilities
    'print_gpu_info',
    'setup_segmentation',
    
    # Core processing
    'process_image',
    
    # Analysis
    'find_optimal_k_elbow',
    'plot_elbow_analysis',
    
    # Visualization
    'generate_outputs',
]
