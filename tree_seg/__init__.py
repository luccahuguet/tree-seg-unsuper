"""
Tree Segmentation Package

A modular tree segmentation system using DINO features and automatic K selection.

Modern API (v2.0+):
    from tree_seg import TreeSegmentation, Config
    
Legacy API (still supported):
    from tree_seg import tree_seg_with_auto_k
"""

# Legacy API (maintained for compatibility)
from .tree_segmentation import tree_seg_with_auto_k, MODELS
from .models import print_gpu_info, setup_segmentation
from .core import process_image
from .analysis import find_optimal_k_elbow, plot_elbow_analysis
from .visualization import generate_outputs

# Modern API (recommended)
from .api import TreeSegmentation, segment_trees
from .core.types import Config, SegmentationResults, ElbowAnalysisResults, OutputPaths
from .core.output_manager import OutputManager

__version__ = "2.0.0"
__author__ = "Tree Segmentation Team"

__all__ = [
    # Modern API (recommended)
    'TreeSegmentation',
    'segment_trees', 
    'Config',
    'SegmentationResults',
    'ElbowAnalysisResults',
    'OutputPaths',
    'OutputManager',
    
    # Legacy API (maintained for compatibility)
    'tree_seg_with_auto_k',
    'MODELS',
    'print_gpu_info',
    'setup_segmentation',
    'process_image',
    'find_optimal_k_elbow',
    'plot_elbow_analysis',
    'generate_outputs',
]
