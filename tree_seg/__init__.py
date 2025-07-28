"""
Tree Segmentation Package - v2.0

Modern unsupervised tree segmentation using DINOv2 and clean architecture.

Usage:
    from tree_seg import TreeSegmentation, Config, segment_trees
    
    # Quick usage
    results = segment_trees("image.jpg", model="base", auto_k=True)
    
    # Advanced usage
    config = Config(model_name="base", elbow_threshold=0.1)
    segmenter = TreeSegmentation(config)
    results, paths = segmenter.process_and_visualize("image.jpg")
"""

# Modern API - clean, type-safe, professional
from .api import TreeSegmentation, segment_trees
from .core.types import Config, SegmentationResults, ElbowAnalysisResults, OutputPaths
from .core.output_manager import OutputManager
from .models import print_gpu_info

__version__ = "2.0.0"
__author__ = "Tree Segmentation Team"

__all__ = [
    # Main API
    'TreeSegmentation',
    'segment_trees',
    
    # Configuration and results
    'Config',
    'SegmentationResults', 
    'ElbowAnalysisResults',
    'OutputPaths',
    'OutputManager',
    
    # Utilities
    'print_gpu_info',
    
    # Version
    '__version__',
]
