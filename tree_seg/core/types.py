"""
Core types and data structures for tree segmentation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path


@dataclass
class Config:
    """
    Centralized configuration for tree segmentation.
    """
    # Input/Output
    input_dir: str = "/kaggle/input/drone-10-best"
    output_dir: str = "/kaggle/working/output"
    filename: Optional[str] = None  # If None, process all images
    
    # Model settings
    model_name: str = "base"  # small/base/large/giant/mega or full DINOv3 name
    version: str = "v3"  # Now using DINOv3
    stride: int = 4
    image_size: int = 1024  # Preprocess resize (square)
    
    # Clustering settings
    auto_k: bool = True
    n_clusters: int = 6  # Used when auto_k=False
    k_range: Tuple[int, int] = (3, 10)
    elbow_threshold: float = 3.5  # Percentage (will be converted to decimal)
    use_pca: bool = False
    pca_dim: Optional[int] = None  # If set, apply PCA to this dimension
    feature_upsample_factor: int = 2  # Upsample HxW feature grid before K-Means
    
    # Visualization settings
    overlay_ratio: int = 4  # 1=opaque, 10=transparent
    edge_width: int = 2
    use_hatching: bool = True
    
    # Web optimization settings
    web_optimize: bool = True  # Auto-optimize images for GitHub Pages (default enabled)
    web_quality: int = 85  # JPEG quality for web
    web_max_width: int = 1200  # Max width for web images
    
    @property
    def model_display_name(self) -> str:
        """Get the DINOv3 model name for loading."""
        model_map = {
            "small": "dinov3_vits16",      # ViT-S/16 (21M params)
            "base": "dinov3_vitb16",       # ViT-B/16 (86M params) - recommended
            "large": "dinov3_vitl16",      # ViT-L/16 (300M params)
            "giant": "dinov3_vith16plus",  # ViT-H+/16 (840M params)
            "mega": "dinov3_vit7b16",      # ViT-7B/16 (6.7B params) - satellite optimized
        }
        return model_map.get(self.model_name, self.model_name)
    
    @property
    def elbow_threshold_decimal(self) -> float:
        """Get elbow_threshold as decimal (percentage / 100)."""
        return self.elbow_threshold / 100.0
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not (1 <= self.overlay_ratio <= 10):
            raise ValueError("overlay_ratio must be between 1 and 10")
        if not (2 <= self.stride <= 8):
            raise ValueError("stride must be between 2 and 8")
        if self.elbow_threshold <= 0:
            raise ValueError("elbow_threshold must be positive")
        if not (2 <= self.n_clusters <= 20):
            raise ValueError("n_clusters must be between 2 and 20")
        if self.image_size < 128 or self.image_size > 2048:
            raise ValueError("image_size must be between 128 and 2048")
        if self.feature_upsample_factor < 1 or self.feature_upsample_factor > 8:
            raise ValueError("feature_upsample_factor must be between 1 and 8")
        if self.pca_dim is not None and (self.pca_dim <= 0 or self.pca_dim > 1024):
            raise ValueError("pca_dim must be between 1 and 1024 when set")


@dataclass
class SegmentationResults:
    """
    Results from image segmentation processing.
    """
    image_np: np.ndarray
    labels_resized: np.ndarray
    n_clusters_used: int
    image_path: str
    processing_stats: Dict[str, Any]
    
    @property
    def success(self) -> bool:
        """Check if segmentation was successful."""
        return self.image_np is not None and self.labels_resized is not None


@dataclass
class ElbowAnalysisResults:
    """
    Results from elbow method K-selection.
    """
    optimal_k: int
    k_values: list
    wcss_values: list
    elbow_idx: int
    pct_decrease: list
    method: str = "elbow"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for plotting functions."""
        return {
            'optimal_k': self.optimal_k,
            'k_values': self.k_values,
            'wcss': self.wcss_values,
            'elbow_idx': self.elbow_idx,
            'pct_decrease': self.pct_decrease,
            'method': self.method
        }


@dataclass 
class OutputPaths:
    """
    Generated output file paths.
    """
    segmentation_legend: str
    edge_overlay: str
    side_by_side: str
    elbow_analysis: Optional[str] = None
    
    def all_paths(self) -> list:
        """Get all non-None paths."""
        paths = [self.segmentation_legend, self.edge_overlay, self.side_by_side]
        if self.elbow_analysis:
            paths.append(self.elbow_analysis)
        return paths
