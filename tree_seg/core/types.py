"""
Core types and data structures for tree segmentation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from ..constants import DEFAULT_K_RANGE


@dataclass
class Config:
    """
    Centralized configuration for tree segmentation.
    """
    # Input/Output
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    filename: Optional[str] = None  # If None, process all images
    
    # Model settings
    model_name: str = "base"  # small/base/large/giant/mega or full DINOv3 name
    stride: int = 4
    image_size: int = 1024  # Preprocess resize (square)
    use_attention_features: bool = True  # Include attention tokens (disable for legacy v1)

    # Pipeline settings
    pipeline: str = "v1_5"  # Pipeline version: "v1_5" (baseline) or "v3" (species clustering)

    # Clustering settings
    clustering_method: str = "kmeans"  # "kmeans", "gmm", or "spectral"
    auto_k: bool = True
    n_clusters: int = 6  # Used when auto_k=False
    k_range: Tuple[int, int] = DEFAULT_K_RANGE
    elbow_threshold: float = 5.0  # Percentage (will be converted to decimal)
    use_pca: bool = False
    pca_dim: Optional[int] = None  # If set, apply PCA to this dimension
    feature_upsample_factor: int = 2  # Upsample HxW feature grid before clustering
    
    # Multi-layer feature extraction
    use_multi_layer: bool = False  # Extract features from multiple layers
    layer_indices: Tuple[int, ...] = (3, 6, 9, 12)  # Layers to extract (base model has 12 layers)
    feature_aggregation: str = "concat"  # "concat", "average", or "weighted"

    # Multi-scale pyramid feature extraction
    use_pyramid: bool = False  # Extract features at multiple image scales
    pyramid_scales: Tuple[float, ...] = (0.5, 1.0, 2.0)  # Scales to process (relative to image_size)
    pyramid_aggregation: str = "concat"  # "concat" or "average"

    # Refinement settings
    refine: Optional[str] = "slic"  # Default to SLIC refinement
    refine_slic_compactness: float = 10.0
    refine_slic_sigma: float = 1.0

    # V2 soft EM refinement (feature space)
    use_soft_refine: bool = False  # Apply soft EM refinement after K-means
    soft_refine_temperature: float = 1.0  # τ - lower = softer boundaries (0.5-2.0)
    soft_refine_iterations: int = 5  # Number of EM iterations (3-5 typical)
    soft_refine_spatial_alpha: float = 0.0  # α - spatial blend weight (0-1, 0=none)

    # Vegetation filtering (works with any method: V1.5, V2, V3)
    apply_vegetation_filter: bool = False  # Enable ExG-based vegetation filtering
    exg_threshold: float = 0.10  # ExG threshold for vegetation classification (0.10 = validated optimal)

    # V3-specific: backward compatibility alias
    @property
    def v3_exg_threshold(self) -> float:
        """Backward compatibility: v3_exg_threshold maps to exg_threshold."""
        return self.exg_threshold

    # Tiling configuration (for large images)
    use_tiling: bool = True  # Auto-enable for large images (automatic above tile_threshold)
    tile_size: int = 2048    # Tile dimension in pixels (square tiles)
    tile_overlap: int = 256  # Overlap between adjacent tiles in pixels
    tile_threshold: int = 2048  # Auto-tile if image dimension exceeds this (px)

    # Downsampling (opt-in for speed)
    downsample_before_tiling: bool = False  # 2× downsample before tiling for faster processing

    # Metrics & benchmarking
    metrics: bool = False  # Collect and expose timing/VRAM info in results
    verbose: bool = False  # Print detailed processing information (default: quiet for benchmarking)
    
    # Visualization settings
    overlay_ratio: int = 4  # 1=opaque, 10=transparent
    edge_width: int = 2
    use_hatching: bool = True
    viz_two_panel: bool = False  # If True, use compact 2-panel GT+overlay viz
    viz_two_panel_opaque: bool = False  # If True, two-panel GT + prediction (opaque)
    
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
        if not isinstance(self.use_attention_features, bool):
            raise ValueError("use_attention_features must be a boolean")
        if self.elbow_threshold <= 0:
            raise ValueError("elbow_threshold must be positive")
        if not (2 <= self.n_clusters <= 50):
            raise ValueError("n_clusters must be between 2 and 50")
        if self.image_size < 128 or self.image_size > 2048:
            raise ValueError("image_size must be between 128 and 2048")
        if self.feature_upsample_factor < 1 or self.feature_upsample_factor > 8:
            raise ValueError("feature_upsample_factor must be between 1 and 8")
        if self.pca_dim is not None and (self.pca_dim <= 0 or self.pca_dim > 1024):
            raise ValueError("pca_dim must be between 1 and 1024 when set")
        if self.refine not in [None, "slic", "slic_skimage", "bilateral"]:
            raise ValueError("refine must be None, 'slic', 'slic_skimage', or 'bilateral'")
        if self.pipeline not in ("v1_5", "v3"):
            raise ValueError("pipeline must be 'v1_5' or 'v3'")

        # Tiling validation
        if self.tile_size < 512 or self.tile_size > 4096:
            raise ValueError("tile_size must be between 512 and 4096")
        if self.tile_overlap < 0 or self.tile_overlap >= self.tile_size // 2:
            raise ValueError("tile_overlap must be between 0 and tile_size/2")
        if self.tile_threshold < self.tile_size:
            raise ValueError("tile_threshold must be >= tile_size")
        
        # Multi-layer validation
        if self.use_multi_layer:
            if not self.layer_indices or len(self.layer_indices) == 0:
                raise ValueError("layer_indices cannot be empty when use_multi_layer is True")
            if any(idx < 1 or idx > 40 for idx in self.layer_indices):
                raise ValueError("layer_indices must be between 1 and 40")
            if self.feature_aggregation not in ("concat", "average", "weighted"):
                raise ValueError("feature_aggregation must be 'concat', 'average', or 'weighted'")

        # V2 soft refinement validation
        if self.use_soft_refine:
            if self.soft_refine_temperature <= 0:
                raise ValueError("soft_refine_temperature must be positive")
            if self.soft_refine_iterations < 1 or self.soft_refine_iterations > 20:
                raise ValueError("soft_refine_iterations must be between 1 and 20")
            if self.soft_refine_spatial_alpha < 0 or self.soft_refine_spatial_alpha > 1:
                raise ValueError("soft_refine_spatial_alpha must be between 0 and 1")


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
    n_clusters_requested: Optional[int] = None
    
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
