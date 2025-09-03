"""
Centralized constants for tree segmentation.
"""

# File handling
SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# Output directories
PNG_SUBDIR = "png"
WEB_SUBDIR = "web"

# Documentation sweeps
DOC_SWEEP_THRESHOLDS = [2.5, 5.0, 10.0, 20.0]

# K-selection
DEFAULT_K_RANGE = (3, 10)
ELBOW_PREFIX = "et"

# Plotting defaults
DPI_SEGMENTATION = 200
DPI_EDGE_OVERLAY = 200
DPI_SIDE_BY_SIDE = 150
DPI_ELBOW = 300
LEGEND_FONT_SIZE = 6
HATCH_PATTERNS = ['/', '\\', '|', '.', 'x', '-']

# Profiles
PROFILE_DEFAULTS = {
    'quality': dict(
        image_size=1280,
        feature_upsample_factor=2,
        pca_dim=None,
        refine='slic',
        refine_slic_compactness=12.0,
        refine_slic_sigma=1.5,
    ),
    'balanced': dict(
        image_size=1024,
        feature_upsample_factor=2,
        pca_dim=None,
        refine='slic',
        refine_slic_compactness=10.0,
        refine_slic_sigma=1.0,
    ),
    'speed': dict(
        image_size=896,
        feature_upsample_factor=1,
        pca_dim=128,
        refine='slic',
        refine_slic_compactness=20.0,
        refine_slic_sigma=1.0,
    ),
}

# Mapping from config keys to CLI flag names used for override detection
PROFILE_FLAG_MAP = {
    'image_size': '--image-size',
    'feature_upsample_factor': '--feature-upsample',
    'pca_dim': '--pca-dim',
    'refine': '--refine',
    'refine_slic_compactness': '--refine-slic-compactness',
    'refine_slic_sigma': '--refine-slic-sigma',
}
