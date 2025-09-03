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

