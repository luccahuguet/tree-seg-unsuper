---
layout: default
title: "Methodology"
nav_order: 2
---

{% include navbar.html %}
{% include navbar-styles.html %}

# Methodology

## Pipeline Overview

Our tree segmentation pipeline consists of four main stages:

1. **Feature Extraction** - DINOv2 Vision Transformer
2. **Clustering** - K-means with automatic K selection
3. **Post-processing** - Upsampling and refinement
4. **Visualization** - Multi-format output generation

## Technical Details

### 1. Feature Extraction

**Model**: DINOv2 Vision Transformer (v1.5)
- **Variant**: Base model (dinov2_vitb14)
- **Features**: Patch embeddings + attention features
- **Dimensionality**: 768-dimensional feature space
- **Stride**: 4 (balance of quality vs. speed)

### 2. Automatic K-Selection

**Elbow Method** with intelligent thresholding:
- **K Range**: 3-8 clusters (narrower range optimized for trees)
- **Threshold**: 0.15 (slightly conservative for stable results)
- **Metric**: Within-Cluster Sum of Squares (WCSS)
- **Optimization**: Percentage decrease analysis

### 3. Clustering Algorithm

**K-means Configuration**:
- **Initialization**: k-means++ for stable results
- **Iterations**: Auto-convergence
- **Random State**: 42 (reproducible results)

### 4. Post-processing

**Upsampling Strategy**:
- **Method**: High-resolution feature upsampling
- **Target**: Original image resolution
- **Interpolation**: Bilinear for smooth boundaries

## Configuration Parameters

```python
config = Config(
    model_name="base",          # DINOv2 variant
    version="v1.5",             # Algorithm version
    stride=4,                   # Feature resolution
    auto_k=True,                # Automatic K selection
    elbow_threshold=0.15,       # Slightly conservative
    k_range=(3, 8),             # Narrower range for trees
    use_pca=False,              # Keep full feature space
    edge_width=2,               # Visualization parameter
    use_hatching=True           # Visual distinction
)
```

## Innovation Points

1. **Modern Architecture**: Type-safe configuration with validation
2. **Smart File Management**: Config-based naming prevents collisions
3. **Flexible API**: Both quick-use and advanced interfaces
4. **Professional Visualization**: Publication-ready outputs

## Implementation Quality

- **Type Safety**: Full type hints and dataclass validation
- **Error Handling**: Graceful fallbacks and clear error messages
- **Performance**: Optimized numpy operations and memory management
- **Maintainability**: Clean separation of concerns and modular design

