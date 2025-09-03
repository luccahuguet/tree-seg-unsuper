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

1. **Feature Extraction** - DINOv3 Vision Transformer
2. **Clustering** - K-means with automatic K selection
3. **Post-processing** - Upsampling and refinement
4. **Visualization** - Multi-format output generation

## Technical Details

### 1. Feature Extraction

**Model**: DINOv3 Vision Transformer (v3)
- **Variant**: Base model (dinov3_vitb16)
- **Features**: Patch embeddings + attention features
- **Dimensionality**: 768-dimensional feature space
- **Stride**: 4 (balance of quality vs. speed)

### 2. Automatic K-Selection

**Elbow Method** with intelligent thresholding:
- **K Range**: 3-10 clusters (optimized for tree species diversity)
- **Threshold**: 5.0% (percentage decrease threshold for diminishing returns)
- **Metric**: Within-Cluster Sum of Squares (WCSS)
- **Optimization**: Percentage decrease analysis with safety bounds

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
    model_name="base",          # DINOv3 variant  
    version="v3",               # Current algorithm version
    stride=4,                   # Feature resolution
    auto_k=True,                # Automatic K selection
    elbow_threshold=5.0,        # 5.0% threshold for diminishing returns
    k_range=(3, 10),            # Extended range for tree species
    use_pca=False,              # Keep full feature space
    edge_width=2,               # Visualization parameter
    web_optimize=True           # Generate web-optimized outputs
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

---

## Pipeline Demonstration

### Segmentation Result with Legend
*Complete segmentation output showing cluster regions and configuration*

![Methodology Segmentation]({{ site.baseurl }}/results/methodology/basic_example_segmentation_legend.jpg)

### Edge Overlay Visualization  
*Original image with colored boundaries showing segmentation accuracy*

![Methodology Edge Overlay]({{ site.baseurl }}/results/methodology/basic_example_edge_overlay.jpg)

### Side-by-Side Comparison
*Original image alongside segmentation result for direct comparison*

![Methodology Side by Side]({{ site.baseurl }}/results/methodology/basic_example_side_by_side.jpg)

### Automatic K-Selection Process
*Elbow method analysis showing optimal cluster selection*

![Methodology Elbow Analysis]({{ site.baseurl }}/results/methodology/basic_example_elbow_analysis.jpg)

---

## See Also

- **[Complete Example]({{ '/complete_example' | relative_url }})**: Full pipeline demonstration with all output types
- **[Parameter Analysis]({{ '/parameter_analysis' | relative_url }})**: Comprehensive study of all 12 configurations and model comparisons
