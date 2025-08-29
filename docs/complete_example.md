---
layout: default
title: "Complete Example"
nav_order: 3
---

{% include navbar.html %}
{% include navbar-styles.html %}

# Complete Example: Full Pipeline Demonstration

This section showcases a complete tree segmentation run with all four output visualizations, demonstrating the entire pipeline from processing to final results.

## Configuration Used

**Image**: forest2.jpeg  
**Configuration**: Basic Example (balanced profile)

- **Model**: DINOv2 Base (`dinov2_vitb14`)
- **Profile**: Balanced (optimized for general use)
- **Stride**: 4 (balanced quality/performance)
- **Elbow Threshold**: 3.5% (optimal for tree species)
- **Auto K-Selection**: Enabled (automatically determined)

---

## Complete Output Set

### 1. Segmentation Legend
*Colored segmentation map with comprehensive cluster legend*

![Segmentation Legend]({{ site.baseurl }}/results/complete_example/basic_example_segmentation_legend.jpg)

**Purpose**: Shows the segmentation result with:
- Color-coded cluster regions
- Detailed legend with cluster information
- Configuration parameters displayed
- Clean, publication-ready visualization

---

### 2. Edge Overlay
*Original image with colored boundaries overlaid*

![Edge Overlay]({{ site.baseurl }}/results/complete_example/basic_example_edge_overlay.jpg)

**Purpose**: Demonstrates boundary accuracy by:
- Overlaying colored edges on original imagery
- Using hatch patterns for region distinction
- Maintaining visual clarity of original features
- Showing precise tree canopy delineation

---

### 3. Side-by-Side Comparison
*Original image alongside segmentation result*

![Side by Side]({{ site.baseurl }}/results/complete_example/basic_example_side_by_side.jpg)

**Purpose**: Enables direct comparison between:
- Original aerial drone imagery (left)
- Segmentation result with legend (right)
- Clear visualization of algorithm performance
- Easy assessment of segmentation quality

---

### 4. Elbow Analysis
*K-selection methodology visualization*

![Elbow Analysis]({{ site.baseurl }}/results/complete_example/basic_example_elbow_analysis.jpg)

**Purpose**: Shows automatic K-selection process:
- WCSS (Within-Cluster Sum of Squares) curve
- Elbow point detection at K=5
- Threshold analysis methodology
- Transparent, reproducible cluster selection

---

## Methodology Validation

This complete example demonstrates:

1. **Input Processing**: High-resolution aerial imagery handling
2. **Feature Extraction**: DINOv2 Vision Transformer features
3. **Clustering**: K-means with automatic K-selection
4. **Visualization**: Multiple output formats for different use cases
5. **Quality Assurance**: Comprehensive documentation and tracking

The results show successful unsupervised segmentation of complex forest imagery with professional-grade output quality suitable for academic presentation.

