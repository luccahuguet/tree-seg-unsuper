---
layout: default
title: "Complete Example"
nav_order: 4
---

<nav class="tree-seg-navbar">
  <div class="navbar-container">
    <a href="{{ '/' | relative_url }}" class="navbar-home">🌳 Tree Segmentation</a>
    <div class="navbar-links">
      <a href="{{ '/methodology' | relative_url }}">Methodology</a>
      <a href="{{ '/results' | relative_url }}">Results</a>
      <a href="{{ '/complete_example' | relative_url }}">Example</a>
      <a href="{{ '/parameter_comparison' | relative_url }}">Comparison</a>
      <a href="{{ '/analysis' | relative_url }}">Analysis</a>
    </div>
  </div>
</nav>

<style>
.tree-seg-navbar {
  background-color: #1e1e1e;
  border-bottom: 2px solid #00ff00;
  padding: 0.5rem 0;
  margin-bottom: 2rem;
}

.navbar-container {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 1rem;
}

.navbar-home {
  color: #00ff00 !important;
  text-decoration: none !important;
  font-weight: bold;
  font-size: 1.2rem;
}

.navbar-links {
  display: flex;
  gap: 1.5rem;
}

.navbar-links a {
  color: #fff !important;
  text-decoration: none !important;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  transition: background-color 0.3s ease;
}

.navbar-links a:hover {
  background-color: #333;
  color: #00ff00 !important;
}

@media (max-width: 768px) {
  .navbar-container {
    flex-direction: column;
    gap: 1rem;
  }
  
  .navbar-links {
    flex-wrap: wrap;
    justify-content: center;
    gap: 1rem;
  }
}
</style>

# Complete Example: Full Pipeline Demonstration

This section showcases a complete tree segmentation run with all four output visualizations, demonstrating the entire pipeline from processing to final results.

## Configuration Used

**Image**: DJI_20250127150117_0029_D.JPG  
**Configuration**: `d111_v1-5_base_str4_et3-0`

- **Model**: DINOv2 Base (`dinov2_vitb14`)
- **Version**: v1.5 (patch + attention features)
- **Stride**: 4 (balanced speed/quality)
- **Clustering**: Elbow threshold 3.0
- **Clusters Found**: 5 (automatically selected)

---

## Complete Output Set

### 1. Segmentation Legend
*Colored segmentation map with comprehensive cluster legend*

![Segmentation Legend]({{ site.baseurl }}/results/complete_example/d111_v1-5_base_str4_et0-15_segmentation_legend.jpg)

**Purpose**: Shows the segmentation result with:
- Color-coded cluster regions
- Detailed legend with cluster information
- Configuration parameters displayed
- Clean, publication-ready visualization

---

### 2. Edge Overlay
*Original image with colored boundaries overlaid*

![Edge Overlay]({{ site.baseurl }}/results/complete_example/d111_v1-5_base_str4_et0-15_edge_overlay.jpg)

**Purpose**: Demonstrates boundary accuracy by:
- Overlaying colored edges on original imagery
- Using hatch patterns for region distinction
- Maintaining visual clarity of original features
- Showing precise tree canopy delineation

---

### 3. Side-by-Side Comparison
*Original image alongside segmentation result*

![Side by Side]({{ site.baseurl }}/results/complete_example/d111_v1-5_base_str4_et0-15_side_by_side.jpg)

**Purpose**: Enables direct comparison between:
- Original aerial drone imagery (left)
- Segmentation result with legend (right)
- Clear visualization of algorithm performance
- Easy assessment of segmentation quality

---

### 4. Elbow Analysis
*K-selection methodology visualization*

![Elbow Analysis]({{ site.baseurl }}/results/complete_example/d111_v1-5_base_str4_et0-15_elbow_analysis.jpg)

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

---

[← Results](results.html) | [Parameter Comparison →](parameter_comparison.html) | [Home](index.html)
