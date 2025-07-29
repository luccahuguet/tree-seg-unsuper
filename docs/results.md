---
layout: default
title: "Results Overview"
nav_order: 3
---

{% include navbar.html %}
{% include navbar-styles.html %}

# Results

## Experiment Overview

Results from processing aerial drone imagery using our DINOv2-based tree segmentation pipeline. All images processed with automatic K-selection using elbow method optimization.

## Results Sections

### [Complete Example: Full Pipeline Demonstration](complete_example.html)
*Comprehensive showcase of all four output types from a single segmentation run*

- **All Output Types**: Segmentation legend, edge overlay, side-by-side, elbow analysis
- **Complete Documentation**: Full parameter tracking and methodology
- **Professional Quality**: Publication-ready visualizations
- **Academic Standard**: Suitable for research presentation

### [Parameter Comparison: Model & Stride Analysis](parameter_comparison.html)
*Systematic comparison of DINOv2 model sizes and stride parameters*

- **Model Comparison**: Small vs Base vs Large DINOv2 variants
- **Stride Analysis**: Resolution vs speed trade-offs (stride 2, 4, 8)
- **Performance Metrics**: Processing time, memory usage, quality assessment
- **Selection Guidelines**: Recommendations for different use cases

---

## Quick Preview

### Sample Configuration: `d111_v1-5_base_str4_et3-0`
- **Image**: DJI_20250127150117_0029_D.JPG
- **Model**: DINOv2 Base (recommended)
- **Stride**: 4 (balanced)
- **Clustering**: Elbow threshold 3.0
- **Clusters Found**: 5 (automatically selected)

### Key Observations
1. **Boundary Quality**: Clean, accurate tree boundaries with minimal over-segmentation
2. **Cluster Coherence**: Meaningful groupings corresponding to distinct tree regions
3. **Automatic K-Selection**: Elbow method successfully identified optimal cluster count
4. **Visual Clarity**: Hatch patterns and color coding provide clear region distinction

### Configuration Parameters Used

```yaml
Model: DINOv2 Base
Version: v1.5
Stride: 4
Elbow Threshold: 0.1
K Range: (3-10)
Edge Width: 2
Hatching: Enabled
```

## Adding Your Results

To add new experimental results:

1. **Place optimized images** in `results/` folder
2. **Copy to docs** using the provided script
3. **Update this page** with new sections following the template above

### File Naming Convention

Our smart naming system encodes all parameters:
```
{hash}_{version}_{model}_{stride}_{clustering}_type.jpg
```

Example: `a3f7_v1-5_base_str4_et0-1_edge_overlay.jpg`
- `a3f7`: Source image hash
- `v1-5`: Algorithm version  
- `base`: Model size
- `str4`: Stride parameter
- `et0-1`: Elbow threshold 0.1

---

[← Methodology](methodology.html) | [Analysis →](analysis.html)