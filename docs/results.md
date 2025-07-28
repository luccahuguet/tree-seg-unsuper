---
layout: default
title: "Results"
---

# Results

## Experiment Overview

Results from processing aerial drone imagery using our DINOv2-based tree segmentation pipeline. All images processed with automatic K-selection using elbow method optimization.

## Sample Results

### Experiment 1: Dense Forest Canopy

**Configuration**: `a3f7_v1-5_base_str4_et0-1`
- **Image**: DJI_20250127150117_0029_D.JPG
- **Clusters Detected**: 5 (automatically selected)
- **Processing Time**: ~45 seconds

#### Visualization Outputs

**Edge Overlay**: Original image with colored boundaries
![Edge Overlay](assets/images/a3f7_v1-5_base_str4_et0-1_edge_overlay.jpg)
*Colored boundaries overlaid on original drone imagery with hatch patterns for region distinction*

**Side-by-Side Comparison**: Original vs. Segmentation
![Side by Side](assets/images/a3f7_v1-5_base_str4_et0-1_side_by_side.jpg)
*Left: Original aerial image | Right: Segmentation map with cluster legend*

**Segmentation Legend**: Detailed cluster visualization
![Segmentation Legend](assets/images/a3f7_v1-5_base_str4_et0-1_segmentation_legend.jpg)
*Segmentation map with comprehensive cluster legend and configuration parameters*

**K-Selection Analysis**: Elbow method visualization
![Elbow Analysis](assets/images/a3f7_v1-5_base_str4_et0-1_elbow_analysis.jpg)
*WCSS curve showing automatic K=5 selection with elbow threshold analysis*

---

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