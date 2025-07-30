---
layout: default
title: "Parameter Comparison"
nav_order: 4
---

{% include navbar.html %}
{% include navbar-styles.html %}

# Parameter Comparison: Model Size and Stride Analysis

This section compares the impact of different DINOv2 model sizes and stride parameters on tree segmentation quality. Results are organized by K-selection method used.

## K-Selection Method Comparison

Two different automatic K-selection methods have been used in this project:

### 🔄 [Curvature Method Results](parameter_comparison_curvature.html)
**Status**: ✅ Complete - Existing results from curvature-based K-selection

- **Method**: Curvature analysis of WCSS curve
- **Threshold**: 0.15 (decimal-based)
- **Behavior**: May exhibit unexpected patterns at different stride values
- **Results**: Full parameter comparison available

### 📊 [Elbow Method Results](parameter_comparison_elbow.html) 
**Status**: 🚧 Pending - Upcoming results from corrected elbow method

- **Method**: Percentage-based diminishing returns analysis
- **Threshold**: 3.5% (intuitive percentage format)
- **Behavior**: More predictable and tunable
- **Results**: To be generated with corrected implementation

---

## Quick Method Comparison

| Aspect | Curvature Method | Elbow Method |
|--------|------------------|--------------|
| **Status** | ✅ Complete | 🚧 Pending |
| **Threshold Format** | Decimal (0.15) | Percentage (3.5%) |
| **Behavior** | Curvature-based | Diminishing returns |
| **Stride Consistency** | ⚠️ Paradoxes observed | ✅ Expected consistency |
| **User Experience** | Less intuitive | More intuitive |
| **Results Available** | Full comparison | Coming soon |

---

## Why Two Methods?

The project initially used a curvature-based approach for automatic K-selection, which generated all current results. However, analysis revealed this wasn't implementing the true "elbow method" and exhibited some unexpected behaviors:

- **Stride 2 Paradox**: Smaller models selected more clusters at stride 2 (inverse of expected)
- **Threshold Confusion**: Using decimals (0.15) instead of intuitive percentages
- **Method Mismatch**: Curvature analysis vs. true elbow method diminishing returns

The corrected elbow method addresses these issues with:
- **True Elbow Logic**: Percentage-based diminishing returns analysis  
- **Intuitive Thresholds**: Users specify 3.5% instead of 0.035
- **Consistent Behavior**: Expected to eliminate stride-dependent paradoxes
- **Predictable Results**: More tunable and understandable K-selection

---

## Navigation

Choose your method of interest:

- **[View Curvature Method Results →](parameter_comparison_curvature.html)** - Complete analysis of existing results
- **[View Elbow Method Results →](parameter_comparison_elbow.html)** - Upcoming corrected implementation results

Both pages follow the same analysis structure for easy comparison once elbow method results are available.