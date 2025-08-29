---
layout: default
title: "Parameter Analysis"
nav_order: 4
---

{% include navbar.html %}
{% include navbar-styles.html %}

# Parameter Analysis: Complete Model and Configuration Study

This section provides comprehensive analysis of all parameters affecting tree segmentation quality, including model sizes, stride values, elbow thresholds, and refinement options.

## Overview: 12-Configuration Sweep

Our systematic analysis covers:
- **1 Basic Example**: Core pipeline demonstration
- **2 Stride Comparisons**: Quality vs speed trade-offs
- **4 Model Size Comparisons**: Feature dimensionality impact
- **3 Elbow Threshold Comparisons**: Clustering sensitivity
- **2 Refinement Comparisons**: Post-processing effects

---

## Model Size Comparison

### DINOv2 Model Variants

| Model | Features | Speed | Quality | Use Case |
|-------|----------|-------|---------|----------|
| **Small** (dinov2_vits14) | 384D | Fastest | Good | Rapid prototyping |
| **Base** (dinov2_vitb14) | 768D | Balanced | Very Good | Recommended default |
| **Large** (dinov2_vitl14) | 1024D | Slower | Excellent | High-quality results |
| **Giant** (dinov2_vitg14) | 1536D | Slowest | Maximum | Research/publication |

### Model Size Results (Stride 2, Quality Profile)

![Small Model]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_small_edge_overlay.jpg)
*Small Model: Fast processing, good for testing*

![Base Model]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_base_edge_overlay.jpg)
*Base Model: Optimal balance of quality and performance*

![Large Model]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_large_edge_overlay.jpg)
*Large Model: Higher quality features, slower processing*

![Giant Model]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_giant_edge_overlay.jpg)
*Giant Model: Maximum quality features, slowest processing*

**Key Finding**: Linear progression in cluster detection capability - larger models consistently identify more granular tree distinctions.

---

## Stride Parameter Analysis

### Quality vs Speed Trade-off

| Stride 2 (Higher Quality) | Stride 4 (Faster Processing) |
|---------------------------|-------------------------------|
| ![Stride 2]({{ site.baseurl }}/results/parameter_comparison/stride/stride_comparison_str2_edge_overlay.jpg) | ![Stride 4]({{ site.baseurl }}/results/parameter_comparison/stride/stride_comparison_str4_edge_overlay.jpg) |

**Analysis**: 
- **Stride 2**: Superior boundary precision, 2x processing time
- **Stride 4**: Acceptable quality, 2x faster processing
- **Recommendation**: Use stride 2 for final results, stride 4 for development

---

## Elbow Threshold Sensitivity

### Clustering Granularity Control

![Conservative (7.0%)]({{ site.baseurl }}/results/parameter_comparison/elbow_threshold/elbow_threshold_7_0_edge_overlay.jpg)
*Conservative Threshold (7.0%): Fewer clusters, broader regions*

![Balanced (3.5%)]({{ site.baseurl }}/results/parameter_comparison/elbow_threshold/elbow_threshold_3_5_edge_overlay.jpg)
*Balanced Threshold (3.5%): Moderate clustering - recommended default*

![Sensitive (1.5%)]({{ site.baseurl }}/results/parameter_comparison/elbow_threshold/elbow_threshold_1_5_edge_overlay.jpg)
*Sensitive Threshold (1.5%): More clusters, finer segmentation*

**Threshold Guidelines**:
- **Conservative (5-10%)**: Broad regions, major tree groups
- **Balanced (3-5%)**: Optimal for most forestry applications  
- **Sensitive (1-3%)**: Fine-grained species differentiation

---

## Refinement Impact Analysis

### Post-Processing Comparison

| With SLIC Refinement | Without Refinement |
|---------------------|-------------------|
| ![With SLIC]({{ site.baseurl }}/results/parameter_comparison/refinement/refine_with_slic_edge_overlay.jpg) | ![No Refinement]({{ site.baseurl }}/results/parameter_comparison/refinement/refine_none_edge_overlay.jpg) |

**Analysis**:
- **With SLIC**: Smoother boundaries, ~15% processing overhead, publication-ready
- **Without**: Faster processing, acceptable for development and testing

---

## Performance Comparison Matrix

| Configuration | Model | Stride | Threshold | Refinement | Expected K | Time | Quality |
|--------------|-------|--------|-----------|------------|------------|------|---------|
| Development | Small | 4 | 3.5% | None | 3-4 | ~15s | Good |
| Balanced | Base | 4 | 3.5% | SLIC | 4-5 | ~35s | Very Good |
| Research | Giant | 2 | 3.5% | SLIC | 5-7 | ~100s | Excellent |

---

## Key Research Findings

### 1. Model Size Impact
- **Linear Progression**: Small (K=3-4) → Giant (K=5-7)
- **Diminishing Returns**: Base→Large shows significant improvement, Large→Giant marginal
- **Optimal Choice**: Base model provides best quality/speed balance

### 2. Elbow Method Validation
- **Consistent Behavior**: Eliminates stride-dependent paradoxes seen in curvature methods
- **Predictable Thresholds**: 1.5%-7.0% range covers full segmentation spectrum
- **Intuitive Configuration**: Percentage-based thresholds (3.5% vs 0.035)

### 3. Configuration Interactions
- **Compound Effects**: Giant + stride 2 + SLIC = maximum quality
- **Efficiency**: Small + stride 4 + no refinement = maximum speed
- **Balance**: Base + stride 4 + default settings = optimal general use

---

## Recommended Profiles

### Development Profile (Fast Iteration)
```python
config = Config(
    model_name="small",
    stride=4,
    elbow_threshold=3.5,
    refine="none",
    profile="speed"
)
```
- **Time**: ~15 seconds
- **Use**: Rapid prototyping, parameter tuning

### Production Profile (Balanced)
```python
config = Config(
    model_name="base", 
    stride=4,
    elbow_threshold=3.5,
    refine="slic",
    profile="balanced"
)
```
- **Time**: ~35 seconds  
- **Use**: Standard forestry analysis

### Research Profile (Maximum Quality)
```python
config = Config(
    model_name="giant",
    stride=2, 
    elbow_threshold=3.5,
    refine="slic",
    profile="quality"
)
```
- **Time**: ~100 seconds
- **Use**: Publication-quality results

---

## Generate Results

To reproduce all parameter analysis results:

```bash
python generate_docs_images.py input/forest2.jpeg
```

This generates all 12 configurations and organizes results into the documentation structure shown above.

---

## Method Comparison: Elbow vs Curvature

### Elbow Method Advantages ✅
- **Eliminated Paradoxes**: Consistent behavior across stride values
- **Intuitive Thresholds**: Percentage-based (3.5% vs 0.035)
- **Predictable Results**: Clear threshold → cluster count relationship
- **Tunable Sensitivity**: Easy to adjust for different applications

### Implementation Details
```python
def find_optimal_k_elbow(features_flat, k_range=(3, 10), elbow_threshold=3.5):
    """
    Find optimal K using percentage-based diminishing returns.
    
    Args:
        elbow_threshold: Percentage threshold (e.g., 3.5 = 3.5%)
    """
    # Calculate percentage decrease between consecutive K values
    for i, pct in enumerate(pct_decrease):
        if pct < elbow_threshold:  # Direct percentage comparison
            optimal_k = k_values[i]
            break
```

The corrected elbow method provides consistent, predictable K-selection across all parameter combinations, eliminating the unexpected behaviors observed in curvature-based approaches.