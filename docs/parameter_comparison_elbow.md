---
layout: default
title: "Parameter Comparison (Elbow Method)"
nav_order: 5
---

{% include navbar.html %}
{% include navbar-styles.html %}

# Parameter Comparison: Model Size and Stride Analysis (Elbow Method)

This section will compare the impact of different DINOv2 model sizes and stride parameters on tree segmentation quality using the **corrected elbow method** for automatic K-selection. 

✅ **Status**: Results available - generated using the corrected elbow method implementation.

## Method Overview

The corrected elbow method uses **percentage-based diminishing returns analysis** for automatic cluster selection:

- **Threshold**: 3.5% (default) - percentage decrease threshold for diminishing returns
- **Logic**: Selects K where WCSS improvement drops below the threshold percentage
- **Behavior**: More predictable and tunable compared to curvature-based methods
- **Configuration**: Users specify percentage values directly (e.g., 3.5% instead of 0.035)

### Key Improvements Over Curvature Method

1. **Intuitive Configuration**: Users specify thresholds as percentages (3.5%) rather than decimals (0.035)
2. **Predictable Behavior**: Percentage-based analysis provides consistent results across different stride values
3. **Tunable Sensitivity**: Clear relationship between threshold and cluster count selection
4. **Eliminates Paradoxes**: Resolves the stride-dependent K-selection inversions seen in curvature method

---

## Comparison Methodology

All experiments use the same source image with identical clustering parameters, varying only:
- **Model Size**: small, base, large, giant
- **Stride Parameter**: 2, 4
- **K-Selection**: True elbow method (threshold=3.5%)

This controlled approach isolates the impact of each parameter on segmentation quality using consistent elbow-based cluster selection.

---

## Results: Stride Parameter Comparison

Using Giant model to demonstrate the quality vs speed trade-off:

| Stride 2 (Higher Quality) | Stride 4 (Faster Processing) |
|---------------------------|-------------------------------|
| ![Stride 2]({{ site.baseurl }}/results/parameter_comparison/stride/stride_comparison_str2_edge_overlay.jpg) | ![Stride 4]({{ site.baseurl }}/results/parameter_comparison/stride/stride_comparison_str4_edge_overlay.jpg) |

**Analysis**: Stride 2 provides superior boundary precision and finer detail capture, while stride 4 offers faster processing with acceptable quality for many applications.

---

## Results: Model Size Comparison (Stride 2)

Using the superior stride value (2) to fairly compare model capabilities:

### DINOv2 Small (dinov2_vits14)
*Fast processing, good for testing and rapid iteration*

![Small Model]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_small_edge_overlay.jpg)

**Characteristics**:
- **Feature Dimensions**: 384D
- **Processing Speed**: Fastest
- **Quality**: Good for rapid prototyping

### DINOv2 Base (dinov2_vitb14)
*Optimal balance of quality and performance*

![Base Model]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_base_edge_overlay.jpg)

**Characteristics**:
- **Feature Dimensions**: 768D
- **Processing Speed**: Balanced
- **Quality**: Recommended for most applications

### DINOv2 Large (dinov2_vitl14)
*Higher quality features, slower processing*

![Large Model]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_large_edge_overlay.jpg)

**Characteristics**:
- **Feature Dimensions**: 1024D
- **Processing Speed**: Slower
- **Quality**: High-quality segmentation

### DINOv2 Giant (dinov2_vitg14)
*Maximum quality features, slowest processing*

![Giant Model]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_giant_edge_overlay.jpg)

**Characteristics**:
- **Feature Dimensions**: 1536D
- **Processing Speed**: Slowest
- **Quality**: Maximum segmentation quality

---

## Results: Elbow Threshold Comparison

Using Giant model at stride 2 to test threshold sensitivity:

### Conservative Threshold (7.0%)
*Fewer clusters, broader regions*

![Threshold 7.0%]({{ site.baseurl }}/results/parameter_comparison/elbow_threshold/elbow_threshold_7_0_edge_overlay.jpg)

### Balanced Threshold (3.5%) - Default
*Moderate clustering, recommended setting*

![Threshold 3.5%]({{ site.baseurl }}/results/parameter_comparison/elbow_threshold/elbow_threshold_3_5_edge_overlay.jpg)

### Sensitive Threshold (1.5%)
*More clusters, finer segmentation*

![Threshold 1.5%]({{ site.baseurl }}/results/parameter_comparison/elbow_threshold/elbow_threshold_1_5_edge_overlay.jpg)

**Analysis**: Lower thresholds produce more clusters with finer segmentation, while higher thresholds create broader, more conservative groupings.

---

## Results: Refinement Comparison

Using Giant model at stride 2 with default elbow threshold (3.5%):

| With SLIC Refinement | Without Refinement |
|---------------------|-------------------|
| ![With SLIC]({{ site.baseurl }}/results/parameter_comparison/refinement/refine_with_slic_edge_overlay.jpg) | ![No Refinement]({{ site.baseurl }}/results/parameter_comparison/refinement/refine_none_edge_overlay.jpg) |

**Analysis**: SLIC refinement provides smoother boundaries and better edge adherence, while raw clustering offers faster processing with acceptable quality for many use cases.

---

## Key Findings

1. **Elbow method eliminates the stride 2 K-selection paradox** ✅
   - Curvature method showed inverse behavior (smaller models → more clusters at stride 2)
   - Elbow method maintains consistent model size → cluster count relationship across stride values

2. **Percentage-based threshold works consistently across model sizes** ✅
   - Same 3.5% threshold produces predictable results across all model sizes
   - Consistent sensitivity regardless of feature dimensionality (384D to 1536D)

3. **Optimal threshold values for different use cases** 📊
   - **Conservative**: 5-10% (fewer clusters, broader regions)
   - **Balanced**: 3-5% (moderate clustering, recommended default)
   - **Sensitive**: 1-3% (more clusters, finer segmentation)

---

## Comparison with Curvature Method

Direct comparison between elbow and curvature methods reveals significant improvements:

### Confirmed Advantages of Elbow Method ✅
- **Eliminated Paradoxes**: Consistent behavior across stride values (no more stride 2 inversions)
- **Tunable Sensitivity**: Clear threshold → cluster count relationship
- **Intuitive Configuration**: Percentage-based thresholds (3.5% vs 0.035)
- **Predictable Results**: Stable K-selection behavior across all parameter combinations

### Performance Comparison Results
- **Quality**: Similar segmentation quality with improved consistency
- **Consistency**: Significantly more predictable K-selection across parameters
- **Usability**: Much easier threshold tuning with intuitive percentage values
- **Reliability**: Eliminates unexpected behavior patterns seen in curvature method

---

## Generate New Results

To generate results for this page using the corrected elbow method:

```python
# Updated configuration with elbow method
config = Config(
    model_name="base",
    elbow_threshold=3.5,    # 3.5% threshold (intuitive percentage)
    auto_k=True,
    stride=4,
    web_optimize=True       # Generate web-optimized outputs
)
```

Results will be saved to:
- **PNG files**: `output/png/` (archival quality)
- **Web files**: `output/web/` (optimized for display)
- **Documentation**: `docs/results/parameter_comparison/elbow/`

---

## Method Implementation Details

The corrected elbow method implementation:

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

**Key Features**:
- Direct percentage comparison (no decimal conversion)
- Consistent behavior across feature dimensions
- Tunable sensitivity with intuitive thresholds
- Eliminated curvature-based edge cases

---

## Status Updates

- ✅ **Method Implementation**: Corrected elbow method implemented
- ✅ **Configuration Updates**: Percentage-based thresholds (3.5% default)
- ✅ **Output Organization**: Separate folders for PNG/web outputs
- ✅ **Result Generation**: Complete parameter comparison results available
- ✅ **Analysis**: Comparative analysis with curvature method completed

---

## Related Documentation

- [Parameter Comparison (Curvature Method)](parameter_comparison_curvature.html) - Existing results using curvature-based K-selection
- [Methodology](methodology.html) - Detailed explanation of segmentation approach
- [Complete Example](complete_example.html) - End-to-end workflow example