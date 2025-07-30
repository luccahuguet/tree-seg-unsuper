---
layout: default
title: "Parameter Comparison (Curvature Method)"
nav_order: 4
---

{% include navbar.html %}
{% include navbar-styles.html %}

# Parameter Comparison: Model Size and Stride Analysis (Curvature Method)

This section compares the impact of different DINOv2 model sizes and stride parameters on tree segmentation quality using the **curvature-based K-selection method**. All results shown were generated using the legacy curvature algorithm for automatic cluster selection.

⚠️ **Method Note**: These results use a curvature-based approach for automatic K-selection, which has been identified as different from the true elbow method. For results using the corrected elbow method implementation, see the [Elbow Method Parameter Comparison](parameter_comparison_elbow.html).

## Comparison Methodology

All experiments use the same source image with identical clustering parameters, varying only:
- **Model Size**: small, base, large, giant
- **Stride Parameter**: 2, 4
- **K-Selection**: Curvature-based algorithm (threshold=0.15)

This controlled approach isolates the impact of each parameter on segmentation quality using consistent curvature-based cluster selection.

---

## Model Size Comparison (Stride 4)

All results use identical parameters (stride=4, curvature_threshold=0.15) with only model size varying, enabling direct quality comparison.

### DINOv2 Small (`dinov2_vits14`)
*Fast processing, good for testing and rapid iteration*

![Small Model]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_small_str4_et0-15_edge_overlay.jpg)

**Characteristics**:
- **Memory**: Lowest GPU memory usage
- **Feature Dimension**: 384
- **Processing Speed**: Fastest
- **Curvature K-Selection**: K=4
- **Use Case**: Rapid prototyping, resource-constrained environments

---

### DINOv2 Base (`dinov2_vitb14`) - Recommended
*Optimal balance of quality and performance*

![Base Model]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_base_str4_et0-15_edge_overlay.jpg)

**Characteristics**:
- **Memory**: Moderate GPU memory usage
- **Feature Dimension**: 768
- **Processing Speed**: Balanced
- **Curvature K-Selection**: K=5
- **Use Case**: Production workflows, research applications

---

### DINOv2 Large (`dinov2_vitl14`)
*Higher quality features, slower processing*

![Large Model]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_large_str4_et0-15_edge_overlay.jpg)

**Characteristics**:
- **Memory**: Higher GPU memory usage
- **Feature Dimension**: 1024
- **Processing Speed**: Slower
- **Curvature K-Selection**: K=5
- **Use Case**: High-quality requirements, detailed analysis

---

### DINOv2 Giant (`dinov2_vitg14`)
*Maximum quality features, slowest processing*

![Giant Model]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_giant_str4_et0-15_edge_overlay.jpg)

**Characteristics**:
- **Memory**: Highest GPU memory usage
- **Feature Dimension**: 1536
- **Processing Speed**: Slowest
- **Curvature K-Selection**: K=6
- **Use Case**: Maximum quality applications, research benchmarks

---

## Analysis: Model Size Impact (Curvature Method)

The comparison reveals clear relationships between model size and segmentation quality using curvature-based K-selection:

### Key Observations

1. **Feature Richness**: Larger models capture more nuanced tree boundary details
2. **Curvature K-Selection**: Optimal cluster count increases with model size (K=4, 5, 5, 6 for Small, Base, Large, Giant respectively)
3. **Clustering Granularity**: Higher-dimensional features enable detection of more distinct tree regions
4. **Computational Trade-offs**: Each model size step significantly increases processing time and memory usage
5. **Diminishing Returns**: Quality improvements become smaller as model size increases
6. **Practical Balance**: Base model provides excellent results for most applications

### Performance vs Quality Trade-offs (Curvature Method)

| Model | Feature Dim | Curvature K | Relative Speed | Memory Usage | Best Use Case |
|-------|-------------|-------------|----------------|--------------|---------------|
| Small | 384 | 4 | Fastest | Lowest | Testing, prototyping |
| Base | 768 | 5 | Balanced | Moderate | Production workflows |
| Large | 1024 | 5 | Slower | Higher | High-quality analysis |
| Giant | 1536 | 6 | Slowest | Highest | Research benchmarks |

---

## Stride Parameter Comparison (Curvature Method)

Comparing stride values shows the resolution vs performance trade-off. Lower stride values provide higher resolution features but require more processing time.

### Stride 2 vs Stride 4 Comparison

**Small Model (dinov2_vits14)**

| Stride 2 | Stride 4 |
|----------|----------|
| ![Small Stride 2]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_small_str2_et0-15_edge_overlay.jpg) | ![Small Stride 4]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_small_str4_et0-15_edge_overlay.jpg) |

**Base Model (dinov2_vitb14)**

| Stride 2 | Stride 4 |
|----------|----------|
| ![Base Stride 2]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_base_str2_et0-15_edge_overlay.jpg) | ![Base Stride 4]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_base_str4_et0-15_edge_overlay.jpg) |

**Large Model (dinov2_vitl14)**

| Stride 2 | Stride 4 |
|----------|----------|
| ![Large Stride 2]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_large_str2_et0-15_edge_overlay.jpg) | ![Large Stride 4]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_large_str4_et0-15_edge_overlay.jpg) |

**Giant Model (dinov2_vitg14)**

| Stride 2 | Stride 4 |
|----------|----------|
| ![Giant Stride 2]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_giant_str2_et0-15_edge_overlay.jpg) | ![Giant Stride 4]({{ site.baseurl }}/results/parameter_comparison/curvature/d111_v1-5_giant_str4_et0-15_edge_overlay.jpg) |

---

## Analysis: Stride Impact (Curvature Method)

The stride parameter significantly affects both processing time and feature resolution with curvature-based K-selection:

### Key Observations

1. **Resolution**: Stride 2 provides ~4x higher spatial resolution than stride 4
2. **Detail Capture**: Lower stride captures finer tree boundary details and smaller trees
3. **Processing Time**: Stride 2 requires ~4x more computation than stride 4
4. **Memory Usage**: Lower stride increases memory requirements proportionally
5. **Quality vs Speed**: Diminishing returns - stride 2 improvements may not justify 4x slowdown
6. **Curvature K-Selection Paradox**: Stride 2 shows inverse relationship - smaller models select more clusters (K=7,6,4,4 for Small,Base,Large,Giant vs K=4,5,5,6 for stride 4)

### Stride Recommendations (Curvature Method)

| Stride | Resolution | Speed | Memory | Curvature Behavior | Best Use Case |
|--------|------------|--------|---------|-------------------|---------------|
| 2 | Highest | Slowest | Highest | Inverse K-selection | Maximum detail requirements |
| 4 | Balanced | Moderate | Moderate | Expected K-selection | Production workflows |
| 8 | Lower | Fastest | Lowest | Not recommended | Rapid prototyping only |

**Note**: Stride 8 is not recommended for production use due to significant quality degradation.

---

## Combined Analysis (Curvature Method)

The edge overlay visualizations enable direct visual comparison of tree boundary detection quality across both model sizes and stride parameters using curvature-based K-selection:

- **Model Size**: At stride 4, larger models select more clusters (K=4,5,5,6 for Small,Base,Large,Giant)
- **Stride Parameter**: Lower stride provides higher resolution but with computational costs
- **Curvature K-Selection Paradox**: Stride 2 reverses the trend - smaller models find more clusters (K=7,6,4,4), suggesting higher resolution may overwhelm larger models' clustering capability when using curvature analysis
- **Sweet Spot**: Base model with stride 4 offers optimal balance for most applications
- **Maximum Quality**: Large model with stride 2 for detailed analysis requiring fine boundaries

---

## Method Comparison

For comparison with the corrected elbow method implementation, see:
- [Parameter Comparison (Elbow Method)](parameter_comparison_elbow.html) - Results using the true elbow method (coming soon)

**Key Differences**:
- **Curvature Method** (this page): Uses curvature analysis for K-selection, may exhibit unexpected behaviors at different stride values
- **Elbow Method** (upcoming): Uses percentage-based diminishing returns analysis for more predictable K-selection behavior