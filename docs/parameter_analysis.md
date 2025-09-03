---
layout: default
title: "Parameter Analysis"
nav_order: 4
---

{% include navbar.html %}
{% include navbar-styles.html %}

# Parameter Analysis: Complete Model and Configuration Study

## V1.5 Foundation: Parameter Analysis for Future Architecture

This comprehensive analysis establishes the baseline performance characteristics for our current DINOv3 + K-means architecture (V1.5), providing the foundation for evaluating future advances in V2 (U2Seg) and beyond. These findings directly inform the design decisions for upcoming architectural versions.

## Overview: 13-Configuration Systematic Study

Our comprehensive analysis covers all critical parameters through systematic comparison:
- **[Basic Example](#basic-example-pipeline-demonstration)**: Core pipeline demonstration
- **[Model Comparison](#model-size-comparison)** (4 configs): Small, Base, Large, Giant feature impact
- **[Stride Analysis](#stride-parameter-analysis)** (2 configs): Quality vs speed trade-offs  
- **[Threshold Sensitivity](#elbow-threshold-sensitivity)** (4 configs): Clustering granularity (2x2 grid)
- **[Refinement Impact](#refinement-impact-analysis)** (2 configs): Post-processing effects
- **[Performance Matrix](#performance-comparison-matrix)**: Speed/quality trade-off guidance
- **[Research Findings](#key-research-findings)**: V1.5 insights and V2+ implications

For configuration profiles, reproduction instructions, and technical implementation details, see **[Technical Implementation]({{ '/technical_implementation' | relative_url }})**.

---

## Basic Example: Pipeline Demonstration

![Basic Example]({{ site.baseurl }}/results/methodology/basic_example_edge_overlay.jpg)
*Configuration: Base model, stride 4, 5.0% threshold, SLIC refinement*

This baseline configuration demonstrates the complete pipeline using balanced settings suitable for most forestry applications. The base model provides good quality with reasonable processing time, making it ideal for methodology demonstration.

---

## Model Size Comparison

### DINOv3 Model Variants

| Small (21M) | Base (86M) |
|-------------|------------|
| ![Small]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_small_edge_overlay.jpg) | ![Base]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_base_edge_overlay.jpg) |
| Fast, good quality | Balanced performance |

| Large (304M) | Giant (1.1B) |
|--------------|--------------|
| ![Large]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_large_edge_overlay.jpg) | ![Giant]({{ site.baseurl }}/results/parameter_comparison/model_size/model_comparison_giant_edge_overlay.jpg) |
| High quality | Maximum quality |

| Model | Features | Parameters | Speed | Quality | Use Case |
|-------|----------|------------|-------|---------|----------|
| **Small** (dinov3_vits16) | 384D | 21M | Fastest | Good | Rapid prototyping |
| **Base** (dinov3_vitb16) | 768D | 86M | Balanced | Very Good | Recommended default |
| **Large** (dinov3_vitl16) | 1024D | 304M | Slower | Excellent | High-quality results |
| **Giant** (dinov3_vith16plus) | 1280D | 1.1B | Slowest | Maximum | Research/publication |

**Key Finding**: Linear progression in cluster detection capability - larger models consistently identify more granular tree distinctions.

---

## Stride Parameter Analysis

### Quality vs Speed Trade-off

| Stride 2 (Higher Quality) | Stride 4 (Faster Processing) |
|---------------------------|-------------------------------|
| ![Stride 2]({{ site.baseurl }}/results/parameter_comparison/stride/stride_comparison_str2_edge_overlay.jpg) | ![Stride 4]({{ site.baseurl }}/results/parameter_comparison/stride/stride_comparison_str4_edge_overlay.jpg) |
| Giant model, stride 2 | Giant model, stride 4 |

**Analysis**: 
- **Stride 2**: Superior boundary precision, 2x processing time
- **Stride 4**: Acceptable quality, 2x faster processing
- **Recommendation**: Use stride 2 for final results, stride 4 for development

---

## Elbow Threshold Sensitivity

### Clustering Granularity Control (2x2 Grid)

| Sensitive (2.5%) | Balanced (5.0%) |
|---------------------|-----------------|
| ![2.5%]({{ site.baseurl }}/results/parameter_comparison/elbow_threshold/elbow_threshold_2_5_edge_overlay.jpg) | ![5.0%]({{ site.baseurl }}/results/parameter_comparison/elbow_threshold/elbow_threshold_5_0_edge_overlay.jpg) |
| More clusters, finer segmentation | Optimal clustering - recommended default |

| Conservative (10.0%) | Very Conservative (20.0%) |
|-----------------|---------------------|
| ![10.0%]({{ site.baseurl }}/results/parameter_comparison/elbow_threshold/elbow_threshold_10_0_edge_overlay.jpg) | ![20.0%]({{ site.baseurl }}/results/parameter_comparison/elbow_threshold/elbow_threshold_20_0_edge_overlay.jpg) |
| Balanced clustering | Fewer clusters, broader regions |

**Threshold Guidelines** (✅ **Validated**):
- **2.5%**: Sensitive detection — more clusters; fine distinctions
- **5.0%**: Balanced clustering — recommended default for most forestry applications
- **10.0%**: Conservative clustering — broader tree groupings
- **20.0%**: Very conservative — major forest regions only
- **Behavior**: Lower thresholds → more clusters, higher thresholds → fewer clusters as expected

---

## Refinement Impact Analysis

### Post-Processing Comparison

| With SLIC Refinement | Without Refinement |
|----------------------|---------------------|
| ![With SLIC]({{ site.baseurl }}/results/parameter_comparison/refinement/refine_with_slic_edge_overlay.jpg) | ![No Refinement]({{ site.baseurl }}/results/parameter_comparison/refinement/refine_none_edge_overlay.jpg) |
| Smoother boundaries, publication-ready | Faster processing, development use |

**Analysis**:
- **With SLIC**: Smoother boundaries, ~15% processing overhead, publication-ready
- **Without**: Faster processing, acceptable for development and testing

---

## Performance Comparison Matrix

| Configuration | Model | Stride | Threshold | Refinement | Expected K | Time | Quality |
|--------------|-------|--------|-----------|------------|------------|------|---------|
| Development | Small | 4 | 5.0% | None | 3-4 | ~15s | Good |
| Balanced | Base | 4 | 5.0% | SLIC | 4-5 | ~35s | Very Good |
| Research | Giant | 2 | 5.0% | SLIC | 5-7 | ~100s | Excellent |

---

## Key Research Findings

### 1. Model Size Impact
- **Feature Quality Varies**: Different models provide varying feature representations affecting clustering quality
- **Diminishing Returns**: Base→Large shows significant improvement, Large→Giant marginal
- **Optimal Choice**: Base model provides best quality/speed balance for V1.5
- **V2+ Implications**: U2Seg's unified approach should improve clustering quality regardless of model size

### 2. Elbow Method Validation (Foundation for Advanced Methods)
- **Consistent Behavior**: Eliminates stride-dependent paradoxes seen in curvature methods
- **Predictable Thresholds**: 2.5%-20.0% range covers broad segmentation spectrum
- **Intuitive Configuration**: Percentage-based thresholds provide direct control over clustering sensitivity
- **Bug Resolution**: Fixed index mapping and safety override issues that masked threshold sensitivity
- **V2+ Integration**: This robust K-selection framework will serve as fallback for U2Seg's adaptive clustering

### 3. Configuration Interactions (Architectural Design Lessons)
- **Compound Effects**: Giant + stride 2 + SLIC = maximum quality baseline
- **Efficiency**: Small + stride 4 + no refinement = maximum speed baseline
- **Balance**: Base + stride 4 + default settings = optimal general use
- **V3+ Planning**: DynaSeg's dynamic fusion should eliminate need for manual configuration trade-offs

### 4. Future Architecture Targets
Based on these V1.5 findings, upcoming versions target:
- **V2 (U2Seg)**: Improve segmentation quality without increasing computational cost
- **V3 (DynaSeg)**: Eliminate manual threshold tuning through dynamic parameter adaptation  
- **V4 (Multispectral)**: Maintain current processing speeds while adding multi-band capability


For detailed configuration profiles, reproduction steps, and technical implementation details, see **[Technical Implementation]({{ '/technical_implementation' | relative_url }})**.

---

## Next Steps

This parameter analysis provides the foundation for advancing to V2+ architectures. The validated threshold sensitivity and performance characteristics will inform the design of:

- **V2 (U2Seg)**: Advanced unified segmentation methods
- **V3 (DynaSeg)**: Dynamic parameter adaptation
- **V4 (Multispectral)**: Multi-band imagery support
