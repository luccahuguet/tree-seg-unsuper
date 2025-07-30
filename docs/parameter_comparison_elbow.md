---
layout: default
title: "Parameter Comparison (Elbow Method)"
nav_order: 5
---

{% include navbar.html %}
{% include navbar-styles.html %}

# Parameter Comparison: Model Size and Stride Analysis (Elbow Method)

This section will compare the impact of different DINOv2 model sizes and stride parameters on tree segmentation quality using the **corrected elbow method** for automatic K-selection. 

🚧 **Status**: Results pending - this page will be populated with new results generated using the corrected elbow method implementation.

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

## Comparison Methodology (Upcoming)

All experiments will use the same source image with identical clustering parameters, varying only:
- **Model Size**: small, base, large, giant
- **Stride Parameter**: 2, 4
- **K-Selection**: True elbow method (threshold=3.5%)

This controlled approach will isolate the impact of each parameter on segmentation quality using consistent elbow-based cluster selection.

---

## Expected Results Structure

### Model Size Comparison (Stride 4)
*Results will show elbow method performance across model sizes*

- **DINOv2 Small**: Expected more predictable K-selection
- **DINOv2 Base**: Balanced performance with elbow method
- **DINOv2 Large**: High-quality results with consistent K-selection  
- **DINOv2 Giant**: Maximum quality with predictable clustering behavior

### Stride Parameter Comparison
*Results will demonstrate elbow method consistency across stride values*

Expected improvements:
- **Consistent K-Selection**: Elbow method should maintain expected model size → cluster count relationship at both stride 2 and 4
- **Eliminated Paradox**: No more inverse K-selection behavior at stride 2
- **Predictable Thresholds**: 3.5% threshold should work consistently across configurations

---

## Key Questions to be Answered

1. **Does elbow method eliminate the stride 2 K-selection paradox?**
   - Curvature method showed inverse behavior (smaller models → more clusters at stride 2)
   - Elbow method should maintain consistent model size → cluster count relationship

2. **How does percentage-based threshold affect different model sizes?**
   - Same 3.5% threshold across all model sizes
   - Expected: consistent sensitivity regardless of feature dimensionality

3. **What are the optimal threshold values for different use cases?**
   - Conservative: 5-10% (fewer clusters)
   - Balanced: 3-5% (moderate clustering)
   - Sensitive: 1-3% (more clusters)

---

## Comparison with Curvature Method

Once results are available, this section will provide direct comparisons:

### Expected Advantages of Elbow Method
- **Eliminated Paradoxes**: Consistent behavior across stride values
- **Tunable Sensitivity**: Clear threshold → cluster count relationship
- **Intuitive Configuration**: Percentage-based thresholds
- **Predictable Results**: Less variation in K-selection behavior

### Performance Comparison
- **Quality**: Expected similar or improved segmentation quality
- **Consistency**: More predictable K-selection across parameters
- **Usability**: Easier threshold tuning with percentage values

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
- 🚧 **Result Generation**: Awaiting new parameter comparison results
- 🚧 **Analysis**: Comparative analysis with curvature method pending

---

## Related Documentation

- [Parameter Comparison (Curvature Method)](parameter_comparison_curvature.html) - Existing results using curvature-based K-selection
- [Methodology](methodology.html) - Detailed explanation of segmentation approach
- [Complete Example](complete_example.html) - End-to-end workflow example