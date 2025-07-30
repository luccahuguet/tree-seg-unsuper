# Parameter Comparison Results - Curvature Method

This folder contains results generated using the **curvature-based K-selection method**.

## Method Details

- **Algorithm**: Curvature analysis of WCSS curve for automatic K-selection
- **Threshold**: 0.15 (decimal format)
- **Generation Date**: Generated before elbow method correction
- **Status**: ✅ Complete dataset

## Files

All files follow the legacy naming convention: `d111_v1-5_{model}_str{stride}_et0-15_edge_overlay.jpg`

Where:
- `d111`: Image hash (first 4 characters)
- `v1-5`: Pipeline version
- `{model}`: Model size (small, base, large, giant)
- `str{stride}`: Stride parameter (2 or 4)
- `et0-15`: Elbow threshold 0.15 (actually curvature threshold)

**Note**: These files use the legacy naming format. New outputs will include the actual K value: `d111_v1-5_{model}_str{stride}_et3-5_k{actual_k}_{type}.{ext}`

## Known Behaviors

### Stride 4 Results (Expected Pattern)
- Small model: K=4
- Base model: K=5  
- Large model: K=5
- Giant model: K=6

### Stride 2 Results (Paradoxical Pattern)
- Small model: K=7 ⚠️
- Base model: K=6 ⚠️
- Large model: K=4
- Giant model: K=4

⚠️ **Note**: The stride 2 results show an inverse relationship where smaller models select more clusters, which is unexpected and may be due to the curvature method's behavior at higher resolutions.

## Usage in Documentation

These results are displayed in:
- [Parameter Comparison (Curvature Method)](../../parameter_comparison_curvature.html)

## Method Comparison

For comparison with the corrected elbow method, see:
- [Parameter Comparison (Elbow Method)](../../parameter_comparison_elbow.html) - Coming soon