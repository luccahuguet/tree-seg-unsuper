# Parameter Comparison Results

This folder contains edge overlay results from multiple runs with different parameter configurations, enabling direct comparison of model performance.

## Expected Files

Place edge overlay files with different configurations:

```
parameter-comparison/
├── {hash}_v1-5_small_str4_{clustering}_edge_overlay.png
├── {hash}_v1-5_base_str4_{clustering}_edge_overlay.png
├── {hash}_v1-5_large_str4_{clustering}_edge_overlay.png
├── {hash}_v1-5_base_str2_{clustering}_edge_overlay.png
├── {hash}_v1-5_base_str8_{clustering}_edge_overlay.png
└── ... (more parameter variations)
```

## Comparison Parameters

### Model Sizes
- **small**: `dinov2_vits14` - Fast, good for testing
- **base**: `dinov2_vitb14` - Best balance (recommended)
- **large**: `dinov2_vitl14` - Higher quality, slower

### Stride Values
- **str2**: Higher resolution, slower processing
- **str4**: Balanced (recommended)
- **str8**: Faster processing, lower resolution

## Purpose

This comparison demonstrates:
- Impact of different DINOv2 model sizes on segmentation quality
- Effect of stride parameter on boundary precision
- Performance trade-offs between speed and quality
- Optimal parameter selection for different use cases