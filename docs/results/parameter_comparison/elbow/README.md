# Parameter Comparison Results - Elbow Method

This folder is reserved for results generated using the **corrected elbow method**.

## Method Details

- **Algorithm**: Percentage-based diminishing returns analysis for automatic K-selection
- **Threshold**: 3.5% (intuitive percentage format)
- **Generation Date**: Pending
- **Status**: 🚧 Awaiting result generation

## Expected Improvements

The corrected elbow method should address issues found in the curvature method:

### Eliminated Paradoxes
- **Consistent K-Selection**: Expected model size → cluster count relationship at all stride values
- **No Stride Inversions**: Larger models should consistently select more clusters

### Expected Stride 4 Results
- Small model: K=3-4 (expected)
- Base model: K=4-5 (expected)
- Large model: K=5-6 (expected)
- Giant model: K=6-7 (expected)

### Expected Stride 2 Results
- Small model: K=3-4 (consistent with stride 4)
- Base model: K=4-5 (consistent with stride 4)
- Large model: K=5-6 (consistent with stride 4)
- Giant model: K=6-7 (consistent with stride 4)

## File Naming Convention

When generated, files will follow: `d111_v1-5_{model}_str{stride}_et3-5_{type}.{ext}`

Where:
- `d111`: Image hash (first 4 characters)
- `v1-5`: Pipeline version
- `{model}`: Model size (small, base, large, giant)
- `str{stride}`: Stride parameter (2 or 4)
- `et3-5`: Elbow threshold 3.5% (true percentage format)
- `{type}`: Output type (edge_overlay, segmentation_legend, etc.)
- `{ext}`: File extension (png for original, jpg for web-optimized)

## Output Organization

Results will be organized with dual-folder structure:
- **PNG files**: High-quality originals in `output/png/`
- **Web files**: Optimized for display in `output/web/`

## Generation Instructions

To generate results for this folder:

```python
# Use corrected elbow method configuration
config = Config(
    model_name="base",           # or "small", "large", "giant"
    elbow_threshold=3.5,         # 3.5% threshold (intuitive)
    auto_k=True,
    stride=4,                    # or 2 for high-resolution
    web_optimize=True            # Generate both PNG and web outputs
)

# Process and save results
segmenter = TreeSegmentation(config)
results, paths = segmenter.process_and_visualize("image.jpg")
```

## Usage in Documentation

These results will be displayed in:
- [Parameter Comparison (Elbow Method)](../../parameter_comparison_elbow.html)

## Method Comparison

For comparison with existing curvature method results, see:
- [Parameter Comparison (Curvature Method)](../../parameter_comparison_curvature.html)