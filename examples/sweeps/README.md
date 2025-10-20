# Documentation Image Generation Sweep

This directory contains sweep configurations for generating all images used in the project documentation.

## Overview

The `docs_image_generation.json` sweep generates comprehensive results for:

- **Methodology**: Pipeline demonstration images
- **Complete Example**: Full workflow visualization  
- **Parameter Comparison**: Model size and stride analysis
- **Analysis**: Performance profile comparisons

## Usage

### 1. Run the Documentation Sweep

```bash
uv run python scripts/generate_docs_images.py data/input/forest2.jpeg
```

This will generate results in `data/output/sweeps/` with organized subdirectories for each configuration and automatically organize them into the proper `docs/results/` structure expected by the Jekyll documentation site.

## Generated Image Structure

After running the script, images will be organized as:

```
docs/results/
├── methodology/
│   ├── basic_example_segmentation_legend.jpg
│   ├── basic_example_edge_overlay.jpg
│   ├── basic_example_side_by_side.jpg
│   └── basic_example_elbow_analysis.jpg
├── complete_example/
│   ├── basic_example_segmentation_legend.jpg
│   ├── basic_example_edge_overlay.jpg
│   ├── basic_example_side_by_side.jpg
│   └── basic_example_elbow_analysis.jpg
├── parameter_comparison/
│   ├── stride/
│   │   ├── stride_comparison_str2_edge_overlay.jpg
│   │   └── stride_comparison_str4_edge_overlay.jpg
│   ├── model_size/
│   │   ├── model_comparison_small_edge_overlay.jpg
│   │   ├── model_comparison_base_edge_overlay.jpg
│   │   ├── model_comparison_large_edge_overlay.jpg
│   │   └── model_comparison_giant_edge_overlay.jpg
│   ├── elbow_threshold/
│   │   ├── elbow_threshold_2_5_edge_overlay.jpg
│   │   ├── elbow_threshold_5_0_edge_overlay.jpg
│   │   ├── elbow_threshold_10_0_edge_overlay.jpg
│   │   └── elbow_threshold_20_0_edge_overlay.jpg
│   └── refinement/
│       ├── refine_with_slic_edge_overlay.jpg
│       └── refine_none_edge_overlay.jpg
└── analysis/
    ├── refine_with_slic_segmentation_legend.jpg (quality profile)
    └── basic_example_edge_overlay.jpg (speed profile)
```

## Sweep Configuration Details (12 Total)

### 1. Basic Example
- **Model**: Base (dinov2_vitb14), **Profile**: Balanced, **Stride**: 4
- **Purpose**: Core pipeline demonstration for methodology and complete example docs

### 2-3. Stride Comparison
- **Model**: Giant (dinov2_vitg14), **Profile**: Quality, **Strides**: 2 vs 4
- **Purpose**: Quality vs speed trade-off analysis

### 4-7. Model Size Comparison  
- **Models**: Small, Base, Large, Giant, **Profile**: Quality, **Stride**: 2
- **Purpose**: Systematic comparison of feature dimensionality impact

### 8-11. Elbow Threshold Comparison
- **Model**: Giant, **Profile**: Quality, **Stride**: 2, **Thresholds**: 2.5%, 5.0%, 10.0%, 20.0%
- **Purpose**: Clustering sensitivity analysis

### 11-12. Refinement Comparison
- **Model**: Giant, **Profile**: Quality, **Stride**: 2, **Refinement**: With/without SLIC
- **Purpose**: Post-processing impact analysis

## Documentation Integration

The generated images are automatically referenced in:

- `docs/methodology.md` - Pipeline demonstration
- `docs/complete_example.md` - Full workflow example
- `docs/parameter_comparison_elbow.md` - Model/stride comparisons  
- `docs/analysis.md` - Performance analysis

## Regenerating Images

To update documentation images:

1. Run the complete generation script: `uv run python scripts/generate_docs_images.py data/input/forest2.jpeg`
2. Commit updated images to repository

The script automatically handles sweep execution and image organization.

## Notes

- Uses `data/input/forest.jpg` as the standard test image for consistency
- Default elbow threshold is 5.0%; threshold comparison includes 2.5%, 5.0%, 10.0%, 20.0%
- Images are generated in web-optimized JPEG format
- Sweep takes approximately 15-20 minutes to complete all configurations
