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
python run_segmentation.py input/forest.jpg base output --sweep sweeps/docs_image_generation.json --verbose
```

This will generate results in `output/sweeps/` with organized subdirectories for each configuration.

### 2. Organize Images for Documentation

```bash
python scripts/organize_docs_images.py
```

This script moves the generated images into the proper `docs/results/` structure expected by the Jekyll documentation site.

## Generated Image Structure

After running both commands, images will be organized as:

```
docs/results/
├── methodology/
│   ├── forest_v1-5_base_str4_et3-5_segmentation_legend.jpg
│   ├── forest_v1-5_base_str4_et3-5_edge_overlay.jpg
│   ├── forest_v1-5_base_str4_et3-5_side_by_side.jpg
│   └── forest_v1-5_base_str4_et3-5_elbow_analysis.jpg
├── complete_example/
│   ├── forest_v1-5_base_str4_et3-5_segmentation_legend.jpg
│   ├── forest_v1-5_base_str4_et3-5_edge_overlay.jpg
│   ├── forest_v1-5_base_str4_et3-5_side_by_side.jpg
│   └── forest_v1-5_base_str4_et3-5_elbow_analysis.jpg
├── parameter_comparison/elbow/
│   ├── forest_v1-5_small_str4_et3-5_edge_overlay.jpg
│   ├── forest_v1-5_base_str4_et3-5_edge_overlay.jpg
│   ├── forest_v1-5_large_str4_et3-5_edge_overlay.jpg
│   ├── forest_v1-5_giant_str4_et3-5_edge_overlay.jpg
│   ├── forest_v1-5_small_str2_et3-5_edge_overlay.jpg
│   ├── forest_v1-5_base_str2_et3-5_edge_overlay.jpg
│   ├── forest_v1-5_large_str2_et3-5_edge_overlay.jpg
│   └── forest_v1-5_giant_str2_et3-5_edge_overlay.jpg
└── analysis/
    ├── forest_v1-5_base_str4_et3-5_segmentation_legend.jpg (quality profile)
    ├── forest_v1-5_base_str4_et3-5_edge_overlay.jpg (speed profile)
    └── forest_v1-5_base_str4_et3-5_elbow_analysis.jpg
```

## Sweep Configuration Details

### Methodology Demo
- **Model**: Base (dinov2_vitb14)
- **Profile**: Balanced
- **Purpose**: Demonstrate core pipeline functionality

### Complete Example  
- **Model**: Base (dinov2_vitb14)
- **Profile**: Balanced
- **Purpose**: Full workflow with all 4 visualization types

### Parameter Comparison
- **Models**: Small, Base, Large, Giant
- **Strides**: 2, 4
- **Purpose**: Systematic comparison of model size and stride impact

### Analysis Profiles
- **Quality Profile**: 1280px, enhanced SLIC, no PCA
- **Speed Profile**: 896px, PCA 128D, optimized SLIC
- **Purpose**: Performance trade-off demonstration

## Documentation Integration

The generated images are automatically referenced in:

- `docs/methodology.md` - Pipeline demonstration
- `docs/complete_example.md` - Full workflow example
- `docs/parameter_comparison_elbow.md` - Model/stride comparisons  
- `docs/analysis.md` - Performance analysis

## Regenerating Images

To update documentation images:

1. Modify sweep configuration if needed
2. Run the sweep: `python run_segmentation.py input/forest.jpg base output --sweep sweeps/docs_image_generation.json`
3. Organize results: `python scripts/organize_docs_images.py`
4. Commit updated images to repository

## Notes

- Uses `input/forest.jpg` as the standard test image for consistency
- All configurations use elbow method with 3.5% threshold
- Images are generated in web-optimized JPEG format
- Sweep takes approximately 15-20 minutes to complete all configurations