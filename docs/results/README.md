# Results Folder

**Place your tree segmentation output images here.**

## Usage

1. **Run your segmentation** using the modern API:
   ```python
   from tree_seg import segment_trees
   results = segment_trees("image.jpg", model="base", auto_k=True)
   ```

2. **Copy output files** from `/kaggle/working/output/` to this `results/` folder

3. **Enable web optimization** during generation:
   ```python
   results = segment_trees("image.jpg", web_optimize=True)
   ```

## Expected Files

Your segmentation will generate files with config-based naming:

```
results/
├── a3f7_v1-5_base_str4_et0-1_segmentation_legend.png
├── a3f7_v1-5_base_str4_et0-1_edge_overlay.png
├── a3f7_v1-5_base_str4_et0-1_side_by_side.png
└── a3f7_v1-5_base_str4_et0-1_elbow_analysis.png
```

## File Naming

The smart naming encodes all parameters:
- `a3f7`: Hash of source image
- `v1-5`: Algorithm version
- `base`: Model size  
- `str4`: Stride parameter
- `et0-1`: Elbow threshold 0.1

## Optimization

The `optimize_images.py` script will:
- ✅ Compress 7MB → 1-2MB (maintain quality)
- ✅ Convert PNG → optimized JPEG
- ✅ Copy to `docs/assets/images/` for GitHub Pages
- ✅ Show compression statistics

Perfect for professional presentation to your professor!