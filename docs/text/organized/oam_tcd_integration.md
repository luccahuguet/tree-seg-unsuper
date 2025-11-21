# OAM-TCD Dataset Integration Guide

**Date:** 2025-11-18
**Status:** ✅ Complete - Ready for V3 validation

---

## Overview

We've integrated the **OAM-TCD (OpenAerialMap Tree Cover Dataset)** for quantitative validation of V3 tree detection performance. This dataset provides 280k+ annotated tree instances across 4,608 high-resolution aerial images at 10cm GSD.

### Why OAM-TCD?

- ✅ **Instance-level tree annotations** (exactly what V3 needs)
- ✅ **COCO-format polygons** (standard, easy to use)
- ✅ **10cm GSD** (matches typical drone imagery)
- ✅ **Large scale** (4,169 train, 439 test images)
- ✅ **CC BY 4.0 license** (open research use)
- ✅ **Diverse biomes** (global coverage)

---

## Dataset Structure

### Location
```
data/oam_tcd/
├── train/              # 4,169 training images
├── test/               # 439 test images
├── dataset_info.json   # Dataset metadata
└── sample_visualization.png  # Example visualization
```

### Features Per Sample
```python
{
  'image_id': int64,                 # Unique image identifier
  'image': PIL.Image,                # RGB image (2048x2048 typically)
  'height': int16,                   # Image height
  'width': int16,                    # Image width
  'annotation': PIL.Image,           # Instance mask (unique ID per tree)
  'coco_annotations': str,           # JSON list of COCO annotations
  'biome_name': str,                 # e.g., "East European forest steppe"
  'lat': float32,                    # Latitude
  'lon': float32,                    # Longitude
  'license': str,                    # e.g., "CC-BY 4.0"
  # ... additional metadata
}
```

### Annotation Categories
- **Category 1**: Individual trees (what V3 targets)
- **Category 2**: Tree canopy groups (closed canopy regions)

**For V3 validation, we use Category 1 only** (individual tree instances).

---

## Dataset Statistics

### Size
- **Total images**: 4,608 (4,169 train + 439 test)
- **Total annotations**: 280,000+ individual trees + 56,000+ canopy groups
- **Resolution**: 10 cm/pixel (0.1m GSD)
- **Image size**: 2048×2048 pixels typical
- **Coverage**: ~20,000 hectares globally

### Geographic Coverage
Diverse biomes including:
- Temperate forests
- Tropical forests
- Mediterranean vegetation
- Steppe ecosystems
- Urban/suburban trees

---

## Usage Instructions

### 1. Download Dataset (Already Done)

```bash
# Download full dataset
uv run python scripts/download_oam_tcd.py

# Download subset for testing
uv run python scripts/download_oam_tcd.py --subset 100
```

### 2. Inspect Dataset

```bash
# View dataset structure and first sample
uv run python scripts/inspect_oam_tcd.py --visualize

# Output: data/oam_tcd/sample_visualization.png
```

### 3. Evaluate V3 Predictions

Once V3 is implemented, evaluate using:

```bash
# Run V3 on OAM-TCD test set (generate predictions)
uv run tree_seg --version v3 \
  --input data/oam_tcd/test/ \
  --output data/oam_tcd/predictions/ \
  --format instance_masks

# Evaluate predictions
uv run python -m tree_seg.evaluation.oam_tcd_eval \
  --dataset data/oam_tcd \
  --predictions data/oam_tcd/predictions/ \
  --output results/oam_tcd_v3_results.json
```

---

## Evaluation Metrics

The evaluation script (`tree_seg/evaluation/oam_tcd_eval.py`) computes:

### Instance-Level Metrics
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall

### IoU Metrics
- **Mean IoU**: Average IoU across all matched instances
- **Median IoU**: Median IoU across all matched instances

### Average Precision
- **AP @ 0.5**: Precision at IoU threshold = 0.5 (COCO standard)
- **AP @ 0.75**: Precision at IoU threshold = 0.75 (stricter)

### Matching Strategy
- Greedy assignment of predictions to ground truth based on IoU
- Can be upgraded to Hungarian matching later if needed
- Default IoU threshold: 0.5 (adjustable via `--iou-threshold`)

---

## Expected Baseline Performance

Based on similar tree detection papers:

| Method | Precision | Recall | F1 | mIoU |
|--------|-----------|--------|----|----- |
| **Supervised (Mask R-CNN)** | 0.80-0.90 | 0.75-0.85 | 0.78-0.87 | 0.65-0.75 |
| **SAM (zero-shot)** | 0.60-0.75 | 0.55-0.70 | 0.58-0.72 | 0.50-0.65 |
| **V3 (target)** | 0.50-0.70 | 0.45-0.65 | 0.48-0.67 | 0.40-0.60 |

**V3 goals:**
- **Precision ≥ 0.50**: Minimize false tree detections
- **Recall ≥ 0.45**: Detect most trees (especially large/dominant ones)
- **F1 ≥ 0.48**: Balanced performance

---

## Integration Timeline

- [x] **Dataset search** (2025-11-18)
- [x] **Download OAM-TCD** (4,608 images)
- [x] **Inspect format** (COCO polygons, instance masks)
- [x] **Create evaluation script** (`oam_tcd_eval.py`)
- [x] **Document integration** (this file)
- [ ] **Implement V3** (next step)
- [ ] **Run V3 evaluation** (after V3 complete)
- [ ] **Compare against V1.5 baseline** (qualitative)
- [ ] **Include in paper results** (Table 2: Tree Detection Performance)

---

## Annotation Format Details

### COCO Annotations (per instance)
```json
{
  "id": 50607,
  "image_id": 661,
  "category_id": 1,              // 1=tree, 2=canopy
  "segmentation": [[x1,y1,x2,y2,...]],  // Polygon coordinates
  "area": 1845966.63,            // Pixel area
  "bbox": [0.0, 140.54, 2048.78, 1912.13],  // [x, y, w, h]
  "iscrowd": 0,
  "extra": {}
}
```

### Instance Mask Format
- PNG image with unique integer ID per tree instance
- 0 = background
- 1, 2, 3, ... = individual tree instances

---

## Files Created

### Scripts
1. **`scripts/download_oam_tcd.py`**
   Downloads dataset from HuggingFace, saves to `data/oam_tcd/`

2. **`scripts/inspect_oam_tcd.py`**
   Inspects dataset structure, creates visualizations

### Evaluation Module
3. **`tree_seg/evaluation/oam_tcd_eval.py`**
   Evaluates V3 predictions against ground truth
   - Classes: `OAMTCDEvaluator`, `TreeDetectionMetrics`
   - Functions: `evaluate_sample()`, `compute_iou()`, `match_predictions_to_ground_truth()`

### Documentation
4. **`docs/text/oam_tcd_integration.md`** (this file)

5. **`docs/text/dataset_search_context.md`**
   Dataset search criteria (used for ChatGPT research)

---

## Known Limitations

### Dataset
- ❌ **No species labels** (RGB only, no multispectral)
- ⚠️ **Variable tree density** (some images have 1 tree, others have 100+)
- ⚠️ **Mixed biomes** (test set includes tropical, temperate, urban)
- ⚠️ **Annotation quality varies** (human-labeled, some boundary imprecision)

### Evaluation
- Current matching is greedy (not Hungarian)
- IoU threshold of 0.5 may be lenient for dense forests
- No evaluation of species classification (dataset lacks labels)

---

## Next Steps

### Immediate (V3 Implementation)
1. Implement V3 tree detection logic:
   - Vegetation filtering (ExG/CIVE)
   - Cluster selection (IoU to veg mask)
   - Instance segmentation (watershed)

2. Run V3 on OAM-TCD test set

3. Evaluate and compare against V1.5

### Future (Post-V3)
1. **Fine-tune evaluation**:
   - Implement Hungarian matching
   - Add AP metrics at multiple IoU thresholds (0.5:0.95)
   - Per-biome breakdown

2. **Error analysis**:
   - Visualize false positives (non-tree detections)
   - Visualize false negatives (missed trees)
   - Identify failure modes (dense canopy, small trees, shadows)

3. **Species validation** (V5):
   - Download Quebec Trees dataset (23k crowns, 14 species)
   - Evaluate species purity metric

---

## Dataset Citation

```bibtex
@article{oam-tcd-2024,
  title={OAM-TCD: A globally diverse dataset of high-resolution tree cover maps},
  author={Restor Foundation},
  journal={arXiv preprint arXiv:2407.11743},
  year={2024},
  url={https://arxiv.org/abs/2407.11743}
}
```

---

## Support

For questions about:
- **Dataset**: See HuggingFace repo https://huggingface.co/restor/tcd
- **Evaluation script**: Check `tree_seg/evaluation/oam_tcd_eval.py` docstrings
- **Integration**: Refer to this document

---

**Status**: Dataset ready for V3 validation. Proceed with V3 implementation! 🚀
