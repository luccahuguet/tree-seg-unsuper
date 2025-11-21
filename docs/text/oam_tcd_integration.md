# OAM-TCD Dataset Integration Guide

**Date:** 2025-11-21 (Updated)
**Status:** ⚠️ Limited Use - Good for qualitative validation, not ideal for V3 quantitative metrics

---

## Overview

We've integrated the **OAM-TCD (OpenAerialMap Tree Cover Dataset)** primarily for **qualitative validation** of V3 vegetation filtering and visual inspection of species clustering results. This dataset provides 280k+ annotated tree instances across 4,608 high-resolution aerial images at 10cm GSD.

**Important:** OAM-TCD has **instance-level annotations** (individual trees), but V3 performs **species-level semantic clustering** (grouping by vegetation type). The dataset can validate vegetation filtering but cannot quantitatively measure species clustering quality.

### Why OAM-TCD?

- ✅ **High-quality RGB imagery** (10cm GSD, matches drone imagery)
- ✅ **Large scale** (4,169 train, 439 test images)
- ✅ **Diverse biomes** (global coverage - temperate, tropical, urban)
- ✅ **CC BY 4.0 license** (open research use)
- ⚠️ **Instance annotations only** (no species labels or semantic regions)
- ⚠️ **Cannot validate species clustering** (our primary V3 goal)

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
- **Category 1**: Individual trees (instance-level)
- **Category 2**: Tree canopy groups (semantic regions of closed canopy)

**Note for V3:** Neither category directly validates species clustering. Category 2 (canopy groups) is closer to our semantic approach but lacks species labels.

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
uv run python scripts/download_dataset_oam.py

# Download subset for testing
uv run python scripts/download_dataset_oam.py --subset 100
```

### 2. Inspect Dataset

```bash
# View dataset structure and first sample
uv run python scripts/inspect_dataset_oam.py --visualize

# Output: data/oam_tcd/sample_visualization.png
```

### 3. Visualize V3 on OAM-TCD Images

**Primary use case:** Qualitative assessment of vegetation filtering and species clustering

```bash
# Simple 2-panel visualization (recommended)
uv run python scripts/visualize_v3_simple.py --image 3828

# Specify OAM-TCD image ID directly
uv run python scripts/visualize_v3_simple.py --image 4363 --style hatching

# Color overlay style
uv run python scripts/visualize_v3_simple.py --image 3828 --style color

# Custom ExG threshold
uv run python scripts/visualize_v3_simple.py --image 3828 --threshold 0.15
```

**Output:** 2-panel image showing original + V3 species clusters with filtering statistics

---

## Validation Approach for V3

**Primary validation methods:**
1. Qualitative visual inspection (species clustering)
2. Quantitative recall measurement (vegetation filtering)

### What OAM-TCD Can Validate:

1. ✅ **Vegetation filter recall** (QUANTITATIVE)
   - Metric: What % of labeled tree pixels are classified as vegetation?
   - Result: **69.9% overall recall** (50-sample evaluation)
   - Interpretation: Our ExG filter captures ~70% of ground truth trees
   - Script: `scripts/evaluate_vegetation_filter_oam.py`
   - **⚠️ Precision is MEANINGLESS** - dataset has many unannotated trees/vegetation

2. ✅ **Cluster visual coherence** (QUALITATIVE)
   - Do clusters align with visually distinct vegetation types?
   - Are similar-looking regions grouped together?
   - Metric: Manual assessment of cluster boundaries

3. ✅ **Diverse scene testing** (QUALITATIVE)
   - Dense forests (image 3828: 0% filtered, all vegetation)
   - Mixed urban/vegetation (image 4363: ~50% filtered)
   - Different biomes and lighting conditions

### What OAM-TCD Cannot Validate:

1. ❌ **Vegetation filter precision**
   - Dataset has **incomplete annotations** (many unlabeled trees/vegetation)
   - False positives are often real vegetation, just unannotated
   - Precision metrics are artificially low and meaningless

2. ❌ **Species clustering accuracy**
   - No species labels available
   - Cannot measure if clusters = actual species boundaries
   - Cannot compute species purity metrics

3. ❌ **Quantitative clustering performance**
   - Instance annotations don't match semantic clustering output
   - No meaningful metrics for species-level segmentation quality

### Recommended Validation Workflow:

```bash
# 1. Quantitative recall evaluation (vegetation filter performance)
uv run python scripts/evaluate_vegetation_filter_oam.py --n-samples 50

# 2. Qualitative visualization on diverse samples
uv run python scripts/visualize_v3_simple.py --image 3828  # Dense forest
uv run python scripts/visualize_v3_simple.py --image 4363  # Mixed scene
uv run python scripts/visualize_v3_simple.py --image 545   # Sparse vegetation

# 3. Manual inspection checklist:
#    - Are filtered regions non-vegetation? (gray/hatched areas)
#    - Do cluster boundaries follow visible species changes?
#    - Are similar textures/colors grouped together?
#    - Appropriate number of clusters for scene complexity?
```

### Vegetation Filter Recall Results

**Evaluation:** 50 random test samples (Category 1: Individual trees only)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Overall Recall** | **69.9%** | Filter captures ~70% of labeled tree pixels |
| Mean Recall | 40.3% ± 45.8% | High variance across images |
| Median Recall | 1.1% | Many images have few/no labeled trees |
| High Recall (≥80%) | 38% of images | Excellent performance on dense forests |
| Zero Recall (0%) | 50% of images | Trees filtered out (dark/shadowed/dead) |

**Key Takeaway:** Vegetation filter successfully captures majority of trees, but struggles with:
- Very dark/shadowed trees (ExG fails on low greenness)
- Dead/brown vegetation (no green signal)
- Urban scenes with sparse isolated trees

---

## Integration Timeline

- [x] **Dataset search** (2025-11-18)
- [x] **Download OAM-TCD** (4,608 images)
- [x] **Inspect format** (COCO polygons, instance masks)
- [x] **Implement V3** (species clustering - 2025-11-21)
- [x] **Create visualization script** (`visualize_v3_simple.py`)
- [x] **Document integration** (this file)
- [x] **Qualitative validation** (tested on diverse samples)
- [ ] **Generate V3 visualizations for paper** (5-10 representative samples)
- [ ] **Include in paper discussion** (qualitative results, dataset limitations)

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
1. **`scripts/download_dataset_oam.py`**
   Downloads dataset from HuggingFace, saves to `data/oam_tcd/`

2. **`scripts/inspect_dataset_oam.py`**
   Inspects dataset structure, creates visualizations

3. **`scripts/visualize_v3_simple.py`** ⭐ **Primary V3 visualization tool**
   Creates 2-panel visualizations for qualitative assessment
   - Left: Original image
   - Right: V3 species clusters with filtering statistics
   - Supports OAM-TCD image IDs directly (e.g., `--image 3828`)

### Documentation
4. **`docs/text/oam_tcd_integration.md`** (this file)

5. **`docs/text/dataset_search_context.md`**
   Dataset search criteria and ideal dataset requirements

---

## Known Limitations

### Dataset
- ❌ **No species labels** (RGB only, no multispectral)
- ❌ **Instance-level annotations only** (no species-level semantic regions)
- ⚠️ **Variable tree density** (some images have 1 tree, others have 100+)
- ⚠️ **Mixed biomes** (test set includes tropical, temperate, urban)
- ⚠️ **Annotation quality varies** (human-labeled, some boundary imprecision)

> **📝 Note on Ideal Dataset (2025-11-21):**
> OAM-TCD provides instance-level tree annotations, which don't directly validate our V3 species clustering approach. An **ideal dataset** would have:
> - **Species-level semantic region annotations** (polygons labeled by species: "pine region", "fir region", etc.)
> - **Drone imagery at 5-20cm GSD** (higher resolution than OAM-TCD's 10cm)
> - **Direct validation metric:** Cluster-species alignment (do our DINOv3 clusters match actual species boundaries?)
>
> OAM-TCD remains valuable for vegetation filtering validation (tree vs non-tree), but doesn't validate species separation quality.

### Current Use
- ✅ **Qualitative validation only** - visual inspection of vegetation filtering and cluster coherence
- ⚠️ **No quantitative metrics** - instance annotations incompatible with semantic clustering
- ⚠️ **Limited scope** - cannot validate species separation accuracy

---

## Current Status & Next Steps

### Completed (V3 Implementation)
1. ✅ V3 species clustering implemented (ExG vegetation filtering)
2. ✅ Visualization tools created (`visualize_v3_simple.py`)
3. ✅ Qualitative validation on diverse samples:
   - Dense forests (3828): 0% filtered, all vegetation
   - Mixed scenes (4363): ~50% filtered, soil/roads removed
   - Sparse vegetation (545): High filtering rate

### For Paper
1. **Generate representative visualizations**:
   - Select 5-10 diverse OAM-TCD samples
   - Show range of filtering percentages (0%, 25%, 50%, 75%+)
   - Demonstrate different biomes and scene types
   - Include filtering statistics annotations

2. **Qualitative discussion**:
   - Document vegetation filtering effectiveness
   - Discuss cluster visual coherence
   - Acknowledge lack of species-level ground truth
   - Emphasize need for species-labeled dataset (future work)

### Future Work (V5+)
1. **Find species-labeled dataset**:
   - Drone imagery with species semantic regions (ideal)
   - OR instance annotations + species labels (Quebec Trees, TreeSatAI)
   - Enable quantitative species clustering metrics

2. **Multispectral enhancement**:
   - NDVI/GNDVI for better vegetation detection
   - Spectral signatures for species distinction

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
- **V3 visualization**: Run `uv run python scripts/visualize_v3_simple.py --help`
- **Integration**: Refer to this document

---

**Status**: ✅ V3 implementation complete. OAM-TCD used for qualitative validation of vegetation filtering. Species-labeled dataset needed for quantitative clustering metrics.
