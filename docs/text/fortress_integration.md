# FORTRESS Dataset Integration

**Date:** 2025-12-02  
**Status:** ✅ Preprocessed and Integrated  
**Source:** Schiefer et al. 2020 (KIT)

---

## Overview

FORTRESS (Forest Tree Species Segmentation) provides **species-level semantic annotations** for aerial forest imagery - exactly what V3 needs for validation.

**Key Advantage:** Unlike OAM-TCD (which has generic "tree groups"), FORTRESS has pixel-accurate labels for 13 distinct tree species.

---

## Dataset Characteristics

**Imagery:**
- RGB orthomosaics (TIF format)
- High-resolution aerial photography
- German temperate forests (KIT region)

**Annotations:**
- 13 species classes (originally vector polygons)
- Non-contiguous species_ID values (0, 2, 4, 5, 6, 8-15)
- Includes deadwood and forest floor classes

**Species Classes:**
| ID | Species |
|----|---------|
| 0 | Picea abies (Spruce) |
| 2 | Acer pseudoplatanus (Maple) |
| 4 | Fagus sylvatica (Beech) |
| 5 | Fraxinus excelsior (Ash) |
| 6 | Quercus spec. (Oak) |
| 8 | Deadwood |
| 9 | Forest floor |
| 10 | Abies alba (Fir) |
| 11 | Larix decidua (Larch) |
| 13 | Pinus sylvestris (Pine) |
| 14 | Pseudotsuga menziesii (Douglas Fir) |
| 15 | Betula pendula (Birch) |

---

## Preprocessing Pipeline

**Original Format:**
```
data/fortress/10.35097-538/data/dataset/
├── orthomosaic/          # RGB TIF files (*_ortho.tif)
└── shapefile/            # Vector polygons (poly_*.shp)
```

**Preprocessing Script:** `scripts/preprocess_fortress.py`

**Output Structure:**
```
data/fortress_processed/
├── images/               # Symlinks to orthomosaics
├── labels/               # Rasterized semantic masks (*_label.tif)
└── species_mapping.txt   # ID → species name mapping
```

**Processing Steps:**
1. Scan shapefiles to collect all species IDs
2. For each orthomosaic-shapefile pair:
   - Rasterize vector polygons to pixel masks
   - Match CRS and dimensions
   - Save as uint8 TIF with species_ID values
3. Create symlinks to original images

**Usage:**
```bash
python scripts/preprocess_fortress.py \
  --data-dir data/fortress/10.35097-538/data/dataset \
  --output-dir data/fortress_processed \
  --extract-ortho  # Optional: extract orthomosaic.zip
```

---

## Dataloader

**Module:** `tree_seg/evaluation/datasets.py`

**Class:** `FortressDataset`

**Interface:**
```python
from tree_seg.evaluation.datasets import FortressDataset

dataset = FortressDataset("data/fortress_processed")

# Get sample
image, label, image_id = dataset[0]
# image: (H, W, 3) RGB array [0-255]
# label: (H, W) species_ID array (int64)
# image_id: str (e.g., "CFB003")

# Dataset info
print(f"Samples: {len(dataset)}")
print(f"Classes: {FortressDataset.NUM_CLASSES}")
print(f"Species: {FortressDataset.CLASS_NAMES}")
```

**Testing:**
```bash
python scripts/test_fortress_dataloader.py
```

---

## Why FORTRESS for V3?

**Perfect Match:**
- ✅ Species-level semantic annotations (not instances)
- ✅ Pixel-accurate ground truth
- ✅ Multiple species in same scene
- ✅ RGB-only (matches V3 input)

**Validation Plan:**
1. Apply V3 to FORTRESS orthomosaics
2. Compare V3 clusters against actual species labels
3. Evaluate if visual similarity clusters align with species boundaries
4. Quantify species separation capability

**Limitations:**
- Smaller dataset than OAM-TCD
- Single geographic region (German forests)
- Temperate species only

---

## Evaluation Modes

FORTRESS supports all evaluation modes via the generic `BenchmarkRunner`:

### 1. Unsupervised Semantic Segmentation (V1.5, V2)
- **Method:** DINOv3 → K-means → SLIC (optional)
- **Evaluation:** Hungarian matching between clusters and species classes
- **Metrics:** mIoU, pixel accuracy, per-class IoU
- **Use case:** Test baseline clustering quality

**How it works:**
- Predictions are cluster indices (0 to K-1)
- Ground truth has species class indices
- Hungarian algorithm finds optimal cluster→species mapping
- Evaluates matched predictions against ground truth

### 2. Species Clustering with Vegetation Filter (V3)
- **Method:** V1.5 + ExG vegetation filtering
- **Evaluation:** Hungarian matching on filtered vegetation clusters only
- **Metrics:** Same as above, but for vegetation regions
- **Use case:** Evaluate species-level semantic segmentation

**Pipeline:**
```
DINOv3 → K-means → SLIC → Vegetation Filter (ExG)
    ↓
Filtered clusters (0=background, 1-N=vegetation)
    ↓
Hungarian matching → mIoU vs ground truth species
```

### 3. Supervised Baseline (V4)
- **Method:** Mask2Former with DINOv3 backbone
- **Evaluation:** Direct class comparison (no Hungarian matching)
- **Use case:** Performance ceiling comparison

---

## Running Benchmarks

### Basic V3 Evaluation

```bash
python scripts/evaluate_fortress.py \
    --dataset data/fortress_processed \
    --method v3 \
    --model base \
    --exg-threshold 0.10 \
    --save-viz
```

**Output:**
- Results JSON: `data/output/results/fortress_v3_*/results.json`
- Visualizations: `data/output/results/fortress_v3_*/visualizations/`

### V1.5 Baseline

```bash
python scripts/evaluate_fortress.py \
    --dataset data/fortress_processed \
    --method v1.5 \
    --model base \
    --elbow-threshold 10.0 \
    --num-samples 5
```

### Configuration Options

**Model size:**
- `--model small` - ViT-S/14 (fastest)
- `--model base` - ViT-B/14 (balanced)
- `--model large` - ViT-L/14 (best quality)

**Clustering:**
- `--clustering kmeans` - K-means only
- `--clustering slic` - K-means + SLIC refinement

**K selection:**
- `--elbow-threshold 5.0` - Auto K via elbow method
- `--fixed-k 20` - Fixed number of clusters

**V3 specific:**
- `--exg-threshold 0.10` - Vegetation filtering threshold

---

## Evaluation Metrics

### mIoU (mean Intersection over Union)
- Primary metric for semantic segmentation quality
- Range: 0.0 (worst) to 1.0 (perfect)
- Averaged across all species classes
- **Hungarian matching:** Automatically finds best cluster→species assignment

### Pixel Accuracy
- Fraction of correctly classified pixels
- Less informative than mIoU (can be high with class imbalance)

### Per-Class IoU
- IoU for each individual species
- Identifies which species are well/poorly segmented
- Useful for understanding model strengths/weaknesses

### Confusion Matrix
- Shows cluster→class assignment frequencies
- Available in results JSON
- Reveals common confusion patterns

---

## Expected Results

**V1.5 Baseline (no vegetation filter):**
- mIoU: ~0.15-0.30 (depends on K and species distribution)
- Many non-vegetation clusters reduce overall performance

**V3 (with vegetation filter):**
- mIoU: ~0.25-0.45 (higher due to vegetation-only focus)
- Background correctly filtered out
- Better species separation in dense forests

**Note:** Actual performance depends on:
- Model size (small/base/large)
- K selection (auto vs fixed)
- SLIC refinement (enabled/disabled)
- ExG threshold (V3 only)

---

## Related Files

**Scripts:**
- `scripts/evaluate_fortress.py` - **New** evaluation script
- `scripts/preprocess_fortress.py` - Dataset preprocessing
- `scripts/inspect_fortress_shapefiles.py` - Shapefile inspection
- `scripts/test_fortress_dataloader.py` - Dataloader testing

**Code:**
- `tree_seg/evaluation/datasets.py` - `FortressDataset` class
- `tree_seg/evaluation/benchmark.py` - Generic `BenchmarkRunner`
- `tree_seg/evaluation/metrics.py` - Evaluation metrics
- `tree_seg/evaluation/cli.py` - Shared CLI arguments

**Docs:**
- `docs/text/dataset_comparison_species_segmentation.md` - Dataset comparison
- `docs/text/v3_species_clustering.md` - V3 architecture
- `docs/text/benchmarking.md` - General benchmarking guide

---

**Status:** Ready for V3 benchmarking 🌲
