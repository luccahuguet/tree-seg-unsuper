# V3 Implementation - Tree-Specific Segmentation

**Date:** 2025-11-18
**Status:** ✅ Implementation Complete - Ready for OAM-TCD Evaluation
**Version:** V3.0

---

## Overview

V3 implements tree-specific segmentation logic that transforms V1.5's general clustering into precise tree detection. It combines RGB-based vegetation indices with IoU-driven cluster selection and watershed instance segmentation.

**Key Innovation:** Exploits tree-specific domain knowledge (spectral signature, shape characteristics) without requiring multispectral data or supervised learning.

---

## Architecture

### Pipeline Flow

```
V1.5 Clusters → Vegetation Filter → Cluster Selection → Instance Segmentation → Tree Instances
     ↓                ↓                    ↓                      ↓                    ↓
  K-means         ExG/CIVE          IoU + VegScore         Watershed           Final Trees
```

### Components

**1. Vegetation Indices** (`tree_seg/tree_focus/vegetation_indices.py`)
- **ExG** (Excess Green): `2*G - R - B` - Highlights vegetation
- **CIVE** (Color Index): `0.441*R - 0.881*G + 0.385*B + 18.787` - Plant segmentation
- **Green Ratio**: `G / (R+G+B)` - Simple but effective

**2. Cluster Selection** (`tree_seg/tree_focus/cluster_selection.py`)
- Computes IoU between clusters and vegetation mask
- Calculates vegetation score within each cluster
- Filters by area (GSD-aware), IoU threshold, vegetation score

**3. Instance Segmentation** (`tree_seg/tree_focus/instance_segmentation.py`)
- Distance transform finds tree centers
- Watershed separates merged crowns
- Shape filtering (circularity, eccentricity, area)

**4. V3 Pipeline** (`tree_seg/tree_focus/v3_pipeline.py`)
- Orchestrates full workflow
- Configurable thresholds and presets
- Returns structured results with statistics

---

## Configuration

### V3Config Parameters

```python
@dataclass
class V3Config:
    # Vegetation filtering
    vegetation_method: str = 'exg'  # 'exg', 'cive', 'green_ratio', 'combined'
    vegetation_threshold: float = 0.0  # Method-specific (None = auto from preset)
    vegetation_preset: str = 'balanced'  # 'permissive', 'balanced', 'strict'

    # Cluster selection
    iou_threshold: float = 0.3  # Min IoU with vegetation
    veg_score_threshold: float = 0.4  # Min vegetation score
    min_tree_area_m2: float = 1.0  # Min tree area (m²)
    max_tree_area_m2: float = 500.0  # Max tree area (m²)

    # Instance segmentation
    watershed_min_distance: int = 10  # Min distance between trees (pixels)
    min_circularity: float = 0.3  # Min shape circularity
    max_eccentricity: float = 0.95  # Max eccentricity

    # Morphology
    morphology_kernel_size: int = 3
    morphology_operation: str = 'close'  # 'open', 'close', 'both', 'none'

    # GSD
    gsd_cm: float = 10.0  # Ground Sample Distance (cm/pixel)
```

### Presets

| Preset | Use Case | IoU | VegScore | Area (m²) | Distance |
|--------|----------|-----|----------|-----------|----------|
| **permissive** | Max recall, catch small trees | 0.2 | 0.3 | 0.5-1000 | 8px |
| **balanced** | Default, balanced P/R | 0.3 | 0.4 | 1.0-500 | 10px |
| **strict** | Max precision, confident trees only | 0.4 | 0.5 | 2.0-300 | 12px |

---

## Usage

### Basic Usage

```python
from tree_seg.tree_focus import V3Pipeline, create_v3_preset
import cv2

# Load image and V1.5 results
image = cv2.imread("path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
v1_clusters = ...  # Load V1.5 cluster labels

# Run V3 with balanced preset
config = create_v3_preset('balanced')
pipeline = V3Pipeline(config)
results = pipeline.process(image, v1_clusters)

# Access results
print(f"Detected {results.num_trees} trees")
print(f"Vegetation coverage: {results.vegetation_mask.mean():.1%}")

# Use instance labels for downstream tasks
instance_mask = results.instance_labels  # (H, W) with unique ID per tree
```

### Custom Configuration

```python
from tree_seg.tree_focus import V3Pipeline, V3Config

config = V3Config(
    vegetation_method='exg',
    iou_threshold=0.35,  # Stricter IoU
    min_tree_area_m2=2.0,  # Larger minimum
    gsd_cm=5.0  # Higher resolution imagery
)

pipeline = V3Pipeline(config)
results = pipeline.process(image, v1_clusters)
```

---

## Outputs

### V3Results Structure

```python
@dataclass
class V3Results:
    # Masks
    vegetation_mask: np.ndarray  # Binary (H, W)
    tree_mask: np.ndarray  # Binary tree mask after selection (H, W)
    instance_labels: np.ndarray  # Instance IDs (H, W), 0=background

    # Statistics
    num_trees: int  # Count of detected trees
    cluster_stats: List[Dict]  # Per-cluster metrics
    instance_stats: List[Dict]  # Per-tree metrics

    # Intermediate (for debugging)
    input_clusters: np.ndarray  # V1.5 clusters
    selected_cluster_ids: List[int]  # Which clusters became trees
```

### Cluster Statistics

Each cluster gets:
- `cluster_id`: Original cluster ID
- `is_tree`: Boolean (passed filters?)
- `iou`: IoU with vegetation mask
- `vegetation_score`: Vegetation likelihood [0, 1]
- `area_m2`: Real-world area
- `coverage_ratio`: Fraction of vegetation mask covered

### Instance Statistics

Each tree instance gets:
- `area`: Area in pixels
- `area_m2`: Area in m²
- `perimeter`: Perimeter length
- `circularity`: Shape circularity [0, 1]
- `eccentricity`: Elongation measure [0, 1]

---

## Files Created

### Core Modules
1. **`tree_seg/tree_focus/__init__.py`** - Package exports
2. **`tree_seg/tree_focus/vegetation_indices.py`** - ExG, CIVE, green ratio
3. **`tree_seg/tree_focus/cluster_selection.py`** - IoU-based filtering
4. **`tree_seg/tree_focus/instance_segmentation.py`** - Watershed + shape filters
5. **`tree_seg/tree_focus/v3_pipeline.py`** - Main pipeline orchestration

### Testing
6. **`scripts/test_v3.py`** - Smoke test with synthetic data

### Documentation
7. **`docs/text/v3_implementation.md`** (this file)

---

## Validation Against OAM-TCD

### Test Setup

```python
from datasets import load_from_disk
from tree_seg.tree_focus import V3Pipeline, create_v3_preset
from tree_seg.evaluation.oam_tcd_eval import OAMTCDEvaluator

# Load OAM-TCD test set
test_data = load_from_disk("data/oam_tcd/test")

# Initialize V3
config = create_v3_preset('balanced')
pipeline = V3Pipeline(config)

# Process test images
for sample in test_data:
    image = sample['image']
    v1_clusters = ...  # Run V1.5 first

    # Run V3
    results = pipeline.process(image, v1_clusters)

    # Save predictions for evaluation
    save_prediction(results.instance_labels, sample['image_id'])

# Evaluate
evaluator = OAMTCDEvaluator("data/oam_tcd")
metrics = evaluator.evaluate("path/to/predictions")
```

### Expected Performance

Based on tree detection literature and V3's design:

| Metric | Conservative | Target | Optimistic |
|--------|--------------|--------|------------|
| **Precision** | 0.45 | 0.55 | 0.65 |
| **Recall** | 0.40 | 0.50 | 0.60 |
| **F1 Score** | 0.42 | 0.52 | 0.62 |
| **Mean IoU** | 0.35 | 0.45 | 0.55 |

**Success criteria**: F1 ≥ 0.48 (beats simple baselines, competitive with zero-shot methods)

---

## Design Decisions

### Why ExG Instead of NDVI?

**NDVI requires NIR** (Near-Infrared), unavailable in RGB-only imagery.
**ExG works on RGB**: `2*G - R - B` exploits the fact that vegetation reflects more green than red/blue.

**Performance**: ExG achieves ~85-90% of NDVI's accuracy for vegetation detection in RGB imagery (Woebbecke et al., 1995).

### Why IoU-Based Selection?

**Robust to cluster imperfections**: Even if K-means creates irregular boundaries, clusters overlapping vegetation are likely trees.

**Complementary to SLIC**: V1.5's SLIC already snaps boundaries to edges. V3's IoU filter adds semantic validation.

### Why Watershed for Instances?

**Standard for touching objects**: Watershed excels at separating merged tree crowns.

**Distance-based markers**: Distance transform naturally finds tree centers (local maxima = canopy peaks).

**No training needed**: Purely geometric, works across tree types and scales.

---

## Tuning Recommendations

### If Precision Too Low (Too many false positives)

1. Increase `iou_threshold` (e.g., 0.3 → 0.4)
2. Increase `veg_score_threshold` (e.g., 0.4 → 0.5)
3. Use 'strict' preset
4. Reduce `max_tree_area_m2` (reject huge clusters)

### If Recall Too Low (Missing trees)

1. Decrease `iou_threshold` (e.g., 0.3 → 0.2)
2. Decrease `veg_score_threshold` (e.g., 0.4 → 0.3)
3. Use 'permissive' preset
4. Reduce `min_tree_area_m2` (catch small trees)
5. Reduce `watershed_min_distance` (split merged crowns better)

### If Instance Segmentation Poor

1. Adjust `watershed_min_distance` (typical tree spacing in pixels)
2. Tune `min_circularity` (relax for irregular crowns)
3. Review `morphology_operation` ('close' fills holes, 'open' removes noise)

---

## Known Limitations

### Current Implementation

1. **RGB-only**: No multispectral (NIR, RedEdge) support yet → V5
2. **No species classification**: Detects trees, doesn't identify species
3. **Static thresholds**: Doesn't adapt to image brightness/contrast
4. **Simple watershed**: Could be enhanced with deep distance estimation

### Failure Modes

**Dense canopy**: Merged crowns may not separate well (watershed limitations)
**Shadows**: Dark shadows reduce green ratio, might reject shaded trees
**Dead trees**: Brown/gray trees fail vegetation filter (need spectral adaptation)
**Small saplings**: Below min_area threshold, filtered out

---

## Next Steps

### Immediate (This Week)

1. **Run on OAM-TCD test set** (439 images)
2. **Evaluate metrics** (precision, recall, F1, mIoU)
3. **Error analysis**: Identify failure modes
4. **Tune thresholds** based on results

### Short-term (Optional)

1. **Compare V3 vs V1.5**: Quantify tree detection improvement
2. **Ablation study**: Test ExG vs CIVE vs combined
3. **Optimize presets**: Data-driven threshold selection

### Future Enhancements (V5+)

1. **Multispectral**: Add NDVI, GNDVI, NDRE for better vegetation detection
2. **Adaptive thresholds**: Auto-tune based on image statistics
3. **Deep distance**: Replace distance transform with learned model
4. **Species classification**: Cluster embeddings by species (V5 metric)

---

## References

### Vegetation Indices
- **ExG**: Woebbecke et al. (1995) "Color indices for weed identification"
- **CIVE**: Kataoka et al. (2003) "Crop growth estimation system using machine vision"

### Watershed Segmentation
- Vincent & Soille (1991) "Watersheds in digital spaces"
- Meyer & Beucher (1990) "Morphological segmentation"

### Tree Detection Benchmarks
- **OAM-TCD**: Restor Foundation (2024) arXiv:2407.11743
- **TreeSatAI**: Schulz et al. (2023) ESSD
- **NEON**: Weinstein et al. (2021) PLOS Computational Biology

---

**Status**: V3 implementation complete. Proceed to OAM-TCD evaluation! 🌳
