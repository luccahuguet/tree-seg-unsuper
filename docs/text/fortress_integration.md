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

## Related Files

**Scripts:**
- `scripts/preprocess_fortress.py` - Dataset preprocessing
- `scripts/inspect_fortress_shapefiles.py` - Shapefile inspection
- `scripts/test_fortress_dataloader.py` - Dataloader testing

**Code:**
- `tree_seg/evaluation/datasets.py` - `FortressDataset` class

**Docs:**
- `docs/text/dataset_comparison_species_segmentation.md` - Dataset comparison
- `docs/text/v3_species_clustering.md` - V3 architecture

---

**Status:** Ready for V3 benchmarking 🌲
