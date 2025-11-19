# V3 Pivot: From Instance to Species Segmentation

## What Happened

**Original V3**: Individual tree crown instance segmentation using watershed
- Vegetation filtering (ExG) → IoU-based cluster selection → Watershed separation
- Goal: Detect and count individual trees
- Result: 2,437 predictions vs 199 ground truth (12x over-detection)

**Critical Discovery**: Wrong problem, wrong approach, wrong dataset

## Why V3 Instance Segmentation Failed

### 1. Misaligned Goal
**What we actually need**: Species-level semantic segmentation
- Cluster regions by species/vegetation type (pines, firs, grass, etc.)
- Separate vegetation from non-vegetation
- Multiple disconnected regions of same species OK

**What V3 was doing**: Individual tree crown detection
- Watershed splitting large vegetation clusters into "trees"
- Counting individual tree instances
- Massive over-detection because it segments every green blob

### 2. Wrong Dataset (OAM-TCD)
OAM-TCD is **not designed for instance segmentation**:
- Contains "group of trees" class (large polygons, not individuals)
- Explicitly does not segment within closed canopy
- Incomplete annotations (visible trees without labels)
- Designed for restoration/coverage, not tree counting

**Our metrics were meaningless**:
- "False positives" were often real trees, just unlabeled
- 0.4% precision reflected dataset incompleteness, not pipeline failure
- Comparing instance predictions against semantic ground truth

### 3. Watershed Over-Segmentation
Root cause of 12x over-detection:
- V1.5 creates large semantic clusters (e.g., "all pines" = 11,000 m²)
- V3 applied area filter (max 1,000 m²) **before** watershed → rejected all clusters
- Fixed by moving area filter to post-watershed
- But this revealed deeper problem: watershed splits vegetation into thousands of tiny "trees"

**Example**: Dense pine forest
- V1.5: 1 large pine cluster ✅ (correct for species segmentation)
- V3: 200 individual "trees" ❌ (over-segments, not what we need)

## V3.1: The Correct Approach

### Architecture
```
DINOv3 features (encode texture, color, species differences)
    ↓
K-means clustering (group visually similar regions)
    ↓
SLIC refinement (clean boundaries)
    ↓
Vegetation filtering (keep only veg clusters)
    ↓
Output: Species-level semantic map
```

**No instance segmentation needed!** DINOv3 + K-means already creates species-level clusters.

### Key Insights

1. **V1.5 was closer to the answer** than V3
   - Already does semantic clustering
   - Just needs vegetation filtering

2. **DINOv3 naturally encodes species differences**
   - Pine texture ≠ fir texture ≠ grass texture
   - Color patterns differ by species
   - With higher K (15-20), clusters may naturally align with species

3. **Multiple regions of same species = different labels**
   - Label 3 = pine patch A
   - Label 7 = pine patch B
   - Both visually similar (pine-like features)
   - Not species classification, just visual clustering

### Evaluation Strategy

**Without labeled species data**, we evaluate on:
1. **Vegetation separation**: Does it filter out soil/roads/buildings?
2. **Visual consistency**: Do clusters align with visible species boundaries?
3. **Feature similarity**: Are similar-looking regions getting similar cluster labels?
4. **Qualitative inspection**: Manual review of outputs

## Implementation Plan

### Phase 1: Test V1.5 with Higher K
- Run V1.5 with K=15, 20, 25 on diverse samples
- Visually inspect if clusters naturally separate species
- Check if DINOv3 features already do the work

### Phase 2: Add Vegetation Filtering
- Compute ExG/NDVI per cluster
- Keep clusters where >50% pixels are vegetation
- Relabel remaining clusters (remove gaps)

### Phase 3 (Optional): Cluster Similarity Merging
- For adjacent clusters with similar DINOv3 features → consider merging
- Helps reduce noise from over-clustering
- May not be needed if higher K already works well

## Lessons Learned

1. **Start with simple baselines** before adding complexity
2. **Understand dataset design** before evaluating on it
3. **Question assumptions** when metrics look wrong
4. **Visual inspection** reveals ground truth issues that metrics hide
5. **User feedback** (black regions, missing annotations) catches problems early

## Files Changed

- `README.md`: Updated project goal to species segmentation
- `docs/text/version_roadmap.md`: Deprecated V3, added V3.1 scope
- `docs/text/v3_pivot.md`: This document

## Next Steps

1. ✅ Update documentation
2. ⏳ Commit current state
3. ⏳ Test V1.5 with K=15-25
4. ⏳ Implement vegetation filtering
5. ⏳ Visual evaluation on diverse samples
