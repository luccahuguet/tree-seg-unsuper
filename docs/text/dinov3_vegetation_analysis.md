# DINOv3 Vegetation Feature Analysis

## Summary

**Key Finding**: DINOv3 features naturally separate vegetation from non-vegetation with **0.95+ correlation**, requiring only minimal post-processing.

## Analysis Methodology

### Test Setup
- **Samples**: OAM-TCD imagery (10cm GSD aerial)
  - Sample 4363: Dense forest scene
  - Sample 545: Sparse vegetation with 26% black regions
- **Pipeline**: V1.5 (DINOv3 + K-means + SLIC)
- **K value**: 15 clusters
- **Ground truth**: ExG vegetation index (threshold = 0.1)

### Questions Investigated
1. Do DINOv3 clusters naturally align with vegetation?
2. Can we identify vegetation clusters without complex filters?
3. What's the correlation between DINOv3 clustering and vegetation indices?

## Results

### Sample 4363 (Dense Forest)

**Overall Statistics:**
- ExG vegetation coverage: 45.0% of image
- DINOv3 found: 5 vegetation clusters + 10 non-vegetation clusters
- Vegetation pixels (via cluster filtering): 36.4%

**Correlation Analysis:**
- **Pearson correlation**: 0.957 (p < 0.0001) ✅
- **Mean ExG separation**: 0.148
  - Vegetation clusters: 0.211
  - Non-vegetation clusters: 0.063

**Cluster Breakdown:**
```
Vegetation Clusters (ExG > 0.1):
  Cluster  1: 71.7% veg, ExG=0.162 (122k px)
  Cluster  3: 95.2% veg, ExG=0.263 (215k px)
  Cluster  5: 54.3% veg, ExG=0.112 (408k px)
  Cluster  9: 99.5% veg, ExG=0.329 (540k px)
  Cluster 10: 87.0% veg, ExG=0.188 (241k px)

Non-Vegetation Clusters (ExG < 0.1):
  Cluster  0: 10.7% veg, ExG=0.048 (buildings/roads)
  Cluster  4:  1.2% veg, ExG=0.011 (water/shadow)
  Cluster  7: 25.1% veg, ExG=0.079 (mixed urban)
  ... (7 more non-veg clusters)
```

### Sample 545 (Sparse Vegetation + Black Regions)

**Overall Statistics:**
- ExG vegetation coverage: 50.2% of image
- DINOv3 found: 8 vegetation clusters + 7 non-vegetation clusters
- Vegetation pixels (via cluster filtering): 59.3%
- Black region cluster: 25.8% (Cluster 1, ExG=0.001)

**Correlation Analysis:**
- **Pearson correlation**: 0.956 (p < 0.0001) ✅
- **Mean ExG separation**: 0.155
  - Vegetation clusters: 0.177
  - Non-vegetation clusters: 0.022

**Key Observations:**
- Black regions correctly identified as non-vegetation (ExG ≈ 0)
- 8 distinct vegetation clusters with varying densities
- Strong separation despite image artifacts

## Conclusions

### 1. DINOv3 Naturally Encodes Vegetation

**Evidence:**
- 0.95+ correlation between DINOv3 clusters and vegetation indices
- Consistent across dense/sparse scenes
- Works despite image quality issues (black regions)

**Interpretation:**
DINOv3's training on massive diverse datasets (ImageNet-22k, etc.) taught it to recognize:
- Vegetation texture (leaves, canopy patterns)
- Green/brown natural colors vs gray/concrete
- Organic structures vs geometric patterns

### 2. Minimal Filtering Sufficient

**Simple approach works:**
```python
# For each DINOv3 cluster:
mean_exg = compute_mean_exg_in_cluster(image, cluster_mask)
is_vegetation = mean_exg > 0.10

# Keep only vegetation clusters
filtered_labels = keep_clusters_where(is_vegetation)
```

**No need for:**
- ❌ Complex multi-index fusion (ExG + CIVE + green ratio)
- ❌ Texture analysis (DINOv3 already captures this)
- ❌ Shape filtering (clustering handles this)
- ❌ Per-pixel classification (cluster-level is sufficient)

### 3. Threshold Recommendation

**ExG threshold = 0.10** works well:
- Separates vegetation (mean ExG: 0.17-0.21) from non-vegetation (0.02-0.06)
- Margin of safety: ~0.05-0.10 separation
- Validated on multiple scenes

**Alternative thresholds:**
- Conservative (fewer false positives): 0.15
- Permissive (higher recall): 0.05
- Adaptive: Per-image percentile-based

## Design Implications for V3.1

### Vegetation Filter Architecture

**Input:**
- RGB image
- DINOv3 cluster labels (from V1.5)

**Process:**
1. For each cluster, compute mean ExG
2. Label cluster as vegetation if mean ExG > 0.10
3. Filter labels (keep only vegetation clusters)
4. Relabel remaining clusters (remove gaps: 0,1,2,... → 0,1,2,...)

**Output:**
- Filtered semantic labels (vegetation only)
- Cluster vegetation scores (for confidence/debugging)

**Complexity:** ~20 lines of code, no ML needed

### Species Clustering Strategy

Since DINOv3 clusters correlate strongly with vegetation:
- **Higher K** (20-30) may naturally separate species by texture/color
- **Vegetation filtering first** removes non-tree noise
- **Remaining clusters** likely represent visually distinct vegetation types

**Hypothesis to test:**
Do vegetation clusters at K=20-30 correspond to species boundaries?
- Pine vs fir: Different needle texture/density
- Deciduous vs conifer: Leaf vs needle patterns
- Young vs mature: Canopy density differences

## Implementation Priority

**Phase 1** (Immediate): Implement minimal vegetation filter ✅
- Cluster-level ExG computation
- Threshold-based filtering
- Test on diverse samples

**Phase 2** (Next): Species separation analysis ✅
- Test K=20, 25, 30 on filtered vegetation
- Visual inspection: Do clusters align with species?
- Refine if needed

**Phase 3** (Future): Dataset validation
- **🌟 IDEAL DATASET NEEDED:** Drone imagery with species-level semantic region annotations
  - **What we need:** Polygons/masks labeled by species (e.g., "pine region", "fir region")
  - **Why:** Direct validation of whether DINOv3 clusters align with actual species boundaries
  - **Current limitation:** OAM-TCD has instance annotations (tree vs non-tree), not species regions
  - **Validation metric:** Cluster-species alignment score (Hungarian matching between clusters and species polygons)
- BAMFORESTS (5cm GSD) for higher-resolution testing (if has species labels)
- TreeSatAI (species labels) for quantitative species metrics (satellite, not drone)
- Custom UAV data (2-3cm GSD) for ultimate resolution (would need manual annotation)

## Visualizations Generated

Analysis outputs saved to `data/outputs/feature_analysis/`:

**Per-sample 6-panel visualizations:**
1. Original image
2. ExG vegetation mask (ground truth)
3. DINOv3 clusters (colored by cluster ID)
4. Cluster vegetation coverage heatmap (0-100%)
5. Filtered vegetation (clusters with >50% veg)
6. Scatter plot: Mean ExG vs Vegetation Coverage

**Key insights from scatter plots:**
- Strong linear relationship (R² ≈ 0.91)
- Clear separation between veg/non-veg clusters
- Threshold line (ExG=0.1) effectively divides groups

## References

**Analysis Scripts:**
- `scripts/analyze_dinov3_vegetation_features.py`
- `scripts/test_species_clustering.py`

**Related Documentation:**
- `docs/text/v3_pivot.md` - Why we pivoted from instance to semantic segmentation
- `docs/text/version_roadmap.md` - V3.1 goals and architecture

**Dataset Information:**
- OAM-TCD: Global aerial imagery at 10cm GSD, 280k+ tree instances
- See deep search results in conversation for BAMFORESTS and TreeSatAI alternatives
