# Research Ideas & Future Directions

This document captures potential improvements and research directions for the tree segmentation system.

---

## Over-Clustering + Region Merging

**Problem:** Some classes exhibit multiple visual patterns within the same semantic category.

**Examples:**
- Tree species with sun-exposed (lighter) vs shaded (darker) canopy regions
- Seasonal variation (flowering vs non-flowering parts of same species)
- Age/health variation (young vs mature foliage)
- Viewing angle effects (top-down vs side-view of same object)

**Current Limitation:** Smart-K uses ground truth class count, forcing DINOv3 features to collapse multi-pattern classes into single clusters. This may cause:
- Poor cluster purity (mixing different visual patterns)
- Suboptimal boundaries (averaging between distinct patterns)
- Information loss (DINOv3 may naturally separate these patterns)

**Proposed Solution:**
1. **Over-cluster**: Use K > K_ground_truth (e.g., K = 2× or 3× ground truth)
2. **Learn co-occurrence**: Identify cluster pairs that consistently appear together spatially
3. **Merge regions**: Post-process by merging clusters with high co-occurrence scores

**Implementation Sketch:**
```python
# Step 1: Over-cluster
k_over = k_ground_truth * 2  # or 3
labels_over = kmeans.fit_predict(features, n_clusters=k_over)

# Step 2: Compute co-occurrence matrix
# For each pair of clusters, measure spatial co-occurrence
co_occurrence = compute_spatial_cooccurrence(labels_over, window_size=50)

# Step 3: Hierarchical merging
# Merge cluster pairs with highest co-occurrence until K = K_ground_truth
merged_labels = hierarchical_merge(labels_over, co_occurrence, target_k=k_ground_truth)
```

**Co-occurrence Metrics to Explore:**
- Jaccard index of spatial neighborhoods
- Conditional probability P(cluster_i | neighbor is cluster_j)
- Graph connectivity on SLIC superpixel adjacency
- Shared boundary length / total boundary length

**Expected Benefits:**
- Better respect for DINOv3's natural feature groupings
- Cleaner cluster boundaries (each sub-cluster has tighter features)
- Automatic discovery of intra-class patterns
- May improve mIoU by reducing forced averaging

**Challenges:**
- Choosing optimal over-clustering factor (2×, 3×, adaptive?)
- Defining "spatial co-occurrence" threshold
- Computational cost of pairwise co-occurrence matrix
- Risk of under-merging (leaving semantically identical clusters separate)

**Validation:**
- Compare against smart-k baseline (K = K_ground_truth)
- Qualitative inspection: do merged clusters make semantic sense?
- Per-class IoU breakdown: does over-cluster+merge help multi-pattern classes?

**Related Work:**
- Hierarchical clustering with automatic cut selection
- Graph-based region merging (Felzenszwalb & Huttenlocher)
- Constrained clustering with spatial priors

**Priority:** Medium (requires smart-k sweep results first to establish baseline)

**Estimated Effort:** 2-3 days (implementation + evaluation)

---

## Validation: Vegetation Filter Integration

**Task:** Verify that current tooling (sweeps, metadb, evaluation pipeline) works correctly with `--vegetation-filter` flag.

**Context:**
- V3.1 implements species-level segmentation via cluster-level ExG filtering
- This is a **different task** than general region segmentation:
  - Adds vegetation filtering step (cluster-level ExG threshold = 0.1)
  - Goal: Separate vegetation species, not just any regions
  - Non-vegetation pixels should be masked/ignored in metrics
  - May require different evaluation metrics or visualization

**Current Status:**
- V3.1 implemented in codebase (see CLAUDE.md)
- `--vegetation-filter` flag exists in CLI
- Unclear if metadb correctly handles this task variant

**Testing Needed:**
1. **Sweep compatibility**: Does `--vegetation-filter` work with sweep system?
   ```bash
   tree-seg sweep data/datasets/fortress_processed --preset quick --vegetation-filter --smart-k
   ```
2. **Metadb hashing**: Are vegetation-filtered runs stored separately from regular runs?
3. **Metrics validity**: Are mIoU/pixel accuracy meaningful when non-veg is masked?
4. **Visualization**: Do viz outputs show vegetation masking correctly?
5. **Cache behavior**: Does changing `--exg-threshold` invalidate cache properly?

**Potential Issues:**
- Metrics may not account for masked regions properly
- Ground truth labels may include non-vegetation classes (buildings, roads, etc.)
- Smart-K behavior unclear: should K count only vegetation classes?
- ExG threshold (0.1) may need tuning per dataset

**Action Items:**
- [ ] Run test sweep with `--vegetation-filter` on FORTRESS
- [ ] Inspect metadb entries: verify hash includes vegetation_filter flag
- [ ] Check visualizations: ensure non-veg masking is visible
- [ ] Compare metrics: vegetation-filtered vs regular segmentation
- [ ] Document in experiments.md: does V3.1 improve species separation?

**Priority:** High (V3.1 is core version, needs validation before research use)

**Estimated Effort:** 1-2 hours (testing + documentation)

---

*Created: 2024-12-08*
*Status: Proposed*
