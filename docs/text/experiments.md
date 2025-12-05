# Improvement Experiments

**Current baseline:** V1.5 + K-means + SLIC + Smart K = 8.9% mIoU, 41.5% pixel acc on FORTRESS CFB003

**Goal:** Test alternative clustering and feature methods to improve unsupervised segmentation quality.

---

## 📊 Recent Results

### Clustering Comparison (Dec 5, 2024)

Tested K-means vs GMM with/without SLIC refinement on FORTRESS CFB003:

| Method | Clustering | Refinement | mIoU | Pixel Acc | Runtime |
|--------|-----------|------------|------|-----------|---------|
| **kmeans_slic** ✅ | K-means | SLIC | **8.86%** | **41.50%** | 179s |
| kmeans | K-means | None | 8.02% | 37.38% | 161s |
| gmm_slic | GMM | SLIC | 5.58% | 36.18% | 203s |
| gmm | GMM | None | 5.14% | 32.89% | 178s |

**Key Findings:**
- ✅ SLIC refinement provides consistent improvement (+0.8-3% mIoU)
- ❌ GMM underperforms K-means significantly (-3% to -6% mIoU)
- 🏆 **Best config:** K-means + SLIC (current baseline)

### Refinement Comparison (Dec 5, 2024)

Tested bilateral filtering vs SLIC on FORTRESS CFB003:

| Method | Refinement | mIoU | Pixel Acc | Runtime |
|--------|-----------|------|-----------|---------|
| **kmeans_slic** ✅ | SLIC | **8.86%** | **41.50%** | 179s |
| kmeans | None | 8.02% | 37.38% | 161s |
| **bilateral** ❌ | Bilateral | **7.81%** | **29.57%** | 137s |

**Key Findings:**
- ❌ Bilateral filtering underperforms: -1.05% mIoU, -11.93% pixel accuracy vs SLIC
- ✅ SLIC remains the best refinement method
- ⚡ Bilateral is faster (137s) but quality loss is unacceptable

**Why GMM failed:**
- Soft assignments may dilute cluster boundaries
- Covariance estimation adds computational cost without benefit
- K-means' hard assignments work better for spatial segmentation

---

## 🥈 Medium Impact Experiments

### D) Better Clustering Algorithms (V6)

- [x] **GMM (Gaussian Mixture Model)** ❌ **FAILED**
  - Replace K-means with probabilistic clustering
  - Soft assignments instead of hard clusters
  - Code: `sklearn.mixture.GaussianMixture`
  - **Result:** -3% to -6% mIoU vs K-means
  - **Status:** Not recommended

- [x] **Spectral Clustering** ❌ **UNDERPERFORMS**
  - Handle non-convex cluster shapes
  - Implemented with 10k subsample + nearest-neighbor propagation
  - Code: `sklearn.cluster.SpectralClustering`
  - **Result:** -1.2% to -1.8% mIoU vs K-means (but +4-8% pixel accuracy)
  - **Status:** Not recommended - subsampling loses spatial structure

- [x] **HDBSCAN** ❌ **FAILED**
  - Density-based, automatic K selection
  - Robust to noise and outliers
  - Code: `--clustering hdbscan`
  - **Result:** Found 0 clusters on CFB003, falls back to K-means
  - **Status:** Not viable - parameters (min_cluster_size=50) too conservative for tree features
  - **Note:** HDBSCAN treats all pixels as noise, likely due to continuous DINOv3 feature space

---

### E) Multi-Scale Features

- [x] **Multi-layer DINOv3 features** ⚠️ **MARGINAL**
  - Extract from layers [3, 6, 9, 12] instead of just last layer
  - Tested concat (3072D → PCA 512D) and average aggregation
  - **Result:** +0.01% mIoU, +0.1% pixel accuracy vs single-layer
  - **Status:** Implemented but marginal improvement on CFB003

- [ ] **Pyramid feature aggregation**
  - Process image at multiple scales (0.5×, 1×, 2×)
  - Aggregate features across scales
  - Expected: +5-8% mIoU

### Multi-Layer Feature Extraction (Dec 5, 2024)

Tested extracting features from multiple DINOv3 layers vs single final layer:

| Method | Layers | Aggregation | mIoU | Pixel Acc | Runtime |
|--------|--------|-------------|------|-----------|---------|
| baseline | 12 only | N/A | 8.863% | 41.50% | 162s |
| multi_concat | [3,6,9,12] | Concat+PCA→512 | **8.876%** | **41.60%** | 179s |
| multi_average | [3,6,9,12] | Average | 8.863% | 41.50% | 161s |

**Key Findings:**
- ⚠️ Multi-layer features provide only **marginal improvement** (+0.01% mIoU)
- Only 0.32% of pixels differ between single vs multi-layer
- Concat+PCA slightly outperforms average aggregation
- Runtime overhead is minimal (~10% for concat+PCA)

**Why improvement is small:**
- Both methods select same K=4 via elbow
- Hungarian matching aligns clusters similarly
- CFB003 may not benefit from multi-scale features
- Need testing on more diverse images

---

### F) Post-Processing with CRF

- [x] **Dense CRF refinement** ❌ **NOT VIABLE**
  - Apply conditional random field to smooth boundaries
  - Use RGB + spatial features for pairwise potentials
  - Code: `pydensecrf` or `SimpleCRF`
  - **Result:** Both libraries fail to compile on Python 3.12+
  - **Status:** Not viable - unmaintained packages, C++ build failures
  - **Note:** CRF has fallen out of favor in modern segmentation

- [x] **Bilateral filtering** ❌ **FAILED**
  - Edge-preserving smoothing on cluster assignments
  - Code: `--clustering bilateral`
  - **Result:** -1.05% mIoU, -11.93% pixel accuracy vs SLIC
  - **Status:** Not recommended - over-smooths boundaries

---

## 📝 Experiment Protocol

For each experiment:
1. Run on FORTRESS CFB003 (single image quick test)
2. Compare against baseline: 8.9% mIoU, 41.5% pixel acc
3. If >+3% improvement, run full evaluation (5-10 samples)
4. Document results in this file

**Running Sweeps:**
```bash
# Test clustering methods
uv run python scripts/evaluate_fortress.py \
  --dataset data/fortress_processed \
  --method v1.5 --model base --stride 4 \
  --num-samples 1 --save-viz \
  --compare-configs --grid clustering --smart-k

# Test tiling configurations
uv run python scripts/evaluate_fortress.py \
  --dataset data/fortress_processed \
  --method v1.5 --model base \
  --num-samples 1 --save-viz \
  --compare-configs --grid tiling_refine --smart-k
```

---

## 📊 Results Log

### Baseline (Updated Dec 5, 2024)
- **Config:** V1.5 + base + K-means + SLIC + Smart K=6
- **mIoU:** 8.86%
- **Pixel Acc:** 41.50%
- **Time:** 179s
- **Notes:** Best performing configuration after clustering comparison

### GMM Clustering Test (Dec 5, 2024)
- **Config:** V1.5 + base + GMM + SLIC + Smart K=6
- **mIoU:** 5.58%
- **Pixel Acc:** 36.18%
- **Time:** 203s
- **Notes:** Underperforms K-means by -3.3% mIoU. Not recommended.

### Bilateral Filtering Test (Dec 5, 2024)
- **Config:** V1.5 + base + K-means + Bilateral + Smart K=6
- **mIoU:** 7.81%
- **Pixel Acc:** 29.57%
- **Time:** 137s
- **Notes:** Underperforms SLIC by -1.05% mIoU and -11.93% pixel accuracy. Not recommended.

### Multi-Layer Feature Extraction (Dec 5, 2024)
- **Config:** V1.5 + base + Multi-layer [3,6,9,12] + Concat+PCA(512) + SLIC
- **mIoU:** 8.876% (+0.01% vs baseline)
- **Pixel Acc:** 41.60% (+0.10% vs baseline)
- **Time:** 179s
- **Notes:** Marginal improvement. Only 0.32% of pixels differ. Multi-layer implemented but not impactful on CFB003.

### Spectral Clustering Test (Dec 5, 2024)
- **Config:** V1.5 + base + Spectral (10k subsample) + SLIC + Smart K
- **Single image (CFB003):** 8.08% mIoU (-1.20%), 45.15% PA (+3.97%)
- **3-image average:** 6.69% mIoU (-0.82%), 44.02% PA (+8.58%)
- **Per-image:** CFB003: -1.20%, CFB008: +1.18%, CFB014: -2.45%
- **Time:** 214s (+11% vs K-means)
- **Notes:** Spectral underperforms K-means on average mIoU but has higher pixel accuracy. Results are **inconsistent** across images (wins 1/3, big loss on CFB014). Subsampling (10k pixels) loses spatial structure. Not recommended.

### HDBSCAN Clustering Test (Dec 5, 2024)
- **Config:** V1.5 + base + HDBSCAN (min_cluster_size=50) + Smart K
- **Result:** Found 0 clusters on CFB003, fell back to K-means (k=4)
- **Notes:** HDBSCAN marked all pixels as noise. Parameters (min_cluster_size=50, min_samples=10) too conservative for continuous DINOv3 feature space. Density-based clustering not suitable for this task. Not recommended.

---

## 🎯 Success Criteria

- **Minimum viable:** +3% mIoU improvement (11.9% total)
- **Good result:** +5% mIoU improvement (13.9% total)
- **Excellent result:** +10% mIoU improvement (18.9% total)

---

## 🛠️ Infrastructure Improvements (Dec 5, 2024)

- ✅ Added `--grid clustering` sweep configuration
- ✅ Fixed visualization overwriting in sweep mode
- ✅ Visualizations now saved with unique config labels
- ✅ Implemented GMM clustering method
- ✅ Added `--smart-k` mode for debugging

---

## 🚀 Next Experiments to Try

1. ~~Spectral Clustering~~ ❌ Done - underperforms K-means
2. ~~Multi-layer features~~ ✅ Done - marginal improvement
3. ~~Bilateral filtering~~ ❌ Done - underperforms SLIC
4. ~~Dense CRF~~ ❌ Done - not viable (build failures)
5. ~~HDBSCAN~~ ❌ Done - found 0 clusters, not viable

**All planned clustering experiments completed. K-means + SLIC remains the best approach.**

*Current Priority: Explore other directions (multi-scale features, different backbones, etc.)*
