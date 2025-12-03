# Improvement Experiments

**Current baseline:** V1.5 + SLIC + Smart K = 9.3% mIoU, 37.2% pixel acc on FORTRESS CFB003

**Goal:** Test alternative clustering and feature methods to improve unsupervised segmentation quality.

---

## 🥈 Medium Impact Experiments

### D) Better Clustering Algorithms (V6)

- [ ] **GMM (Gaussian Mixture Model)**
  - Replace K-means with probabilistic clustering
  - Soft assignments instead of hard clusters
  - Code: `sklearn.mixture.GaussianMixture`
  - Expected: +3-7% mIoU

- [ ] **Spectral Clustering**
  - Handle non-convex cluster shapes
  - Better for spatial data with complex boundaries
  - Code: `sklearn.cluster.SpectralClustering`
  - Expected: +3-7% mIoU

- [ ] **HDBSCAN**
  - Density-based, automatic K selection
  - Robust to noise and outliers
  - Code: `import hdbscan`
  - Expected: +2-5% mIoU

---

### E) Multi-Scale Features

- [ ] **Multi-layer DINOv3 features**
  - Extract from layers [3, 6, 9, 12] instead of just last layer
  - Combine with PCA or concatenation
  - Captures both fine texture and semantic context
  - Expected: +5-10% mIoU

- [ ] **Pyramid feature aggregation**
  - Process image at multiple scales (0.5×, 1×, 2×)
  - Aggregate features across scales
  - Expected: +5-8% mIoU

---

### F) Post-Processing with CRF

- [ ] **Dense CRF refinement**
  - Apply conditional random field to smooth boundaries
  - Use RGB + spatial features for pairwise potentials
  - Code: `import pydensecrf.densecrf as dcrf`
  - Expected: +3-5% mIoU

- [ ] **Bilateral filtering**
  - Edge-preserving smoothing on cluster assignments
  - Already available as `--clustering bilateral`
  - Expected: +2-3% mIoU

---

## 📝 Experiment Protocol

For each experiment:
1. Run on FORTRESS CFB003 (single image quick test)
2. Compare against baseline: 9.3% mIoU, 37.2% pixel acc
3. If >+3% improvement, run full evaluation (5-10 samples)
4. Document results below

---

## 📊 Results Log

### Baseline
- **Date:** 2024-12-03
- **Config:** V1.5 + base + SLIC + Smart K=6
- **mIoU:** 9.3%
- **Pixel Acc:** 37.2%
- **Time:** 177s

### [Experiment name]
- **Date:**
- **Config:**
- **mIoU:**
- **Pixel Acc:**
- **Time:**
- **Notes:**

---

## 🎯 Success Criteria

- **Minimum viable:** +3% mIoU improvement (12.3% total)
- **Good result:** +5% mIoU improvement (14.3% total)
- **Excellent result:** +10% mIoU improvement (19.3% total)

---

## 🚀 Quick Start

```bash
# Test GMM clustering
python experiments/test_gmm_clustering.py

# Test multi-scale features
python experiments/test_multiscale_features.py

# Test CRF post-processing
python experiments/test_crf_postprocess.py
```

*Experiment scripts to be created as needed.*
