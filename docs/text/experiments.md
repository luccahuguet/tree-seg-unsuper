# Improvement Experiments

**Current baseline:** V1.5 + K-means + SLIC + Smart K = 8.9% mIoU, 41.5% pixel acc on FORTRESS CFB003

**Goal:** Test alternative clustering and feature methods to improve unsupervised segmentation quality.

## 🎛️ CLI Structure (Composable)

The evaluation CLI uses composable flags at different abstraction levels:

```bash
# Clustering algorithm (what groups pixels)
--clustering kmeans|gmm|spectral|hdbscan  # Default: kmeans

# Refinement method (how to improve boundaries)
--refine none|slic|soft-em|bilateral|soft-em+slic  # Default: slic

# Task modifier (what to segment)
--vegetation-filter  # Enable species-level segmentation (V3)

# Supervised mode (different approach)
--supervised  # Use Mask2Former instead of clustering (V4)
```

**Examples:**
```bash
# V1.5 baseline: K-means + SLIC
tree-seg eval data/fortress

# V2: K-means + soft EM refinement
tree-seg eval data/fortress --refine soft-em

# Experiment: GMM + soft EM
tree-seg eval data/fortress --clustering gmm --refine soft-em

# V3 task: Species segmentation
tree-seg eval data/fortress --vegetation-filter

# No refinement: just clustering
tree-seg eval data/fortress --refine none
```

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

## 🎯 V2 Soft EM Refinement (Implemented)

- [x] **Soft EM Refinement (V2)** ✅ **IMPLEMENTED**
  - Refine K-means clusters using temperature-scaled softmax
  - Iterative EM updates in DINOv3 feature space
  - Optional spatial blending with neighbors
  - Code: `tree_seg/clustering/head_refine.py`
  - Usage: `--refine soft-em` or `--refine soft-em+slic`
  - **Parameters:**
    - Temperature τ = 1.0 (controls boundary softness)
    - Iterations = 5 (typical 3-5)
    - Spatial blend α = 0.0 (disabled by default)
  - **Status:** Ready for testing on FORTRESS
  - **Expected:** +1-3% mIoU over baseline K-means

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

- [x] **Pyramid feature aggregation** ❌ **FAILED**
  - Process image at multiple scales (0.5×, 1.0×, 2.0×)
  - Concatenate and PCA-reduce aggregated features
  - Code: `--use-pyramid --pyramid-scales 0.5,1.0,2.0`
  - **Result:** -0.06% mIoU, -4.7% pixel accuracy vs baseline
  - **Status:** Not viable - requires disabling tiling (lower resolution)
  - **Note:** Resolution loss outweighs multi-scale benefits

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
# Test clustering methods with new sweep command
uv run tree-seg sweep data/fortress_processed -c all -r slic,none --num-samples 1 --save-viz --smart-k

# Test specific clustering vs refinement combinations
uv run tree-seg sweep data/fortress_processed -c kmeans,gmm -r slic,soft-em,none --num-samples 1 --save-viz

# Test V2 soft EM refinement
uv run tree-seg eval data/fortress_processed --refine soft-em --num-samples 1 --save-viz

# Test refinement combinations
uv run tree-seg eval data/fortress_processed --refine soft-em+slic --num-samples 1 --save-viz

# Experiment: GMM with soft EM
uv run tree-seg eval data/fortress_processed --clustering gmm --refine soft-em --num-samples 1 --save-viz

# Use preset for comprehensive testing
uv run tree-seg sweep data/fortress_processed --preset paper --num-samples 3 --save-viz --smart-k
```

Metadata tips:
- `--save-labels/--no-save-labels` controls NPZ dumps under `labels/` (on by default).
- Runs auto-log into the metadata bank (best-effort) under `results/`. Query with `uv run tree-seg results --tags kmeans,slic --sort mIoU --top 5` or use `--hash <id> --render` to regenerate viz.

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

### Pyramid Multi-Scale Features (Dec 5, 2024)
- **Config:** V1.5 + base + Pyramid [0.5×, 1.0×, 2.0×] + Concat+PCA(1536) + SLIC + Smart K
- **mIoU:** 8.80% (-0.06% vs baseline)
- **Pixel Acc:** 36.80% (-4.70% vs baseline)
- **Time:** 51.6s (faster due to no tiling, but lower resolution)
- **Notes:** Multi-scale features fail to compensate for resolution loss from disabling tiling (1024×1024 vs 9372×9372). Concatenated 4608D features reduced to 1536D via PCA. Not recommended unless tiling support added.

### SLIC Parameter Sweep (Dec 5, 2024)
- **Configs Tested:** compactness=[5, 10, 20], sigma=[0.5, 1.0, 2.0]
- **Results (CFB003):**
  - All configs: 9.2-9.3% mIoU, 37.1-37.3% PA
  - Baseline (c=10, σ=1.0): 9.3% mIoU, 174s (fastest)
  - Other configs: 191-198s (slower)
- **Notes:** SLIC parameters have **negligible impact** on segmentation quality. Baseline settings are optimal for speed with no quality loss. Parameter tuning does not improve results.

---

## 🎯 Success Criteria

- **Minimum viable:** +3% mIoU improvement (11.9% total)
- **Good result:** +5% mIoU improvement (13.9% total)
- **Excellent result:** +10% mIoU improvement (18.9% total)

---

## 🛠️ Infrastructure Improvements (Dec 2024)

- ✅ Added new `sweep` command with multiplicative parameter exploration (Dec 7)
- ✅ Curated sweep presets in presets.toml (quick/clustering/refine/models/etc.)
- ✅ Support for 'all' keyword to expand parameter options
- ✅ Fixed visualization overwriting in sweep mode
- ✅ Visualizations now saved with unique config labels
- ✅ Implemented GMM clustering method
- ✅ Added `--smart-k` mode for debugging

---

## 🚀 Next Experiments to Try

**Completed:**
1. ~~Spectral Clustering~~ ❌ Done - underperforms K-means
2. ~~Multi-layer features~~ ✅ Done - marginal improvement
3. ~~Bilateral filtering~~ ❌ Done - underperforms SLIC
4. ~~Dense CRF~~ ❌ Done - not viable (build failures)
5. ~~HDBSCAN~~ ❌ Done - found 0 clusters, not viable
6. ~~Pyramid features~~ ❌ Done - resolution loss outweighs multi-scale benefits

**Parameter Tuning (Low-Effort):**
- [x] **SLIC parameter sweep** ✅ **NO IMPROVEMENT**
  - Tested compactness: 5.0, 10.0 (current), 20.0
  - Tested sigma: 0.5, 1.0 (current), 2.0
  - **Result:** All configs 9.2-9.3% mIoU (identical)
  - **Status:** Default parameters (c=10, σ=1.0) are optimal

- [ ] **Tile overlap optimization**
  - Current: 256px
  - Test: 128px, 384px, 512px
  - Expected: Reduce stitching artifacts, +0.5-1% mIoU

**DINOv3 Linear Segmentation Head (Supervised/Transfer Learning):**

Official DINOv3 repository includes linear segmentation head. Multiple approaches possible:

- [ ] **Option 1: Supervised training from scratch**
  - Train linear head (Conv2d 1×1) on FORTRESS labels with ViT-B/16 backbone
  - Architecture: ~10K params (13 classes × 768D features)
  - Fits in 32GB RAM (86M backbone + 10K head params)
  - Code: Follow `dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml`
  - Expected: +30-50% mIoU (supervised upper bound)
  - **Pros:** Shows DINOv3 feature quality ceiling for FORTRESS
  - **Cons:** Changes problem from unsupervised to supervised
  - **Note:** ViT-7B doesn't fit in 32GB; use existing ViT-B/16 backbone

- [x] **Option 2: Pre-trained vegetation filtering** ❌ **BLOCKED**
  - Use ADE20K pre-trained head (150 classes including "tree")
  - Official checkpoint: ViT-7B/16 + Mask2Former on ADE20K
  - Extract "tree" class predictions as vegetation mask
  - **Result:** ViT-7B + M2F doesn't fit in 32GB RAM
  - **Alternative:** Could try with smaller backbone (ViT-B/16) if M2F checkpoint available
  - **Status:** Blocked by memory - would need >32GB or smaller model

- [ ] **Option 3: Transfer learning (fine-tune)**
  - Load ADE20K pre-trained head
  - Replace final layer (150 → 13 classes)
  - Fine-tune only last layer on FORTRESS
  - Expected: +20-40% mIoU (faster than training from scratch)
  - **Pros:** Leverages pre-training, less data needed
  - **Cons:** Still requires supervised training

**Note:** Options 1 & 3 are supervised methods. Option 2 could improve unsupervised baseline.

**All unsupervised clustering experiments completed. K-means + SLIC remains the best unsupervised approach.**

## 📈 Supervised Baselines (for comparison)

- **sklearn LogisticRegression** (`--head sklearn`): mIoU ≈ 0.082, PA ≈ 0.536 on `fortress_processed` (stride=4, 47 tiles, 100k sample cap, max_iter=3000).
- **Torch Linear MLP** (`--head linear`): best observed so far mIoU ≈ 0.074, PA ≈ 0.236 (stride=2, 2M patches, lr=5e-4, patience 20, val_split 0.1).
- **sklearn MLPClassifier** (`--head mlp`):
  - Peak (no early stop, likely overfit): mIoU ≈ 0.427, PA ≈ 0.953 with `--stride 2 --max-patches 2000000 --epochs 400 --patience 0 --val-split 0 --lr 5e-4 --mlp-use-xy`.
  - Best with early stopping (non-overfitting peak, reproducible): mIoU ≈ 0.278, PA ≈ 0.833 with `--stride 2 --max-patches 2000000 --epochs 400 --patience 5 --val-split 0.02 --lr 5e-4 --mlp-use-xy`.

Use these supervised numbers as the bar for unsupervised DINO improvements.

---

## 🧭 Possible Next: K-means Successors (Not Tried)

Shortlist from `docs/text/kmeans_successors.md` that could be drop-in tests if we revisit unsupervised clustering:
- **Spherical + soft k-means**: swap to cosine distance with temperature-sharpened soft assignments; trivial change, expected cleaner clusters on DINO features.
- **DP-means**: k-means variant with a λ penalty that auto-selects K; good when K varies across tiles; medium effort.
- **Regularized k-means (Potts on SLIC graph)**: add spatial smoothness via α-expansion on the SLIC adjacency; should reduce speckle/edge noise.
