# Planned Sweeps for Configuration Space Exploration

**Dataset:** FORTRESS (fortress_processed)
**Goal:** Systematically explore parameter combinations to find optimal unsupervised configuration and feed metadata bank

---

## 🎯 Quick Reference

**Baselines:**
- **Auto-K (Real Unsupervised):** K-means + SLIC = 8.86% mIoU, 41.50% pixel accuracy (single image CFB003)
- **Smart-K (Oracle Mode):** K-means + SLIC = 7.61% mIoU, 34.2% pixel accuracy (3 samples avg, 9.75% best on CFB003)
- **Supervised MLP:** 27.8% mIoU, 83.3% pixel accuracy (ceiling)

**Sweep syntax:**
```bash
uv run tree-seg sweep data/datasets/fortress_processed -c <methods> -r <refinements> [OPTIONS]
```

---

## ✅ Completed Sweeps

### Quick Validation (Smart-K) - Dec 8, 2024

**Command:**
```bash
uv run tree-seg sweep data/datasets/fortress_processed --preset quick --smart-k --num-samples 3 --save-viz --name quick_smartk
```

**Results (3 samples: CFB003, CFB008, CFB014):**

| Config | mIoU | Pixel Acc | Per-Img Time | Total Time |
|--------|------|-----------|--------------|------------|
| **kmeans_slic** ✅ | **7.61%** | **34.2%** | 33s | 99s |
| kmeans_none | 7.03% | 32.6% | 5s | 15s |
| gmm_slic | 6.35% | 29.3% | 39s | 118s |
| gmm_none | 6.17% | 28.5% | 13s | 38s |

**Per-Image Breakdown (kmeans_slic):**
- CFB003: **9.75% mIoU**, 35.8% PA, K=6 (best, matches/exceeds baseline)
- CFB008: 6.43% mIoU, 48.0% PA, K=4
- CFB014: 6.64% mIoU, 18.8% PA, K=10

**Key Findings:**
- ✅ **K-means > GMM** by ~1.26% mIoU (confirms previous experiments)
- ✅ **SLIC helps** (+0.58% mIoU for kmeans, +0.18% for gmm)
- ✅ **Smart-K working correctly** (uses ground truth K per image: 4, 6, 10)
- ✅ **CFB003 performance matches baseline** (9.75% vs 8.86% from experiments.md)
- ⚠️ **High variance across images** (6.4% to 9.75% mIoU)
- 📊 **Metadb populated** with 4 configs (hashes: b0cc1f2406, 4a40a17681, f3b47a9143, e0c44c584a)

**Status:** Complete. Metadb integration validated. Ready for comprehensive sweeps.

---

## 📊 Systematic Sweeps

### Core Method Comparison

- [ ] **All Clustering × All Refinement (35 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed -c all -r all --num-samples 3 --save-viz
  ```
  - Purpose: Comprehensive comparison of every clustering + refinement combination
  - Expected runtime: ~2-3 hours (35 configs × 3 samples)
  - Will identify: Best clustering algorithm, best refinement method, interaction effects

- [ ] **Clustering Methods Baseline (7 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed -c all -r slic --num-samples 5 --save-viz
  ```
  - Purpose: Compare all clustering algorithms with standard SLIC refinement
  - Expected: K-means likely to win, but quantify performance gap

- [ ] **Refinement Methods Comparison (5 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed -c kmeans -r all --num-samples 5 --save-viz
  ```
  - Purpose: Deep dive into refinement methods with best clustering (K-means)
  - Expected: SLIC or soft-em+slic to perform best

### Model Architecture

- [ ] **DINOv3 Model Size Impact (4 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed --model all -c kmeans -r slic --num-samples 5 --save-viz
  ```
  - Purpose: Quantify mIoU vs model size tradeoff
  - Models: small (ViT-S/14), base (ViT-B/14), large (ViT-L/14), mega (ViT-g/14)
  - Expected: Larger models = better features, but diminishing returns

- [ ] **Model × Stride Optimization (12 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed --model base,large --stride all -c kmeans -r slic --num-samples 3 --save-viz
  ```
  - Purpose: Find optimal stride for each model size
  - Expected: Stride 4 good for base, stride 2 may help large model

### Feature Extraction

- [ ] **Stride Deep Dive (3 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed --stride all -c kmeans -r slic --num-samples 5 --save-viz
  ```
  - Purpose: Impact of feature grid resolution on segmentation quality
  - Strides: 2 (dense, slow), 4 (balanced), 8 (coarse, fast)
  - Expected: Stride 4 optimal for speed/quality tradeoff

- [ ] **Elbow Threshold Sweep (5 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed --elbow-threshold all -c kmeans -r slic --num-samples 5 --save-viz
  ```
  - Purpose: Optimize auto-K selection sensitivity
  - Thresholds: 2.5 (sensitive), 5.0 (default), 10.0, 20.0, 50.0 (conservative)
  - Expected: Lower thresholds = more clusters, may improve mIoU

### Spatial Processing

- [ ] **Tiling Impact (4 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed --tiling all --model base,large -c kmeans -r slic --num-samples 3 --save-viz
  ```
  - Purpose: Quantify tiling overhead vs quality improvement
  - Expected: Tiling on = better quality, tiling off = faster

- [ ] **Tiling × Refinement (8 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed --tiling all -r slic,bilateral,soft-em,none -c kmeans --num-samples 3 --save-viz
  ```
  - Purpose: Does refinement interact with tiling strategy?
  - Expected: SLIC + tiling may reduce stitching artifacts

### Advanced Combinations

- [ ] **Top 3 Clustering × Top 3 Refinement (9 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed -c kmeans,gmm,spectral -r slic,soft-em,soft-em+slic --num-samples 5 --save-viz
  ```
  - Purpose: Focus on most promising methods (from initial results)
  - Use after running "All Clustering × All Refinement" sweep

- [ ] **Preset: Paper Configurations (9 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed --preset paper --num-samples 5 --save-viz
  ```
  - Purpose: Standard configurations for final paper comparison
  - Configs: kmeans/gmm/spectral × none/slic/soft-em

- [ ] **Best Config × All Models (4 configs)**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed --model all -c kmeans -r slic --num-samples 10 --save-viz
  ```
  - Purpose: Establish upper bound with best clustering/refinement
  - Use after identifying best method combination

---

## 🔬 Focused Investigations

### Clustering Algorithm Deep Dives

- [ ] **Spherical K-means Variants**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed -c kmeans,spherical -r slic,none --num-samples 5 --save-viz
  ```
  - Purpose: Test cosine distance vs Euclidean for DINOv3 features
  - Expected: Spherical may work better on normalized features

- [ ] **Auto-K Clustering Methods**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed -c dpmeans -r slic --num-samples 5 --save-viz
  ```
  - Purpose: Evaluate automatic K selection algorithm (DP-means)
  - Note: HDBSCAN removed (found 0 clusters, always fell back to K-means)

- [ ] **Potts Regularization Test**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed -c kmeans,potts -r slic,none --num-samples 5 --save-viz
  ```
  - Purpose: Test spatial smoothness regularization
  - Expected: May reduce noise but over-smooth boundaries

### Refinement Strategy Tests

- [ ] **Soft EM Parameter Exploration**
  ```bash
  uv run tree-seg eval data/datasets/fortress_processed -c kmeans -r soft-em --num-samples 5 --save-viz
  ```
  - Purpose: Baseline soft-em performance
  - Follow-up: Tune temperature, iterations, spatial alpha

- [ ] **Combined Refinement Impact**
  ```bash
  uv run tree-seg sweep data/datasets/fortress_processed -c kmeans -r slic,soft-em,soft-em+slic --num-samples 5 --save-viz
  ```
  - Purpose: Does combining soft-em + slic improve over either alone?
  - Expected: Combination may provide best of both worlds

---

## 📈 Validation Sweeps

### Statistical Significance

- [ ] **Full Dataset Sweep: Best Config (47 samples)**
  ```bash
  uv run tree-seg eval data/datasets/fortress_processed -c kmeans -r slic --save-viz
  ```
  - Purpose: Establish baseline statistics on full dataset
  - Use after finding best config from sweeps

- [ ] **Full Dataset Sweep: Top 3 Configs**
  - Run top 3 configurations on all 47 samples
  - Purpose: Ensure results generalize across entire dataset
  - Statistical significance testing

### Reproducibility

- [ ] **Repeat Best Config 3× (Random Seed Variation)**
  - Purpose: Verify results are stable across runs
  - Check K-means initialization sensitivity

---

## 🎯 Priority Order

**Week 1: Core methods**
1. All Clustering × All Refinement (comprehensive)
2. Model Size Impact (architecture comparison)
3. Stride Deep Dive (feature resolution)

**Week 2: Optimization**
4. Elbow Threshold Sweep (auto-K tuning)
5. Top 3 × Top 3 (focused refinement)
6. Tiling Impact (spatial processing)

**Week 3: Validation**
7. Full Dataset Sweep: Best Config
8. Statistical significance testing
9. Paper preset configurations

---

## 📊 Expected Outcomes

**Metadata bank growth:** ~150-200 experiment runs
**Key insights:**
- Optimal clustering algorithm for FORTRESS
- Best refinement strategy
- Model size vs performance tradeoff
- Optimal auto-K threshold
- Tiling impact quantification

**Final deliverable:** Ranked configuration table for paper with statistical significance
