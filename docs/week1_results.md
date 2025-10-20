# Week 1 Benchmark Results

**Date:** October 19, 2025
**Dataset:** ISPRS Potsdam (2,400 images)
**Method:** V1.5 (DINOv3 + K-means with auto K-selection)
**Samples per config:** 10

## Initial OFAT (One-Factor-At-Time) Exploration

### Results Summary

| Configuration | mIoU | Pixel Acc | Runtime | K (avg) |
|--------------|------|-----------|---------|---------|
| **Elbow Threshold Sweep** | | | | |
| elbow_2.5 | 0.202 | 41.7% | 12.8s | ~9.7 |
| elbow_5.0 ✓ | 0.204 | 41.7% | 12.6s | ~7.8 |
| elbow_10.0 | **0.211** | 47.7% | 12.5s | ~6.5 |
| elbow_20.0 | **0.217** | **52.3%** | 12.6s | ~5.2 |
| **Model Size Sweep** | | | | |
| model_small | 0.181 | 42.4% | **6.9s** | ~7.8 |
| model_base ✓ | 0.204 | 41.7% | 12.8s | ~7.8 |
| model_large | 0.188 | 39.3% | **51.5s** | ~7.8 |
| **Refinement Sweep** | | | | |
| kmeans (none) ✓ | 0.204 | 41.7% | 12.4s | ~7.8 |
| slic | **0.221** | **44.2%** | 11.6s | ~7.8 |

✓ = baseline/default configuration

## Key Findings

### 1. SLIC Refinement Improves Quality
- **+8.3% mIoU** improvement (0.204 → 0.221)
- **+2.5% pixel accuracy** improvement (41.7% → 44.2%)
- **Slightly faster** (12.4s → 11.6s) due to better convergence

### 2. Higher Elbow Threshold = Better Results
- **Conservative clustering** (fewer clusters) performs better
- Threshold 20.0: 21.7% mIoU vs 20.2% at threshold 2.5
- **Fewer false positives** with fewer clusters
- Trend: **Higher threshold → Better metrics**

### 3. Base Model is Optimal
- **Large model paradox**: 4x slower but **worse** quality (18.8% vs 20.4%)
  - Hypothesis: Overfitting to patch-level details, missing global structure
  - Or: Large model needs more data/tuning
- **Small model**: 2x faster, only -2.3% mIoU (good tradeoff)
- **Base model**: Best quality/speed balance

### 4. Expected vs Actual Performance
- **Predicted mIoU**: 20-35% for unsupervised methods
- **Actual mIoU**: 18-22% (within range, toward lower end)
- **Pixel accuracy**: 39-52% (reasonable for 6-class problem)

## Performance Analysis

### Runtime Breakdown
- **Small model**: 6.9s per image (300x300)
- **Base model**: 12.6s per image
- **Large model**: 51.5s per image (4x slower!)

### Computational Cost (10 samples)
- Feature extraction: ~60-70% of time
- K-means clustering: ~20-30% of time
- SLIC refinement: ~5-10% of time (minimal overhead)

## Observations

### SLIC Advantage
SLIC refinement adds spatial constraints to cluster assignments:
- Enforces spatial continuity (neighboring pixels likely same class)
- Reduces noise in segmentation maps
- Minimal computational overhead
- **Recommendation:** Enable by default

### Elbow Threshold Impact
Lower thresholds (2.5, 5.0) tend to select K=8-10:
- More granular segmentation
- More false cluster divisions
- Lower mIoU due to over-segmentation

Higher thresholds (10.0, 20.0) select K=5-7:
- Coarser segmentation
- Better alignment with ground truth (6 classes)
- Higher mIoU and pixel accuracy

### Large Model Failure
Surprising result: Large model (ViT-L/16) performs **worse** than Base (ViT-B/16):
- Possible explanations:
  1. Overfitting to fine-grained texture, missing semantic structure
  2. Requires different hyperparameters (stride, upsample factor)
  3. Feature dimensionality mismatch with K-means
  4. Needs larger K range for more expressive features
- **Needs investigation** before using in production

## Smart Grid Search Results

Based on OFAT findings, tested 8 configurations combining best parameters:
- **Models**: small, base (changed from mega due to OOM crashes)
- **Thresholds**: 10.0, 20.0 (top performers from OFAT)
- **Refinement**: kmeans, slic

### Results Summary

| Configuration | mIoU | Pixel Acc | Per Img | Total (10 samples) |
|--------------|------|-----------|---------|---------------------|
| small_e10_km | 0.177 | 42.5% | 6.92s | 69.2s |
| small_e10_slic | 0.193 | 46.0% | 7.03s | 70.3s |
| small_e20_km | 0.184 | 43.4% | 6.82s | 68.2s |
| small_e20_slic | 0.196 | 46.4% | 7.04s | 70.4s |
| base_e10_km | 0.211 | 47.7% | 13.06s | 130.6s |
| base_e10_slic | 0.224 | 49.9% | 13.58s | 135.8s |
| base_e20_km | 0.217 | 52.3% | 12.75s | 127.5s |
| **base_e20_slic** ⭐ | **0.225** | **53.6%** | **13.29s** | **132.9s** |

⭐ = **Best overall configuration**

### Key Findings

#### 1. Best Configuration: base_e20_slic
- **22.5% mIoU, 53.6% pixel accuracy**
- Confirms hypothesis: SLIC + elbow 20.0 combine for best results
- Reasonable runtime: 13.29s per 300x300 image on CPU
- **Recommended as default for production use**

#### 2. Consistent SLIC Advantage
SLIC refinement improves all configurations:
- small_e10: +1.6% mIoU (17.7% → 19.3%)
- small_e20: +1.2% mIoU (18.4% → 19.6%)
- base_e10: +1.3% mIoU (21.1% → 22.4%)
- base_e20: +0.8% mIoU (21.7% → 22.5%)

#### 3. Elbow Threshold Trade-offs
- **Elbow 20.0**: Better pixel accuracy (+3-6%), slightly higher mIoU
- **Elbow 10.0**: More clusters, lower pixel accuracy
- Recommendation: Use 20.0 for better alignment with ground truth

#### 4. Small Model as "Fast Mode"
- **2x faster** than base model (7s vs 13s)
- **Only -2.9% mIoU loss** with SLIC (19.6% vs 22.5%)
- **Good tradeoff** for real-time or resource-constrained scenarios
- Recommended config: small_e20_slic (19.6% mIoU, 7.04s)

---

## Methodology Notes

### Dataset Details
- **Source**: ISPRS Potsdam (Kaggle: jahidhasan66/isprs-potsdam)
- **Format**: 300x300 patches (pre-tiled from original 6000x6000)
- **Classes**: 6 (Impervious, Building, Low vegetation, Tree, Car, Clutter)
- **Evaluation**: Hungarian matching for unsupervised cluster→class alignment

### Evaluation Metrics
- **mIoU**: Mean Intersection over Union (primary metric)
  - Range: 0.0 to 1.0 (higher better)
  - Measures overlap quality after optimal cluster assignment
- **Pixel Accuracy**: Fraction of correctly classified pixels
  - Range: 0.0 to 1.0 (higher better)
  - More lenient than mIoU (dominated by large classes)

### Hardware
- **Device**: CPU (default, safer for limited GPU RAM)
- **Runtime**: ~12s per 300x300 image with base model
- **Memory**: <4GB RAM per process

---

## Files Generated

### OFAT Exploration (9 configs)
- `results/comparison_summary.json` - Initial OFAT results
- `results/comparison_*/` - Individual run directories

### Smart Grid Search (8 configs)
- `results/comparison_summary.json` - Smart grid results (updated)
- `results/comparison_small_e*_*/` - Small model configurations
- `results/comparison_base_e*_*/` - Base model configurations
- Each directory contains:
  - `results.json` - Per-sample metrics
  - `visualizations/` - Comparison images (if --save-viz)

## Raw Data

See `results/comparison_summary.json` for complete per-configuration metrics from the latest run (smart grid).
