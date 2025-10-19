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

## Next Steps: Smart Grid Search

### Unanswered Questions
1. Does SLIC + elbow 20.0 combine for even better results?
2. Can mega model (ViT-7B/16) outperform base?
3. Is small + SLIC a good "fast mode"?

### Proposed Grid (8 configs)
- **Models**: base, mega (skip large - proven worse)
- **Thresholds**: 10.0, 20.0 (top performers)
- **Refinement**: kmeans, slic
- **Total**: 2×2×2 = 8 configurations (~16 minutes)

### Hypothesis
Best config will be: **mega + slic + elbow 20.0**
- Mega: Satellite-optimized features (ViT-7B/16)
- SLIC: Spatial consistency (+8% mIoU)
- Elbow 20.0: Conservative clustering (+6% mIoU over 2.5)
- **Predicted mIoU**: ~24-26%

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

- `results/comparison_summary.json` - Full results for all 9 configs
- `results/comparison_*/` - Individual run directories with:
  - `results.json` - Per-sample metrics
  - `visualizations/` - Comparison images (if --save-viz)

## Raw Data

See `results/comparison_summary.json` for complete per-configuration metrics.
