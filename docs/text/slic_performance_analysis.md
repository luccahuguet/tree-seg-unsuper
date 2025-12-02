# SLIC Refinement Performance Analysis

**Date:** 2025-12-02  
**Dataset:** FORTRESS (9372×9372 pixel orthomosaics)

---

## Problem

SLIC edge-aware refinement significantly improves segmentation quality but has unacceptable runtime on large aerial images.

---

## Benchmark Results (CFB003, K=10)

| Configuration | mIoU | Pixel Acc | Runtime | Speedup |
|---------------|------|-----------|---------|---------|
| V1.5 baseline | 7.7% | 24.1% | 34s | 1.0× |
| **V1.5 + SLIC** | **9.1%** | **28.5%** | **196s** | **0.17×** |
| V3 (veg filter) | 6.4% | 24.3% | 36s | 0.94× |

**SLIC Impact:**
- ✅ **+18% mIoU improvement** (9.1% vs 7.7%)
- ✅ **+18% pixel accuracy improvement** 
- ❌ **5.7× slower** (196s vs 34s per image)
- ❌ **Full dataset: 2.7 hours vs 27 minutes**

---

## Root Cause

**Image dimensions:** 9372 × 9372 = 87.8M pixels

**Original SLIC implementation:**
```python
n_segments = max(100, int((h * w) / (48 * 48)))
# = 38,122 superpixels for FORTRESS images
```

**Problems:**
1. Too many superpixels (38K+) caused infinite hang
2. Even with MAX_SEGMENTS=2000 cap, still very slow
3. SLIC complexity: O(n_iterations × n_pixels × sqrt(n_segments))

**Bug fix applied:**
```python
MAX_SEGMENTS = 2000
n_segments = min(n_segments, MAX_SEGMENTS)
```

This prevented hanging but runtime is still 5.7× slower.

---

## Why SLIC Helps

SLIC performs **majority voting within edge-aware superpixels**:

1. K-means creates coarse clusters based on DINOv3 features
2. SLIC creates superpixels respecting image edges
3. Each superpixel adopts the majority cluster label
4. Result: Cleaner boundaries, less noise

**Effect on FORTRESS:**
- Better species boundary alignment (+18% mIoU)
- Reduces over-segmentation artifacts
- Respects tree edges from aerial imagery

---

## Alternatives Needed

**Requirements:**
- Edge-aware refinement quality similar to SLIC
- Runtime comparable to baseline (~30-40s)
- Works with 9000×9000 pixel images

---

## Research Findings: Fast Edge-Aware Refinement Alternatives

### 1. **Fast-SLIC** ⚡ (Most Promising)
**Python Library:** `fast-slic` (https://github.com/Algy/fast-slic)
- **Speed:** 7-20× faster than standard SLIC
- **Optimizations:** AVX2 CPU instructions, parallel processing
- **Real-time capable:** Can process video streams
- **Status:** Mature Python package on PyPI
- **Estimated runtime:** ~10-30s (vs 196s)

**Implementation:**
```python
from fast_slic import Slic
slic = Slic(num_components=2000, compactness=10)
segments = slic.iterate(image)
```

### 2. **Dense CRF (Conditional Random Fields)** 🎯
**Python Library:** `pydensecrf` (https://github.com/lucasb-eyer/pydensecrf)
- **Speed:** Fast C++ implementation with Python wrapper
- **Method:** Fully-connected CRF with Gaussian edge potentials
- **Use case:** Standard post-processing in semantic segmentation
- **Quality:** Often better than SLIC for segmentation refinement
- **Estimated runtime:** ~5-15s per image

**Advantages:**
- Considers global image structure (not just local superpixels)
- Smooths within regions while preserving edges
- Used in DeepLab and other SOTA segmentation methods

### 3. **Bilateral Filter** 🔧 (Simplest)
**Built-in:** OpenCV `cv2.bilateralFilter()`
- **Speed:** Very fast (C++ implementation)
- **Method:** Edge-preserving blur based on spatial + intensity distance
- **Quality:** Good for noise reduction, moderate edge preservation
- **Estimated runtime:** <5s per image

**Trade-off:** Less sophisticated than superpixels, but much faster

### 4. **Felzenszwalb's Graph-Based Segmentation** 
**Built-in:** `skimage.segmentation.felzenszwalb`
- **Speed:** Faster than SLIC
- **Method:** Efficient graph-based clustering
- **Already available:** No new dependencies
- **Estimated runtime:** ~30-60s

### 5. **Quickshift**
**Built-in:** `skimage.segmentation.quickshift`
- **Speed:** Moderate (slower than Felzenszwalb, faster than standard SLIC)
- **Method:** Mode-seeking in 5D color+location space
- **Already available:** No new dependencies

### 6. **Resize Strategy** 💡 (Creative Solution)
**Approach:** Downsample → SLIC → Upsample
- **Speed:** Proportional to pixel count (4× smaller = 16× faster)
- **Trade-off:** Some edge precision loss
- **Implementation:**
  1. Resize to 50% (9372 → 4686 pixels)
  2. Run SLIC with MAX_SEGMENTS=2000
  3. Resize segments back to original size
- **Estimated runtime:** ~12-25s (vs 196s)

---

## Recommended Solution

**1st Choice: Dense CRF** ✅
- Best quality/speed trade-off for semantic segmentation
- Industry-standard post-processing method
- Fast C++ implementation
- Expected: 9%+ mIoU in <60s

**2nd Choice: Fast-SLIC** ✅
- Direct SLIC replacement with minimal code changes
- Proven 7-20× speedup
- Maintains superpixel-based approach

**3rd Choice: Bilateral Filter** ✅
- Simplest implementation (one OpenCV call)
- Fastest option (<5s)
- May achieve 8-8.5% mIoU (vs 9.1%)

**Fallback: Resize + SLIC**
- If other methods don't work
- Guaranteed to be faster
- Controllable quality vs speed trade-off

---

## Next Steps

1. **Implement Dense CRF refinement** as primary replacement
2. **Benchmark** on FORTRESS CFB003 sample
3. **Compare** quality (mIoU) vs runtime
4. **If successful:** Use for full dataset evaluation

Expected outcome: **8.5-9.5% mIoU in 30-60 seconds** (vs current 7.7% in 34s or 9.1% in 196s)

---

## Related Files

- [`tree_seg/core/segmentation.py`](file:///home/lucca/pjs/science/tree-seg-unsuper/tree_seg/core/segmentation.py#L339-L388) - `_refine_with_slic()` implementation
- [`scripts/evaluate_fortress.py`](file:///home/lucca/pjs/science/tree-seg-unsuper/scripts/evaluate_fortress.py) - FORTRESS evaluation
- [`docs/text/fortress_integration.md`](file:///home/lucca/pjs/science/tree-seg-unsuper/docs/text/fortress_integration.md) - FORTRESS dataset docs
