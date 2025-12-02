## Update: Fast Refinement Experiments

**Date:** 2025-12-02

### 1. Bilateral Filter 📉
- **Runtime:** 36.34s (vs 33.9s baseline) - Very fast overhead (~2.4s).
- **Metrics:** 7.7% mIoU, 24.1% Acc.
- **Result:** **Identical to baseline.** No improvement.
- **Analysis:** The bilateral filter smoothing didn't change the majority vote enough to affect metrics, or the parameters (`d=9, sigma=75`) were too conservative for the coarse DINOv3 features.

### 2. OpenCV SLIC (Initial) 🐌
- **Status:** Terminated after >5 minutes.
- **Issue:** Used `region_size=48` (default), which for 9372x9372 image results in **~38,000 superpixels**.
- **Comparison:** Previous `skimage` test used `MAX_SEGMENTS=2000` (approx `region_size=210`).
- **Correction Needed:** Must calculate `region_size` dynamically to enforce a maximum number of segments (e.g., 2000) to achieve target speed.

### 3. Dense CRF 🚫
- **Status:** Blocked.
- **Reason:** `pydensecrf2` fails to build on Python 3.12/3.13 due to C API incompatibilities.

### 4. Optimized OpenCV SLIC 🚀 (Winner!)
- **Optimization 1:** Dynamic `region_size` to cap segments at 2000.
- **Optimization 2:** Vectorized majority voting using `np.histogram2d` (eliminated O(K*N) loop).
- **Runtime:** **47.7s** (vs 196s Skimage, 149s Unoptimized OpenCV).
- **Metrics:** **9.4% mIoU** (Best result!), 29.4% Acc.
- **Overhead:** Only **+14s** vs baseline.
- **Speedup:** **4.1× faster** than Skimage SLIC.

---

## Final Recommendation

**Use OpenCV SLIC with optimizations:**
- **Method:** `v1.5` + `refine="slic"`
- **Implementation:** `cv2.ximgproc.createSuperpixelSLIC`
- **Settings:** `compactness=10`, `MAX_SEGMENTS=2000`
- **Performance:** 9.4% mIoU @ ~48s/image

This provides the best trade-off between accuracy (+22% relative improvement over baseline) and speed (acceptable overhead).
