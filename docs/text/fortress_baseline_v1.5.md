# FORTRESS Baseline: V1.5 + OpenCV SLIC

**Date:** 2025-12-03
**Status:** Established Baseline

## Configuration
- **Method:** `v1.5` (Unsupervised Clustering)
- **Model:** `dinov3_vitb16` (Base)
- **Refinement:** `slic` (OpenCV `cv2.ximgproc`)
- **Clustering:** KMeans (K=10 fixed for evaluation)
- **Dataset:** `fortress_processed` (Fixed version with `IGNORE_INDEX=255`)

## Key Parameters
| Parameter | Value | Description |
|---|---|---|
| `stride` | 4 | Feature extraction stride |
| `elbow_threshold` | 5.0 | Auto-K threshold (if dynamic) |
| `compactness` | 10.0 | SLIC superpixel compactness |
| `MAX_SEGMENTS` | 2000 | Cap on superpixels per image |
| `region_size` | Dynamic | Calculated from MAX_SEGMENTS |

## Performance (Sample CFB003)
| Metric | Value | vs. Raw Baseline |
|---|---|---|
| **mIoU** | **10.0%** | +30% (vs 7.7%) |
| **Pixel Accuracy** | **34.6%** | +43% (vs 24.1%) |
| **Runtime** | **46.7s** | +13s overhead |

## Improvements Implemented
1.  **Fast Refinement:** Replaced Skimage SLIC (196s) with optimized OpenCV SLIC (47s).
2.  **Dataset Fix:** Corrected preprocessing to mark unannotated background as `IGNORE_INDEX` (255) instead of Class 0 (Spruce).
3.  **Visualization:** Aligned prediction colors with Ground Truth using Hungarian matching.

## Reproduction Command
```bash
uv run python scripts/evaluate_fortress.py \
    --dataset data/fortress_processed \
    --method v1.5 \
    --model base \
    --num-samples 1 \
    --fixed-k 10 \
    --clustering slic \
    --save-viz
```
