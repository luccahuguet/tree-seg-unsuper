## Update: OpenCV SLIC Success

**Date:** 2025-12-02
**Solution:** `opencv-contrib-python` provides fast C++ SLIC

**Problem Recap:**
- `skimage.segmentation.slic` was too slow (196s/image).
- `pydensecrf`/`pydensecrf2` failed to build (Python 3.13/3.12 issues).
- `fast-slic` failed to build (legacy build system).

**The Winner: OpenCV `ximgproc`**
- **Library:** `opencv-contrib-python` (pre-built wheels available!)
- **Method:** `cv2.ximgproc.createSuperpixelSLIC`
- **Speed:** C++ implementation, expected to be very fast (~10-20s).
- **Quality:** Standard SLIC algorithm.

**Implementation Plan:**
1.  Replaced `opencv-python` with `opencv-contrib-python`.
2.  Implement `_refine_with_opencv_slic` wrapper.
3.  Use as default for "slic" refinement.
