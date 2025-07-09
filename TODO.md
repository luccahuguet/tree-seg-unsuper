# TODO List - Tree Segmentation Project

- [x] Test larger DINO models: Experiment with `dinov2_vitb14`, `dinov2_vitl14`, or `dinov2_vitg14` for potentially better feature extraction
- [x] Add edge overlay visualization: Create a new visualization that doesn't color the image, but overlays edge borders on top of each segmented region
- [x] Add min region size filter: Filter out regions smaller than a certain percentage of the image area
- [x] fix region size filter: start by making the regions black for debugging purposes
- [ ] Add option to turn off hatching
- [ ] Test without PCA: Run segmentation without PCA dimensionality reduction and compare results to see if the 128-dimension limit is necessary

## 📊 Current Status
- [x] Automatic K selection using elbow method
- [x] Enhanced elbow detection with multiple methods
- [x] Tree-optimized K range (3-10)
- [x] Beautiful analysis plots
- [x] Streamlined configuration

## 🚀 Optional Future Enhancements
- [ ] Test different stride values for resolution vs performance trade-offs
- [ ] Experiment with different attention feature combinations
- [ ] Add validation metrics for segmentation quality
