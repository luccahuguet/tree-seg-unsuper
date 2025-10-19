# Paper Timeline: End of Year

| Week | Week Starting | Task | Description |
|------|---------------|------|-------------|
| **W1** | **Oct 6** | Benchmark baseline evaluation | Select benchmark dataset (NEON AOP / BAMForest / IDTrees), download and prepare data, run V1.5 baseline with multiple configurations, compute mIoU and pixel accuracy scores |
| **W2** | **Oct 13** | Implement V2 (U2Seg) | Integrate codebase, adapt for DINOv3 features, test on sample images, verify output format |
| **W3** | **Oct 20** | Implement V3 (DynaSeg) | Integrate codebase, adapt for DINOv3 features, configure dynamic weighting parameters, test on sample images |
| **W4** | **Oct 27** | Implement V4 (DINOv3 Mask2Former) | Load pretrained segmentor via torch.hub, setup inference pipeline, adapt to drone imagery resolution, test zero-shot performance |
| **W5** | **Nov 3** | Run full evaluation + analyze results | Execute all methods (V1.5, V2, V3, V4) on complete test set, compute metrics (mIoU, boundary precision, pixel accuracy), generate visualizations, compare results and identify strengths/weaknesses |
| **W6** | **Nov 10** | Draft methodology + results sections | Complete methodology writeup (V1.5/V2/V3/V4 pipelines, parameters), draft results section with tables and figures |
| **W7** | **Nov 17** | Draft related work + introduction | Literature review section (unsupervised segmentation, self-supervised models, forestry applications), introduction and abstract |
| **W8** | **Nov 24** | Draft discussion + complete first draft | Discussion section (interpret findings, limitations, V5 outlook), integrate all sections into complete draft |
| **W9** | **Dec 1** | Internal review and revisions | Review complete draft, address gaps, improve flow and coherence, refine arguments |
| **W10** | **Dec 8** | Final polishing for semester deadline | Format report, finalize references and figures, proofread, submit semester report by Dec 15 |

---

## Pipeline Architecture Overview

### V1: Visual Feature Clustering (K-Means)
- **Task 1:** Deep Feature Extraction (Model: DINOv3)
  - Method: Extract dense DINOv3 feature maps from RGB drone images
  - Use upsampling techniques
- **Task 2:** Unsupervised Segmentation
  - Method: Apply K-Means to DINOv3 feature vectors (pixels/patches)
- **Comment:** Baseline using visual features and K-Means for RGB images

---

### V1.5: Visual Feature + Attention Clustering (K-Means) [Current]
- **Task 1:** Deep Feature Extraction (Model: DINOv3)
  - Extract both patch features and attention features from DINOv3
  - Concatenate patch and attention features for each patch
- **Task 2:** Unsupervised Segmentation
  - Apply K-Means to the concatenated feature vectors
  - Automatic K-selection using elbow method with forest-specific thresholds (5.0% default)
- **Task 3:** Cluster Visualization
  - Output PCA scatter plots of features colored by cluster
  - Side-by-side, edge overlay, and segmentation legend visualizations
- **Comment:** Enhanced baseline. Incorporates contextual information via attention, provides cluster visualization for qualitative analysis

---

### V2: Advanced Visual Feature Clustering (U2Seg)
- **Task 1:** Unsupervised Segmentation
  - Method: Apply U2Seg model to RGB drone images
  - U2Seg uses self-supervised features, compatible with DINOv3
- **Comment:** Uses U2Seg, optimized for RGB imagery
  - Open-source code: https://github.com/u2seg/U2Seg
  - Expected competitive mIoU performance

---

### V3: Advanced Visual Feature Clustering (DynaSeg)
- **Task 1:** Deep Feature Extraction (Model: DINOv3)
  - Method: Extract dense feature maps from RGB drone images
- **Task 2:** Unsupervised Segmentation
  - Method: Apply DynaSeg model to extracted features
  - DynaSeg balances feature similarity and spatial continuity
- **Comment:** Uses DynaSeg's dynamic weighting for segmentation
  - Open-source code: https://github.com/RyersonMultimediaLab/DynaSeg
  - Strong benchmark performance expected

---

### V4: Supervised Segmentation (DINOv3 Mask2Former) [NEW - For Comparison]
- **Task 1:** Supervised Segmentation Head
  - Method: Use pretrained DINOv3 + Mask2Former segmentation head
  - Trained on ADE20K dataset (150 semantic classes)
  - Zero-shot transfer to tree segmentation task
- **Architecture:**
  - **Backbone**: Frozen DINOv3 features (ViT-7B/16 or ViT-L/16)
  - **Pixel Decoder**: MSDeformAttnPixelDecoder (6 transformer layers)
  - **Transformer Decoder**: MultiScaleMaskedTransformerDecoder (9 decoder layers, 100 queries)
- **Implementation:**
  - Load via: `torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', ...)`
  - Inference code: `dinov3/dinov3/eval/segmentation/inference.py`
- **Performance:** 63.0% mIoU on ADE20K (SOTA for frozen backbone)
- **Comment:** Supervised baseline for comparison. Tests whether task-specific unsupervised methods (V1.5) can match or exceed zero-shot supervised performance on tree segmentation

---

## Key Research Question

**Can unsupervised methods (V1.5) specialized for tree segmentation match or exceed the zero-shot performance of supervised segmentation heads (V4) trained on general scenes?**

### Comparison Framework
- **V1.5 (Unsupervised)**: DINOv3 features + K-means clustering with automatic K-selection
- **V4 (Supervised)**: DINOv3 features + Mask2Former head trained on ADE20K

### Expected Outcomes
- **V1.5 Advantages**: No training data required, domain-specific K-selection tuning
- **V4 Advantages**: Learned semantic boundaries, richer training signal
- **Research Gap**: Understanding when unsupervised specialization outperforms supervised generalization

---

### V5: Multispectral Extension (Future Work)
- **Task 1:** Multispectral Feature Extraction
  - Method: Adapt DINOv3 or train custom adapter for multispectral bands
  - Utilize near-infrared (NIR), red-edge, and other spectral bands
- **Task 2:** Enhanced Unsupervised Segmentation
  - Method: Apply V1.5 pipeline (K-means + attention) to multispectral features
  - Alternative: Extend V2 (U2Seg) or V3 (DynaSeg) for multispectral data
- **Task 3:** RGB vs Multispectral Comparison
  - Compare segmentation quality between RGB-only and multispectral approaches
  - Assess vegetation indices (NDVI, GNDVI) integration
- **Comment:** Future extension addressing secondary research question
  - Spectral bands provide complementary information for tree species differentiation
  - NEON AOP dataset supports multispectral analysis
  - Expected improvement: 5-15% mIoU gain over RGB-only methods

---

## Dataset Benchmarks

- **NEON Aerial Observation Platform (AOP)** [Recommended]
  - High-resolution RGB aerial imagery of U.S. forest sites
  - Includes tree crown and species annotations for some regions
  - Ideal for ground-truth subset and mIoU/Pixel Accuracy evaluation
  - Accessible: https://data.neonscience.org/data-products/DP3.30010.001

- **BAMForest / IDTrees** [Additional consideration]
  - Tree-specific datasets for validation

---

## Evaluation Plan

1. **Implement V4 (DINOv3 + Mask2Former) on RGB drone imagery**
   - Use pretrained segmentor from DINOv3 codebase
   - Apply zero-shot inference on tree segmentation task

2. **Compare V1.5 vs V4 outputs on RGB drone images**
   - Assess tree crown/species patch segmentation
   - Qualitative analysis of boundary quality

3. **Create ground-truth labeled subset**
   - Compare performance with Pixel Accuracy and mIoU
   - Use Hungarian algorithm for unsupervised evaluation
   - **Expected mIoU:** V1.5 (20–30%), V4 (30–50%) for trees
   - **Expected Pixel Accuracy:** V1.5 (55–65%), V4 (65–80%)

4. **Analyze trade-offs:**
   - Zero training data (V1.5) vs supervised pretraining (V4)
   - Domain specialization vs generalization
   - Computational cost comparison

---

## Relevant Links and References

- DINOv3: https://github.com/facebookresearch/dinov3
- U2Seg: https://github.com/u2seg/U2Seg
- DynaSeg: https://github.com/RyersonMultimediaLab/DynaSeg
- NEON AOP: https://data.neonscience.org/data-products/DP3.30010.001

---

**Summary:**
- **V1**: Patch features only
- **V1.5**: Patch + attention features, automatic K-selection (**current unsupervised baseline**)
- **V2**: U2Seg (future work)
- **V3**: DynaSeg (future work)
- **V4**: DINOv3 + Mask2Former supervised head (**new supervised comparison baseline**)
- **V5**: Multispectral extension (future work - secondary research question)
