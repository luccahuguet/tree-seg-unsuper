# Paper Timeline: End of Year

| Week | Week Starting | Task | Description |
|------|---------------|------|-------------|
| **W1** | **Oct 6** | Benchmark baseline evaluation | Select benchmark dataset (NEON AOP / BAMForest / IDTrees), download and prepare data, run V1.5 baseline with multiple configurations, compute mIoU and pixel accuracy scores |
| **W2** | **Oct 13** | Implement V2 (Head Refinement) | Add soft/EM head refinement with spatial blending, validate gains on mIoU and edge-F over V1.5 |
| **W3** | **Oct 20** | Implement V3 (Tree Focus) | Build vegetation-gated clustering, apply shape filters, generate instance masks with DT + watershed |
| **W4** | **Oct 27** | Prototype V4 (SAM Polisher) | Configure auto prompts, vegetation gating, and precision guardrails for optional SAM refinement |
| **W5** | **Nov 3** | Run full evaluation + analyze results | Execute V1.5–V4 on the test set, compute metrics (mIoU, edge-F, pixel accuracy), generate visualizations, compare strengths/weaknesses |
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

### V2: Head Refinement (Soft/EM)
- **Task 1:** Prototype refinement module
  - Initialize with K-means clusters from V1.5
  - Run soft/EM updates with temperature (τ) for 3–5 iterations
- **Task 2:** Spatial blending
  - Apply a single smoothing/blending pass (weight α) to reinforce local consistency
- **Evaluation:** Expect higher mIoU and edge-F with minimal runtime overhead
- **Deliverable:** `head_refine` module, A/B runner, metrics report vs V1.5

---

### V3: Tree-Focused Segmentation (RGB)
- **Task 1:** Vegetation gating
  - Compute ExG/CIVE indices to isolate vegetation pixels
  - Select clusters by IoU to vegetation mask + green ratio thresholds
- **Task 2:** Instance shaping
  - Apply shape/area filters using GSD-aware thresholds
  - Use distance transform + watershed to emit instance masks
- **Task 3:** Optional refinement
  - Keep SLIC snapping; consider light CRF/bilateral smoothing
- **Evaluation:** Track tree precision/recall and edge-F relative to V2
- **Deliverable:** Binary tree mask, instance mask, per-tile CSV

---

### V4: SAM Polisher (Optional Assistant)
- **Task 1:** Prompt generation
  - Use connected-component centroids as positive points, optional boundary rings as negatives
  - Provide optional bounding boxes derived from cluster extents
- **Task 2:** Vegetation-aware gating
  - Apply NDVI/green masks (or ExG) to constrain SAM expansion
  - Record pre/post outputs and prompt metadata for audits
- **Task 3:** Precision guardrails
  - Enforce configurable thresholds: ≥ X% edge-F gain with ≤ Y% precision loss
- **Deliverable:** `sam_polish` stage with toggleable integration

---

## Key Research Question

**Can incremental, unsupervised refinements (V2/V3) close the accuracy gap enough that optional SAM polishing (V4) becomes a targeted assist instead of a dependency?**

### Comparison Framework
- **V1.5 (Baseline)**: DINOv3 features + K-means clustering with automatic K-selection
- **V2 (Head Refinement)**: Soft/EM updates with spatial blending
- **V3 (Tree Focus)**: Vegetation-informed filtering and instance generation
- **V4 (SAM Polisher)**: Prompted boundary cleanup gated by precision/edge metrics

### Expected Outcomes
- **V2 Advantages**: Minimal code change, measurable boosts to mIoU/edge-F
- **V3 Advantages**: Better tree precision/recall and cleaner instances for forestry tasks
- **V4 Role**: Optional safety net for thin structures when metrics justify the extra cost
- **Research Gap**: Quantifying how far unsupervised refinements can go before SAM or multispectral inputs become necessary

---

### V5: Multispectral Extension (MSI)
- **V5a Vegetation Gating**
  - Compute NDVI/GNDVI/NDRE to strengthen vegetation masks
  - Constrain SAM growth and tree-focus stages using MSI thresholds
- **V5b Late Fusion**
  - Concatenate normalized MSI indices with DINO tokens
  - Reuse the V2 head refinement without retraining DINO
- **Evaluation:** Compare RGB vs MSI runs for tree precision/recall and species purity while monitoring edge-F
- **Comment:** Future extension once RGB roadmap stabilizes; expected 5–15% mIoU gain when multispectral data is available

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

1. **Ship V2 head refinement**
   - Implement soft/EM updates + spatial blend and validate ≥ V1.5 on mIoU + edge-F
   - Record runtime/VRAM deltas alongside metrics

2. **Add V3 tree-focused pipeline**
   - Generate vegetation masks (ExG/CIVE), apply cluster gating, and run DT + watershed instances
   - Track tree precision/recall, edge-F, and instance stats (count, mean area)

3. **Optional V4 SAM polishing**
   - Auto-generate prompts, enforce vegetation gating, and log precision/edge-F trade-offs
   - Save pre/post overlays and prompt metadata for audits
   - Run on high-memory infrastructure (≥64 GB RAM or 48 GB GPU); local CPU boxes may crash loading ViT-7B + Mask2Former

4. **Ground-truth subset & scoring**
   - Maintain Hungarian-aligned mIoU, pixel accuracy, and edge-F
   - Add tree P/R and instance metrics for V3+, capturing qualitative galleries

5. **Analyze trade-offs**
   - Minimal-lift refinements (V2) vs heuristic-heavy tree focus (V3) vs SAM assistance (V4)
   - Computational cost comparison and decision thresholds for deploying optional stages

---

## Relevant Links and References

- DINOv3: https://github.com/facebookresearch/dinov3
- Soft k-means / EM overview: https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf
- Vegetation indices (ExG/CIVE/NDVI): https://www.indexdatabase.de/db/i.php
- SAM (Segment Anything): https://segment-anything.com/
- NEON AOP: https://data.neonscience.org/data-products/DP3.30010.001

---

**Summary:**
- **V1**: Patch features only
- **V1.5**: Patch + attention features, automatic K-selection (**current unsupervised baseline**)
- **V2**: Head refinement (soft/EM + spatial blend)
- **V3**: Tree-focused segmentation (vegetation gating + instances)
- **V4**: SAM polisher (optional assistant)
- **V5**: Multispectral extension (vegetation gating + late fusion)
- **V6**: K-means successors (spherical/soft, DP-means spike)
