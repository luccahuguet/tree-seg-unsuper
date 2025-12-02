# Version Roadmap (Gate-Driven)

## Project Goal: Species-Level Semantic Segmentation

**Primary objective**: Segment aerial imagery into species-level vegetation regions where:
- Visually similar vegetation (same species) is grouped together
- Different species are separated (pines ≠ firs ≠ birch)
- Non-vegetation (soil, roads, buildings) is filtered out

**NOT instance segmentation**: We don't separate individual tree crowns. We cluster by species/type.

## Sequencing Overview
`V1.5 → V3 (species clustering) → V2 (optional refinement) → V5 (multispectral) → V6 (clustering variants)`

**Status Update**: V3 instance segmentation (watershed) was wrong approach for species clustering. Pivoting to V3.

Cross-cutting standards:
- Fixed data splits and random seeds
- Cached DINOv3 tokens for reproducibility
- Runtime/VRAM logging alongside metrics
- Consistent overlays, error galleries, and version tags

---

## Pipeline Composition (Updated)

```
DINOv3 Feature Extraction (captures texture, color, pattern)
    ↓
[Clustering Layer] — V1.5 (K-means) finds visually similar regions
    ↓
[Refinement Layer] — Optional: V2 soft/EM refinement (feature space)
    ↓
[Boundary Snapping] — SLIC (image space) for clean edges
    ↓
[Vegetation Filtering] — V3: Keep only vegetation clusters, merge similar species
    ↓
[Multispectral] — V5 MSI fusion (NDVI/GNDVI/NDRE for better species distinction)
    ↓
[Clustering Variants] — V6 (spherical/soft/DP-means alternatives to K-means)
```

**Pipeline philosophy**: DINOv3 features already encode species-level differences (texture/color). Our job is to cluster and filter these features, not segment individual objects.

**Valid pipeline combinations:**
- **V1.5 alone**: Baseline semantic clustering (may already separate species with high K)
- **V1.5 → V3**: Semantic clusters + vegetation filtering + species merging
- **V1.5 → V2 → V3**: Refined clusters + vegetation filtering
- **V1.5 → V3 → V5**: RGB clusters + multispectral enhancement
- **V6 → V3**: Alternative clustering + vegetation filtering

---

## Sequencing Rationale

**Why V1.5 → V2 → V3 ordering?**

1. **V1.5 first (locked baseline)**: Establishes reference metrics and frozen artifacts. No changes allowed once locked.

2. **V2 before V3 (general before specific)**:
   - V2 improves semantic coherence in feature space (general clustering improvement)
   - V3 applies tree-specific domain knowledge (specialized filtering)
   - Progressive refinement: broad improvements first, then domain-specific tuning
   - V3's gate explicitly compares against V2 ("higher precision/recall than V2")

3. **Why not V3 before V2?**:
   - V3 performs IoU filtering against vegetation masks—works better with improved clusters
   - V2 is simpler to implement (soft/EM iterations vs vegetation indices + watershed)
   - Clearer validation: test general refinement first, then add tree logic
   - Better debugging: if V3 struggles, know whether it's tree logic or bad input clusters

4. **V6 parallel to main sequence**: Research spike, findings feed back into clustering algorithm choice. Can run V6 experiments while implementing V2/V3.

5. **SLIC throughout**: Already implemented in V1.5 as default. V2 complements it (feature space vs image space). Both can be combined.

---

## V1.5 — Baseline (Active)
- **Goal:** Provide a locked reference point.
- **Scope:** DINOv3 feature extraction → K-means clustering (plus optional SLIC), tiny-component pruning, visualization overlays, PCA plots.
- **Metrics:** Hungarian-aligned mIoU, edge-F score.
- **Deliverables:** `v1.5` tag, baseline report, frozen overlays/metrics.
- **Gate:** No further tweaks once metrics and artifacts are locked.

---

## V2 — DINO Head (Unsupervised Refine)
- **Goal:** Beat plain K-means with minimal lift.
- **Scope:** Initialize with K-means, run soft/EM refinement (temperature τ, 3–5 iterations), apply a single spatial blend (weight α), optional SLIC vote out.
- **Feature space refinement:** V2 operates on DINOv3 embeddings (semantic feature space), complementary to SLIC which operates on RGB pixels (image space). Both can be combined for improved results.
- **Deliverables:** `head_refine` module, A/B evaluation script.
- **Gate:** Demonstrated gains on both mIoU and edge-F over V1.5 at comparable runtime/VRAM. Test with and without SLIC to find best configuration.

---

## V3 — Tree Instance Segmentation (DEPRECATED)
- **Status:** ❌ Wrong approach for species-level semantic segmentation
- **Original Goal:** Individual tree crown detection via watershed
- **What we learned:** OAM-TCD has incomplete instance annotations; watershed creates too many false positives; doesn't align with species clustering goal
- **Pivot:** Moving to V3 for semantic species clustering instead

## V3 — Species-Level Semantic Clustering (Complete ✅)
- **Goal:** Segment vegetation into species-level regions without instance separation
- **Status:** ✅ Implementation complete - minimal vegetation filtering working well
- **Scope:**
  - V1.5 semantic clusters (K=15-30 for species granularity)
  - **Vegetation filtering**: Cluster-level ExG thresholding (mean ExG > 0.10)
  - Output: Semantic map where each label = distinct species/vegetation type
  - **Important**: Multiple disconnected regions of same species will have different labels (e.g., label 3 = pine patch A, label 7 = pine patch B). We're not doing species classification, just clustering by visual similarity.
- **Key Finding:** DINOv3 + K-means already does most of the work! Simple cluster-level ExG filter sufficient (no complex multi-index fusion needed).
- **Implementation:**
  - Module: `tree_seg/vegetation_filter.py` (~150 lines)
  - Integration: Config parameter `v3_exg_threshold=0.10`, pipeline flag `pipeline="v3"`
  - Testing: `scripts/test_v3_filter.py` with 4-panel visualization
- **Validation:**
  - Sample 4363: 20→9 clusters, 55.4% filtered (soil/roads)
  - Sample 545: 20→13 clusters, 39.8% filtered (black regions/roads)
  - Successfully removes non-vegetation while preserving species diversity
- **Gate:** ✅ Passed qualitative assessment - filtered clusters align with visible species boundaries, ExG removes non-vegetation effectively
- **Documentation:**
  - Feature analysis: `docs/text/dinov3_vegetation_analysis.md`
  - Implementation details: `docs/text/v3_pivot.md`

---

## V4 — Supervised Baseline (Mask2Former) *(Comparison Point)*
- **Goal:** Establish performance ceiling using pretrained supervised segmentation.
- **Scope:** DINOv3 ViT-7B/16 backbone + Mask2Former head pretrained on ADE20K (150 classes). Zero-shot inference on tree imagery for comparison against unsupervised methods (V1.5-V3).
- **Deliverables:** Benchmark results comparing V4 against V1.5/V2/V3 on mIoU, edge-F, and tree-specific metrics.
- **Implementation status:** ✅ Implemented in `tree_seg/models/mask2former.py`
- **Gate:** Document performance vs unsupervised approaches for paper discussion. No improvement gate—serves as reference point.
- **Implementation note:** Mask2Former checkpoints must be accessible. If direct downloads from `dl.fbaipublicfiles.com` are blocked, set `DINOV3_MASK2FORMER_WEIGHTS` and `DINOV3_BACKBONE_WEIGHTS` to local paths before running.
- **Model availability:** Only ViT-7B/16 has pretrained Mask2Former weights. Smaller models (ViT-B/16, ViT-L/16) do not have publicly available Mask2Former checkpoints.
- **Resource warning:** Loading the official ViT-7B backbone (~26 GB) plus Mask2Former head (~3.5 GB) consumes more than 40 GB of RAM in float32. Expect CPU-only workstations with ≤32 GB to crash; run V4 benchmarks on a high-memory remote server or GPU box (≥64 GB RAM or A100-class GPU) and record the requirement in documentation.

---

## V5 — Multispectral (MSI)
- **V5a Vegetation Gating**
  - Use NDVI/GNDVI/NDRE to strengthen the vegetation mask and cap SAM growth.
- **V5b Late Fusion**
  - Concatenate MSI indices (normalized) with DINOv3 tokens, then run the V2 head without retraining DINO.
- **Deliverables:** MSI index module, fusion toggles, comparative plots against V3/V4.
- **Gate:** Improved tree precision/recall **or** higher species purity while keeping edge-F stable.

---

## V6 — K-Means Successors *(Time-boxed Spike)*
- **Goal:** Explore alternative clustering algorithms to replace vanilla K-means.
- **Scope:** Test clustering algorithm variants: (1) Spherical k-means (cosine metric for normalized DINOv3 tokens), (2) Soft k-means (as clustering algorithm, not refinement), (3) DP-means (automatic K selection).
- **Clarification:** Soft k-means in V6 is a clustering algorithm choice. This is distinct from V2's soft/EM refinement which operates on K-means output. V6 outputs can feed into V2 refinement.
- **Gate:** Adopt only if both mIoU and edge-F exceed V2 at similar runtime/VRAM; otherwise archive findings.

