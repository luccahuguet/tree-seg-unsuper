# Version Roadmap (Gate-Driven)

## Sequencing Overview
`V1.5 → V2 → V3 → V4 → V5a → V5b → (optional) V6`

Cross-cutting standards:
- Fixed data splits and random seeds
- Cached DINOv3 tokens for reproducibility
- Runtime/VRAM logging alongside metrics
- Consistent overlays, error galleries, and version tags

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
- **Deliverables:** `head_refine` module, A/B evaluation script.
- **Gate:** Demonstrated gains on both mIoU and edge-F over V1.5 at comparable runtime/VRAM.

---

## V3 — Tree Focus (RGB Only, No SAM)
- **Goal:** Produce robust tree/non-tree masks plus instances.
- **Scope:** Vegetation prefiltering (ExG/CIVE), cluster selection driven by IoU to veg mask + green ratio, shape/area filters with GSD awareness, distance-transform + watershed instance split, optional CRF or bilateral smoothing, retain SLIC snapping.
- **Deliverables:** Binary tree mask, instance mask, per-tile CSV summaries.
- **Gate:** Higher tree precision/recall than V2 without regressing edge-F.

---

## V4 — SAM Polisher / Assistant *(Optional)*
- **Goal:** Recover thin structures and sharpen boundaries.
- **Scope:** Auto prompts from connected-component centroids (positive) and rings (negative), optional boxes, vegetation prior gating, optional interactive click UI, persist pre/post images and prompt logs.
- **Deliverables:** `sam_polish` stage and configuration.
- **Gate:** Configurable threshold such that edge-F improves by ≥ *X%* with precision loss ≤ *Y%*.
- **Implementation note:** Mask2Former checkpoints must be accessible. If direct downloads from `dl.fbaipublicfiles.com` are blocked, set `DINOV3_MASK2FORMER_WEIGHTS` and `DINOV3_BACKBONE_WEIGHTS` to local paths before running.
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
- **Goal:** Probe simple clustering upgrades.
- **Scope:** Start with spherical + soft k-means; add DP-means (auto-K) if warranted.
- **Gate:** Adopt only if both mIoU and edge-F exceed V2 at similar runtime/VRAM; otherwise archive findings.
