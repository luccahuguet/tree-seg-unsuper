# K-Means Successors for DINOv3 Tokens

Yep—there *are* “k-means successors” that stay simple but fix its biggest pain points (cosine space, choosing K, spatial smoothness). Here’s a short list tuned for the current DINOv3 feature pipeline.

| Candidate | Why it’s a successor | Drop-in cost | Auto-K? | Speed | What to expect vs k-means |
| --- | --- | ---: | ---: | ---: | --- |
| **Spherical k-means** (cosine) | DINOv3 tokens live on the unit sphere; cosine is a better metric than L2 | **Low** (L2-normalize, replace distance with dot-sim) | No | **Fast** | Cleaner clusters, fewer “angle” mistakes; great first upgrade |
| **Soft k-means / EM** (with temperature) | Soft assignments + prototype refinement handle overlaps better | Low | No | Fast | Crisper boundaries and more stable results across tiles; easy win |
| **DP-means** | k-means with a penalty that automatically picks K | Medium (needs λ) | **Yes** | Med | Good when K varies tile-to-tile; fewer under/over-cluster cases |
| **X-means / G-means** | Start small, split clusters by BIC (X) or Gaussianity (G) | Medium | **Yes** | Med | More stable K without hand-tuning; still k-means-like |
| **Power Iteration Clustering (PIC)** | Graph-based k-means cousin; captures non-convex structure | Medium (kNN graph) | No | Med | Better separation along roads/canopy ridges than vanilla k-means |
| **HDBSCAN (on kNN graph)** | Density-based, no K; can mark noise | Medium–High | **Yes** | Slow–Med | Good for dumping background/noise; pair with superpixels/RAG to scale |
| **Regularized k-means (Potts/CRF)** | k-means plus spatial smoothness term solved with graph-cuts | Medium | No | Med | Noticeably cleaner edges; great for aerial/tree imagery |

## Recommended Order of Attack
1. **Spherical + Soft k-means (temperature-sharpened)** — trivial change, usually the biggest bang for buck on ViT/DINO tokens.
2. **DP-means** — removes the “pick K” loop; tune one λ (start with median pairwise distance × 0.7 after L2-normalizing).
3. **Regularized k-means** — add a Potts term on a 4/8-neighbor grid or SLIC RAG; solve once with α-expansion (PyMaxflow). Keeps the pipeline but fixes speckle/edges.
4. **PIC** if non-convex shapes (e.g., winding canopy edges) still merge or split oddly.

## Tiny Implementation Notes (DINOv3-Friendly)
- **Spherical/Soft k-means**
  - L2-normalize tokens: `f_i ← f_i / ||f_i||`.
  - Assign with cosine: `p_ik ∝ exp(τ * c_k · f_i)` (τ ≈ 10–20), update `c_k ← normalize(∑_i p_ik f_i)` for 3–5 EM iterations.
- **DP-means**
  - Start with one center; when `min_k ||f_i − c_k||² > λ`, spawn a new center at `f_i`. Iterate assignments/updates as usual.
- **Regularized k-means (Potts)**
  - Energy: `Σ_i ||f_i − c_{z_i}||² + β Σ_(i∼j) [z_i ≠ z_j]`.
  - Alternate center updates with α-expansion graph-cut label updates; set β ≈ 0.3–0.7 and use SLIC adjacency to reduce pixels.
- **PIC**
  - Build a mutual kNN graph (k ≈ 10–20) with cosine weights; one power-iteration step yields an embedding; cluster that with spherical k-means.

## Quick Defaults to Get You Moving
- Normalize tokens and prefer cosine everywhere.
- Initial K (if needed): 6–12; or switch to DP-means with `λ ≈ 0.7 × median(||f_i − f_j||²)`.
- For Potts regularization: reuse the existing SLIC graph, start β at 0.5, and run a single α-expansion pass.
- Evaluate exactly as today (mIoU/Hungarian + edge F). Keep runtime in check with FAISS for kNN builds if you explore PIC/HDBSCAN.

Need a drop-in implementation (e.g., spherical soft k-means)? Flag it, and we can stub a 40-line version that slots where the current K-means call lives.
