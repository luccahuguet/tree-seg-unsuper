# TODO

- Verify v1/v2/v3/v4 separation; consider abandoning the version split if it no longer maps cleanly to the current pipelines.
- Schedule untried experiments from `experiments.md`:
  - Tile overlap optimization (128/384/512px)
  - Elbow threshold sweep (0.1, 0.5, 1.0, 3.0, 10.0)
  - Feature stride optimization (stride=2)
  - Potential k-means successors: spherical+soft k-means, DP-means, regularized k-means (Potts on SLIC graph)
- Plan a larger sweep on FORTRESS (3–10 images) to identify the current peak performer across clustering/refine/tiling/stride variants.
- Add cache reuse for eval/segment: check metadata hash before running, return existing results/viz if found; support `--force/--no-cache`; skip cached configs in sweeps; handle partial artifacts (regen viz from labels).
- Future (optional): meta-learning on metadata bank
  - Add dataset feature descriptors (resolution stats, tile counts, class counts, color/entropy/exg)
  - Train a simple ranker/nearest-neighbor recommender for configs per dataset
  - (Optional) BO/surrogate to propose next configs under runtime constraints
  - Expose via `tree-seg results --recommend --dataset <name>`
