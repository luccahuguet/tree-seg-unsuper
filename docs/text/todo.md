# TODO

- Verify v1/v2/v3/v4 separation; consider abandoning the version split if it no longer maps cleanly to the current pipelines.
- Schedule untried experiments from `experiments.md`:
  - Tile overlap optimization (128/384/512px)
  - Elbow threshold sweep (0.1, 0.5, 1.0, 3.0, 10.0)
  - Feature stride optimization (stride=2)
  - Potential k-means successors: spherical+soft k-means, DP-means, regularized k-means (Potts on SLIC graph)
