# Week 2 Plan – Head Refinement (V2)

## Objective
- Deliver the V2 head-refinement module (soft/EM + spatial blend) that reuses the existing DINOv3 feature pipeline, preserves V1.5 artifacts, and clears the accuracy gate (mIoU + edge-F uplift).

## Workstreams
- **Refinement Core**
  - [ ] Add `tree_seg/clustering/head_refine.py` with soft/EM updates seeded from the existing K-means centroids.
  - [ ] Expose temperature (τ), iteration count, and spatial blend (α) via config/profile defaults.
- **Pipeline Integration**
  - [ ] Wire the new stage through `tree_seg/core/segmentation.py` and update CLI/profile presets for `--model head-refine`.
  - [ ] Ensure the elbow initializer remains available as a fallback option.
- **Outputs & UX**
  - [ ] Confirm outputs flow through existing visualization/output managers with identical naming/layout.
  - [ ] Document new flags and defaults in README, CLI help, and roadmap docs.
- **Validation**
  - [ ] Run smoke tests on the Week 1 benchmark tiles, logging timing, VRAM usage, and qualitative segmentation notes.
  - [ ] Capture representative visuals and metrics for `docs/results/` and note open issues or regressions.

## Dependencies
- [ ] Existing DINOv3 feature extractor and elbow-based initializer.
- [ ] Configuration dataclasses updated with τ/α parameters.
- [ ] Visualization/output utilities and profiling helpers confirmed compatible with the new branch.

## Deliverables
- [ ] Executable CLI path (`--model head-refine` or profile toggle) with configuration defaults.
- [ ] Updated documentation, including quick start, methodology references, and profile tables.
- [ ] Validation notes with screenshots/metrics and a list of open issues.

## Timeline
- **Mon–Tue**
  - [ ] Implement refinement core and configuration plumbing.
- **Wed**
  - [ ] Ensure outputs/visualizations align with conventions.
  - [ ] Run initial smoke tests and resolve blockers.
- **Thu**
  - [ ] Collect metrics, curate visuals, and draft documentation updates.
- **Fri**
  - [ ] Finalize validation notes, review against benchmarking checklist, and log follow-up tasks.

## Risks & Mitigations
- [ ] **Parameter sensitivity**: Sweep τ/α values on smoke tiles; keep defaults conservative and document guidance.
- [ ] **Feature mismatch**: Maintain the elbow fallback and add assertions around tensor shapes to surface incompatibilities early.
- [ ] **Performance regressions**: Use the metrics flag to profile inference; capture before/after timing to guide optimizations.
- [ ] **Documentation gaps**: Stage updates in the docs before merging so onboarding remains accurate.
