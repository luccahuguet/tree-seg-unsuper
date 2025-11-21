# Roadmap Reassessment Notes

## Current Architecture Snapshot
- **V1.5 – DINOv3 + K-Means (Active)**  
  Uses dense DINOv3 features, elbow-based K selection, and SLIC refinement for optional edge alignment. Outputs (masks, overlays, PCA plots) are stable and documented.
- **V2 – Head Refinement (Planned)**  
  Lightweight soft/EM refinement over K-means initialization with a spatial blend, chosen after reassessing U2Seg accuracy claims.
- **V3 – Tree Focus (Planned)**  
  Vegetation gating, shape filters, and instance generation tuned for forestry imagery.

## Why U2Seg & DynaSeg Were Deferred
- Accuracy uncertainty: available benchmarks do not confirm that either method improves segmentation quality on forestry/aerial imagery relative to our V1.5 baseline.
- Integration cost: significant adaptation would be required before we can even test whether accuracy improves.

## Exploration Goals
- [ ] Compile a shortlist of alternative unsupervised or self-supervised segmentation papers with public code and permissive licenses.
- [ ] Prioritize methods with reported accuracy gains on remote-sensing or forestry datasets.
- [ ] Evaluate quantitative metrics and qualitative examples to determine whether legacy candidates (U2Seg/DynaSeg) or new alternatives are likely to outperform the V1.5 baseline.

## Potential Roadmap Adjustments
- Revisited roadmap now captured in `docs/text/version_roadmap.md`, emphasizing gate-driven progression and accuracy-driven evaluation.
- Continue validating alternatives, but prioritize those that can satisfy the new V2+ gates without sacrificing accuracy.

## Next Steps
- [ ] Schedule a focused literature/code review sprint to populate the candidate list.
- [ ] Draft a comparison matrix (method, repo link, reported accuracy, forestry-specific evidence).
- [ ] Propose an updated version roadmap once at least two viable alternatives demonstrate a realistic shot at improving accuracy over V1.5.
