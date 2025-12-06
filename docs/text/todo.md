# TODO

- [ ] Plan a larger sweep on FORTRESS (3–10 imgs) to identify the current peak performer across clustering/refine/tiling/stride variants.
- [ ] Docs: update stale site content to reflect current CLI, metadata/results features, and removed version split.
- [ ] Supervised results: log fortress_processed metrics for sklearn (mIoU~0.082), torch MLP (mIoU~0.062, XY variant worse), sklearn MLP (peak mIoU~0.427/PA~0.953 no early stop; ~0.278/0.833 with patience=5,val_split=0.02; ~0.111/0.608 with patience=3,val_split=0.01) into metadata/results.
