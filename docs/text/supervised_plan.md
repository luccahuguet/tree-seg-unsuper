# Supervised Baselines (Status)

## Implemented Heads
- **sklearn LogisticRegression** (`--head sklearn`): mIoU ≈ 0.082, PA ≈ 0.536 on `fortress_processed` (stride=4, 47 tiles, 100k sample cap, max_iter=3000).
- **Torch Linear MLP** (`--head linear`): 2-layer MLP (1024 hidden, dropout 0.1), max_patches=1M, early stopping. Best observed so far: mIoU ≈ 0.074, PA ≈ 0.236 (stride=2, 2M patches, lr=5e-4, patience 20, val_split 0.1).
- **sklearn MLPClassifier** (`--head mlp`): nonlinear head with early stopping (current defaults hidden (2048,1024,512), max_iter=400). Best observed: mIoU ≈ 0.427, PA ≈ 0.953 on `fortress_processed` with stride=2, 2M patches, lr=5e-4, patience=0, val_split=0.0, `--mlp-use-xy` (training ~14m on GPU). Early-stopping “non-overfitting peak”: patience=5/val_split=0.02 → mIoU ≈ 0.278/PA ≈ 0.833. Stricter patience/val splits dropped to ~0.111 mIoU / 0.608 PA.

## CLI
- Supervised mode: `tree-seg eval ... --supervised --supervised-head {sklearn,linear,mlp}`
- Shortcut: `tree-seg eval-super DATASET --head {sklearn,linear,mlp}` with concise flags:
  - `--stride`, `--epochs` (max_iter for MLP heads), `--max-patches`, `--lr`, `--hidden-dim`, `--dropout`, `--patience`, `--val-split`, `--ignore-index`, `--num-samples`

Example commands:
```bash
# Quick linear head (defaults, stride=4)
tree-seg eval-super data/datasets/fortress_processed --head linear

# Stronger linear head
tree-seg eval-super data/datasets/fortress_processed --head linear \
  --stride 2 --max-patches 2000000 --epochs 200 --patience 20 --val-split 0.1 --lr 5e-4

# Sklearn MLP head (nonlinear)
tree-seg eval-super data/datasets/fortress_processed --head mlp --stride 2 --max-patches 2000000 --epochs 200
```

## Notes
- Ignore handling: 255 is auto-treated as ignore if present; override with `--ignore-index`.
- Patch budget: linear defaults to 1M; increase cautiously for memory (2M used in best linear run).
- Early stopping: defaults patience=5, val_split=0.1 for linear; sklearn MLP uses its own early_stopping.

## Next Steps
- Run full-dataset experiment for sklearn MLP head and log metrics.
- Compare heads with stride=2 vs stride=4 to find best default.
- Optional: add checkpointing/caching for supervised runs to avoid re-extracting features.
- Add supervised augmentation (Albumentations) on train split only (flips/rot90, light color jitter/blur).***
