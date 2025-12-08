# Benchmarking & Dataset Setup Guide

End-to-end reference for acquiring the ISPRS Potsdam dataset, authenticating with Kaggle, and running the segmentation benchmark suite.

---

## 1. Kaggle Access & Dataset Download

1. **Create an API token**
   - Visit https://www.kaggle.com/settings → *API* → **Create New Token**
   - Move the downloaded `kaggle.json` into place and lock permissions:
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

2. **Download the ISPRS Potsdam dataset**
   ```bash
   uv run python scripts/download_dataset_isprs.py
   ```
   The helper script will download ~20 GB, extract, organize under `data/datasets/isprs_potsdam/{images,labels}/`, clean temp files, and verify integrity.

3. **Alternate dataset options**
   ```bash
   # Smaller subset
   uv run python scripts/download_dataset_isprs.py \
     --dataset-id trito12/potsdam-vaihingen-isprs

   # Urban segmentation bundle
   uv run python scripts/download_dataset_isprs.py \
     --dataset-id aletbm/urban-segmentation-isprs
   ```

4. **Useful script flags**
   ```bash
   uv run python scripts/download_dataset_isprs.py --keep-zips        # retain archives
   uv run python scripts/download_dataset_isprs.py --skip-download    # re-organize only
   uv run python scripts/download_dataset_isprs.py --output-dir data/my_custom_dir
   uv run python scripts/download_dataset_isprs.py --help
   ```

Troubleshooting reminders:
- `401 Unauthorized` → ensure `~/.kaggle/kaggle.json` exists and has `600` permissions.
- `kaggle: command not found` → always invoke through the provided script.
- Slow downloads → try an alternate dataset ID or ensure a stable connection.

---

## 2. Running Benchmarks

The benchmark harness evaluates segmentation quality (mIoU, pixel accuracy) and runtime across configurations.

### Quick start
```bash
# Baseline V1.5 run on 5 samples (GPU if available)
uv run python scripts/evaluate_semantic_segmentation.py \
  --dataset data/datasets/isprs_potsdam \
  --method v1.5 \
  --num-samples 5 \
  --save-viz

# Force CPU if necessary
FORCE_CPU=1 uv run python scripts/evaluate_semantic_segmentation.py \
  --dataset data/datasets/isprs_potsdam \
  --method v1.5 \
  --num-samples 5 \
  --save-viz
```

**Recommended production baseline (base model, elbow 20.0, SLIC refinement, 10 samples):**

```bash
uv run python scripts/evaluate_semantic_segmentation.py \
  --dataset data/datasets/isprs_potsdam \
  --method v1.5 \
  --model base \
  --clustering slic \
  --elbow-threshold 20.0 \
  --num-samples 10 \
  --save-viz \
  --output-dir data/outputs/results/base_e20_slic_run
```

Outputs land in `data/outputs/results/<method>_<model>_<timestamp>/`:
- `results.json` – metrics per image and aggregate statistics
- `visualizations/` – optional overlays when `--save-viz` is used

### Core flags

| Flag | Purpose | Default |
|------|---------|---------|
| `--dataset PATH` | Root containing `images/` and `labels/` | Required |
| `--method` | Segmentation method (`v1`, `v1.5`, `v2`, `v3`) | `v1.5` |
| `--model` | Backbone size (`small`, `base`, `large`, `mega`) | `base` |
| `--clustering` | Post-clustering (`kmeans`, `slic`) | `kmeans` |
| `--stride` | Feature stride | `4` |
| `--elbow-threshold` | Auto-K elbow percentage | `5.0` |
| `--fixed-k` | Disable auto-K and force cluster count | None |
| `--num-samples` | Limit evaluation set | all images |
| `--save-viz` | Persist PNG comparisons | disabled |
| `--output-dir` | Override results directory | timestamped folder |

### Common scenarios

```bash
# Compare elbow thresholds
uv run python scripts/evaluate_semantic_segmentation.py --dataset data/datasets/isprs_potsdam --elbow-threshold 2.5 --num-samples 5
uv run python scripts/evaluate_semantic_segmentation.py --dataset data/datasets/isprs_potsdam --elbow-threshold 20.0 --num-samples 5

# Model comparison
uv run python scripts/evaluate_semantic_segmentation.py --dataset data/datasets/isprs_potsdam --model small --num-samples 5
uv run python scripts/evaluate_semantic_segmentation.py --dataset data/datasets/isprs_potsdam --model mega --num-samples 5

# K-means vs. SLIC refinement
uv run python scripts/evaluate_semantic_segmentation.py --dataset data/datasets/isprs_potsdam --clustering slic --num-samples 5

# Full run with explicit destination
uv run python scripts/evaluate_semantic_segmentation.py \
  --dataset data/datasets/isprs_potsdam \
  --method v1.5 \
  --model base \
  --clustering slic \
  --elbow-threshold 20.0 \
  --num-samples 10 \
  --save-viz \
  --output-dir data/outputs/results/base_e20_slic_run
```

### Comparison grid mode

```bash
uv run python scripts/evaluate_semantic_segmentation.py \
  --dataset data/datasets/isprs_potsdam \
  --compare-configs \
  --num-samples 5
```

Automatically sweeps several elbow thresholds, model sizes, and clustering strategies, saving a summary to `data/outputs/results/comparison_summary.json`.

### Metrics primer
- **mIoU** (mean Intersection over Union): primary quality metric, typically 0.20–0.35 for unsupervised V1.5 on Potsdam.
- **Pixel Accuracy**: secondary metric, usually 55–70 % for V1.5.
- **Runtime**: seconds per image; strongly influenced by model size and stride.

Predicted clusters are matched to ground-truth labels using the Hungarian algorithm before computing metrics.

---

## 3. Evaluation Roadmap (condensed)

1. **Dataset readiness**
   - Verify `data/datasets/isprs_potsdam/{images,labels}` structure.
   - Decide on tiling/downsampling strategy for 6000×6000 TIFFs if needed.

2. **Benchmark infrastructure**
   - Core metric implementations (`tree_seg/evaluation/metrics.py`)
   - Batch runner (`tree_seg/evaluation/benchmark.py`) with Hungarian matching
   - CLI entry point (`scripts/evaluate_semantic_segmentation.py`)

3. **Baseline experiments**
   - V1.5 with multiple elbow thresholds and models
   - Persist `results.json` tables and visual comparisons

4. **Future comparisons**
   - Re-run the same suite with V2 (head refinement), V3 (tree focus), and optional V4 (SAM polisher)
   - Track mIoU, pixel accuracy, runtime, memory, edge-F, and cluster stability

5. **Suggested timeline**
   - Days 1–2: dataset download + verification
   - Days 3–4: metric + benchmark implementation
   - Day 5: execute V1.5 baselines
   - Days 6–7: analyze results, prepare write-up, and plan V2/V3 evaluations under the gate-driven roadmap

With the dataset in place and the benchmark script configured, you can iterate quickly on new segmentation approaches and track improvements quantitatively.
