# Benchmark System Usage Guide

## Overview

The benchmark system allows you to quantitatively evaluate segmentation methods (V1, V1.5, V2, V3) using standard metrics (mIoU, pixel accuracy) on aerial imagery datasets like ISPRS Potsdam.

## Quick Start

### 1. Download ISPRS Potsdam Dataset

Follow the instructions in `data/isprs_potsdam/README.md` to download and organize the dataset.

### 2. Run a Simple Benchmark

```bash
# Run V1.5 baseline on 5 sample images
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --method v1.5 \
  --num-samples 5 \
  --save-viz
```

### 3. View Results

Results are saved to `results/<method>_<model>_<timestamp>/`:
- `results.json` - Detailed metrics for each image
- `visualizations/` - Side-by-side comparisons (if --save-viz was used)

## Command Reference

### Basic Usage

```bash
uv run python scripts/run_benchmark.py \
  --dataset <path> \
  --method <v1|v1.5|v2|v3> \
  [OPTIONS]
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Path to dataset directory | Required |
| `--method` | Segmentation method version | `v1.5` |
| `--model` | DINOv3 model (small/base/large/mega) | `base` |
| `--clustering` | Clustering method (kmeans/slic) | `kmeans` |
| `--stride` | Feature extraction stride | `4` |
| `--elbow-threshold` | Auto K-selection threshold | `5.0` |
| `--fixed-k` | Fixed number of clusters (disables auto K) | None |
| `--num-samples` | Number of images to evaluate | All |
| `--save-viz` | Save visualization images | False |
| `--quiet` | Suppress progress output | False |

### Examples

**Test different elbow thresholds:**
```bash
# Conservative (fewer clusters)
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --elbow-threshold 20.0 \
  --num-samples 5

# Sensitive (more clusters)
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --elbow-threshold 2.5 \
  --num-samples 5
```

**Compare different models:**
```bash
# Small model (fastest)
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --model small \
  --num-samples 5

# Mega model (highest quality, ViT-7B/16)
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --model mega \
  --num-samples 5
```

**Test SLIC vs K-means:**
```bash
# K-means (default)
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --clustering kmeans \
  --num-samples 5

# SLIC (spatial clustering)
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --clustering slic \
  --num-samples 5
```

**Run full evaluation with visualizations:**
```bash
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --method v1.5 \
  --model large \
  --save-viz \
  --output-dir results/v1.5_full_evaluation
```

### Comparison Mode

Automatically run multiple configurations and generate comparison table:

```bash
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --compare-configs \
  --num-samples 5
```

This tests:
- Different elbow thresholds (2.5, 5.0, 10.0, 20.0)
- Different models (small, base, large)
- Different clustering methods (kmeans, slic)

Results saved to `results/comparison_summary.json`

## Understanding Results

### Metrics

**mIoU (mean Intersection over Union)**
- Primary metric for segmentation quality
- Range: 0.0 to 1.0 (higher is better)
- Measures overlap between predicted and ground truth segments
- Expected values:
  - V1.5 unsupervised: 0.20 - 0.35
  - V4 supervised (future): 0.35 - 0.50

**Pixel Accuracy**
- Secondary metric
- Range: 0.0 to 1.0 (higher is better)
- Fraction of correctly classified pixels
- Expected values:
  - V1.5 unsupervised: 0.55 - 0.70
  - V4 supervised (future): 0.70 - 0.85

**Runtime**
- Seconds per image
- Depends on image size, model size, and stride
- Typical values:
  - Small model, stride 8: ~5-10s
  - Base model, stride 4: ~15-30s
  - Mega model, stride 4: ~60-120s

### Hungarian Matching

The benchmark uses the **Hungarian algorithm** to match predicted clusters to ground truth classes:

1. Your method predicts K clusters (e.g., 8 clusters)
2. Ground truth has C classes (ISPRS: 6 classes)
3. Hungarian algorithm finds optimal cluster→class assignment
4. Metrics are computed after applying this mapping

This is necessary because unsupervised methods don't know class labels a priori.

## Results File Format

`results.json` structure:
```json
{
  "dataset": "isprs_potsdam",
  "method": "v1.5_kmeans",
  "config": {
    "version": 1.5,
    "model": "DINOv3 ViT-B/14",
    "stride": 4,
    "elbow_threshold": 5.0,
    ...
  },
  "metrics": {
    "mean_miou": 0.287,
    "mean_pixel_accuracy": 0.623,
    "mean_runtime": 18.4
  },
  "samples": [
    {
      "image_id": "top_potsdam_2_10",
      "miou": 0.312,
      "pixel_accuracy": 0.658,
      "num_clusters": 7,
      "runtime_seconds": 17.2
    },
    ...
  ]
}
```

## Week 1 Evaluation Plan

### Day 1-2: Dataset Setup
```bash
# Download ISPRS Potsdam dataset
# See data/isprs_potsdam/README.md for instructions

# Verify dataset is accessible
uv run python scripts/test_benchmark.py
```

### Day 3: Quick Validation (5 samples)
```bash
# Test V1.5 baseline on subset
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --method v1.5 \
  --num-samples 5 \
  --save-viz
```

### Day 4: Full V1.5 Evaluation
```bash
# Run on all samples with default config
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --method v1.5 \
  --save-viz \
  --output-dir results/v1.5_baseline
```

### Day 5: Configuration Sweep
```bash
# Compare different configurations
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --compare-configs \
  --output-dir results/v1.5_sweep
```

### Day 6-7: Analysis
- Review results in `results/` directory
- Analyze which configs work best
- Document findings for paper methodology section

## Comparing V1 vs V1.5 vs V2 vs V3

Once you implement V2 and V3, you can compare all methods:

```bash
# V1 (patch features only)
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --method v1 \
  --output-dir results/comparison_v1

# V1.5 (patch + attention features)
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --method v1.5 \
  --output-dir results/comparison_v1.5

# V2 (U2Seg) - when implemented
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --method v2 \
  --output-dir results/comparison_v2

# V3 (DynaSeg) - when implemented
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --method v3 \
  --output-dir results/comparison_v3
```

## Troubleshooting

### "Dataset not found"
- Check that dataset is downloaded to `data/isprs_potsdam/`
- Verify `images/` and `labels/` subdirectories exist
- See `data/isprs_potsdam/README.md` for download instructions

### Out of memory errors
- Use smaller model: `--model small`
- Increase stride: `--stride 8`
- Reduce number of samples: `--num-samples 5`

### Slow performance
- Use smaller model: `--model small`
- Increase stride: `--stride 8`
- Process fewer samples: `--num-samples 10`

### Low mIoU scores
- This is expected for unsupervised methods!
- V1.5 typically achieves 0.20-0.35 mIoU
- Try adjusting `--elbow-threshold` (lower = more clusters)
- Different clustering methods: `--clustering slic`

## Python API Usage

You can also use the benchmark system programmatically:

```python
from pathlib import Path
from tree_seg.core.types import Config
from tree_seg.evaluation import run_benchmark

# Create config
config = Config(
    version=1.5,
    model_name="base",
    stride=4,
    elbow_threshold=5.0,
    auto_k=True,
)

# Run benchmark
results = run_benchmark(
    config=config,
    dataset_path=Path("data/isprs_potsdam"),
    output_dir=Path("results/my_test"),
    num_samples=5,
    save_visualizations=True,
    verbose=True,
)

# Access results
print(f"Mean mIoU: {results.mean_miou:.3f}")
print(f"Mean Pixel Accuracy: {results.mean_pixel_accuracy:.3f}")

for sample in results.samples:
    print(f"{sample.image_id}: {sample.miou:.3f}")
```

## Next Steps

1. Download ISPRS Potsdam dataset
2. Run initial validation with 5 samples
3. Analyze results and adjust configurations
4. Run full evaluation for Week 1 deliverable
5. Document findings in paper methodology section
6. Prepare infrastructure for V2/V3 comparison
