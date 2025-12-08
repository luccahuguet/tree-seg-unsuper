# CLI Usage Guide

Command-line entrypoints (`tree-seg segment|eval|results`) for running tree segmentation workflows with sensible defaults and configurable profiles.

---

## 1. Installation Prerequisites

```bash
# Clone and install dependencies
git clone https://github.com/luccahuguet/tree-seg-unsuper.git
cd tree-seg-unsuper
uv sync
```

All commands below assume you run them from the project root.

---

## 2. Fast Start

```bash
# Process every image in data/inputs/ with the balanced profile
UV_CACHE_DIR=.uv_cache uv run tree-seg segment

# Inspect available flags
UV_CACHE_DIR=.uv_cache uv run tree-seg --help
```

Defaults:
- Input directory: `data/inputs`
- Output directory: `data/outputs`
- Model: DINOv3 Base (`dinov3_vitb16`)
- Balanced profile (`image_size=1024`, `feature_upsample_factor=2`, `refine=slic`)

---

## 3. Common Workflows

### 3.1 Single Image, Custom Output
```bash
UV_CACHE_DIR=.uv_cache uv run tree-seg segment \
  data/inputs/forest2.jpeg \
  base \
  --output-dir data/outputs/my_run
```

### 3.2 Quality vs. Speed Profiles
```bash
# Highest quality (larger resize, more refinement)
UV_CACHE_DIR=.uv_cache uv run tree-seg segment --profile quality

# Fastest runtime / lowest memory
UV_CACHE_DIR=.uv_cache uv run tree-seg segment --profile speed
```

### 3.3 Manual Overrides
```bash
UV_CACHE_DIR=.uv_cache uv run tree-seg segment \
  --image-size 1280 \
  --feature-upsample 1 \
  --pca-dim 128 \
  --refine none \
  --metrics
```

### 3.4 Quiet Mode (for scripts)
```bash
UV_CACHE_DIR=.uv_cache uv run tree-seg segment --quiet
```

---

## 4. Sweep Execution

The CLI can apply multiple configurations in sequence via `--sweep`. Create a YAML/JSON file listing overrides:

```yaml
- name: quality_large
  profile: quality
  model: large
- name: balanced_base
  profile: balanced
  model: base
- name: speed_small
  profile: speed
  model: small
  refine: none
```

Run the sweep:

```bash
UV_CACHE_DIR=.uv_cache uv run tree-seg segment \
  data/inputs \
  giant \
  --output-dir data/outputs \
  --sweep sweeps/example.yaml \
  --metrics
```

Results appear under `data/outputs/<sweep-prefix>/<name>/` (default prefix: `sweeps`).

For the curated documentation sweep:

```bash
UV_CACHE_DIR=.uv_cache uv run python scripts/generate_docs_images.py data/inputs/forest2.jpeg
```

---

## 5. Helpful Options Overview

| Flag | Description | Default |
|------|-------------|---------|
| `image_path` | Single file or directory to process | `data/inputs` |
| `model` | Model size (`small`, `base`, `large`, `giant`, `mega` or full ID) | `base` |
| `output_dir` | Destination folder | `data/outputs` |
| `--image-size` | Resize dimension before feature extraction | `1024` |
| `--feature-upsample` | Upsample factor for feature grid | `2` |
| `--pca-dim` | Apply PCA before clustering (`None` = disabled) | `None` |
| `--refine` | Refinement mode (`slic` or `none`) | `slic` |
| `--metrics` | Print timing & VRAM statistics | off |
| `--elbow-threshold` | Percentage for automatic K selection | `5.0` |
| `--sweep` | Path to sweep file (JSON/YAML) | `None` |
| `--sweep-prefix` | Subfolder under output for sweeps | `sweeps` |
| `--clean-output` | Remove existing contents before writing | off |
| `--verbose` / `--quiet` | Control console output detail | verbose |
| `--save-labels` | Save predicted labels (NPZ) for regen/metadata | on |
| `--save-metadata` | Store run in metadata bank (`results/`) | on |
| `--metadata-dir` | Where to store metadata/results index | `results` |
| `--tag` | User tags to attach to metadata entries | `[]` |

---

## 6. Option Precedence & Compatibility

- **Profiles first, explicit flags second**: Supplying `--profile` loads defaults unless you also pass the corresponding flag (`--image-size`, `--feature-upsample`, etc.), in which case the explicit flag wins.
- **Sweeps inherit CLI defaults**: When `--sweep` is provided, the top-level CLI arguments (model, image size, output dir, etc.) become the baseline for every sweep item. Each item can override specific fields (including selecting its own `profile`). Reserved keys: `name`, `model`, and `profile`.
- **Per-sweep outputs**: `--output_dir` defines the base directory; each sweep entry writes to `data/outputs/<prefix>/<name>/`. Outdirectories are cleared automatically; the global `--clean-output` flag is only honored for non-sweep runs.
- **Verbose vs. quiet**: Passing `--quiet` disables verbose logging even if `--verbose` is set earlier.
- **Metrics during sweeps**: `--metrics` applies to both single runs and all sweep items.

## 7. Benchmark CLI (`tree-seg eval`)

Evaluate on labeled datasets with rich progress, metadata, and label dumps:

```bash
# Baseline on FORTRESS with viz + labels
UV_CACHE_DIR=.uv_cache uv run tree-seg eval data/datasets/fortress_processed \
  --model base \
  --clustering kmeans \
  --refine slic \
  --num-samples 5 \
  --save-viz \
  --save-labels
```

Parameter sweep with metadata per config:

```bash
UV_CACHE_DIR=.uv_cache uv run tree-seg sweep data/datasets/fortress_processed \
  --preset tiling \
  --num-samples 1 \
  --save-viz --save-labels
```

Notes:
- Outputs land in `data/outputs/results/<dataset>_<method>_<timestamp>/` unless `--output-dir` is provided.
- `--save-labels/--no-save-labels` controls NPZ dumps under `labels/`; metadata is stored best-effort in `results/`.
- Progress bar shows ETA using cached runtimes (scaled by hardware tier).

## 8. Results CLI (`tree-seg results`)

Query the metadata bank, regenerate visuals, and manage index size:

```bash
# List top runs for fortress kmeans+slic
uv run tree-seg results --dataset fortress --tags kmeans,slic --sort mIoU --top 5

# Show details for a hash
uv run tree-seg results --hash abc123 --show-config

# Regenerate visualizations from labels
uv run tree-seg results --hash abc123 --render

# Nearest-config ETA lookup
uv run tree-seg results --nearest '{"clustering":"kmeans","model":"base","stride":4,"tiling":false}'

# Maintenance
uv run tree-seg results --compact
uv run tree-seg results --prune-older-than 30

# Regenerate all missing visualizations
uv run tree-seg results --sync-all-viz

# Regenerate visualizations for all runs (overwrite/regenerate)
uv run tree-seg results --render-all
```

Notes:
- `--render` works per-hash; `--sync-all-viz` scans the index and regenerates only missing PNGs from stored labels, skipping runs that already have viz. It does not rerun inference.
- `--render-all` regenerates visualizations for every indexed run using stored labels (overwrites existing regen outputs).

## 9. Environment Helpers

- **Force GPU**: set `FORCE_GPU=1` to prefer CUDA/MPS if available.
- **CPU fallback**: default behavior automatically runs on CPU when GPUs are missing or exhausted.
- **Cache directories**: set `UV_CACHE_DIR` (for dep resolution) and `PRE_COMMIT_HOME` (for git hooks) when working in limited environments.

---

## 10. Troubleshooting Tips

- Ensure input files have supported extensions (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`).
- Use `--clean-output` if you need fresh directories between runs.
- If sweeps appear empty, confirm your sweep file has at least one item and is valid YAML/JSON.
- When running inside automated environments, pass `--quiet` to reduce logging noise and keep outputs deterministic.

With these CLI commands you can iterate quickly, automate parameter searches, and integrate tree segmentation into larger pipelines.
