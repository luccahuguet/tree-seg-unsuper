# CLI Usage Guide

Command-line entrypoints for running tree segmentation workflows with sensible defaults and configurable profiles.

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
# Process every image in data/input/ with the balanced profile
UV_CACHE_DIR=.uv_cache uv run python main.py

# Inspect available flags
UV_CACHE_DIR=.uv_cache uv run python main.py --help
```

Defaults:
- Input directory: `data/input`
- Output directory: `data/output`
- Model: DINOv3 Base (`dinov3_vitb16`)
- Balanced profile (`image_size=1024`, `feature_upsample_factor=2`, `refine=slic`)

---

## 3. Common Workflows

### 3.1 Single Image, Custom Output
```bash
UV_CACHE_DIR=.uv_cache uv run python main.py \
  data/input/forest2.jpeg \
  base \
  data/output/my_run
```

### 3.2 Quality vs. Speed Profiles
```bash
# Highest quality (larger resize, more refinement)
UV_CACHE_DIR=.uv_cache uv run python main.py --profile quality

# Fastest runtime / lowest memory
UV_CACHE_DIR=.uv_cache uv run python main.py --profile speed
```

### 3.3 Manual Overrides
```bash
UV_CACHE_DIR=.uv_cache uv run python main.py \
  --image-size 1280 \
  --feature-upsample 1 \
  --pca-dim 128 \
  --refine none \
  --metrics
```

### 3.4 Quiet Mode (for scripts)
```bash
UV_CACHE_DIR=.uv_cache uv run python main.py --quiet
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
UV_CACHE_DIR=.uv_cache uv run python main.py \
  data/input \
  giant \
  data/output \
  --sweep sweeps/example.yaml \
  --metrics
```

Results appear under `data/output/<sweep-prefix>/<name>/` (default prefix: `sweeps`).

For the curated documentation sweep:

```bash
UV_CACHE_DIR=.uv_cache uv run python scripts/generate_docs_images.py data/input/forest2.jpeg
```

---

## 5. Helpful Options Overview

| Flag | Description | Default |
|------|-------------|---------|
| `image_path` | Single file or directory to process | `data/input` |
| `model` | Model size (`small`, `base`, `large`, `giant`, `mega` or full ID) | `base` |
| `output_dir` | Destination folder | `data/output` |
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

---

## 6. Option Precedence & Compatibility

- **Profiles first, explicit flags second**: Supplying `--profile` loads defaults unless you also pass the corresponding flag (`--image-size`, `--feature-upsample`, etc.), in which case the explicit flag wins.
- **Sweeps inherit CLI defaults**: When `--sweep` is provided, the top-level CLI arguments (model, image size, output dir, etc.) become the baseline for every sweep item. Each item can override specific fields (including selecting its own `profile`). Reserved keys: `name`, `model`, and `profile`.
- **Per-sweep outputs**: `--output_dir` defines the base directory; each sweep entry writes to `data/output/<prefix>/<name>/`. Outdirectories are cleared automatically; the global `--clean-output` flag is only honored for non-sweep runs.
- **Verbose vs. quiet**: Passing `--quiet` disables verbose logging even if `--verbose` is set earlier.
- **Metrics during sweeps**: `--metrics` applies to both single runs and all sweep items.

## 7. Benchmark CLI (`scripts/bench.py`)

The benchmark runner shares the same configuration backbone as `main.py`, but evaluates segmentation quality against labeled datasets. Treat it as a companion CLI:

```bash
# Recommended baseline on ISPRS Potsdam (10 samples, saves visuals)
UV_CACHE_DIR=.uv_cache uv run python scripts/bench.py \
  --dataset data/isprs_potsdam \
  --method v1.5 \
  --model base \
  --clustering slic \
  --elbow-threshold 20.0 \
  --num-samples 10 \
  --save-viz \
  --output-dir data/output/results/base_e20_slic_run
```

Need a quick comparison sweep?

```bash
UV_CACHE_DIR=.uv_cache uv run python scripts/bench.py \
  --dataset data/isprs_potsdam \
  --method v1.5 \
  --compare-configs \
  --num-samples 5
```

Key notes:
- Defaults mirror the main CLI (balanced behaviour, stride 4, elbow threshold 5.0). Adjust them explicitly via flags (`--stride`, `--elbow-threshold`, `--clustering`, etc.).
- Outputs go to `data/output/results/<method>_<model>_<timestamp>/` unless you supply `--output-dir` (as shown above).
- Comparison mode (`--compare-configs`) iterates over a pre-defined grid of settings; top-level flags seed the base config, while each comparison case applies its overrides.
- For quick CSV-style profiling (runtime/VRAM per image) see `scripts/utils/benchmark_profiles.py`, which simply wraps `segment_trees` with metrics enabled.

Because the benchmark CLI is a separate script, you can invoke it directly, or add an alias (e.g., `uvx tree-seg-bench`) if you want it to feel like a subcommand.

## 8. Environment Helpers

- **Force GPU**: set `FORCE_GPU=1` to prefer CUDA/MPS if available.
- **CPU fallback**: default behavior automatically runs on CPU when GPUs are missing or exhausted.
- **Cache directories**: set `UV_CACHE_DIR` (for dep resolution) and `PRE_COMMIT_HOME` (for git hooks) when working in limited environments.

---

## 9. Troubleshooting Tips

- Ensure input files have supported extensions (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`).
- Use `--clean-output` if you need fresh directories between runs.
- If sweeps appear empty, confirm your sweep file has at least one item and is valid YAML/JSON.
- When running inside automated environments, pass `--quiet` to reduce logging noise and keep outputs deterministic.

With these CLI commands you can iterate quickly, automate parameter searches, and integrate tree segmentation into larger pipelines.
