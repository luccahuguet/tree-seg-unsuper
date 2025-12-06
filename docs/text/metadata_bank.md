# Experiment Tracking & Metadata Bank

## Goal

Persistent, queryable cache of experiment metadata (configs → runtime, metrics, assets) to:
1. Speed UX (loading bars with accurate ETAs on repeat runs)
2. Avoid re-running identical experiments
3. Enable comparison and discovery of past results
4. Support future meta-learning

## Storage Structure

```
results/
├── index.jsonl              # Append-only experiment log for fast queries
└── by-hash/                 # Content-addressable storage
    ├── a3f7c2d1/
    │   ├── meta.json        # Config + timing + metrics + tags
    │   └── labels.npz       # Compressed segmentation labels
    └── b8e4f1a9/
        └── ...
```

## Config Hashing

Deterministic hash from config (excluding git SHA to maximize cache hits):

```python
import hashlib
import json

# Keys that MUST be included in hash (affect runtime/outputs)
# Mirror all params from Config that affect runtime or outputs
HASH_KEYS = [
    # Dataset & model
    "dataset",
    "model",
    # Clustering
    "clustering",
    "k",
    "smart_k",
    "elbow_threshold",
    # Refinement
    "refine",
    "soft_refine",           # V2 soft-em toggle
    "soft_refine_temperature",
    "soft_refine_iterations",
    "soft_refine_spatial_alpha",
    # SLIC refinement params
    "slic_n_segments",
    "slic_compactness",
    "slic_sigma",
    # Task modifiers
    "vegetation_filter",
    "supervised",            # V4 Mask2Former flag
    # Feature extraction
    "stride",
    "tiling",
    "tile_size",             # Tile dimensions
    "tile_overlap",          # Overlap between tiles
    "image_size",            # Tuple (H, W) or int
    # Pyramid/multi-scale
    "pyramid",
    "pyramid_scales",
    "pyramid_aggregation",
    "multi_layer",           # Multi-layer feature extraction
    "layer_weights",
    # Grid sweep context
    "grid_label",
]

# Defaults for normalization (missing keys get these values)
HASH_DEFAULTS = {
    # Tiling
    "tiling": False,
    "tile_size": 512,
    "tile_overlap": 0,
    # Pyramid
    "pyramid": False,
    "pyramid_scales": [1],
    "pyramid_aggregation": "mean",
    "multi_layer": False,
    "layer_weights": None,
    # Soft refinement
    "soft_refine": False,
    "soft_refine_spatial_alpha": 0.0,
    # SLIC (None = use library defaults)
    "slic_n_segments": None,
    "slic_compactness": 10.0,
    "slic_sigma": 1.0,
    # Flags
    "vegetation_filter": False,
    "supervised": False,
    "smart_k": False,
}

def normalize_config(config: dict) -> dict:
    """Normalize config for consistent hashing. Single source of truth."""
    normalized = {}
    for key in HASH_KEYS:
        value = config.get(key, HASH_DEFAULTS.get(key))
        if value is None:
            continue
        # Freeze mutable types
        if isinstance(value, list):
            value = tuple(value)
        normalized[key] = value
    return normalized

def config_hash(config: dict) -> str:
    """Generate unique hash for a configuration."""
    normalized = normalize_config(config)
    # Canonical JSON: sorted keys, no whitespace
    # Convert tuples back to lists for JSON serialization
    serializable = {k: list(v) if isinstance(v, tuple) else v
                    for k, v in normalized.items()}
    canonical = json.dumps(serializable, sort_keys=True, separators=(',', ':'))
    # 10 hex chars = 40 bits, safe for ~1M runs before collision risk
    return hashlib.sha256(canonical.encode()).hexdigest()[:10]
```

Git SHA stored in `meta.json` for reproducibility, but not part of hash.

## Tag System

**Source of truth**: Auto-tags derived from normalized config dict. User tags stored separately.

```python
def derive_tags(config: dict) -> list[str]:
    """Generate auto-tags from config. Source of truth for CLI/API."""
    tags = []
    # Dataset
    if "dataset" in config:
        tags.append(config["dataset"])
    # Clustering
    if "clustering" in config:
        tags.append(config["clustering"])
    # K value
    if "k" in config:
        tags.append(f"k{config['k']}")
    # Refinement
    if "refine" in config:
        tags.append(config["refine"])
    # Model
    if "model" in config:
        tags.append(config["model"])
    # Stride
    if "stride" in config:
        tags.append(f"stride-{config['stride']}")
    # Flags
    if config.get("vegetation_filter"):
        tags.append("veg-filter")
    if config.get("smart_k"):
        tags.append("smart-k")
    return tags

# User-supplied tags passed through untouched
user_tags = ["v1.5-baseline", "paper-figure-3"]
all_tags = derive_tags(config) + user_tags
```

Query by tags:
```bash
tree-seg results --tags kmeans,fortress
tree-seg results --tags paper-figure-3
```

## meta.json Schema

```json
{
    "hash": "a3f7c2d1ab",
    "created_at": "2024-12-06T14:30:00Z",
    "git_sha": "abc123def",
    "config": {
        "dataset": "fortress",
        "clustering": "kmeans",
        "k": 5,
        "refine": "slic",
        "vegetation_filter": false,
        "model": "base",
        "stride": 4,
        "tiling": false,
        "image_size": [1024, 1024],
        "smart_k": false
    },
    "tags": {
        "auto": ["fortress", "kmeans", "k5", "slic", "stride-4", "base"],
        "user": ["v1.5-baseline"]
    },
    "samples": {
        "num_samples": 12,
        "per_sample_stats": true
    },
    "hardware": {
        "gpu": "NVIDIA RTX 4090",
        "gpu_tier": "extreme",  # low|mid|high|extreme for ETA scaling
        "cpu": "AMD Ryzen 9 5900X",
        "cpu_threads": 24,
        "ram_gb": 64
    },
    "timing": {
        "feature_extraction_s": 12.5,
        "clustering_s": 0.3,
        "refinement_s": 1.2,
        "total_s": 14.0,
        "per_sample_mean_s": 1.17
    },
    "metrics": {
        "mIoU": 0.415,
        "pixel_accuracy": 0.623,
        "per_class_iou": {"tree": 0.52, "grass": 0.41, "building": 0.31}
    },
    "artifacts": {
        "labels": "labels.npz",           # Single-image datasets
        "labels_dir": "labels/",          # Multi-image: labels/{sample_id}.npz
        "sample_ids": ["img001", "img002", "img003"]  # For multi-image lookup
    },
    "notes": null
}
```

## index.jsonl Format

Append-only log with summary for fast queries:

```jsonl
{"hash":"a3f7c2d1","dataset":"fortress","tags":["kmeans","k5","slic"],"mIoU":0.415,"total_s":14.0,"created_at":"2024-12-06T14:30:00Z"}
{"hash":"b8e4f1a9","dataset":"fortress","tags":["gmm","k5","slic"],"mIoU":0.398,"total_s":15.2,"created_at":"2024-12-06T14:35:00Z"}
```

## Runtime Fallback

When exact hash not found, fall back to nearest config match:

```python
def lookup_nearest(config: dict) -> Optional[dict]:
    """Find closest matching config for ETA estimation."""
    # Priority weights (higher = more important for runtime)
    WEIGHTS = {
        "model": 10,       # Model size dominates runtime
        "image_size": 8,   # Image dimensions heavily affect time
        "tiling": 7,       # Tiling vs full image processing
        "stride": 6,       # Feature extraction density
        "num_samples": 5,  # Dataset size
        "clustering": 3,   # Algorithm choice
        "refine": 2,       # Refinement method
        "k": 1,            # Number of clusters (minor)
    }
    # Score candidates by weighted match, pick highest
    ...
```

This enables reasonable ETAs even for new config combinations.

### GPU Tier Classification

For ETA scaling across hardware:

```python
GPU_TIERS = {
    "extreme": ["A100", "H100", "RTX 4090", "RTX 3090"],
    "high": ["RTX 4080", "RTX 3080", "A6000", "V100"],
    "mid": ["RTX 4070", "RTX 3070", "RTX 2080", "T4"],
    "low": ["RTX 3060", "GTX 1080", "P100", "CPU"],
}

# Rough scaling factors (relative to "high" tier)
TIER_SCALE = {"extreme": 0.6, "high": 1.0, "mid": 1.5, "low": 3.0}
```

ETA lookup scales stored timing by tier ratio: `eta = stored_time * (my_tier_scale / stored_tier_scale)`

## Artifacts Policy

| Artifact | Store? | Rationale |
|----------|--------|-----------|
| `meta.json` | Always | Essential for queries |
| `labels.npz` | Default | ~100KB, enables viz regeneration |
| Images (PNG) | Never | Regenerate on demand |
| Features | Never | Too large, re-extract if needed |

**Labels toggle**: For large datasets (satellite imagery with 100+ tiles), labels can grow large. Use `--no-labels` to skip:

```bash
tree-seg eval data/huge-dataset --no-labels  # Only save meta.json
```

Note: Without labels, visualization regeneration is not possible. ETAs still work via meta.json timing.

Regenerate visualizations (requires labels):
```bash
tree-seg results --hash a3f7c2d1ab --render
```

## Retention & Compaction

- **Deduplication**: Same hash → same directory (automatic)
- **Compaction**: Periodic `tree-seg results --compact` to remove orphaned entries
- **Cleanup**: `tree-seg results --prune --older-than 30d` for old experiments
- **Size limit**: Optional `--max-size 1GB` to auto-prune oldest

## CLI Interface

### Automatic Save (after eval)

```bash
tree-seg eval data/fortress --clustering kmeans --k 5 --refine slic
# → Saves to results/by-hash/{hash}/
# → Appends to results/index.jsonl
# → Output: Saved to a3f7c2d1 (mIoU: 0.415, 14.0s)
```

### Query Results

```bash
# By config fields
tree-seg results --clustering kmeans --dataset fortress

# By tags
tree-seg results --tags paper-figure-3

# Best results
tree-seg results --dataset fortress --sort mIoU --top 5

# Details for specific hash
tree-seg results --hash a3f7c2d1

# Compare two runs
tree-seg results --compare a3f7c2d1 b8e4f1a9
```

### Management

```bash
# Regenerate visualization
tree-seg results --hash a3f7c2d1 --render

# Export to CSV
tree-seg results --dataset fortress --export results.csv

# Cleanup
tree-seg results --compact
tree-seg results --prune --older-than 30d
```

### Current Implementation Status (Dec 6, 2024)
- ✅ Metadata storage: hash/index/meta.json, auto-tags, hardware info, artifacts (results/viz/labels paths)
- ✅ Label dumping: `--save-labels/--no-save-labels` in `tree-seg eval`, saved under `<output>/labels/{image_id}.npz`
- ✅ Results CLI: `tree-seg results` to list/filter by dataset/tags, sort, or show details for a hash (with config dump)
- ⚠️ Not yet: visualization regeneration, feature caching, nearest-config ETA scaling in CLI, integration with `segment` command, compaction/prune commands (stubs in doc only)

## API Surface

Module: `tree_seg/metadata/`

```python
# tree_seg/metadata/store.py
def store_run(results, config, dataset_id, tags=None, notes=None) -> str:
    """Store experiment results, return hash."""

# tree_seg/metadata/load.py
def lookup(hash: str) -> Optional[dict]:
    """Load meta.json for given hash."""

def lookup_nearest(config: dict) -> Optional[dict]:
    """Find closest config match for ETA fallback."""

# tree_seg/metadata/query.py
def query(dataset=None, tags=None, sort_by=None, limit=None) -> List[dict]:
    """Query index.jsonl with filters."""

def latest_by_tag(tag: str, limit: int = 5) -> List[dict]:
    """Get most recent runs with given tag."""
```

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] `tree_seg/metadata/store.py` - store_run, config hashing
- [ ] `tree_seg/metadata/load.py` - lookup, lookup_nearest
- [ ] `tree_seg/metadata/query.py` - query, latest_by_tag
- [ ] labels.npz compression/decompression
- [ ] Auto-derive tags from config

### Phase 2: CLI Integration
- [ ] Auto-save after `tree-seg eval`
- [ ] `tree-seg results` subcommand
- [ ] Query filtering (--clustering, --dataset, --tags)
- [ ] Rich table output

### Phase 3: Advanced Features
- [ ] Runtime fallback (nearest config matching)
- [ ] Visualization regeneration (--render)
- [ ] Export to CSV/markdown
- [ ] Comparison view (--compare)

### Phase 4: Maintenance
- [ ] Compaction (--compact)
- [ ] Pruning (--prune)
- [ ] Size limits

### Phase 5: Future (Low Priority)
- [ ] Meta-learning: rank configs by success for dataset family
- [ ] Online ETA model learning from hardware + config
- [ ] Web dashboard for browsing runs

## Open Questions

1. **Cache invalidation**: Use `--force` to recompute, or auto-invalidate on code changes?
   - ✅ Decided: `--force` flag, store git SHA for reference only

2. **Multi-image datasets**: Store per-image labels or aggregate only?
   - Current preference: Per-image labels in subdirectory (`labels/{sample_id}.npz`)

3. **Nearest-config similarity**: Exact Hamming distance or weighted by importance?
   - ✅ Decided: Weighted (model > image_size > tiling > stride > clustering > k)

4. **Hardware normalization**: How to compare ETAs across different hardware?
   - ✅ Decided: GPU tier buckets (low/mid/high/extreme) with scaling factors

---

*Created: 2024-12-06*
*Status: Planning*
*Merged from: claude_ideas.md + codex_ideas.md*
*Refined with: Codex feedback (hash inputs, length, schema, artifacts toggle, ETA weights)*
*Final polish: Config normalization, labels path structure, GPU tier buckets*
