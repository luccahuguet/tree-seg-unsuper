# Experiment Tracking & Metadata Bank

## Problem Statement

With the composable CLI (`--clustering`, `--refine`, `--vegetation-filter`, etc.), we have many possible configuration combinations. We need:

1. **Timing references** - Know expected runtime for progress bars on repeat runs
2. **Result caching** - Avoid re-running identical experiments
3. **Discoverability** - Find past results by config parameters
4. **Comparison** - Query best results across experiments

## Proposed Architecture

### Storage Structure

```
results/
├── index.jsonl              # Append-only experiment log
├── by-hash/                 # Content-addressable storage
│   ├── a3f7c2d1/
│   │   ├── meta.json        # Config + timing + metrics
│   │   ├── labels.npz       # Compressed segmentation labels
│   │   └── features.npz     # Optional: cached DINOv3 features
│   └── b8e4f1a9/
│       └── ...
└── by-experiment/           # Optional: human-readable symlinks
    └── fortress-kmeans-k5-slic → ../by-hash/a3f7c2d1/
```

### Config Hashing

Deterministic hash from canonical config for deduplication:

```python
import hashlib
import json

def config_hash(config: dict) -> str:
    """Generate unique hash for a configuration."""
    # Canonical JSON: sorted keys, no whitespace
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]
```

Config dict structure:
```python
{
    "dataset": "fortress",
    "clustering": "kmeans",
    "k": 5,
    "refine": "slic",
    "vegetation_filter": false,
    "model": "base",
    "stride": 4,
    # ... other relevant params
}
```

### What to Store

| Artifact | Store? | Rationale |
|----------|--------|-----------|
| `meta.json` | Always | Small, essential for queries |
| `labels.npz` | Always | ~100KB compressed, enables image regeneration |
| Images (PNG) | Never | Regenerate on demand from labels |
| `features.npz` | Optional | Large but saves DINOv3 inference time |
| Raw metrics | Always | mIoU, accuracy, per-class scores |
| Timing data | Always | For progress bar estimates |

### meta.json Schema

```json
{
    "hash": "a3f7c2d1",
    "created_at": "2024-12-06T14:30:00Z",
    "config": {
        "dataset": "fortress",
        "clustering": "kmeans",
        "k": 5,
        "refine": "slic",
        "vegetation_filter": false
    },
    "timing": {
        "feature_extraction_s": 12.5,
        "clustering_s": 0.3,
        "refinement_s": 1.2,
        "total_s": 14.0
    },
    "metrics": {
        "mIoU": 0.415,
        "pixel_accuracy": 0.623,
        "per_class_iou": {"tree": 0.52, "grass": 0.41, "building": 0.31}
    },
    "artifacts": {
        "labels": "labels.npz",
        "features": null
    }
}
```

### index.jsonl Format

Append-only log for fast queries without traversing directories:

```jsonl
{"hash":"a3f7c2d1","dataset":"fortress","clustering":"kmeans","k":5,"refine":"slic","mIoU":0.415,"created_at":"2024-12-06T14:30:00Z"}
{"hash":"b8e4f1a9","dataset":"fortress","clustering":"gmm","k":5,"refine":"slic","mIoU":0.398,"created_at":"2024-12-06T14:35:00Z"}
{"hash":"c9d2e3f4","dataset":"oam-tcd","clustering":"kmeans","k":8,"refine":"soft-em","mIoU":0.512,"created_at":"2024-12-06T15:00:00Z"}
```

Benefits:
- Git-friendly (append-only, merge-friendly)
- Grep-able (`grep "kmeans" index.jsonl`)
- No database dependency
- Human-readable

## CLI Interface

### Saving Results (Automatic)

Results saved automatically after `tree-seg eval`:

```bash
tree-seg eval data/fortress --clustering kmeans --k 5 --refine slic
# → Saves to results/by-hash/{hash}/
# → Appends to results/index.jsonl
```

### Querying Results

```bash
# Find all kmeans experiments
tree-seg results --clustering kmeans

# Find best mIoU on fortress
tree-seg results --dataset fortress --sort mIoU --top 5

# Get details for specific hash
tree-seg results --hash a3f7c2d1

# Compare two configs
tree-seg results --compare a3f7c2d1 b8e4f1a9
```

### Regenerating Images

```bash
# Regenerate visualization from cached labels
tree-seg results --hash a3f7c2d1 --regenerate

# Export comparison table
tree-seg results --dataset fortress --export results.csv
```

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] `tree_seg/results/store.py` - ResultStore class
- [ ] Config hashing function
- [ ] meta.json read/write
- [ ] labels.npz compression/decompression
- [ ] index.jsonl append/query

### Phase 2: CLI Integration
- [ ] Auto-save after `tree-seg eval`
- [ ] `tree-seg results` subcommand
- [ ] Query filtering (--clustering, --dataset, etc.)
- [ ] Table output with Rich

### Phase 3: Advanced Features
- [ ] Symlink generation for human traversal
- [ ] Feature caching (optional, for speed)
- [ ] Result comparison views
- [ ] Export to CSV/markdown

### Phase 4: Future (Low Priority)
- [ ] Metalearning: predict best config for new dataset
- [ ] Auto-tuning: suggest next experiment based on results
- [ ] Web dashboard (if needed)

## Design Decisions

### Why JSONL over SQLite?
- Git-friendly (text, append-only)
- No extra dependency
- Human-readable and grep-able
- Sufficient for expected scale (~1000s of experiments)

### Why hash-based storage?
- Automatic deduplication
- Deterministic: same config = same location
- Easy cache invalidation (delete by hash)
- No naming conflicts

### Why not save images?
- Labels are ~100KB, images are ~1MB+
- Can regenerate any visualization from labels
- Saves significant storage over time
- Labels preserve full information

### Why optional feature caching?
- DINOv3 features are large (~50MB per image)
- But inference is slow (~10-15s per image)
- Trade-off: storage vs. time
- Let user decide based on their constraints

## Example Workflow

```bash
# Run experiment
tree-seg eval data/fortress --clustering kmeans --k 5 --refine slic
# Output: Saved to results/by-hash/a3f7c2d1/ (mIoU: 0.415)

# Run another
tree-seg eval data/fortress --clustering gmm --k 5 --refine slic
# Output: Saved to results/by-hash/b8e4f1a9/ (mIoU: 0.398)

# Compare results
tree-seg results --dataset fortress --sort mIoU
# ┌──────────┬────────────┬───┬───────┬───────┐
# │ hash     │ clustering │ k │ refine│ mIoU  │
# ├──────────┼────────────┼───┼───────┼───────┤
# │ a3f7c2d1 │ kmeans     │ 5 │ slic  │ 0.415 │
# │ b8e4f1a9 │ gmm        │ 5 │ slic  │ 0.398 │
# └──────────┴────────────┴───┴───────┴───────┘

# Re-run same config (uses cache for timing estimate)
tree-seg eval data/fortress --clustering kmeans --k 5 --refine slic
# Output: Using cached result from a3f7c2d1 (run with --force to recompute)
```

## Open Questions

1. **Cache invalidation**: When should cached results be invalidated?
   - Code changes? (hard to detect)
   - Explicit `--force` flag? (current preference)
   - TTL-based expiry?

2. **Feature caching granularity**: Cache per-image or per-dataset?
   - Per-image: more flexible, larger storage
   - Per-dataset: less flexible, smaller storage

3. **Multi-image datasets**: How to handle datasets with many images?
   - Store per-image labels separately?
   - Aggregate metrics only?

4. **Version tracking**: Should we track code version/git commit?
   - Helps with reproducibility
   - Adds complexity

---

*Created: 2024-12-06*
*Status: Planning*
