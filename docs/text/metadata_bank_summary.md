# Metadata Bank - Quick Reference

## Purpose
Cache experiment results to avoid re-running identical configs and provide accurate runtime ETAs.

## Storage Structure
```
results/
├── index.jsonl              # Fast searchable log
└── by-hash/{hash}/          # Content-addressable storage
    ├── meta.json            # Config + timing + metrics + tags
    └── labels.npz           # Compressed segmentation labels
```

## How It Works

**Config Hashing:**
- 10-character deterministic hash from config parameters
- Excludes git SHA to maximize cache hits
- Includes all runtime-affecting params: model, stride, clustering, refinement, tiling, etc.

**Auto-Tagging:**
- Derived from config: `["fortress", "kmeans", "k5", "slic", "stride-4", "base"]`
- User tags: `["v1.5-baseline", "paper-figure-3"]`

**Hardware-Aware ETAs:**
- GPU tiers: extreme/high/mid/low with scaling factors
- Nearest-config matching when exact hash not found
- Weighted similarity: model (10) > image_size (8) > tiling (7) > stride (6) > clustering (3) > refine (2) > k (1)

## CLI Usage

**Auto-save (automatic):**
```bash
tree-seg eval data/datasets/fortress --clustering kmeans
# → Saves to results/by-hash/{hash}/ automatically
```

**Query results:**
```bash
# Find best runs
tree-seg results --sort mIoU --top 5

# Filter by tags
tree-seg results --tags kmeans,fortress

# View specific run
tree-seg results --hash a3f7c2d1

# Regenerate visualization (requires labels.npz)
tree-seg results --hash a3f7c2d1 --render

# Export to CSV
tree-seg results --export results.csv
```

**Management:**
```bash
# Remove orphaned entries
tree-seg results --compact

# Cleanup old experiments
tree-seg results --prune-older-than 30d
```

## Label Storage

**Default:** Labels saved to `<output>/labels/{image_id}.npz` (~100KB per image)

**Disable for large datasets:**
```bash
tree-seg eval data/huge-dataset --no-save-labels
```

Note: Without labels, visualization regeneration (`--render`) won't work, but ETAs still function via meta.json timing data.

## Implementation Status (Dec 6, 2024)

✅ **Complete:**
- Metadata storage with auto-tags
- Label dumping with toggle
- Results CLI (list, filter, sort, details, render, export)
- Nearest-config ETA lookup
- Compaction and age-based pruning

⚠️ **Deferred:**
- Comparison view (`--compare`)
- Size-based pruning (only age-based currently)
- Web dashboard
- Feature caching (not needed - full-run caching prevents recompute)

## Key Files
- `tree_seg/metadata/store.py` - Save runs
- `tree_seg/metadata/load.py` - Load/lookup with nearest-config fallback
- `tree_seg/metadata/query.py` - Query index with filters

---

*Summarized from: metadata_bank.md*
*Created: 2024-12-07*
