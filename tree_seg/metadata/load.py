"""Load helpers for metadata bank."""

import json
from pathlib import Path
from typing import Dict, Optional

from tree_seg.metadata.store import normalize_config


def lookup(hash_id: str, base_dir: Path | str = "results") -> Optional[Dict]:
    """Load meta.json for a given hash."""
    meta_path = Path(base_dir) / "by-hash" / hash_id / "meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open() as f:
        return json.load(f)


def _load_index(base_dir: Path | str = "results") -> list[Dict]:
    """Load all entries from index.jsonl."""
    index_path = Path(base_dir) / "index.jsonl"
    if not index_path.exists():
        return []
    entries = []
    with index_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def lookup_nearest(
    config: Dict[str, object], base_dir: Path | str = "results"
) -> Optional[Dict]:
    """
    Find the closest matching config in the index using weighted matches.

    Returns the index entry for the closest match, or None if empty.
    """
    entries = _load_index(base_dir)
    if not entries:
        return None

    target = normalize_config(config)

    WEIGHTS = {
        "model": 10,
        "image_size": 8,
        "tiling": 7,
        "stride": 6,
        "k": 5,
        "clustering": 4,
        "refine": 3,
        "smart_k": 2,
        "pyramid": 1,
    }

    def score(entry_config: Dict[str, object]) -> int:
        s = 0
        for key, weight in WEIGHTS.items():
            if target.get(key) == entry_config.get(key):
                s += weight
        return s

    best_entry = None
    best_score = -1
    for entry in entries:
        entry_config = entry.get("config", {})
        entry_score = score(entry_config)
        if entry_score > best_score:
            best_score = entry_score
            best_entry = entry
    return best_entry
