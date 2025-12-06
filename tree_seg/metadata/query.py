"""Query helpers for metadata bank."""

import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_index(base_dir: Path | str = "results") -> List[Dict]:
    index_path = Path(base_dir) / "index.jsonl"
    if not index_path.exists():
        return []
    entries: List[Dict] = []
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


def query(
    dataset: Optional[str] = None,
    tags: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    limit: Optional[int] = None,
    base_dir: Path | str = "results",
) -> List[Dict]:
    """Query index.jsonl with simple filters."""
    entries = _load_index(base_dir)

    def match(entry: Dict) -> bool:
        if dataset and entry.get("dataset") != dataset:
            return False
        if tags:
            entry_tags = set(entry.get("tags", []))
            if not set(tags).issubset(entry_tags):
                return False
        return True

    results = [e for e in entries if match(e)]

    if sort_by:
        results.sort(key=lambda x: x.get(sort_by, 0), reverse=True)

    if limit is not None:
        results = results[:limit]
    return results


def latest_by_tag(tag: str, limit: int = 5, base_dir: Path | str = "results") -> List[Dict]:
    """Return most recent entries containing the tag."""
    entries = query(tags=[tag], base_dir=base_dir)
    entries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return entries[:limit]


def compact(base_dir: Path | str = "results") -> int:
    """
    Compact the index by removing entries whose meta.json is missing.

    Returns number of entries removed.
    """
    index_path = Path(base_dir) / "index.jsonl"
    if not index_path.exists():
        return 0

    entries = _load_index(base_dir)
    kept = []
    removed = 0
    for entry in entries:
        hash_id = entry.get("hash")
        meta_path = Path(base_dir) / "by-hash" / hash_id / "meta.json"
        if meta_path.exists():
            kept.append(entry)
        else:
            removed += 1

    if removed:
        with index_path.open("w") as f:
            for e in kept:
                f.write(json.dumps(e) + "\n")
    return removed


def prune_older_than(days: int, base_dir: Path | str = "results") -> int:
    """
    Prune entries older than N days (index + meta directories).

    Returns number of entries removed.
    """
    from datetime import datetime, timezone, timedelta

    index_path = Path(base_dir) / "index.jsonl"
    if not index_path.exists():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = _load_index(base_dir)
    kept = []
    removed = 0
    for entry in entries:
        created_at = entry.get("created_at")
        try:
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except Exception:
            kept.append(entry)
            continue
        if created_dt < cutoff:
            hash_id = entry.get("hash")
            meta_dir = Path(base_dir) / "by-hash" / hash_id
            if meta_dir.exists():
                try:
                    import shutil

                    shutil.rmtree(meta_dir)
                except Exception:
                    pass
            removed += 1
        else:
            kept.append(entry)

    if removed:
        with index_path.open("w") as f:
            for e in kept:
                f.write(json.dumps(e) + "\n")
    return removed
