"""Query helpers for metadata bank."""

import csv
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


def latest_by_tag(
    tag: str, limit: int = 5, base_dir: Path | str = "results"
) -> List[Dict]:
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


def purge_all(base_dir: Path | str = "results") -> int:
    """
    Delete ALL metadata (index + all by-hash directories).

    Returns number of entries removed.
    """
    import shutil

    base_path = Path(base_dir)
    index_path = base_path / "index.jsonl"
    by_hash_dir = base_path / "by-hash"

    # Count entries before deletion
    entries = _load_index(base_dir) if index_path.exists() else []
    count = len(entries)

    # Remove all by-hash directories
    if by_hash_dir.exists():
        shutil.rmtree(by_hash_dir)
        by_hash_dir.mkdir(parents=True, exist_ok=True)

    # Clear index file
    if index_path.exists():
        index_path.write_text("")

    return count


def export_to_csv(
    entries: List[Dict], csv_path: Path | str, base_dir: Path | str = "results"
) -> int:
    """
    Export query results to a CSV file with per-image granularity and upsert behavior.

    Loads full metadata for each entry to extract per-sample statistics.
    Creates one row per (config, image) combination.
    If the CSV exists, existing rows are updated by (hash, image_id) and new rows are appended.

    Returns number of total rows in the CSV after export.
    """
    fieldnames = [
        "hash",
        "dataset",
        "image_id",
        "mIoU",
        "pixel_accuracy",
        "num_clusters",
        "runtime_s",
        "tags",
        "type",
        "created_at",
    ]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = Path(base_dir)

    # Read existing rows if CSV exists (for upsert)
    # Key by (hash, image_id) tuple to prevent duplicates
    existing_rows = {}
    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hash_id = row.get("hash")
                image_id = row.get("image_id")
                if hash_id and image_id:
                    existing_rows[(hash_id, image_id)] = row

    # Load full metadata and extract per-sample stats
    for entry in entries:
        hash_id = entry.get("hash", "")
        if not hash_id:
            continue

        # Load full meta.json to get per_sample_stats
        meta_path = base_dir / "by-hash" / hash_id / "meta.json"
        if not meta_path.exists():
            continue

        try:
            with meta_path.open() as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        dataset = meta.get("dataset", "")
        created_at = meta.get("created_at", "")
        tags_list = (meta.get("tags", {}).get("auto") or []) + (
            meta.get("tags", {}).get("user") or []
        )
        tags_str = ",".join(tags_list)
        entry_type = entry.get("type", "benchmark")

        # Extract per-sample stats
        samples = meta.get("samples", {})
        per_sample_stats = samples.get("per_sample_stats", [])

        for sample in per_sample_stats:
            image_id = sample.get("image_id", "")
            if not image_id:
                continue

            key = (hash_id, image_id)
            existing_rows[key] = {
                "hash": hash_id,
                "dataset": dataset,
                "image_id": image_id,
                "mIoU": sample.get("miou", ""),
                "pixel_accuracy": sample.get("pixel_accuracy", ""),
                "num_clusters": sample.get("num_clusters", ""),
                "runtime_s": sample.get("runtime_seconds", ""),
                "tags": tags_str,
                "type": entry_type,
                "created_at": created_at,
            }

    # Write all rows back
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows.values():
            writer.writerow(row)

    return len(existing_rows)
