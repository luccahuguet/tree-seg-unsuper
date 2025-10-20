#!/usr/bin/env python3
"""
Benchmark different segmentation settings across a directory of images.

Writes a CSV with runtime, VRAM, and configuration per image.

Usage:
  uv run python scripts/utils/benchmark_profiles.py \
    --input data/input --output-csv bench.csv \
    --profiles quality balanced speed \
    --models small base

Use --keep-outputs to preserve generated images. Otherwise, outputs go to a temp folder and are removed.
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path so we can import tree_seg
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tree_seg import segment_trees  # noqa: E402
from tree_seg.presets import PRESETS as PROFILES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Benchmark segmentation profiles")
    ap.add_argument('--input', required=True, help='Input directory of images')
    ap.add_argument('--output-csv', required=True, help='Path to write CSV results')
    ap.add_argument('--profiles', nargs='+', default=['balanced'], choices=list(PROFILES.keys()))
    ap.add_argument('--models', nargs='+', default=['base'])
    ap.add_argument('--keep-outputs', action='store_true', help='Keep generated images')
    args = ap.parse_args()

    fieldnames = [
        'timestamp', 'image', 'profile', 'model',
        'image_size', 'feature_upsample_factor', 'pca_dim', 'refine', 'refine_slic_compactness', 'refine_slic_sigma',
        'k', 'time_total_s', 'time_features_s', 'time_kselect_s', 'time_kmeans_s', 'time_refine_s',
        'grid_H', 'grid_W', 'n_vectors', 'n_features', 'device', 'peak_vram_mb'
    ]

    first_write = not os.path.exists(args.output_csv)
    with open(args.output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if first_write:
            writer.writeheader()

        base_output_dir = Path("data/output") / "benchmarks"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        for profile in args.profiles:
            cfg = PROFILES[profile]
            for model in args.models:
                out_dir = base_output_dir / f"{profile}_{model}"
                results = segment_trees(
                    args.input,
                    model=model,
                    auto_k=True,
                    output_dir=str(out_dir),
                    metrics=True,
                    **cfg,
                )
                for res, _paths in results:
                    stats = res.processing_stats
                    row = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'image': os.path.basename(res.image_path),
                        'profile': profile,
                        'model': model,
                        'image_size': cfg['image_size'],
                        'feature_upsample_factor': cfg['feature_upsample_factor'],
                        'pca_dim': cfg['pca_dim'],
                        'refine': cfg['refine'],
                        'refine_slic_compactness': cfg['refine_slic_compactness'],
                        'refine_slic_sigma': cfg['refine_slic_sigma'],
                        'k': res.n_clusters_used,
                        'time_total_s': stats.get('time_total_s'),
                        'time_features_s': stats.get('time_features_s'),
                        'time_kselect_s': stats.get('time_kselect_s'),
                        'time_kmeans_s': stats.get('time_kmeans_s'),
                        'time_refine_s': stats.get('time_refine_s'),
                        'grid_H': stats.get('grid_H'),
                        'grid_W': stats.get('grid_W'),
                        'n_vectors': stats.get('n_vectors'),
                        'n_features': stats.get('n_features'),
                        'device': stats.get('device'),
                        'peak_vram_mb': stats.get('peak_vram_mb'),
                    }
                    writer.writerow(row)

                if not args.keep_outputs:
                    shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
