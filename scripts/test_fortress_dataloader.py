#!/usr/bin/env python3
"""Test FORTRESS dataset loader."""

import sys
from pathlib import Path
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tree_seg.evaluation.datasets import FortressDataset

if __name__ == "__main__":
    # Load dataset
    dataset_path = Path("data/fortress_processed")
    
    print(f"Loading FORTRESS dataset from {dataset_path}...")
    dataset = FortressDataset(dataset_path)
    
    print("\nDataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {dataset.NUM_CLASSES}")
    print("\nClass mapping:")
    for class_id in sorted(dataset.CLASS_NAMES.keys()):
        print(f"  {class_id}: {dataset.CLASS_NAMES[class_id]}")
    
    # Test loading first sample
    print("\nLoading first sample...")
    image, label, image_id = dataset[0]
    
    print("\nFirst sample info:")
    print(f"  Image ID: {image_id}")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min()}, {image.max()}]")
    print(f"  Label shape: {label.shape}")
    print(f"  Label dtype: {label.dtype}")
    print(f"  Unique labels: {sorted(np.unique(label).tolist())}")
    
    # Check label distribution
    unique, counts = np.unique(label, return_counts=True)
    print("\nLabel distribution (first sample):")
    for val, count in zip(unique, counts):
        pct = 100 * count / label.size
        class_name = dataset.CLASS_NAMES.get(val, f"Unknown ({val})")
        print(f"  {val:2d} ({class_name:30s}): {count:8d} pixels ({pct:5.2f}%)")
    
    print("\n✅ Dataset test passed!")
