#!/usr/bin/env python3
"""
Test V3 Integration with Main Pipeline

Verifies that V3 pipeline works through the TreeSegmentation API.
"""

from tree_seg import TreeSegmentation, Config
import numpy as np
from PIL import Image
from pathlib import Path


def test_v3_integration():
    """Test V3 through main API."""
    print("=" * 80)
    print("V3 Integration Test")
    print("=" * 80)

    # Create synthetic test image
    print("\n1. Creating test image...")
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    # Add green "trees"
    for cy, cx in [(150, 150), (350, 350)]:
        for r in range(50):
            y, x = np.ogrid[:512, :512]
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            test_image[mask] = [50, 150, 50]  # Green

    # Save test image
    test_path = Path("data/output/test_v3_integration.jpg")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(test_image).save(test_path)
    print(f"   Saved to: {test_path}")

    # Test V1.5 pipeline (baseline)
    print("\n2. Testing V1.5 (baseline)...")
    config_v1_5 = Config(
        pipeline="v1_5",
        auto_k=True,
        elbow_threshold=10.0,
        verbose=True
    )
    seg_v1_5 = TreeSegmentation(config_v1_5)
    results_v1_5 = seg_v1_5.process_single_image(str(test_path))
    print(f"   V1.5 Result: {results_v1_5.n_clusters_used} clusters")

    # Test V3 pipeline (tree-specific)
    print("\n3. Testing V3 (tree-specific)...")
    config_v3 = Config(
        pipeline="v3",
        v3_preset="balanced",
        auto_k=True,
        elbow_threshold=10.0,
        verbose=True
    )
    seg_v3 = TreeSegmentation(config_v3)
    results_v3 = seg_v3.process_single_image(str(test_path))
    print("   V3 Result: Instance segmentation complete")

    print("\n" + "=" * 80)
    print("✓ V3 Integration test passed!")
    print("=" * 80)

    print("\nResults:")
    print(f"  V1.5: {results_v1_5.n_clusters_used} clusters")
    print(f"  V3: Instance mask shape {results_v3.labels_resized.shape}")
    print("\nV3 pipeline successfully integrated into main API!")


if __name__ == "__main__":
    test_v3_integration()
