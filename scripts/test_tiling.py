#!/usr/bin/env python3
"""
Test tiling implementation on synthetic and real images.
"""

import numpy as np

from tree_seg.models.tiling import TileManager, TileConfig


def test_tile_extraction():
    """Test basic tile extraction."""
    print("=" * 60)
    print("Test 1: Tile Extraction")
    print("=" * 60)

    # Create a synthetic 4096×4096 image
    image = np.random.rand(4096, 4096, 3).astype(np.float32)
    print(f"Image shape: {image.shape}")

    # Configure tiling
    config = TileConfig(
        tile_size=2048,
        overlap=256,
        auto_tile_threshold=2048
    )
    manager = TileManager(config)

    # Check if tiling is needed
    h, w = image.shape[:2]
    should_tile = manager.should_tile(h, w)
    print(f"Should tile: {should_tile}")

    # Get grid info
    grid_info = manager.get_grid_info(h, w)
    print(f"Grid info: {grid_info}")

    # Extract tiles
    tiles = manager.extract_tiles(image)
    print(f"Number of tiles extracted: {len(tiles)}")

    # Verify tile shapes
    for i, tile_info in enumerate(tiles[:3]):  # Check first 3 tiles
        print(f"  Tile {i}: shape={tile_info.shape}, "
              f"coords=({tile_info.x_start}, {tile_info.y_start}, "
              f"{tile_info.x_end}, {tile_info.y_end})")

    # Expected: 3×3 = 9 tiles (due to overlap)
    expected_tiles = 9
    assert len(tiles) == expected_tiles, f"Expected {expected_tiles} tiles, got {len(tiles)}"
    print("✓ Tile extraction test passed!")


def test_feature_stitching():
    """Test feature stitching with weighted blending."""
    print("\n" + "=" * 60)
    print("Test 2: Feature Stitching (Basic)")
    print("=" * 60)

    # Simplified test: Just verify the stitching function runs without errors
    # Real coordinate mapping will be tested in end-to-end tests

    # Create synthetic tile features
    tile_size = 2048
    feat_size = 128  # Feature grid size for a 2048px tile
    feat_dim = 768  # DINOv3 base feature dimension

    # Single tile test (simplest case)
    tile_features = [np.ones((feat_size, feat_size, feat_dim), dtype=np.float32)]
    tile_coords = [(0, 0, tile_size, tile_size)]

    print("Testing with 1 tile")
    print(f"Tile feature shape: {tile_features[0].shape}")

    # Configure tiling
    config = TileConfig(tile_size=tile_size, overlap=256)
    manager = TileManager(config)

    # Output shape should match tile feature size
    output_h, output_w = feat_size, feat_size

    # Stitch features
    stitched = manager.stitch_features(
        tile_features,
        tile_coords,
        output_shape=(output_h, output_w)
    )

    print(f"Stitched shape: {stitched.shape}")

    # Verify no NaN values
    assert not np.isnan(stitched).any(), "Stitched features contain NaN!"
    assert stitched.shape == (output_h, output_w, feat_dim), "Shape mismatch!"

    # For single tile, should have full coverage
    non_zero = (stitched.sum(axis=-1) > 0).sum()
    total = output_h * output_w
    coverage = 100.0 * non_zero / total
    print(f"Coverage: {coverage:.1f}% of output has non-zero features")

    # Single tile should have 100% coverage
    assert coverage >= 95.0, f"Coverage too low: {coverage:.1f}%"

    print("✓ Feature stitching test passed!")
    print(f"  Value range: [{stitched.min():.2f}, {stitched.max():.2f}]")
    print("  Note: Multi-tile coordinate mapping will be tested end-to-end")


def test_edge_weights():
    """Test 2D weight computation for blending."""
    print("\n" + "=" * 60)
    print("Test 3: Edge Weight Computation")
    print("=" * 60)

    config = TileConfig(tile_size=2048, overlap=256)
    manager = TileManager(config)

    # Compute weights for a tile
    tile_shape = (128, 128)  # Feature space
    overlap_feat = 16  # 256px overlap / 16 = 16 features

    weights = manager._compute_2d_weights(tile_shape, overlap_feat)

    print(f"Weight map shape: {weights.shape}")
    print(f"Weight range: [{weights.min():.2f}, {weights.max():.2f}]")

    # Check corners (should be 0.5 * 0.5 = 0.25)
    corner_weight = weights[0, 0]
    print(f"Corner weight: {corner_weight:.2f}")
    assert 0.2 <= corner_weight <= 0.3, f"Corner weight should be ~0.25, got {corner_weight}"

    # Check center (should be 1.0)
    center_weight = weights[64, 64]
    print(f"Center weight: {center_weight:.2f}")
    assert center_weight == 1.0, f"Center weight should be 1.0, got {center_weight}"

    print("✓ Edge weight test passed!")


def test_fortress_realistic():
    """Test with FORTRESS-like dimensions."""
    print("\n" + "=" * 60)
    print("Test 4: FORTRESS Realistic Test (9372×9372)")
    print("=" * 60)

    # FORTRESS image dimensions
    h, w = 9372, 9372
    print(f"Image size: {h}×{w}")

    config = TileConfig(
        tile_size=2048,
        overlap=256,
        auto_tile_threshold=2048
    )
    manager = TileManager(config)

    # Get grid info
    grid_info = manager.get_grid_info(h, w)
    print(f"Grid: {grid_info['n_tiles_y']}×{grid_info['n_tiles_x']} = {grid_info['n_tiles']} tiles")

    # Expected: 6×6 = 36 tiles (with overlap, need more tiles to cover the image)
    assert 30 <= grid_info['n_tiles'] <= 40, f"Unexpected tile count: {grid_info['n_tiles']}"

    print("✓ FORTRESS realistic test passed!")


if __name__ == "__main__":
    print("Testing Tiling Implementation")
    print("=" * 60)

    try:
        test_tile_extraction()
        test_feature_stitching()
        test_edge_weights()
        test_fortress_realistic()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
