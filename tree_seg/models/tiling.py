"""
Tiling Module for Large Image Processing

Handles tile extraction and feature stitching for large aerial imagery
that exceeds DINOv3's optimal resolution.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class TileConfig:
    """Configuration for tile-based processing."""

    tile_size: int = 2048  # Tile dimension (square)
    overlap: int = 256  # Overlap between adjacent tiles
    auto_tile_threshold: int = 2048  # Auto-enable tiling if image > this size
    blend_mode: str = "linear"  # Blending mode: "linear" or "none"


@dataclass
class TileInfo:
    """Information about a single extracted tile."""

    tile_array: np.ndarray  # (H, W, 3) RGB tile
    x_start: int  # Start x coordinate in full image
    y_start: int  # Start y coordinate in full image
    x_end: int  # End x coordinate in full image
    y_end: int  # End y coordinate in full image

    @property
    def shape(self) -> Tuple[int, int]:
        """Get tile shape (H, W)."""
        return self.tile_array.shape[:2]


class TileManager:
    """
    Manages tile extraction and feature stitching for large images.

    Usage:
        config = TileConfig(tile_size=2048, overlap=256)
        manager = TileManager(config)

        if manager.should_tile(h, w):
            tiles = manager.extract_tiles(image_np)
            # Process each tile...
            stitched = manager.stitch_features(tile_features, tile_coords, output_shape)
    """

    def __init__(self, config: TileConfig):
        """
        Initialize TileManager.

        Args:
            config: Tiling configuration
        """
        self.config = config
        self.tile_size = config.tile_size
        self.overlap = config.overlap
        self.stride = self.tile_size - self.overlap  # Effective step size

    def should_tile(self, h: int, w: int) -> bool:
        """
        Check if image dimensions require tiling.

        Args:
            h: Image height
            w: Image width

        Returns:
            True if tiling is needed, False otherwise
        """
        return (
            h > self.config.auto_tile_threshold or w > self.config.auto_tile_threshold
        )

    def extract_tiles(self, image_np: np.ndarray) -> List[TileInfo]:
        """
        Extract overlapping tiles from image in a grid pattern.

        Args:
            image_np: Full image array (H, W, 3)

        Returns:
            List of TileInfo objects with tile data and coordinates
        """
        h, w = image_np.shape[:2]
        tiles = []

        # Calculate grid dimensions
        # We want tiles to cover the entire image with overlap
        y_positions = self._calculate_tile_positions(h)
        x_positions = self._calculate_tile_positions(w)

        # Extract tiles in row-major order
        for y_start in y_positions:
            y_end = min(y_start + self.tile_size, h)

            for x_start in x_positions:
                x_end = min(x_start + self.tile_size, w)

                # Extract tile
                tile_array = image_np[y_start:y_end, x_start:x_end]

                # Pad if tile is smaller than tile_size (edge case)
                if (
                    tile_array.shape[0] < self.tile_size
                    or tile_array.shape[1] < self.tile_size
                ):
                    tile_array = self._pad_tile(tile_array, self.tile_size)

                tiles.append(
                    TileInfo(
                        tile_array=tile_array,
                        x_start=x_start,
                        y_start=y_start,
                        x_end=x_end,
                        y_end=y_end,
                    )
                )

        return tiles

    def _calculate_tile_positions(self, dim_size: int) -> List[int]:
        """
        Calculate starting positions for tiles along one dimension.

        Args:
            dim_size: Size of dimension (height or width)

        Returns:
            List of starting positions
        """
        positions = []
        pos = 0

        while pos < dim_size:
            positions.append(pos)

            # Move by stride, but ensure we don't skip the end
            next_pos = pos + self.stride

            # If we're near the end, place final tile at the edge
            if next_pos + self.tile_size > dim_size and pos + self.tile_size < dim_size:
                # Add final tile that ends exactly at dim_size
                positions.append(dim_size - self.tile_size)
                break

            pos = next_pos

            # Stop if we've covered the entire dimension
            if pos + self.tile_size >= dim_size:
                break

        return positions

    def _pad_tile(self, tile: np.ndarray, target_size: int) -> np.ndarray:
        """
        Pad tile to target size (for edge tiles).

        Args:
            tile: Tile array (H, W, 3)
            target_size: Target dimension

        Returns:
            Padded tile (target_size, target_size, 3)
        """
        h, w = tile.shape[:2]

        # Calculate padding amounts
        pad_h = target_size - h
        pad_w = target_size - w

        # Pad with edge values (reflection)
        padded = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        return padded

    def stitch_features(
        self,
        tile_features: List[np.ndarray],
        tile_coords: List[Tuple[int, int, int, int]],
        output_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Stitch tile features with weighted blending in overlap regions.

        Args:
            tile_features: List of (H_tile, W_tile, D) feature arrays
            tile_coords: List of (x_start, y_start, x_end, y_end) in image coordinates
            output_shape: Target (H, W) for stitched feature map

        Returns:
            Stitched features (H, W, D)
        """
        if len(tile_features) == 0:
            raise ValueError("No tile features provided")

        # Get feature dimension
        feat_dim = tile_features[0].shape[-1]

        # Initialize output arrays
        output_h, output_w = output_shape
        full_features = np.zeros((output_h, output_w, feat_dim), dtype=np.float32)
        weight_sum = np.zeros((output_h, output_w, 1), dtype=np.float32)

        # Calculate feature scale factor (image pixels per feature pixel)
        # This is the ratio between tile size in pixels and tile size in features
        if len(tile_features) > 0:
            first_tile_feat_h, first_tile_feat_w = tile_features[0].shape[:2]
            # Assume tiles were processed at self.tile_size
            feat_scale_h = self.tile_size / first_tile_feat_h
            feat_scale_w = self.tile_size / first_tile_feat_w
        else:
            feat_scale_h = feat_scale_w = 1.0

        # Process each tile
        for features, (x0, y0, x1, y1) in zip(tile_features, tile_coords):
            tile_feat_h, tile_feat_w = features.shape[:2]

            # Calculate feature coordinates corresponding to image coordinates
            feat_x0 = int(x0 / feat_scale_w)
            feat_y0 = int(y0 / feat_scale_h)
            feat_x1 = min(int(x1 / feat_scale_w), output_w)
            feat_y1 = min(int(y1 / feat_scale_h), output_h)

            # Extract the relevant portion of features (handle edge tiles)
            actual_feat_h = feat_y1 - feat_y0
            actual_feat_w = feat_x1 - feat_x0

            tile_features_cropped = features[:actual_feat_h, :actual_feat_w, :]

            # Compute 2D blend weights for this tile
            blend_weights = self._compute_2d_weights(
                (actual_feat_h, actual_feat_w),
                int(self.overlap / feat_scale_h),  # Overlap in feature space
            )

            # Accumulate weighted features
            full_features[feat_y0:feat_y1, feat_x0:feat_x1] += (
                tile_features_cropped * blend_weights[..., None]
            )
            weight_sum[feat_y0:feat_y1, feat_x0:feat_x1] += blend_weights[..., None]

        # Normalize by weight sum (avoid division by zero)
        full_features = full_features / np.maximum(weight_sum, 1e-8)

        return full_features

    def _compute_2d_weights(
        self, tile_shape: Tuple[int, int], overlap: int
    ) -> np.ndarray:
        """
        Compute 2D weight map for a tile with linear tapering at edges.

        Creates weights that are 1.0 in the center and taper to 0.5 at edges
        in overlap regions. This ensures smooth blending when tiles overlap.

        Args:
            tile_shape: (H, W) shape of tile in feature space
            overlap: Overlap width in feature space

        Returns:
            weights (H, W) with values [0.5 ... 1.0 ... 0.5]
        """
        h, w = tile_shape

        # Create 1D weight profiles (1.0 everywhere initially)
        weight_h = np.ones(h, dtype=np.float32)
        weight_w = np.ones(w, dtype=np.float32)

        # Taper edges in overlap regions
        if overlap > 0 and overlap < h and overlap < w:
            # Linear taper from 0.5 to 1.0 over overlap distance
            edge_weights = np.linspace(0.5, 1.0, overlap, dtype=np.float32)

            # Apply to edges
            weight_h[:overlap] = np.minimum(weight_h[:overlap], edge_weights)
            weight_h[-overlap:] = np.minimum(weight_h[-overlap:], edge_weights[::-1])
            weight_w[:overlap] = np.minimum(weight_w[:overlap], edge_weights)
            weight_w[-overlap:] = np.minimum(weight_w[-overlap:], edge_weights[::-1])

        # Compute 2D weights via outer product
        weights_2d = np.outer(weight_h, weight_w)

        return weights_2d

    def get_grid_info(self, h: int, w: int) -> dict:
        """
        Get information about the tile grid for an image.

        Args:
            h: Image height
            w: Image width

        Returns:
            Dictionary with grid information
        """
        y_positions = self._calculate_tile_positions(h)
        x_positions = self._calculate_tile_positions(w)

        return {
            "n_tiles": len(y_positions) * len(x_positions),
            "n_tiles_y": len(y_positions),
            "n_tiles_x": len(x_positions),
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "stride": self.stride,
        }
