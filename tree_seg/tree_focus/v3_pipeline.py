"""
V3 Pipeline: Tree-Specific Segmentation

Integrates:
1. Vegetation prefiltering (ExG/CIVE)
2. Cluster selection (IoU-based)
3. Instance segmentation (watershed)
4. Shape/area filtering (GSD-aware)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

from tree_seg.tree_focus.vegetation_indices import create_vegetation_mask, THRESHOLDS
from tree_seg.tree_focus.cluster_selection import select_tree_clusters
from tree_seg.tree_focus.instance_segmentation import (
    separate_tree_instances,
    morphological_cleanup
)


@dataclass
class V3Config:
    """Configuration for V3 tree detection."""

    # Vegetation filtering
    vegetation_method: str = 'exg'  # 'exg', 'cive', 'green_ratio', 'combined'
    vegetation_threshold: Optional[float] = None  # If None, uses method default
    vegetation_preset: str = 'balanced'  # 'permissive', 'balanced', 'strict'

    # Cluster selection
    iou_threshold: float = 0.3  # Minimum IoU with vegetation mask
    veg_score_threshold: float = 0.4  # Minimum vegetation score
    min_tree_area_m2: float = 1.0  # Minimum tree area (m²)
    max_tree_area_m2: float = 500.0  # Maximum tree area (m²)

    # Instance segmentation
    watershed_min_distance: int = 10  # Min distance between tree centers (pixels)
    min_circularity: float = 0.3  # Minimum shape circularity
    max_eccentricity: float = 0.95  # Maximum eccentricity

    # Morphological cleanup
    morphology_kernel_size: int = 3
    morphology_operation: str = 'close'  # 'open', 'close', 'both', 'none'

    # GSD (Ground Sample Distance)
    gsd_cm: float = 10.0  # cm/pixel (10cm = typical drone imagery)

    def __post_init__(self):
        """Set vegetation threshold from preset if not specified."""
        if self.vegetation_threshold is None:
            self.vegetation_threshold = THRESHOLDS[self.vegetation_preset][self.vegetation_method]


@dataclass
class V3Results:
    """Results from V3 tree detection."""

    # Masks
    vegetation_mask: np.ndarray  # Binary vegetation mask (H, W)
    tree_mask: np.ndarray  # Binary tree mask after cluster selection (H, W)
    instance_labels: np.ndarray  # Instance labels with unique ID per tree (H, W)

    # Statistics
    num_trees: int  # Number of detected trees
    cluster_stats: List[Dict]  # Stats for each cluster
    instance_stats: List[Dict]  # Stats for each tree instance

    # Intermediate results (for debugging/visualization)
    input_clusters: np.ndarray  # Input cluster labels from V1.5
    selected_cluster_ids: List[int]  # IDs of clusters selected as trees


class V3Pipeline:
    """V3 tree detection pipeline."""

    def __init__(self, config: V3Config = None):
        """
        Initialize V3 pipeline.

        Args:
            config: V3 configuration (uses defaults if None)
        """
        self.config = config if config is not None else V3Config()

    def process(
        self,
        image: np.ndarray,
        cluster_labels: np.ndarray
    ) -> V3Results:
        """
        Run full V3 pipeline.

        Args:
            image: RGB image (H, W, 3)
            cluster_labels: Cluster labels from V1.5 (H, W) with 0=background

        Returns:
            V3Results with all outputs and statistics
        """
        # Step 1: Create vegetation mask
        vegetation_mask = create_vegetation_mask(
            image,
            method=self.config.vegetation_method,
            threshold=self.config.vegetation_threshold
        )

        # Optional: Morphological cleanup of vegetation mask
        if self.config.morphology_operation != 'none':
            vegetation_mask = morphological_cleanup(
                vegetation_mask,
                kernel_size=self.config.morphology_kernel_size,
                operation=self.config.morphology_operation
            )

        # Step 2: Select tree clusters based on vegetation overlap
        tree_mask, cluster_stats = select_tree_clusters(
            image=image,
            cluster_labels=cluster_labels,
            vegetation_mask=vegetation_mask,
            iou_threshold=self.config.iou_threshold,
            veg_score_threshold=self.config.veg_score_threshold,
            min_area_m2=self.config.min_tree_area_m2,
            max_area_m2=self.config.max_tree_area_m2,
            gsd_cm=self.config.gsd_cm
        )

        # Step 3: Separate into individual tree instances
        # Convert area thresholds to pixels
        min_area_pixels = int(self.config.min_tree_area_m2 / (self.config.gsd_cm / 100.0) ** 2)
        max_area_pixels = int(self.config.max_tree_area_m2 / (self.config.gsd_cm / 100.0) ** 2)

        instance_labels, num_trees, instance_stats = separate_tree_instances(
            tree_mask=tree_mask,
            min_distance=self.config.watershed_min_distance,
            min_area_pixels=min_area_pixels,
            max_area_pixels=max_area_pixels,
            gsd_cm=self.config.gsd_cm
        )

        # Extract selected cluster IDs
        selected_cluster_ids = [
            stat['cluster_id'] for stat in cluster_stats if stat['is_tree']
        ]

        return V3Results(
            vegetation_mask=vegetation_mask,
            tree_mask=tree_mask,
            instance_labels=instance_labels,
            num_trees=num_trees,
            cluster_stats=cluster_stats,
            instance_stats=instance_stats,
            input_clusters=cluster_labels,
            selected_cluster_ids=selected_cluster_ids
        )

    def process_from_v1_results(
        self,
        image: np.ndarray,
        v1_segmentation: np.ndarray
    ) -> V3Results:
        """
        Process from V1.5 segmentation results.

        Args:
            image: RGB image (H, W, 3)
            v1_segmentation: V1.5 cluster labels (H, W)

        Returns:
            V3Results
        """
        return self.process(image, v1_segmentation)


def create_v3_preset(preset: str) -> V3Config:
    """
    Create V3 configuration from preset.

    Args:
        preset: 'permissive', 'balanced', or 'strict'

    Returns:
        V3Config object
    """
    if preset == 'permissive':
        # Catch more trees (higher recall, lower precision)
        return V3Config(
            vegetation_preset='permissive',
            iou_threshold=0.2,
            veg_score_threshold=0.3,
            min_tree_area_m2=0.5,
            max_tree_area_m2=1000.0,
            watershed_min_distance=8,
        )

    elif preset == 'balanced':
        # Default balanced configuration
        return V3Config(
            vegetation_preset='balanced',
            iou_threshold=0.3,
            veg_score_threshold=0.4,
            min_tree_area_m2=1.0,
            max_tree_area_m2=500.0,
            watershed_min_distance=10,
        )

    elif preset == 'strict':
        # Only confident trees (higher precision, lower recall)
        return V3Config(
            vegetation_preset='strict',
            iou_threshold=0.4,
            veg_score_threshold=0.5,
            min_tree_area_m2=2.0,
            max_tree_area_m2=300.0,
            watershed_min_distance=12,
        )

    else:
        raise ValueError(f"Unknown preset: {preset}. Use 'permissive', 'balanced', or 'strict'")


if __name__ == "__main__":
    # Example usage
    print("V3 Pipeline module loaded")
    print("\nAvailable presets:")
    for preset in ['permissive', 'balanced', 'strict']:
        config = create_v3_preset(preset)
        print(f"  {preset}: IoU≥{config.iou_threshold}, VegScore≥{config.veg_score_threshold}")
