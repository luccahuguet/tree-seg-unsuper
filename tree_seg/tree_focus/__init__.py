"""Tree Focus (V3) - Tree-specific segmentation logic."""

from tree_seg.tree_focus.vegetation_indices import (
    excess_green_index,
    color_index_vegetation_extraction,
    green_ratio,
    create_vegetation_mask,
    compute_vegetation_score,
    THRESHOLDS,
)

from tree_seg.tree_focus.cluster_selection import (
    select_tree_clusters,
    compute_cluster_iou,
    filter_clusters_by_vegetation,
)

from tree_seg.tree_focus.instance_segmentation import (
    separate_tree_instances,
    watershed_segmentation,
    filter_instances_by_shape,
)

__all__ = [
    # Vegetation indices
    'excess_green_index',
    'color_index_vegetation_extraction',
    'green_ratio',
    'create_vegetation_mask',
    'compute_vegetation_score',
    'THRESHOLDS',
    # Cluster selection
    'select_tree_clusters',
    'compute_cluster_iou',
    'filter_clusters_by_vegetation',
    # Instance segmentation
    'separate_tree_instances',
    'watershed_segmentation',
    'filter_instances_by_shape',
]
