"""Soft EM refinement for K-means clusters (V2)."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax


def soft_em_refine(
    features: np.ndarray,
    initial_labels: np.ndarray,
    n_clusters: int,
    temperature: float = 1.0,
    iterations: int = 5,
    spatial_blend_alpha: float = 0.0,
    height: int = None,
    width: int = None,
) -> np.ndarray:
    """
    Refine K-means clusters using soft EM iterations with optional spatial blending.

    Algorithm:
    1. Initialize cluster centers from hard K-means assignments
    2. For each iteration:
       - E-step: Compute soft assignments using temperature-scaled softmax
       - M-step: Update cluster centers using soft-weighted averages
       - Optional: Blend with spatial neighbors (if alpha > 0)
    3. Convert soft assignments back to hard labels

    Args:
        features: Feature vectors (N, D) where N = H*W
        initial_labels: Hard K-means labels (N,)
        n_clusters: Number of clusters
        temperature: τ - controls softness (lower = softer boundaries)
        iterations: Number of EM iterations (typically 3-5)
        spatial_blend_alpha: α - weight for spatial smoothing (0=none, 1=full)
        height: Image height (required if spatial_blend_alpha > 0)
        width: Image width (required if spatial_blend_alpha > 0)

    Returns:
        Refined hard labels (N,)
    """
    # Validate inputs
    if spatial_blend_alpha > 0 and (height is None or width is None):
        raise ValueError("height and width required for spatial blending")

    N, D = features.shape

    # Initialize cluster centers from hard K-means labels
    centers = np.zeros((n_clusters, D))
    for k in range(n_clusters):
        mask = initial_labels == k
        if mask.sum() > 0:
            centers[k] = features[mask].mean(axis=0)
        else:
            # Handle empty cluster - reinitialize randomly
            centers[k] = features[np.random.randint(N)]

    # Soft EM refinement
    soft_assignments = None
    for iteration in range(iterations):
        # E-step: Compute soft assignments with temperature scaling
        distances = cdist(features, centers, metric='euclidean')  # (N, K)

        # Apply temperature-scaled softmax (lower temp = softer)
        # Note: We use negative distances because softmax gives higher prob to larger values
        soft_assignments = softmax(-distances / temperature, axis=1)  # (N, K)

        # Optional: Spatial blending
        if spatial_blend_alpha > 0:
            soft_assignments = _spatial_blend(
                soft_assignments,
                alpha=spatial_blend_alpha,
                height=height,
                width=width
            )

        # M-step: Update cluster centers using soft weights
        for k in range(n_clusters):
            weights = soft_assignments[:, k:k+1]  # (N, 1)
            weight_sum = weights.sum()

            if weight_sum > 1e-8:  # Avoid division by zero
                centers[k] = (features * weights).sum(axis=0) / weight_sum
            # else: keep previous center

    # Convert soft assignments to hard labels
    final_labels = soft_assignments.argmax(axis=1)

    return final_labels


def _spatial_blend(
    soft_assignments: np.ndarray,
    alpha: float,
    height: int,
    width: int
) -> np.ndarray:
    """
    Blend soft assignments with spatial neighbors for smoother boundaries.

    Args:
        soft_assignments: Soft cluster assignments (N, K)
        alpha: Blending weight (0=no blend, 1=full spatial averaging)
        height: Image height
        width: Image width

    Returns:
        Spatially blended soft assignments (N, K)
    """
    n_clusters = soft_assignments.shape[1]

    # Reshape to 2D spatial grid
    assignments_2d = soft_assignments.reshape(height, width, n_clusters)

    # Simple spatial smoothing: average with 4-connected neighbors
    blended = assignments_2d.copy()

    # For each pixel, average with neighbors
    for i in range(height):
        for j in range(width):
            neighbors = []

            # Collect valid neighbors
            if i > 0:
                neighbors.append(assignments_2d[i-1, j])
            if i < height - 1:
                neighbors.append(assignments_2d[i+1, j])
            if j > 0:
                neighbors.append(assignments_2d[i, j-1])
            if j < width - 1:
                neighbors.append(assignments_2d[i, j+1])

            if neighbors:
                neighbor_avg = np.mean(neighbors, axis=0)
                # Blend current assignment with neighbor average
                blended[i, j] = (1 - alpha) * assignments_2d[i, j] + alpha * neighbor_avg

    # Renormalize to ensure probabilities sum to 1
    blended = blended / blended.sum(axis=2, keepdims=True)

    # Reshape back to (N, K)
    return blended.reshape(-1, n_clusters)


def _spatial_blend_fast(
    soft_assignments: np.ndarray,
    alpha: float,
    height: int,
    width: int
) -> np.ndarray:
    """
    Fast vectorized spatial blending using convolution.

    This is a faster alternative to the loop-based _spatial_blend.
    Uses scipy.ndimage for efficient neighbor averaging.
    """
    from scipy.ndimage import uniform_filter

    n_clusters = soft_assignments.shape[1]

    # Reshape to 2D spatial grid
    assignments_2d = soft_assignments.reshape(height, width, n_clusters)

    # Apply spatial smoothing per cluster
    smoothed = np.zeros_like(assignments_2d)
    for k in range(n_clusters):
        # Uniform filter averages with 3x3 neighborhood
        smoothed[:, :, k] = uniform_filter(
            assignments_2d[:, :, k],
            size=3,
            mode='reflect'
        )

    # Blend original with smoothed
    blended = (1 - alpha) * assignments_2d + alpha * smoothed

    # Renormalize
    blended = blended / blended.sum(axis=2, keepdims=True)

    return blended.reshape(-1, n_clusters)
