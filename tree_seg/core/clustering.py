"""Clustering utilities for tree_seg segmentation."""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def _make_kmeans(n_clusters: int) -> KMeans:
    """Factory to enforce consistent KMeans defaults across variants."""
    return KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")


def run_kmeans(
    features_flat: np.ndarray, n_clusters: int, verbose: bool = False
) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with K-means (k={n_clusters})...")
    kmeans = _make_kmeans(n_clusters)
    return kmeans.fit_predict(features_flat)


def run_spherical_kmeans(
    features_flat: np.ndarray, n_clusters: int, verbose: bool = False
) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with spherical k-means (cosine) (k={n_clusters})...")
    norms = np.linalg.norm(features_flat, axis=1, keepdims=True) + 1e-8
    features_norm = features_flat / norms
    kmeans = _make_kmeans(n_clusters)
    return kmeans.fit_predict(features_norm)


def run_dpmeans(
    features_flat: np.ndarray,
    n_clusters: int,
    max_iter: int = 20,
    verbose: bool = False,
    max_centers: int = 150,
) -> np.ndarray:
    """DP-means clustering with automatic K selection.

    Note: This is a simplified implementation that may struggle with very
    large/diverse datasets. Consider using kmeans with smart-k instead.
    """
    if verbose:
        print(
            f"🎯 Clustering with DP-means (auto K via lambda; init k={n_clusters})..."
        )

    # Estimate lambda threshold from sample (memory-efficient)
    sample_size = min(
        2000, features_flat.shape[0]
    )  # Larger sample for better lambda estimate
    idx = np.random.choice(features_flat.shape[0], sample_size, replace=False)
    sample = features_flat[idx]

    # Compute pairwise distances using matrix multiplication (more efficient)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b
    norms_sq = np.sum(sample**2, axis=1)
    pdists = norms_sq[:, None] + norms_sq[None, :] - 2 * np.dot(sample, sample.T)
    pdists = np.maximum(pdists, 0)  # Fix numerical errors

    median_dist = np.median(pdists)
    lam = 0.7 * median_dist

    # Initialize with random center
    centers = np.array([features_flat[np.random.choice(features_flat.shape[0])]])
    assignments = np.zeros(features_flat.shape[0], dtype=int)

    for iteration in range(max_iter):
        # Memory-efficient distance computation: compute min distance incrementally
        min_dists = np.full(features_flat.shape[0], np.inf)
        assignments = np.zeros(features_flat.shape[0], dtype=int)

        for k, center in enumerate(centers):
            # Compute distance to this center only
            dists = np.sum((features_flat - center) ** 2, axis=1)
            # Update assignments where this center is closer
            closer = dists < min_dists
            min_dists[closer] = dists[closer]
            assignments[closer] = k

        # Create at most ONE new center per iteration if any point too far
        far_points = min_dists > lam
        if far_points.any() and len(centers) < max_centers:
            far_idx = np.where(far_points)[0][0]
            centers = np.vstack([centers, features_flat[far_idx : far_idx + 1]])
            if verbose:
                print(
                    f"   Iteration {iteration + 1}: Added center (total={len(centers)})"
                )
        elif len(centers) >= max_centers:
            if verbose:
                print(f"   Reached max_centers={max_centers}, stopping growth")
            break

        # Update centers as means of assignments
        for k in range(len(centers)):
            mask = assignments == k
            if mask.any():
                centers[k] = features_flat[mask].mean(axis=0)

    if verbose:
        print(f"   Final: {len(centers)} clusters")

    return assignments


def run_potts_kmeans(
    features_flat: np.ndarray,
    n_clusters: int,
    H: int,
    W: int,
    beta: float = 0.5,
    iters: int = 2,
    verbose: bool = False,
) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with regularized k-means (Potts, k={n_clusters})...")
    kmeans = _make_kmeans(n_clusters)
    labels = kmeans.fit_predict(features_flat).reshape(H, W)
    smoothed = labels.copy()
    for _ in range(iters):
        for y in range(H):
            for x in range(W):
                neighbors = []
                if y > 0:
                    neighbors.append(smoothed[y - 1, x])
                if y < H - 1:
                    neighbors.append(smoothed[y + 1, x])
                if x > 0:
                    neighbors.append(smoothed[y, x - 1])
                if x < W - 1:
                    neighbors.append(smoothed[y, x + 1])
                neighbor_votes = np.bincount(neighbors, minlength=n_clusters)
                data_cost = np.sum(
                    (features_flat[y * W + x] - kmeans.cluster_centers_) ** 2, axis=1
                )
                smooth_cost = beta * (len(neighbors) - neighbor_votes)
                total_cost = data_cost + smooth_cost
                smoothed[y, x] = int(np.argmin(total_cost))
    return smoothed.reshape(-1)


def run_gmm(
    features_flat: np.ndarray, n_clusters: int, verbose: bool = False
) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with GMM (n_components={n_clusters})...")
    gmm = GaussianMixture(
        n_components=n_clusters,
        random_state=42,
        covariance_type="diag",
        reg_covar=1e-6,
    )
    return gmm.fit_predict(features_flat)


def run_spectral(
    features_flat: np.ndarray,
    n_clusters: int,
    verbose: bool = False,
    max_samples: int = 10000,
) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with Spectral Clustering (n_clusters={n_clusters})...")

    n_samples = features_flat.shape[0]
    if n_samples > max_samples:
        if verbose:
            print(
                f"   Subsampling {max_samples} of {n_samples} pixels for affinity matrix..."
            )
        np.random.seed(42)
        subsample_idx = np.random.choice(n_samples, max_samples, replace=False)
        features_subsample = features_flat[subsample_idx]
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=10,
            random_state=42,
            assign_labels="kmeans",
        )
        subsample_labels = spectral.fit_predict(features_subsample)
        if verbose:
            print(f"   Propagating labels to all {n_samples} pixels...")
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(features_subsample)
        _, indices = nn.kneighbors(features_flat)
        return subsample_labels[indices.flatten()]

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=10,
        random_state=42,
        assign_labels="kmeans",
    )
    return spectral.fit_predict(features_flat)


def cluster_features(
    features_flat: np.ndarray,
    method: str,
    n_clusters: int,
    H: int | None = None,
    W: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    method = (method or "kmeans").lower()
    if method == "gmm":
        return run_gmm(features_flat, n_clusters, verbose=verbose)
    if method == "spectral":
        return run_spectral(features_flat, n_clusters, verbose=verbose)
    if method == "dpmeans":
        return run_dpmeans(features_flat, n_clusters, verbose=verbose)
    if method == "spherical":
        return run_spherical_kmeans(features_flat, n_clusters, verbose=verbose)
    if method == "potts":
        if H is None or W is None:
            raise ValueError(
                "Potts regularization requires H and W for the feature grid."
            )
        return run_potts_kmeans(
            features_flat, n_clusters, H, W, verbose=verbose
        ).reshape(-1)
    return run_kmeans(features_flat, n_clusters, verbose=verbose)
