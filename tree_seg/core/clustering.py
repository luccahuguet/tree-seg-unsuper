"""Clustering utilities for tree_seg segmentation."""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import hdbscan


def _make_kmeans(n_clusters: int) -> KMeans:
    """Factory to enforce consistent KMeans defaults across variants."""
    return KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")


def run_kmeans(features_flat: np.ndarray, n_clusters: int, verbose: bool = False) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with K-means (k={n_clusters})...")
    kmeans = _make_kmeans(n_clusters)
    return kmeans.fit_predict(features_flat)


def run_spherical_kmeans(features_flat: np.ndarray, n_clusters: int, verbose: bool = False) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with spherical k-means (cosine) (k={n_clusters})...")
    norms = np.linalg.norm(features_flat, axis=1, keepdims=True) + 1e-8
    features_norm = features_flat / norms
    kmeans = _make_kmeans(n_clusters)
    return kmeans.fit_predict(features_norm)


def run_dpmeans(features_flat: np.ndarray, n_clusters: int, max_iter: int = 10, verbose: bool = False) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with DP-means (auto K via lambda; init k={n_clusters})...")
    sample_size = min(5000, features_flat.shape[0])
    idx = np.random.choice(features_flat.shape[0], sample_size, replace=False)
    sample = features_flat[idx]
    pdists = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1)
    median_dist = np.median(pdists)
    lam = 0.7 * median_dist

    centers = [features_flat[np.random.choice(features_flat.shape[0])]]
    assignments = np.zeros(features_flat.shape[0], dtype=int)
    for _ in range(max_iter):
        dists = np.stack([np.sum((features_flat - c) ** 2, axis=1) for c in centers], axis=1)
        min_dists = dists.min(axis=1)
        assignments = dists.argmin(axis=1)
        new_centers = features_flat[min_dists > lam]
        if len(new_centers):
            centers.extend(list(new_centers))
        for k in range(len(centers)):
            mask = assignments == k
            if mask.any():
                centers[k] = features_flat[mask].mean(axis=0)
    return assignments


def run_potts_kmeans(features_flat: np.ndarray, n_clusters: int, H: int, W: int, beta: float = 0.5, iters: int = 2, verbose: bool = False) -> np.ndarray:
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
                data_cost = np.sum((features_flat[y * W + x] - kmeans.cluster_centers_) ** 2, axis=1)
                smooth_cost = beta * (len(neighbors) - neighbor_votes)
                total_cost = data_cost + smooth_cost
                smoothed[y, x] = int(np.argmin(total_cost))
    return smoothed.reshape(-1)


def run_gmm(features_flat: np.ndarray, n_clusters: int, verbose: bool = False) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with GMM (n_components={n_clusters})...")
    gmm = GaussianMixture(
        n_components=n_clusters,
        random_state=42,
        covariance_type="diag",
        reg_covar=1e-6,
    )
    return gmm.fit_predict(features_flat)


def run_spectral(features_flat: np.ndarray, n_clusters: int, verbose: bool = False, max_samples: int = 10000) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with Spectral Clustering (n_clusters={n_clusters})...")

    n_samples = features_flat.shape[0]
    if n_samples > max_samples:
        if verbose:
            print(f"   Subsampling {max_samples} of {n_samples} pixels for affinity matrix...")
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


def run_hdbscan(features_flat: np.ndarray, n_clusters: int, verbose: bool = False, max_samples: int = 10000) -> np.ndarray:
    if verbose:
        print("🎯 Clustering with HDBSCAN (automatic K detection)...")

    def _cluster_and_resolve(data: np.ndarray) -> np.ndarray:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            cluster_selection_epsilon=0.0,
            metric="euclidean",
        )
        labels_local = clusterer.fit_predict(data)
        unique_labels = np.unique(labels_local[labels_local >= 0])
        return labels_local, unique_labels

    n_samples = features_flat.shape[0]
    if n_samples > max_samples:
        if verbose:
            print(f"   Subsampling {max_samples} of {n_samples} pixels...")
        np.random.seed(42)
        subsample_idx = np.random.choice(n_samples, max_samples, replace=False)
        features_subsample = features_flat[subsample_idx]
        subsample_labels, unique_labels = _cluster_and_resolve(features_subsample)
        if verbose:
            print(f"   HDBSCAN found {len(unique_labels)} clusters")
        if len(unique_labels) == 0:
            if verbose:
                print(f"   ⚠️  HDBSCAN found no clusters, falling back to K-means (k={n_clusters})")
            return run_kmeans(features_flat, n_clusters, verbose=False)
        if verbose:
            print(f"   Propagating labels to all {n_samples} pixels...")
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(features_subsample[subsample_labels >= 0])
        _, indices = nn.kneighbors(features_flat)
        return subsample_labels[subsample_labels >= 0][indices.flatten()]

    labels, unique_labels = _cluster_and_resolve(features_flat)
    if verbose:
        print(f"   HDBSCAN found {len(unique_labels)} clusters")
    if len(unique_labels) == 0:
        if verbose:
            print(f"   ⚠️  HDBSCAN found no clusters, falling back to K-means (k={n_clusters})")
        return run_kmeans(features_flat, n_clusters, verbose=False)

    if np.any(labels == -1):
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(features_flat[labels >= 0])
        _, indices = nn.kneighbors(features_flat[labels == -1])
        labels[labels == -1] = labels[labels >= 0][indices.flatten()]
    return labels


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
    if method == "hdbscan":
        return run_hdbscan(features_flat, n_clusters, verbose=verbose)
    if method == "dpmeans":
        return run_dpmeans(features_flat, n_clusters, verbose=verbose)
    if method == "spherical":
        return run_spherical_kmeans(features_flat, n_clusters, verbose=verbose)
    if method == "potts":
        if H is None or W is None:
            raise ValueError("Potts regularization requires H and W for the feature grid.")
        return run_potts_kmeans(features_flat, n_clusters, H, W, verbose=verbose).reshape(-1)
    return run_kmeans(features_flat, n_clusters, verbose=verbose)
