"""Clustering utilities for tree_seg segmentation."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def run_kmeans(features_flat: np.ndarray, n_clusters: int, verbose: bool = False) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with K-means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    return kmeans.fit_predict(features_flat)


def run_spherical_kmeans(features_flat: np.ndarray, n_clusters: int, verbose: bool = False) -> np.ndarray:
    if verbose:
        print(f"🎯 Clustering with spherical k-means (cosine) (k={n_clusters})...")
    norms = np.linalg.norm(features_flat, axis=1, keepdims=True) + 1e-8
    features_norm = features_flat / norms
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
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
