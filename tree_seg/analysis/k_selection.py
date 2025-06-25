"""
Automatic K Selection for K-means Clustering
Implements elbow method and silhouette analysis for optimal cluster number selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict
import os

def find_optimal_k_elbow(features_flat: np.ndarray, k_range: Tuple[int, int] = (2, 15)) -> Tuple[int, Dict]:
    """
    Find optimal number of clusters using the elbow method.

    Args:
        features_flat: Flattened feature array (n_samples, n_features)
        k_range: Tuple of (min_k, max_k) to test

    Returns:
        optimal_k: Best number of clusters
        scores: Dictionary with WCSS values and elbow point info
    """
    min_k, max_k = k_range
    k_values = range(min_k, max_k + 1)
    wcss = []

    print(f"Testing K values from {min_k} to {max_k} using elbow method...")

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(features_flat)
        wcss.append(kmeans.inertia_)
        print(f"K={k}: WCSS={wcss[-1]:.2f}")

    # Find elbow point using second derivative
    wcss_array = np.array(wcss)
    first_diff = np.diff(wcss_array)
    second_diff = np.diff(first_diff)

    # Find the point with maximum second derivative (sharpest bend)
    elbow_idx = np.argmax(np.abs(second_diff)) + 1  # +1 because we lost 2 points in diff
    optimal_k = k_values[elbow_idx]

    scores = {
        'k_values': list(k_values),
        'wcss': wcss,
        'elbow_idx': elbow_idx,
        'optimal_k': optimal_k,
        'method': 'elbow'
    }

    print(f"Elbow method suggests optimal K = {optimal_k}")
    return optimal_k, scores

def find_optimal_k_silhouette(features_flat: np.ndarray, k_range: Tuple[int, int] = (2, 15)) -> Tuple[int, Dict]:
    """
    Find optimal number of clusters using silhouette analysis.

    Args:
        features_flat: Flattened feature array (n_samples, n_features)
        k_range: Tuple of (min_k, max_k) to test

    Returns:
        optimal_k: Best number of clusters
        scores: Dictionary with silhouette scores
    """
    min_k, max_k = k_range
    k_values = range(min_k, max_k + 1)
    silhouette_scores = []

    print(f"Testing K values from {min_k} to {max_k} using silhouette analysis...")

    for k in k_values:
        if k == 1:
            silhouette_scores.append(0)  # Silhouette not defined for k=1
            continue

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(features_flat)

        # Calculate silhouette score
        try:
            silhouette_avg = silhouette_score(features_flat, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"K={k}: Silhouette Score={silhouette_avg:.3f}")
        except Exception as e:
            print(f"K={k}: Error calculating silhouette score - {e}")
            silhouette_scores.append(0)

    # Find K with highest silhouette score
    optimal_k = k_values[np.argmax(silhouette_scores)]

    scores = {
        'k_values': list(k_values),
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k,
        'method': 'silhouette'
    }

    print(f"Silhouette analysis suggests optimal K = {optimal_k}")
    return optimal_k, scores

def find_optimal_k(features_flat: np.ndarray, k_range: Tuple[int, int] = (2, 15),
                  method: str = 'elbow') -> Tuple[int, Dict]:
    """
    Find optimal number of clusters using specified method.

    Args:
        features_flat: Flattened feature array (n_samples, n_features)
        k_range: Tuple of (min_k, max_k) to test
        method: 'elbow', 'silhouette', or 'both'

    Returns:
        optimal_k: Best number of clusters
        scores: Dictionary with evaluation metrics
    """
    if method == 'elbow':
        return find_optimal_k_elbow(features_flat, k_range)
    elif method == 'silhouette':
        return find_optimal_k_silhouette(features_flat, k_range)
    elif method == 'both':
        # Run both methods and return the one with better justification
        elbow_k, elbow_scores = find_optimal_k_elbow(features_flat, k_range)
        silhouette_k, silhouette_scores = find_optimal_k_silhouette(features_flat, k_range)

        # For now, prefer silhouette if it's reasonable, otherwise use elbow
        if silhouette_k >= 2 and silhouette_k <= max(k_range):
            print(f"Both methods suggest: Elbow={elbow_k}, Silhouette={silhouette_k}")
            print("Using silhouette result as it's more quantitative.")
            return silhouette_k, silhouette_scores
        else:
            print(f"Silhouette method failed, using elbow result: K={elbow_k}")
            return elbow_k, elbow_scores
    else:
        raise ValueError(f"Unknown method: {method}. Use 'elbow', 'silhouette', or 'both'")

def plot_k_selection_analysis(scores: Dict, output_dir: str, output_prefix: str):
    """
    Plot K selection analysis and save to file.

    Args:
        scores: Dictionary with K selection results
        output_dir: Directory to save plots
        output_prefix: Prefix for output filenames
    """
    method = scores['method']
    k_values = scores['k_values']

    if method == 'elbow':
        wcss = scores['wcss']
        optimal_k = scores['optimal_k']
        elbow_idx = scores['elbow_idx']

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, wcss, 'bo-', linewidth=2, markersize=8)
        plt.plot(k_values[elbow_idx], wcss[elbow_idx], 'ro', markersize=12, label=f'Optimal K = {optimal_k}')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title('Elbow Method for Optimal K Selection')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{output_prefix}_elbow_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved elbow analysis plot: {plot_path}")

    elif method == 'silhouette':
        silhouette_scores = scores['silhouette_scores']
        optimal_k = scores['optimal_k']

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
        optimal_idx = k_values.index(optimal_k)
        plt.plot(k_values[optimal_idx], silhouette_scores[optimal_idx], 'ro', markersize=12, label=f'Optimal K = {optimal_k}')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis for Optimal K Selection')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{output_prefix}_silhouette_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved silhouette analysis plot: {plot_path}")

    elif method == 'both':
        # Plot both methods side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Elbow plot
        wcss = scores.get('wcss', [])
        if wcss:
            elbow_k = scores.get('optimal_k_elbow', scores['optimal_k'])
            elbow_idx = scores.get('elbow_idx', 0)
            ax1.plot(k_values, wcss, 'bo-', linewidth=2, markersize=8)
            ax1.plot(k_values[elbow_idx], wcss[elbow_idx], 'ro', markersize=12, label=f'Elbow K = {elbow_k}')
            ax1.set_xlabel('Number of Clusters (K)')
            ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
            ax1.set_title('Elbow Method')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

        # Silhouette plot
        silhouette_scores = scores.get('silhouette_scores', [])
        if silhouette_scores:
            silhouette_k = scores.get('optimal_k_silhouette', scores['optimal_k'])
            optimal_idx = k_values.index(silhouette_k) if silhouette_k in k_values else 0
            ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
            ax2.plot(k_values[optimal_idx], silhouette_scores[optimal_idx], 'ro', markersize=12, label=f'Silhouette K = {silhouette_k}')
            ax2.set_xlabel('Number of Clusters (K)')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Analysis')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{output_prefix}_k_selection_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved K selection analysis plot: {plot_path}")

def validate_k_selection(optimal_k: int, k_range: Tuple[int, int], method: str) -> int:
    """
    Validate and potentially adjust the selected K value.

    Args:
        optimal_k: Selected optimal K
        k_range: Valid K range
        method: Method used for selection

    Returns:
        validated_k: Validated K value
    """
    min_k, max_k = k_range

    if optimal_k < min_k:
        print(f"Warning: Selected K={optimal_k} is below minimum {min_k}. Using K={min_k}")
        return min_k
    elif optimal_k > max_k:
        print(f"Warning: Selected K={optimal_k} is above maximum {max_k}. Using K={max_k}")
        return max_k
    else:
        return optimal_k