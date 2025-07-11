"""
Elbow method for automatic K selection in tree segmentation clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans


def find_optimal_k_elbow(features_flat, k_range=(3, 10), elbow_threshold=3.0):
    """
    Find optimal K using enhanced elbow method optimized for tree segmentation.

    Args:
        features_flat: Flattened feature array
        k_range: Tuple of (min_k, max_k) - default (3,10) optimized for tree species
        elbow_threshold: Percentage threshold for diminishing returns (lower = more sensitive)

    Returns:
        optimal_k: Best number of clusters
        scores: Dictionary with analysis results
    """
    min_k, max_k = k_range
    k_values = list(range(min_k, max_k + 1))
    wcss = []

    print(f"🔍 Testing K values from {min_k} to {max_k} using elbow method...")

    # Calculate WCSS for each K
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(features_flat)
        wcss.append(kmeans.inertia_)
        print(f"   K={k}: WCSS={wcss[-1]:.2f}")

    # Enhanced elbow detection with multiple methods
    wcss_array = np.array(wcss)

    # Method 1: Second derivative (curvature)
    if len(wcss_array) >= 3:
        first_diff = np.diff(wcss_array)
        second_diff = np.diff(first_diff)
        curvature_idx = np.argmax(np.abs(second_diff)) + 1  # +1 due to diff operations
    else:
        curvature_idx = 0

    # Method 2: Percentage decrease threshold
    pct_decrease = []
    for i in range(1, len(wcss_array)):
        pct = (wcss_array[i-1] - wcss_array[i]) / wcss_array[i-1] * 100
        pct_decrease.append(pct)

    # Find where percentage decrease drops below threshold (diminishing returns)
    threshold_idx = 0
    for i, pct in enumerate(pct_decrease):
        if pct < elbow_threshold:  # Less than threshold% improvement
            threshold_idx = i
            break

    # Choose the more conservative estimate (earlier elbow)
    elbow_idx = min(int(curvature_idx), int(threshold_idx)) if threshold_idx > 0 else int(curvature_idx)

    # Safety bounds
    elbow_idx = max(0, min(int(elbow_idx), len(k_values) - 1))
    optimal_k = k_values[elbow_idx]

    # Validate result
    if optimal_k < 3:
        print(f"⚠️  Optimal K={optimal_k} seems too low for tree species, using K=3")
        optimal_k = 3
        elbow_idx = k_values.index(3) if 3 in k_values else 0
    elif optimal_k > 8:
        print(f"⚠️  Optimal K={optimal_k} seems high for typical tree species, using K=8")
        optimal_k = min(8, max(k_values))
        elbow_idx = k_values.index(optimal_k)

    print(f"📊 Elbow method suggests optimal K = {optimal_k}")

    return optimal_k, {
        'k_values': k_values,
        'wcss': wcss,
        'elbow_idx': elbow_idx,
        'optimal_k': optimal_k,
        'pct_decrease': pct_decrease,
        'method': 'elbow'
    }


def plot_elbow_analysis(scores, output_dir, output_prefix, elbow_threshold=3.0):
    """
    Create enhanced elbow plot with additional analysis information.
    """
    k_values = scores['k_values']
    wcss = scores['wcss']
    elbow_idx = scores['elbow_idx']
    optimal_k = scores['optimal_k']
    pct_decrease = scores.get('pct_decrease', [])

    # Create subplot with elbow curve and percentage decrease
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Main elbow plot
    ax1.plot(k_values, wcss, 'bo-', linewidth=3, markersize=10, alpha=0.7)
    ax1.plot(k_values[elbow_idx], wcss[elbow_idx], 'ro', markersize=15,
             label=f'Optimal K = {optimal_k}', zorder=5)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    ax1.set_title('Tree Species Clustering - Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    # Add annotations
    ax1.annotate(f'Selected K = {optimal_k}',
                xy=(k_values[elbow_idx], wcss[elbow_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Percentage decrease plot
    if pct_decrease:
        ax2.plot(k_values[1:], pct_decrease, 'go-', linewidth=2, markersize=8)
        ax2.axhline(y=elbow_threshold, color='r', linestyle='--', alpha=0.7, label=f'{elbow_threshold}% Threshold')
        ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax2.set_ylabel('WCSS Improvement (%)', fontsize=12)
        ax2.set_title('Diminishing Returns Analysis', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f"{output_prefix}_elbow_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 Saved elbow analysis: {plot_path}")
    return plot_path 