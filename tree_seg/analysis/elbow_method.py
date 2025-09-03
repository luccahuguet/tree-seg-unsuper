"""
Elbow method for automatic K selection in tree segmentation clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib
from sklearn.cluster import KMeans


def find_optimal_k_elbow(features_flat, k_range=(3, 10), elbow_threshold=3.5):
    """
    Find optimal K using enhanced elbow method optimized for tree segmentation.

    Args:
        features_flat: Flattened feature array
        k_range: Tuple of (min_k, max_k) - default (3,10) optimized for tree species
        elbow_threshold: Percentage threshold for diminishing returns (3.5 = 3.5%, lower = more sensitive)

    Returns:
        optimal_k: Best number of clusters
        scores: Dictionary with analysis results
    """
    min_k, max_k = k_range
    k_values = list(range(min_k, max_k + 1))
    wcss = []

    # Check for NaN values and clean if necessary
    if np.isnan(features_flat).any():
        print("⚠️  Warning: Features contain NaN values (likely from random weights)")
        print("🧹 Cleaning NaN values for clustering...")
        
        # Replace NaN with zeros or mean values
        features_flat = np.nan_to_num(features_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        # If all features are zeros, add some small random noise for clustering
        if np.all(features_flat == 0):
            print("🎲 Adding small random noise to zero features...")
            features_flat += np.random.normal(0, 0.001, features_flat.shape)

    print(f"🔍 Testing K values from {min_k} to {max_k} using elbow method...")

    # Calculate WCSS for each K
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(features_flat)
        wcss.append(kmeans.inertia_)
        print(f"   K={k}: WCSS={wcss[-1]:.2f}")

    # Percentage decrease threshold method
    wcss_array = np.array(wcss)
    
    # Calculate percentage decrease between consecutive K values
    pct_decrease = []
    for i in range(1, len(wcss_array)):
        pct = (wcss_array[i-1] - wcss_array[i]) / wcss_array[i-1] * 100
        pct_decrease.append(pct)

    # Find where percentage decrease drops below threshold (diminishing returns)
    threshold_idx = len(pct_decrease) - 1  # Default to last K if none found
    for i, pct in enumerate(pct_decrease):
        if pct < elbow_threshold:  # Direct percentage comparison
            threshold_idx = i
            break

    # Fix index mapping: pct_decrease[i] represents transition from k_values[i] to k_values[i+1]
    # So we want k_values[threshold_idx + 1] as the optimal K
    elbow_idx = threshold_idx + 1

    # Safety bounds
    elbow_idx = max(0, min(int(elbow_idx), len(k_values) - 1))
    optimal_k = k_values[elbow_idx]

    # Validate result (safety override removed for investigation)
    if optimal_k < 3:
        print(f"⚠️  Optimal K={optimal_k} seems too low for tree species, using K=3")
        optimal_k = 3
        elbow_idx = k_values.index(3) if 3 in k_values else 0
    # Note: Upper bound safety override removed to investigate threshold sensitivity

    print(f"📊 Elbow method suggests optimal K = {optimal_k}")

    return optimal_k, {
        'k_values': k_values,
        'wcss': wcss,
        'elbow_idx': elbow_idx,
        'optimal_k': optimal_k,
        'pct_decrease': pct_decrease,
        'method': 'elbow'
    }


def plot_elbow_analysis(scores, output_dir, output_prefix, elbow_threshold=3.0, 
                        model_name=None, version=None, stride=None, n_clusters=None, 
                        auto_k=True, image_path=None):
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

    # Generate config-based filename if all parameters are provided
    if all([model_name, version, stride, image_path]) and n_clusters:
        from ..utils.config import parse_model_info
        
        # Generate filename hash
        filename = os.path.basename(image_path)
        file_hash = hashlib.sha1(filename.encode()).hexdigest()[:4]
        
        # Parse model info
        _, nickname, _ = parse_model_info(model_name)
        model_nick = nickname.lower()
        
        # Format version (replace dots with hyphens)
        version_str = version.replace(".", "-")
        
        # Build config-based filename
        components = [file_hash, version_str, model_nick, f"str{stride}"]
        
        if auto_k and elbow_threshold is not None:
            # Format elbow threshold without float artifacts (e.g., 3.5)
            et_clean = (f"{elbow_threshold:.2f}".rstrip('0').rstrip('.')).replace('.', '-')
            et_str = f"et{et_clean}"
            components.append(et_str)
        
        # Always add the actual K value used
        components.append(f"k{n_clusters}")
        
        config_filename = "_".join(components) + "_elbow_analysis"
        plot_path = os.path.join(output_dir, f"{config_filename}.png")
    else:
        # Fallback to old naming
        plot_path = os.path.join(output_dir, f"{output_prefix}_elbow_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 Saved elbow analysis: {plot_path}")
    return plot_path 