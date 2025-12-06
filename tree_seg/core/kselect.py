"""Helpers for K selection and PCA preprocessing."""

from __future__ import annotations

import torch


def maybe_run_pca(features_flat, use_pca: bool, pca_dim: int | None, verbose: bool):
    """Optionally apply PCA to reduce feature dimensionality."""
    effective_pca_dim = None
    if pca_dim is not None and pca_dim > 0:
        effective_pca_dim = min(pca_dim, features_flat.shape[-1])
    elif use_pca and features_flat.shape[-1] > 128:
        effective_pca_dim = 128

    if effective_pca_dim is not None and effective_pca_dim < features_flat.shape[-1]:
        if verbose:
            print(f"Running PCA to {effective_pca_dim} dims...")
        features_flat_tensor = torch.tensor(features_flat, dtype=torch.float32)
        mean = features_flat_tensor.mean(dim=0)
        features_flat_centered = features_flat_tensor - mean
        _U, _S, V = torch.pca_lowrank(features_flat_centered, q=effective_pca_dim, center=False)
        features_flat = (features_flat_centered @ V[:, :effective_pca_dim]).numpy()
        if verbose:
            print(f"PCA-reduced features shape: {features_flat.shape}")
    else:
        if verbose:
            print(f"Using {features_flat.shape[-1]}-D features (no PCA)")

    return features_flat


def clean_features(features_flat, verbose: bool):
    """Clean NaNs/Infs and add noise if everything collapses to zero."""
    if not hasattr(features_flat, "reshape"):
        return features_flat

    if (features_flat != features_flat).any():  # NaN check
        if verbose:
            print("⚠️  Warning: Features contain NaN values for main clustering")
            print("🧹 Cleaning NaN values...")
        import numpy as np

        features_flat = np.nan_to_num(features_flat, nan=0.0, posinf=0.0, neginf=0.0)
        if np.all(features_flat == 0):
            if verbose:
                print("🎲 Adding small random noise to zero features...")
            features_flat += np.random.normal(0, 0.001, features_flat.shape)
    return features_flat
