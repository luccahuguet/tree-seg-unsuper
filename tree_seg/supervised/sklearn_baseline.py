"""Supervised baselines on DINOv3 features (sklearn + PyTorch heads)."""

from pathlib import Path
from typing import Optional

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from tree_seg.evaluation.benchmark import BenchmarkResults
from tree_seg.supervised.utils import (
    _evaluate_predictions,
    _make_spinner,
    _maybe_append_xy,
    _stop_progress,
    flatten_features_labels,
    predict_linear_head,
    train_linear_head,
    load_and_extract_features,
)


def train_sklearn_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    ignore_index: Optional[int] = None,
    max_samples: int = 100_000,
    max_iter: int = 3000,
    tol: float = 1e-3,
) -> LogisticRegression:
    """Train a per-pixel logistic regression classifier."""
    X, y = flatten_features_labels(
        features,
        labels,
        ignore_index=ignore_index,
        max_samples=max_samples,
    )

    print(f"Training on {len(y):,} pixels...")
    clf = LogisticRegression(
        max_iter=max_iter,
        tol=tol,
        solver="lbfgs",
        n_jobs=1,
        verbose=0,
    )
    clf.fit(X, y)
    return clf


def predict_sklearn(
    clf: LogisticRegression,
    features: np.ndarray,
) -> np.ndarray:
    """Predict per-pixel classes for a single image."""
    H, W, D = features.shape
    X = features.reshape(-1, D)
    pred = clf.predict(X)
    return pred.reshape(H, W)


def evaluate_sklearn_baseline(
    dataset_path: Path | str,
    model_name: str = "base",
    stride: int = 4,
    ignore_index: Optional[int] = None,
    max_samples: int = 100_000,
    verbose: bool = True,
    num_samples: Optional[int] = None,
    train_ratio: float = 1.0,
) -> BenchmarkResults:
    """
    Full pipeline: extract features, train sklearn LR, evaluate.

    Args:
        dataset_path: Path to dataset directory
        model_name: DINOv3 model to use (base, large, giant)
        stride: Feature extraction stride
        ignore_index: Label to ignore in GT masks
        max_samples: Max training samples
        verbose: Print progress
        num_samples: Limit number of samples to process
    """
    (
        config,
        device,
        all_features,
        all_masks_resized,
        num_classes,
        dataset_name,
        ignore_idx,
    ) = load_and_extract_features(
        dataset_path=dataset_path,
        model_name=model_name,
        stride=stride,
        ignore_index=ignore_index,
        verbose=verbose,
        num_samples=num_samples,
        train_ratio=train_ratio,
    )

    if verbose:
        print("\nTraining supervised classifier...")
    train_progress, _ = _make_spinner("Training Logistic Regression", verbose)

    clf = train_sklearn_classifier(
        all_features,
        all_masks_resized,
        ignore_idx,
        max_samples,
        max_iter=3000,
        tol=1e-3,
    )

    _stop_progress(train_progress)

    if verbose:
        print("\nEvaluating predictions...")

    return _evaluate_predictions(
        features=list(all_features),
        masks=list(all_masks_resized),
        num_classes=num_classes,
        ignore_index=ignore_idx,
        method_name="supervised-sklearn",
        dataset_name=dataset_name,
        config=config,
        predict_fn=lambda feat: predict_sklearn(clf, feat),
        verbose=verbose,
    )


def train_mlp_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    ignore_index: Optional[int] = None,
    max_samples: int = 1_000_000,
    max_iter: int = 200,
    lr: float = 1e-3,
    hidden_sizes: tuple[int, ...] = (2048, 1024, 512),
    alpha: float = 1e-4,
    val_split: float = 0.1,
    patience: int = 10,
) -> tuple[MLPClassifier, Optional[StandardScaler]]:
    """Train a nonlinear MLP classifier on flattened patch features."""
    X, y = flatten_features_labels(
        features,
        labels,
        ignore_index=ignore_index,
        max_samples=max_samples,
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    use_early_stop = val_split > 0 and patience > 0
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        max_iter=max_iter,
        early_stopping=use_early_stop,
        validation_fraction=val_split if use_early_stop else 0.1,
        n_iter_no_change=max(1, patience) if use_early_stop else 10,
        alpha=alpha,
        verbose=False,
        random_state=0,
    )
    clf.fit(X, y)
    return clf, scaler


def predict_mlp(
    clf: MLPClassifier,
    features: np.ndarray,
    scaler: Optional[StandardScaler],
) -> np.ndarray:
    """Predict per-pixel classes for a single image using MLP head."""
    H, W, D = features.shape
    X = features.reshape(-1, D)
    if scaler is not None:
        X = scaler.transform(X)
    pred = clf.predict(X)
    return pred.reshape(H, W)


def evaluate_mlp_baseline(
    dataset_path: Path | str,
    model_name: str = "base",
    stride: int = 4,
    ignore_index: Optional[int] = None,
    verbose: bool = True,
    num_samples: Optional[int] = None,
    max_samples: int = 2_000_000,
    max_iter: int = 400,
    lr: float = 1e-3,
    hidden_dim: int = 2048,
    use_xy: bool = False,
    val_split: float = 0.1,
    patience: int = 10,
    train_ratio: float = 1.0,
) -> BenchmarkResults:
    """Train/evaluate a sklearn MLPClassifier head on DINOv3 features."""
    (
        config,
        device,
        train_features,
        train_masks,
        holdout_features,
        holdout_masks,
        num_classes,
        dataset_name,
        ignore_idx,
    ) = load_and_extract_features(
        dataset_path=dataset_path,
        model_name=model_name,
        stride=stride,
        ignore_index=ignore_index,
        verbose=verbose,
        num_samples=num_samples,
        train_ratio=train_ratio,
    )

    train_progress, _ = _make_spinner("Training MLP head", verbose)

    # Optionally append XY coords to each patch vector
    features_for_head = _maybe_append_xy(train_features, use_xy)

    clf, scaler = train_mlp_classifier(
        features_for_head,
        train_masks,
        ignore_index=ignore_idx,
        max_samples=max_samples,
        max_iter=max_iter,
        lr=lr,
        hidden_sizes=(hidden_dim, max(hidden_dim // 2, 1)),
        val_split=val_split,
        patience=patience,
    )

    _stop_progress(train_progress)

    if verbose:
        print("\nEvaluating predictions...")

    holdout_eval = (
        list(_maybe_append_xy(holdout_features, use_xy))
        if len(holdout_features) > 0
        else []
    )
    all_eval_features = list(features_for_head) + holdout_eval
    all_eval_masks = list(train_masks) + (
        list(holdout_masks) if len(holdout_masks) > 0 else []
    )
    sample_ids = [
        f"img_{idx:03d}" + ("_holdout" if idx >= len(train_masks) else "")
        for idx in range(len(all_eval_features))
    ]

    return _evaluate_predictions(
        features=all_eval_features,
        masks=all_eval_masks,
        num_classes=num_classes,
        ignore_index=ignore_idx,
        method_name="supervised-mlp",
        dataset_name=dataset_name,
        config=config,
        predict_fn=lambda feat: predict_mlp(clf, feat, scaler),
        verbose=verbose,
        sample_ids=sample_ids,
    )


def evaluate_linear_head(
    dataset_path: Path | str,
    model_name: str = "base",
    stride: int = 4,
    ignore_index: Optional[int] = None,
    verbose: bool = True,
    num_samples: Optional[int] = None,
    epochs: int = 100,
    batch_size: int = 4096,
    max_patches: int = 1_000_000,
    lr: float = 1e-3,
    patience: int = 5,
    val_split: float = 0.1,
    hidden_dim: int = 2048,
    dropout: float = 0.1,
    use_xy: bool = False,
    train_ratio: float = 1.0,
) -> BenchmarkResults:
    """Train/evaluate a tiny PyTorch LinearHead on DINOv3 features."""
    (
        config,
        device,
        train_features,
        train_masks,
        holdout_features,
        holdout_masks,
        num_classes,
        dataset_name,
        ignore_idx,
    ) = load_and_extract_features(
        dataset_path=dataset_path,
        model_name=model_name,
        stride=stride,
        ignore_index=ignore_index,
        verbose=verbose,
        num_samples=num_samples,
        train_ratio=train_ratio,
    )

    if verbose:
        print("\nTraining supervised classifier...")
    train_progress, _ = _make_spinner("Preparing LinearHead", verbose)

    bar = (
        Progress(
            TextColumn("[bold magenta]Training LinearHead[/bold magenta]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        if verbose
        else None
    )
    if bar:
        bar.start()

    model = train_linear_head(
        train_features,
        train_masks,
        num_classes=num_classes,
        ignore_index=ignore_idx,
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=1e-4,
        max_patches=max_patches,
        progress=bar,
        patience=patience,
        val_split=val_split,
        use_class_weights=True,
        hidden_dims=(hidden_dim, max(hidden_dim // 2, 1), max(hidden_dim // 4, 1)),
        dropout=dropout,
        use_xy=use_xy,
    )

    _stop_progress(train_progress)
    if verbose and bar:
        for task in bar.tasks:
            if not task.finished:
                bar.update(task.id, advance=task.total)
        bar.stop()

    if verbose:
        print("\nEvaluating predictions...")

    all_eval_features = list(train_features) + (
        list(holdout_features) if len(holdout_features) > 0 else []
    )
    all_eval_masks = list(train_masks) + (
        list(holdout_masks) if len(holdout_masks) > 0 else []
    )
    sample_ids = [
        f"img_{idx:03d}" + ("_holdout" if idx >= len(train_masks) else "")
        for idx in range(len(all_eval_features))
    ]

    return _evaluate_predictions(
        features=all_eval_features,
        masks=all_eval_masks,
        num_classes=num_classes,
        ignore_index=ignore_idx,
        method_name="supervised-linear-head",
        dataset_name=dataset_name,
        config=config,
        predict_fn=lambda feat: predict_linear_head(model, feat, device),
        verbose=verbose,
        sample_ids=sample_ids,
    )
