"""Supervised baselines on DINOv3 features (sklearn + PyTorch heads)."""

import copy
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tree_seg.core.features import extract_features
from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import BenchmarkResults
from tree_seg.evaluation.runner import detect_dataset_type, load_dataset
from tree_seg.models.preprocessing import init_model_and_preprocess
from tree_seg.supervised.data import load_dataset as load_supervised_dataset
from tree_seg.supervised.data import resize_masks_to_features
from tree_seg.supervised.utils import (
    _evaluate_predictions,
    _make_bar_progress,
    _make_spinner,
    _maybe_append_xy,
    _remap_masks_contiguous,
    _stop_progress,
    flatten_features_labels,
)


def _set_torch_threads():
    """Leave one core idle to reduce contention on CPU runs."""
    try:
        cpu_total = os.cpu_count() or 1
        torch.set_num_threads(max(1, cpu_total - 1))
        torch.set_num_interop_threads(max(1, cpu_total - 1))
    except Exception:
        pass


def _load_and_extract_features(
    dataset_path: Path | str,
    model_name: str,
    stride: int,
    ignore_index: Optional[int],
    verbose: bool,
    num_samples: Optional[int] = None,
    train_ratio: float = 1.0,
):
    """Shared feature extraction + mask resizing for supervised heads."""
    dataset_path = Path(dataset_path)
    config = Config(
        model_name=model_name,
        stride=stride,
        n_clusters=1,
        clustering_method="kmeans",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_torch_threads()
    if verbose:
        print(f"Using device: {device}")

    model, preprocess = init_model_and_preprocess(
        config.model_display_name, stride, device
    )

    dataset = None
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    dataset_name = dataset_path.name
    ignore_idx = ignore_index

    detected_type = detect_dataset_type(dataset_path)
    try:
        # Prefer custom downsampled loader for known large datasets
        if detected_type in {"fortress", "isprs"}:
            raise RuntimeError("Use custom loader for large datasets")

        dataset, detected_type = load_dataset(
            dataset_path, detect_dataset_type(dataset_path)
        )
        dataset_name = (
            dataset.dataset_path.name if hasattr(dataset, "dataset_path") else dataset_name
        )
        ignore_idx = getattr(dataset, "IGNORE_INDEX", ignore_idx)
    except Exception as exc:
        if verbose:
            print(f"Using custom dataset loader (reason: {exc})...")
        max_side = max(config.image_size * 2, 1024)
        images, masks, _class_names = load_supervised_dataset(
            dataset_path, max_side=max_side
        )
        dataset = None

    # Build a single iterable of (image, mask) pairs to avoid duplicated loops
    if dataset is not None:
        total_samples = len(dataset)
        sample_total = min(total_samples, num_samples) if num_samples else total_samples
        if verbose:
            print(
                f"Loaded dataset: {dataset_name} ({sample_total}/{total_samples} samples)"
            )
        pairs = [dataset[idx][:2] for idx in range(sample_total)]
    else:
        if num_samples:
            images = images[:num_samples]
            masks = masks[:num_samples]
        sample_total = len(images)
        if verbose:
            print(f"Loaded {sample_total} image/mask pairs")
        pairs = list(zip(images, masks))

    progress, feat_task = _make_bar_progress("Extracting features", sample_total, verbose)
    all_features = []
    all_masks = []

    for idx, (image, mask) in enumerate(pairs):
        _start = time.time()
        features_np, H, W = extract_features(
            image_np=image,
            model=model,
            preprocess=preprocess,
            stride=stride,
            device=device,
            use_attention_features=config.use_attention_features,
            use_multi_layer=config.use_multi_layer,
            layer_indices=config.layer_indices,
            feature_aggregation=config.feature_aggregation,
            use_pyramid=config.use_pyramid,
            pyramid_scales=config.pyramid_scales,
            pyramid_aggregation=config.pyramid_aggregation,
            verbose=False,
        )
        _ = time.time() - _start

        if progress and feat_task is not None:
            progress.update(
                feat_task,
                advance=1,
                description=(
                    f"[bold cyan]Extracting features[/bold cyan] ({idx+1}/{sample_total})"
                ),
            )

        all_features.append(features_np)
        all_masks.append(mask)

    _stop_progress(progress)

    all_features = np.stack(all_features)  # (N, H, W, D)

    if verbose:
        print(f"Resizing masks to feature resolution: {all_features.shape[1:3]}")

    all_masks_resized = resize_masks_to_features(
        all_masks, target_size=all_features.shape[1:3]
    )

    # Auto-detect ignore index if not provided (common: 255 for unlabeled)
    if ignore_index is None:
        if 255 in np.unique(all_masks_resized):
            ignore_idx = 255
        else:
            ignore_idx = None

    remapped_masks, num_classes, ignore_for_eval = _remap_masks_contiguous(
        all_masks_resized, ignore_idx
    )

    # Split into train/holdout sets
    n_total = len(all_features)
    n_train = max(1, int(n_total * train_ratio))
    train_features = all_features[:n_train]
    train_masks = remapped_masks[:n_train]
    holdout_features = all_features[n_train:] if n_train < n_total else []
    holdout_masks = remapped_masks[n_train:] if n_train < n_total else []

    return (
        config,
        device,
        train_features,
        train_masks,
        holdout_features,
        holdout_masks,
        num_classes,
        dataset_name,
        ignore_for_eval,
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
    ) = _load_and_extract_features(
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
    ) = _load_and_extract_features(
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


def train_linear_head(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int],
    device: torch.device,
    batch_size: int = 4096,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_patches: int = 1_000_000,
    progress: Optional[Progress] = None,
    val_split: float = 0.0,
    patience: int = 0,
    min_delta: float = 0.0,
    use_class_weights: bool = True,
    hidden_dims: tuple[int, ...] = (2048, 1024, 512),
    dropout: float = 0.1,
    use_xy: bool = False,
):
    """Train a simple Linear -> CrossEntropy head with optional early stopping."""
    features = _maybe_append_xy(features, use_xy)
    X, y = flatten_features_labels(
        features,
        labels,
        ignore_index=ignore_index,
        max_samples=max_patches,
    )

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y.astype(np.int64))

    # Train/val split
    if val_split > 0 and len(y_tensor) > 1:
        split = int(len(y_tensor) * (1 - val_split))
        indices = torch.randperm(len(y_tensor))
        train_idx, val_idx = indices[:split], indices[split:]
        train_ds = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_ds = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
    else:
        train_ds = TensorDataset(X_tensor, y_tensor)
        val_ds = None

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    layers = []
    layers.append(nn.LayerNorm(X.shape[1]))
    in_dim = X.shape[1]
    for hd in hidden_dims:
        layers.extend([nn.Linear(in_dim, hd), nn.ReLU(), nn.Dropout(dropout)])
        in_dim = hd
    layers.append(nn.Linear(in_dim, num_classes))
    model = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_class_weights:
        counts = torch.bincount(y_tensor, minlength=num_classes).float()
        weights = 1.0 / torch.clamp(counts, min=1.0)
        weights = weights / weights.sum() * num_classes
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    total_steps = epochs * len(loader)
    task_id = None
    if progress:
        task_id = progress.add_task(
            "[bold magenta]Training LinearHead[/bold magenta]",
            total=total_steps,
        )

    best_state = None
    best_val = float("inf")
    no_improve = 0

    model.train()
    for epoch in range(epochs):
        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            if progress and task_id is not None:
                progress.update(
                    task_id,
                    advance=1,
                    description=(
                        f"[bold magenta]Training LinearHead[/bold magenta] "
                        f"(epoch {epoch+1}/{epochs})"
                    ),
                )

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    logits = model(xb)
                    val_loss += loss_fn(logits, yb).item() * len(xb)
                    val_count += len(xb)
            val_loss /= max(1, val_count)
            model.train()

            if val_loss + min_delta < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if patience and no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def _predict_linear_head(
    model: nn.Module,
    feature_map: np.ndarray,
    device: torch.device,
):
    """Run linear head on a single feature map."""
    H, W, D = feature_map.shape
    feat = torch.from_numpy(feature_map.reshape(-1, D).astype(np.float32)).to(
        device, non_blocking=True
    )
    with torch.no_grad():
        logits = model(feat)
        pred = torch.argmax(logits, dim=1).cpu().numpy().reshape(H, W)
    return pred


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
    ) = _load_and_extract_features(
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
        predict_fn=lambda feat: _predict_linear_head(model, feat, device),
        verbose=verbose,
        sample_ids=sample_ids,
    )
