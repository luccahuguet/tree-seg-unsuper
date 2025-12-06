"""Supervised baselines on DINOv3 features (sklearn + PyTorch heads)."""

import os
import time
import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from tree_seg.core.features import extract_features
from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import BenchmarkResults, BenchmarkSample
from tree_seg.evaluation.metrics import evaluate_segmentation
from tree_seg.evaluation.runner import detect_dataset_type, load_dataset
from tree_seg.models.preprocessing import init_model_and_preprocess
from tree_seg.supervised.data import load_dataset as load_supervised_dataset
from tree_seg.supervised.data import resize_masks_to_features
from sklearn.neural_network import MLPClassifier


def _remap_masks_contiguous(
    masks: np.ndarray, ignore_index: Optional[int]
) -> tuple[np.ndarray, int, int]:
    """
    Remap mask labels to contiguous ids starting at 0.

    Returns remapped masks, num_classes, and ignore_index for evaluation
    (or -1 when not used).
    """
    uniques = np.unique(masks)
    if ignore_index is not None:
        uniques = uniques[uniques != ignore_index]
    uniques_sorted = sorted(int(u) for u in uniques.tolist())

    if not uniques_sorted:
        return masks, 0, -1

    max_val = int(np.max(masks))
    lookup = np.full(max_val + 1, -1, dtype=np.int32)
    for new_id, old_val in enumerate(uniques_sorted):
        lookup[old_val] = new_id

    remapped = lookup[masks]
    ignore_eval = -1
    if ignore_index is not None:
        remapped[masks == ignore_index] = ignore_index
        ignore_eval = ignore_index

    num_classes = len(uniques_sorted)
    return remapped, num_classes, ignore_eval


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

    if dataset is not None:
        total_samples = len(dataset)
        sample_total = min(total_samples, num_samples) if num_samples else total_samples
        if verbose:
            print(
                f"Loaded dataset: {dataset_name} ({sample_total}/{total_samples} samples)"
            )
            progress = Progress(
                TextColumn("[bold cyan]Extracting features[/bold cyan]"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            progress.start()
            feat_task = progress.add_task("features", total=sample_total)
        else:
            progress = None
            feat_task = None

        all_features = []
        all_masks = []
        indices = range(sample_total)

        for idx in indices:
            image, gt_mask, _image_id = dataset[idx]
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

            if verbose:
                progress.update(
                    feat_task,
                    advance=1,
                    description=(
                        f"[bold cyan]Extracting features[/bold cyan] ({idx+1}/{sample_total})"
                    ),
                )

            all_features.append(features_np)
            all_masks.append(gt_mask)

        if progress:
            progress.stop()
    else:
        total_samples = len(images)
        if num_samples:
            images = images[:num_samples]
            masks = masks[:num_samples]
        sample_total = len(images)
        if verbose:
            print(f"Loaded {sample_total} image/mask pairs")
            progress = Progress(
                TextColumn("[bold cyan]Extracting features[/bold cyan]"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            progress.start()
            feat_task = progress.add_task("features", total=sample_total)
        else:
            progress = None
            feat_task = None

        all_features = []
        all_masks = []

        for idx, (image, mask) in enumerate(zip(images, masks)):
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

            if verbose:
                progress.update(
                    feat_task,
                    advance=1,
                    description=(
                        f"[bold cyan]Extracting features[/bold cyan] ({idx+1}/{sample_total})"
                    ),
                )

            all_features.append(features_np)
            all_masks.append(mask)

        if progress:
            progress.stop()

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
    X = features.reshape(-1, features.shape[-1])
    y = labels.flatten()

    if ignore_index is not None:
        valid_mask = y != ignore_index
        X = X[valid_mask]
        y = y[valid_mask]

    if len(y) > max_samples:
        indices = np.random.choice(len(y), max_samples, replace=False)
        X = X[indices]
        y = y[indices]

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
    )

    if verbose:
        print("\nTraining supervised classifier...")
        train_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]Training Logistic Regression[/bold magenta]"),
            TimeElapsedColumn(),
        )
        train_progress.start()
        train_task = train_progress.add_task("train", total=None)
    else:
        train_progress = None
        train_task = None

    clf = train_sklearn_classifier(
        all_features,
        all_masks_resized,
        ignore_idx,
        max_samples,
        max_iter=3000,
        tol=1e-3,
    )

    if train_progress:
        train_progress.update(
            train_task, advance=1, description="[bold green]Training complete[/bold green]"
        )
        train_progress.stop()

    if verbose:
        print("\nEvaluating predictions...")
        progress = Progress(
            TextColumn("[bold green]Evaluating[/bold green]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        progress.start()
        eval_task = progress.add_task("eval", total=len(all_features))
    else:
        progress = None
        eval_task = None

    samples = []
    for idx in range(len(all_features)):
        start_time = time.time()
        pred = predict_sklearn(clf, all_features[idx])
        elapsed = time.time() - start_time

        eval_result = evaluate_segmentation(
            pred,
            all_masks_resized[idx],
            num_classes=num_classes,
            ignore_index=ignore_idx,
            use_hungarian=False,
        )

        sample = BenchmarkSample(
            image_id=f"img_{idx:03d}",
            miou=eval_result.miou,
            pixel_accuracy=eval_result.pixel_accuracy,
            per_class_iou=eval_result.per_class_iou,
            num_clusters=len(np.unique(pred)),
            runtime_seconds=elapsed,
            image_shape=all_features[idx].shape[:2],
        )
        samples.append(sample)

        if progress:
            progress.update(
                eval_task,
                advance=1,
                description=(
                    f"[bold green]Evaluating[/bold green] ({idx+1}/{len(all_features)})"
                ),
            )

    mean_miou = np.mean([s.miou for s in samples])
    mean_pixel_accuracy = np.mean([s.pixel_accuracy for s in samples])
    mean_runtime = np.mean([s.runtime_seconds for s in samples])

    if progress:
        progress.stop()

    results = BenchmarkResults(
        dataset_name=dataset_name,
        method_name="supervised-sklearn",
        config=config,
        samples=samples,
        mean_miou=mean_miou,
        mean_pixel_accuracy=mean_pixel_accuracy,
        mean_runtime=mean_runtime,
        total_samples=len(samples),
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print("Final Results:")
        print(f"  Mean mIoU: {mean_miou:.3f}")
        print(f"  Mean Pixel Accuracy: {mean_pixel_accuracy:.3f}")
        print(f"  Mean Runtime: {mean_runtime:.3f}s")
        print(f"{'=' * 60}")

    return results


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
    X = features.reshape(-1, features.shape[-1])
    y = labels.flatten()

    if ignore_index is not None:
        valid_mask = y != ignore_index
        X = X[valid_mask]
        y = y[valid_mask]

    if len(y) > max_samples:
        indices = np.random.choice(len(y), max_samples, replace=False)
        X = X[indices]
        y = y[indices]

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
        train_ratio=1.0,
    )

    if verbose:
        train_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]Training MLP head[/bold magenta]"),
            TimeElapsedColumn(),
        )
        train_progress.start()
        train_task = train_progress.add_task("train", total=None)
    else:
        train_progress = None
        train_task = None

    # Optionally append XY coords to each patch vector
    features_for_head = train_features
    if use_xy:
        n, h, w, d = train_features.shape
        yy, xx = np.meshgrid(
            np.linspace(0, 1, h, dtype=np.float32),
            np.linspace(0, 1, w, dtype=np.float32),
            indexing="ij",
        )
        coords = np.stack([yy, xx], axis=-1)  # H,W,2
        coords = np.broadcast_to(coords, (n, h, w, 2))
        features_for_head = np.concatenate([train_features, coords], axis=-1)

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

    if train_progress:
        train_progress.update(
            train_task, advance=1, description="[bold green]Training complete[/bold green]"
        )
        train_progress.stop()

    if verbose:
        print("\nEvaluating predictions...")
        progress = Progress(
            TextColumn("[bold green]Evaluating[/bold green]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        progress.start()
        eval_task = progress.add_task("eval", total=len(all_features))
    else:
        progress = None
        eval_task = None

    samples = []
    all_eval_features = list(features_for_head) + (
        list(holdout_features) if len(holdout_features) > 0 else []
    )
    all_eval_masks = list(train_masks) + (
        list(holdout_masks) if len(holdout_masks) > 0 else []
    )
    for idx in range(len(all_eval_features)):
        start_time = time.time()
        pred = predict_mlp(
            clf,
            all_eval_features[idx],
            scaler,
        )
        elapsed = time.time() - start_time

        eval_result = evaluate_segmentation(
            pred,
            all_eval_masks[idx],
            num_classes=num_classes,
            ignore_index=ignore_idx,
            use_hungarian=False,
        )

        sample = BenchmarkSample(
            image_id=f"img_{idx:03d}"
            + ("_holdout" if idx >= len(train_masks) else ""),
            miou=eval_result.miou,
            pixel_accuracy=eval_result.pixel_accuracy,
            per_class_iou=eval_result.per_class_iou,
            num_clusters=len(np.unique(pred)),
            runtime_seconds=elapsed,
            image_shape=all_eval_features[idx].shape[:2],
        )
        samples.append(sample)

        if progress:
            progress.update(
                eval_task,
                advance=1,
                description=(
                    f"[bold green]Evaluating[/bold green] ({idx+1}/{len(all_features)})"
                ),
            )

    mean_miou = np.mean([s.miou for s in samples])
    mean_pixel_accuracy = np.mean([s.pixel_accuracy for s in samples])
    mean_runtime = np.mean([s.runtime_seconds for s in samples])

    if progress:
        progress.stop()

    results = BenchmarkResults(
        dataset_name=dataset_name,
        method_name="supervised-mlp",
        config=config,
        samples=samples,
        mean_miou=mean_miou,
        mean_pixel_accuracy=mean_pixel_accuracy,
        mean_runtime=mean_runtime,
        total_samples=len(samples),
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print("Final Results:")
        print(f"  Mean mIoU: {mean_miou:.3f}")
        print(f"  Mean Pixel Accuracy: {mean_pixel_accuracy:.3f}")
        print(f"  Mean Runtime: {mean_runtime:.3f}s")
        print(f"{'=' * 60}")

    return results


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
    X = features.reshape(-1, features.shape[-1]).astype(np.float32)
    y = labels.reshape(-1)

    if ignore_index is not None:
        valid = y != ignore_index
        X = X[valid]
        y = y[valid]

    if len(y) > max_patches:
        idx = np.random.choice(len(y), max_patches, replace=False)
        X = X[idx]
        y = y[idx]

    # Optionally append XY coords to each patch vector
    if use_xy:
        n_total = features.shape[0]
        h, w = features.shape[1:3]
        yy, xx = np.meshgrid(
            np.linspace(0, 1, h, dtype=np.float32),
            np.linspace(0, 1, w, dtype=np.float32),
            indexing="ij",
        )
        coords = np.stack([yy, xx], axis=-1)  # H,W,2
        coords_flat = np.broadcast_to(coords, (n_total, h, w, 2)).reshape(-1, 2)
        # Align coords with the flattened X, then filter by valid mask below
        if X.shape[0] == coords_flat.shape[0]:
            X = np.concatenate([X, coords_flat], axis=1)

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

        # Early stopping on val loss
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
        train_ratio=1.0,
    )

    if verbose:
        print("\nTraining supervised classifier...")
        train_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]Preparing LinearHead[/bold magenta]"),
            TimeElapsedColumn(),
        )
        train_progress.start()
        # We replace the spinner with a bar once data loader is known
        train_progress.stop()

        bar = Progress(
            TextColumn("[bold magenta]Training LinearHead[/bold magenta]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        bar.start()
    else:
        train_progress = None
        bar = None

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

    if bar:
        for task in bar.tasks:
            if not task.finished:
                bar.update(task.id, advance=task.total)
        bar.stop()

    if verbose:
        print("\nEvaluating predictions...")
        progress = Progress(
            TextColumn("[bold green]Evaluating[/bold green]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        progress.start()
        eval_task = progress.add_task("eval", total=len(all_features))
    else:
        progress = None
        eval_task = None

    samples = []
    # Evaluate on train + holdout (if any)
    all_eval_features = list(train_features) + (list(holdout_features) if len(holdout_features) > 0 else [])
    all_eval_masks = list(train_masks) + (list(holdout_masks) if len(holdout_masks) > 0 else [])

    for idx in range(len(all_eval_features)):
        start_time = time.time()
        pred = _predict_linear_head(model, all_eval_features[idx], device)
        elapsed = time.time() - start_time

        eval_result = evaluate_segmentation(
            pred,
            all_eval_masks[idx],
            num_classes=num_classes,
            ignore_index=ignore_idx,
            use_hungarian=False,
        )

        sample = BenchmarkSample(
            image_id=f"img_{idx:03d}",
            miou=eval_result.miou,
            pixel_accuracy=eval_result.pixel_accuracy,
            per_class_iou=eval_result.per_class_iou,
            num_clusters=len(np.unique(pred)),
            runtime_seconds=elapsed,
            image_shape=all_features[idx].shape[:2],
        )
        samples.append(sample)

        if progress:
            progress.update(
                eval_task,
                advance=1,
                description=(
                    f"[bold green]Evaluating[/bold green] ({idx+1}/{len(all_features)})"
                ),
            )

    mean_miou = np.mean([s.miou for s in samples])
    mean_pixel_accuracy = np.mean([s.pixel_accuracy for s in samples])
    mean_runtime = np.mean([s.runtime_seconds for s in samples])

    if progress:
        progress.stop()

    results = BenchmarkResults(
        dataset_name=dataset_name,
        method_name="supervised-linear-head",
        config=config,
        samples=samples,
        mean_miou=mean_miou,
        mean_pixel_accuracy=mean_pixel_accuracy,
        mean_runtime=mean_runtime,
        total_samples=len(samples),
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print("Final Results:")
        print(f"  Mean mIoU: {mean_miou:.3f}")
        print(f"  Mean Pixel Accuracy: {mean_pixel_accuracy:.3f}")
        print(f"  Mean Runtime: {mean_runtime:.3f}s")
        print(f"{'=' * 60}")

    return results
