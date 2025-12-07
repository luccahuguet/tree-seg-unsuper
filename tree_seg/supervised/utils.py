"""Shared helpers for supervised baselines (progress, XY, evaluation, feature loading)."""

import copy
import os
import time
from pathlib import Path
from typing import Callable, Optional

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

from tree_seg.core.features import extract_features
from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import BenchmarkResults, BenchmarkSample
from tree_seg.evaluation.metrics import evaluate_segmentation
from tree_seg.evaluation.runner import detect_dataset_type, load_dataset
from tree_seg.models.preprocessing import init_model_and_preprocess
from tree_seg.supervised.data import load_dataset as load_supervised_dataset
from tree_seg.supervised.data import resize_masks_to_features
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _set_torch_threads():
    """Leave one core idle to reduce contention on CPU runs."""
    try:
        cpu_total = os.cpu_count() or 1
        torch.set_num_threads(max(1, cpu_total - 1))
        torch.set_num_interop_threads(max(1, cpu_total - 1))
    except Exception:
        pass


def load_and_extract_features(
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
        features_np, _, _ = extract_features(
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

    if ignore_index is None:
        ignore_idx = 255 if 255 in np.unique(all_masks_resized) else None

    remapped_masks, num_classes, ignore_for_eval = _remap_masks_contiguous(
        all_masks_resized, ignore_idx
    )

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


def _remap_masks_contiguous(
    masks: np.ndarray, ignore_index: Optional[int]
) -> tuple[np.ndarray, int, int]:
    """Remap mask labels to contiguous ids starting at 0."""
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


def _make_bar_progress(label: str, total: int, enabled: bool):
    """Create a Rich progress bar with consistent columns."""
    if not enabled:
        return None, None
    progress = Progress(
        TextColumn(f"[bold cyan]{label}[/bold cyan]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    progress.start()
    task = progress.add_task(label.lower(), total=total)
    return progress, task


def _make_spinner(label: str, enabled: bool):
    """Create a spinner progress indicator."""
    if not enabled:
        return None, None
    progress = Progress(
        SpinnerColumn(),
        TextColumn(f"[bold magenta]{label}[/bold magenta]"),
        TimeElapsedColumn(),
    )
    progress.start()
    task = progress.add_task(label.lower(), total=None)
    return progress, task


def _stop_progress(progress: Optional[Progress]):
    if progress:
        progress.stop()


def _maybe_append_xy(features: np.ndarray, use_xy: bool) -> np.ndarray:
    """Optionally append normalized XY coordinates to each patch vector."""
    if not use_xy or features is None or len(features) == 0:
        return features
    n, h, w, _ = features.shape
    yy, xx = np.meshgrid(
        np.linspace(0, 1, h, dtype=np.float32),
        np.linspace(0, 1, w, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([yy, xx], axis=-1)  # H,W,2
    coords = np.broadcast_to(coords, (n, h, w, 2))
    return np.concatenate([features, coords], axis=-1)


def flatten_features_labels(
    features: np.ndarray,
    labels: np.ndarray,
    ignore_index: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (N,H,W,D) features and (N,H,W) labels with optional ignore filter and subsampling."""
    X = features.reshape(-1, features.shape[-1]).astype(np.float32)
    y = labels.reshape(-1)

    if ignore_index is not None:
        valid = y != ignore_index
        X = X[valid]
        y = y[valid]

    if max_samples and len(y) > max_samples:
        idx = np.random.choice(len(y), max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y


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
    """Train a simple MLP head with optional early stopping."""
    features = _maybe_append_xy(features, use_xy)
    X, y = flatten_features_labels(
        features,
        labels,
        ignore_index=ignore_index,
        max_samples=max_patches,
    )

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y.astype(np.int64))

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

    layers = [nn.LayerNorm(X.shape[1])]
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
        for _, (xb, yb) in enumerate(loader):
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


def predict_linear_head(
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


def _evaluate_predictions(
    features: list[np.ndarray],
    masks: list[np.ndarray],
    num_classes: int,
    ignore_index: Optional[int],
    method_name: str,
    dataset_name: str,
    config: Config,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    verbose: bool,
    sample_ids: Optional[list[str]] = None,
) -> BenchmarkResults:
    """Shared evaluation loop for supervised heads."""
    progress, task = _make_bar_progress("Evaluating", len(features), verbose)

    samples: list[BenchmarkSample] = []
    for idx, (feat, mask) in enumerate(zip(features, masks)):
        pred, elapsed = _predict_single(feat, predict_fn)
        eval_result = evaluate_segmentation(
            pred,
            mask,
            num_classes=num_classes,
            ignore_index=ignore_index,
            use_hungarian=False,
        )

        image_id = sample_ids[idx] if sample_ids else f"img_{idx:03d}"
        samples.append(
            BenchmarkSample(
                image_id=image_id,
                miou=eval_result.miou,
                pixel_accuracy=eval_result.pixel_accuracy,
                per_class_iou=eval_result.per_class_iou,
                num_clusters=len(np.unique(pred)),
                runtime_seconds=elapsed,
                image_shape=mask.shape,
            )
        )

        if progress and task is not None:
            progress.update(
                task,
                advance=1,
                description=(
                    f"[bold green]Evaluating[/bold green] "
                    f"({idx+1}/{len(features)})"
                ),
            )

    _stop_progress(progress)
    return _summarize_results(
        samples=samples,
        dataset_name=dataset_name,
        method_name=method_name,
        config=config,
        verbose=verbose,
    )


def _predict_single(
    feature_map: np.ndarray, predict_fn: Callable[[np.ndarray], np.ndarray]
) -> tuple[np.ndarray, float]:
    start = time.time()
    pred = predict_fn(feature_map)
    return pred, time.time() - start


def _summarize_results(
    samples: list[BenchmarkSample],
    dataset_name: str,
    method_name: str,
    config: Config,
    verbose: bool,
) -> BenchmarkResults:
    mean_miou = np.mean([s.miou for s in samples]) if samples else 0.0
    mean_pixel_accuracy = (
        np.mean([s.pixel_accuracy for s in samples]) if samples else 0.0
    )
    mean_runtime = np.mean([s.runtime_seconds for s in samples]) if samples else 0.0

    results = BenchmarkResults(
        dataset_name=dataset_name,
        method_name=method_name,
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
