"""Shared helpers for supervised baselines (progress, XY, evaluation)."""

import time
from typing import Callable, Optional

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import BenchmarkResults, BenchmarkSample
from tree_seg.evaluation.metrics import evaluate_segmentation


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
    """
    Flatten (N,H,W,D) features and (N,H,W) labels with optional ignore filter and subsampling.
    Returns X (float32) and y (int).
    """
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
