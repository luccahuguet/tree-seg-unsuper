"""Quick supervised baseline using sklearn on DINOv3 features."""

import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from tree_seg.core.features import extract_features
from tree_seg.core.types import Config
from tree_seg.evaluation.benchmark import BenchmarkResults, BenchmarkSample
from tree_seg.evaluation.runner import detect_dataset_type, load_dataset
from tree_seg.evaluation.metrics import evaluate_segmentation
from tree_seg.models.preprocessing import init_model_and_preprocess
from tree_seg.supervised.data import load_dataset as load_supervised_dataset
from tree_seg.supervised.data import resize_masks_to_features


def train_sklearn_classifier(
    features: np.ndarray,  # (N, H, W, D) or list of (H, W, D)
    labels: np.ndarray,  # (N, H, W) integer class labels
    ignore_index: int = 255,
    max_samples: int = 100_000,  # Subsample for memory
    multiclass_mode: str = "auto",
) -> LogisticRegression:
    """
    Train a per-pixel logistic regression classifier.

    Args:
        features: DINOv3 patch features, shape (N, H, W, D)
        labels: Ground truth masks, shape (N, H, W)
        ignore_index: Label to ignore (unlabeled pixels)
        max_samples: Max pixels to train on (subsample if needed)

    Returns:
        Trained LogisticRegression classifier
    """
    # Flatten to (N*H*W, D) and (N*H*W,)
    X = features.reshape(-1, features.shape[-1])
    y = labels.flatten()

    # Remove ignored pixels
    valid_mask = y != ignore_index
    X = X[valid_mask]
    y = y[valid_mask]

    # Subsample if too many pixels
    if len(y) > max_samples:
        indices = np.random.choice(len(y), max_samples, replace=False)
        X = X[indices]
        y = y[indices]

    # Train classifier
    print(f"Training on {len(y):,} pixels...")
    if multiclass_mode == "auto":
        multiclass_mode = "multinomial"

    clf = LogisticRegression(
        max_iter=1000,
        multi_class=multiclass_mode,
        solver="lbfgs",
        n_jobs=1,  # single-process to avoid joblib shm issues in constrained envs
        verbose=0,
    )
    clf.fit(X, y)
    return clf


def predict_sklearn(
    clf: LogisticRegression,
    features: np.ndarray,  # (H, W, D) single image
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
    ignore_index: int = 255,
    max_samples: int = 100_000,
    verbose: bool = True,
) -> BenchmarkResults:
    """
    Full pipeline: extract features, train classifier, evaluate.

    Args:
        dataset_path: Path to dataset directory
        model_name: DINOv3 model to use (base, large, giant)
        stride: Feature extraction stride
        ignore_index: Label to ignore in GT masks
        max_samples: Max training samples
        verbose: Print progress

    Returns:
        BenchmarkResults with metrics
    """
    dataset_path = Path(dataset_path)

    # Create config for feature extraction
    config = Config(
        model_name=model_name,
        stride=stride,
        n_clusters=1,  # Not used for supervised
        clustering_method="kmeans",  # Not used
    )

    # Load model and preprocessing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Leave one core idle to reduce contention on CPU runs
    try:
        cpu_total = os.cpu_count() or 1
        torch.set_num_threads(max(1, cpu_total - 1))
        torch.set_num_interop_threads(max(1, cpu_total - 1))
    except Exception:
        pass
    if verbose:
        print(f"Using device: {device}")

    model, preprocess = init_model_and_preprocess(config.model_display_name, stride, device)

    # Load dataset using evaluation infrastructure
    dataset = None
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    class_names: list[str] = []
    dataset_name = dataset_path.name
    ignore_idx = ignore_index

    detected_type = detect_dataset_type(dataset_path)
    try:
        # Use evaluation dataset for known types (fortress/isprs) unless we want custom downsampling
        if detected_type in {"fortress", "isprs"}:
            # Prefer custom downsampled loader to avoid huge RAM spikes
            raise RuntimeError("Use custom loader for large datasets")

        dataset, detected_type = load_dataset(dataset_path, detect_dataset_type(dataset_path))
        dataset_name = dataset.dataset_path.name if hasattr(dataset, "dataset_path") else dataset_name
        ignore_idx = getattr(dataset, "IGNORE_INDEX", ignore_idx)
    except Exception as exc:
        # Fall back to custom loader
        if verbose:
            print(f"Using custom dataset loader (reason: {exc})...")
        # Cap longest side to control memory (roughly 2× preprocess size)
        max_side = max(config.image_size * 2, 1024)
        images, masks, class_names = load_supervised_dataset(dataset_path, max_side=max_side)
        dataset = None

    if dataset is not None:
        # Use evaluation dataset
        total_samples = len(dataset)
        if verbose:
            print(f"Loaded dataset: {dataset_name} ({total_samples} samples)")
            progress = Progress(
                TextColumn("[bold cyan]Extracting features[/bold cyan]"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            progress.start()
            feat_task = progress.add_task("features", total=total_samples)
        else:
            progress = None
            feat_task = None

        # Extract features for all images
        all_features = []
        all_masks = []

        for idx in range(total_samples):
            image, gt_mask, _image_id = dataset[idx]

            # Extract features
            start_time = time.time()
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
            elapsed = time.time() - start_time

            if verbose:
                progress.update(feat_task, advance=1, description=f"[bold cyan]Extracting features[/bold cyan] ({idx+1}/{total_samples})")

            all_features.append(features_np)
            all_masks.append(gt_mask)

        if progress:
            progress.stop()
    else:
        # Use custom loaded data
        total_samples = len(images)
        if verbose:
            print(f"Loaded {total_samples} image/mask pairs")
            progress = Progress(
                TextColumn("[bold cyan]Extracting features[/bold cyan]"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            progress.start()
            feat_task = progress.add_task("features", total=total_samples)
        else:
            progress = None
            feat_task = None

        all_features = []
        all_masks = []

        for idx, (image, mask) in enumerate(zip(images, masks)):
            start_time = time.time()
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
            elapsed = time.time() - start_time

            if verbose:
                progress.update(feat_task, advance=1, description=f"[bold cyan]Extracting features[/bold cyan] ({idx+1}/{total_samples})")

            all_features.append(features_np)
            all_masks.append(mask)

        if progress:
            progress.stop()

    # Stack features
    all_features = np.stack(all_features)  # (N, H, W, D)

    # Resize masks to feature resolution
    if verbose:
        print(f"Resizing masks to feature resolution: {all_features.shape[1:3]}")

    all_masks_resized = resize_masks_to_features(
        all_masks, target_size=all_features.shape[1:3]
    )

    # Determine class count after resizing (robust to non-contiguous labels)
    valid_pixels = all_masks_resized[all_masks_resized != ignore_idx]
    num_classes = int(valid_pixels.max()) + 1 if valid_pixels.size > 0 else 1

    # Train classifier
    if verbose:
        print("\nTraining supervised classifier...")

    clf = train_sklearn_classifier(
        all_features,
        all_masks_resized,
        ignore_idx,
        max_samples,
        multiclass_mode="multinomial",
    )

    # Predict and evaluate
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

        # Evaluate
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
            progress.update(eval_task, advance=1, description=f"[bold green]Evaluating[/bold green] ({idx+1}/{len(all_features)})")

    # Aggregate results
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
