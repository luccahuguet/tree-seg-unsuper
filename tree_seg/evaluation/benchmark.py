"""Benchmark runner for evaluating segmentation methods."""

import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image
import threading

from contextlib import contextmanager, redirect_stdout, redirect_stderr

from tree_seg.api import TreeSegmentation
from tree_seg.core.types import Config
from tree_seg.evaluation.datasets import ISPRSPotsdamDataset, SegmentationDataset
from tree_seg.evaluation.metrics import EvaluationResults, evaluate_segmentation
from tree_seg.utils.runtime_cache import RuntimeCache


@dataclass
class BenchmarkSample:
    """Results for a single benchmark sample."""

    image_id: str
    miou: float
    pixel_accuracy: float
    per_class_iou: dict
    num_clusters: int
    runtime_seconds: float
    image_shape: tuple


@dataclass
class BenchmarkResults:
    """Complete benchmark results across all samples."""

    dataset_name: str
    method_name: str
    config: Config
    samples: List[BenchmarkSample]
    mean_miou: float
    mean_pixel_accuracy: float
    mean_runtime: float
    total_samples: int


class BenchmarkRunner:
    """
    Runner for benchmarking segmentation methods on datasets.

    Handles:
    - Loading dataset samples
    - Running segmentation
    - Computing evaluation metrics
    - Aggregating results
    """

    def __init__(
        self,
        config: Config,
        dataset: SegmentationDataset,
        output_dir: Optional[Path] = None,
        save_visualizations: bool = False,
        use_smart_k: bool = False,
        model_cache: Optional[dict] = None,
        suppress_logs: bool = False,
    ):
        """
        Initialize benchmark runner.

        Args:
            config: Configuration for segmentation method
            dataset: Dataset to evaluate on (any dataset implementing SegmentationDataset protocol)
            output_dir: Optional directory to save results and visualizations
            save_visualizations: Whether to save visualization images
            use_smart_k: If True, set K to match the number of classes in each image's ground truth
            model_cache: Optional dict to cache models across runs (key: (model_name, stride, image_size))
        """
        self.config = config
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_visualizations = save_visualizations
        self.use_smart_k = use_smart_k
        self.model_cache = model_cache if model_cache is not None else {}
        self.runtime_cache = RuntimeCache()
        self.suppress_logs = suppress_logs
        # Lazy segmenter creation
        self.segmenter = None

        # Create output directories if needed
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if save_visualizations:
                (self.output_dir / "visualizations").mkdir(exist_ok=True)

    def _maybe_load_cached_model(self):
        """Load model from cache if available, otherwise initialize and cache it."""
        model_key = (
            self.config.model_display_name,
            self.config.stride,
            self.config.image_size,
        )

        if model_key in self.model_cache:
            # Reuse cached model
            cached_model, cached_preprocess = self.model_cache[model_key]
            self.segmenter.model = cached_model
            self.segmenter.preprocess = cached_preprocess
        else:
            # Initialize model and add to cache
            self.segmenter.initialize_model()
            self.model_cache[model_key] = (
                self.segmenter.model,
                self.segmenter.preprocess,
            )

    def _create_segmenter(self, config: Config) -> TreeSegmentation:
        if self.suppress_logs:
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                return TreeSegmentation(config)
        return TreeSegmentation(config)

    def run_single_sample(
        self, idx: int, verbose: bool = True, suppress_output: bool = False
    ) -> tuple[BenchmarkSample, EvaluationResults]:
        """
        Run benchmark on a single dataset sample.

        Args:
            idx: Sample index in dataset
            verbose: Whether to print progress

        Returns:
            Tuple of (BenchmarkSample, EvaluationResults)
        """
        # Load image and ground truth
        image, gt_labels, image_id = self.dataset[idx]

        if verbose:
            print(f"Processing {image_id} ({idx + 1}/{len(self.dataset)})...")

        # Handle smart K mode
        if self.use_smart_k:
            # Count unique classes in ground truth (excluding ignore index)
            unique_classes = np.unique(gt_labels)
            if self.dataset.IGNORE_INDEX is not None:
                unique_classes = unique_classes[unique_classes != self.dataset.IGNORE_INDEX]
            num_gt_classes = len(unique_classes)

            # Create modified config with smart K
            from dataclasses import replace
            smart_config = replace(
                self.config,
                n_clusters=num_gt_classes,
                auto_k=False,
            )

            if verbose:
                print(f"  Smart K mode: Using K={num_gt_classes} (matched to ground truth)")

            # Create temporary segmenter with smart config
            segmenter = self._create_segmenter(smart_config)

            # Try to reuse cached model
            model_key = (
                smart_config.model_display_name,
                smart_config.stride,
                smart_config.image_size,
            )
            if model_key in self.model_cache:
                cached_model, cached_preprocess = self.model_cache[model_key]
                segmenter.model = cached_model
                segmenter.preprocess = cached_preprocess
                if verbose:
                    print(f"  ♻️  Reusing cached model: {smart_config.model_display_name}")
            else:
                # Initialize and cache
                segmenter.initialize_model()
                self.model_cache[model_key] = (segmenter.model, segmenter.preprocess)
        else:
            if self.segmenter is None:
                self.segmenter = self._create_segmenter(self.config)
                self._maybe_load_cached_model()
            segmenter = self.segmenter

        # Run segmentation with timing
        start_time = time.time()
        if suppress_output:
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                results = segmenter.segment_image(image)
        else:
            results = segmenter.segment_image(image)
        runtime = time.time() - start_time

        # Get segmentation labels
        pred_labels = results.labels_resized

        # Resize prediction to match ground truth if needed
        if pred_labels.shape != gt_labels.shape:
            pred_labels_resized = np.array(
                Image.fromarray(pred_labels.astype(np.uint8)).resize(
                    (gt_labels.shape[1], gt_labels.shape[0]), Image.NEAREST
                )
            )
        else:
            pred_labels_resized = pred_labels

        # Evaluate
        eval_results = evaluate_segmentation(
            pred_labels=pred_labels_resized,
            gt_labels=gt_labels,
            num_classes=self.dataset.NUM_CLASSES,
            ignore_index=self.dataset.IGNORE_INDEX,
            use_hungarian=True,
        )

        # Create sample result
        sample_result = BenchmarkSample(
            image_id=image_id,
            miou=eval_results.miou,
            pixel_accuracy=eval_results.pixel_accuracy,
            per_class_iou=eval_results.per_class_iou,
            num_clusters=results.n_clusters_used,
            runtime_seconds=runtime,
            image_shape=image.shape,
        )

        if verbose:
            print(
                f"  mIoU: {eval_results.miou:.3f}, "
                f"Pixel Acc: {eval_results.pixel_accuracy:.3f}, "
                f"K: {results.n_clusters_used}, "
                f"Time: {runtime:.2f}s"
            )

        # Save visualization if requested
        if self.save_visualizations and self.output_dir:
            self._save_visualization(
                image_id,
                image,
                pred_labels_resized,
                gt_labels,
                eval_results,
                runtime_seconds=runtime,
            )

        return sample_result, eval_results

    def run(
        self,
        num_samples: Optional[int] = None,
        start_idx: int = 0,
        verbose: bool = True,
    ) -> BenchmarkResults:
        """
        Run benchmark on entire dataset (or subset).

        Args:
            num_samples: Number of samples to evaluate (None = all)
            start_idx: Starting index in dataset
            verbose: Whether to print progress

        Returns:
            BenchmarkResults with aggregated metrics
        """
        # Determine sample range
        end_idx = len(self.dataset) if num_samples is None else start_idx + num_samples
        end_idx = min(end_idx, len(self.dataset))

        if verbose:
            print(f"\nRunning benchmark on {end_idx - start_idx} samples...")
            print(f"Dataset: {self.dataset.dataset_path.name}")
            if self.config.version == "v4":
                print("Method: v4 (Mask2Former decoder)")
            else:
                print(f"Method: {self.config.version} (refine={self.config.refine})")
            print(f"Model: {self.config.model_display_name}")
            print(f"Config: stride={self.config.stride}, " f"elbow_threshold={self.config.elbow_threshold}\n")

        dataset_name = getattr(self.dataset, "dataset_path", None)
        dataset_name = dataset_name.name if dataset_name else "unknown_dataset"

        # Load runtime estimate from cache
        est_key = self.runtime_cache.make_key(self.config)
        cached_mean = self.runtime_cache.get_mean_runtime(est_key)
        run_key = self.runtime_cache.make_run_key(dataset_name, self.config, end_idx - start_idx)
        cached_total = self.runtime_cache.get_total_runtime(run_key)

        # Progress bar setup
        total_samples = end_idx - start_idx
        sample_results = []
        est_total = cached_total if cached_total else (cached_mean * total_samples if cached_mean else None)
        est_remaining = est_total

        def _format_eta(seconds: float) -> str:
            if seconds is None:
                return "?"
            seconds = max(seconds, 0)
            minutes = int(seconds // 60)
            sec = seconds % 60
            if minutes == 0:
                return f"{sec:04.1f}s"
            return f"{minutes}m {sec:04.1f}s"

        @contextmanager
        def maybe_progress(total):
            try:
                from tqdm import tqdm
            except ImportError:
                yield None
                return
            bar = tqdm(total=total, desc="Benchmark", unit="img")
            try:
                if est_total:
                    bar.set_postfix(eta=_format_eta(est_total))
                yield bar
            finally:
                bar.close()

        stop_heartbeat = threading.Event()

        def heartbeat(bar, start_time, est_total_val):
            if bar is None or est_total_val is None:
                return
            while not stop_heartbeat.is_set():
                elapsed = time.time() - start_time
                remaining = est_total_val - elapsed
                bar.set_postfix(eta=_format_eta(remaining))
                stop_heartbeat.wait(1.0)

        total_start = time.time()

        with maybe_progress(total_samples) as bar:
            if bar is not None and est_total is not None:
                threading.Thread(target=heartbeat, args=(bar, total_start, est_total), daemon=True).start()
            for idx in range(start_idx, end_idx):
                t0 = time.time()
                sample_verbose = verbose and bar is None
                sample_result, eval_results = self.run_single_sample(
                    idx, 
                    verbose=sample_verbose, 
                    suppress_output=bar is not None
                )
                dt = time.time() - t0
                sample_results.append(sample_result)

                # Update ETA
                if bar is not None:
                    bar.update(1)
                    if est_remaining is not None:
                        est_remaining = max(est_remaining - dt, 0)
                    else:
                        est_remaining = dt * (total_samples - len(sample_results))
                    bar.set_postfix(eta=_format_eta(est_remaining))
                    # Light per-sample line without breaking the bar
                    bar.write(
                        f"{sample_result.image_id}: mIoU={sample_result.miou:.3f}, "
                        f"PA={sample_result.pixel_accuracy:.3f}, "
                        f"K={sample_result.num_clusters}, "
                        f"Time={sample_result.runtime_seconds:.1f}s"
                    )
        stop_heartbeat.set()

        # Compute aggregated metrics
        mean_miou = np.mean([s.miou for s in sample_results])
        mean_pixel_acc = np.mean([s.pixel_accuracy for s in sample_results])
        mean_runtime = np.mean([s.runtime_seconds for s in sample_results])
        total_runtime = sum([s.runtime_seconds for s in sample_results])

        # Persist runtime mean for future ETA
        if len(sample_results) > 0:
            self.runtime_cache.update(est_key, mean_runtime)
            self.runtime_cache.update_total(run_key, total_runtime)

        # Create benchmark results
        refine_str = "mask2former" if self.config.version == "v4" else (self.config.refine if self.config.refine else "kmeans")
        results = BenchmarkResults(
            dataset_name=self.dataset.dataset_path.name,
            method_name=f"{self.config.version}_{refine_str}",
            config=self.config,
            samples=sample_results,
            mean_miou=mean_miou,
            mean_pixel_accuracy=mean_pixel_acc,
            mean_runtime=mean_runtime,
            total_samples=len(sample_results),
        )

        if verbose:
            print("\n" + "=" * 60)
            print("BENCHMARK RESULTS")
            print("=" * 60)
            print(f"Mean mIoU: {mean_miou:.3f}")
            print(f"Mean Pixel Accuracy: {mean_pixel_acc:.3f}")
            print(f"Mean Runtime: {mean_runtime:.2f}s")
            print(f"Total Samples: {len(sample_results)}")
            print("=" * 60 + "\n")

        return results

    def _save_visualization(
        self,
        image_id: str,
        image: np.ndarray,
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
        eval_results: EvaluationResults,
        runtime_seconds: float | None = None,
    ):
        """Save visualization comparing prediction and ground truth."""
        from ..visualization.plotting import plot_evaluation_comparison
        
        output_path = self.output_dir / "visualizations" / f"{image_id}_comparison.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plot_evaluation_comparison(
            image=image,
            pred_labels=pred_labels,
            gt_labels=gt_labels,
            eval_results=eval_results,
            dataset_class_names=getattr(self.dataset, "CLASS_NAMES", None),
            ignore_index=self.dataset.IGNORE_INDEX,
            output_path=str(output_path),
            image_id=image_id,
            config=self.config,
            runtime_seconds=runtime_seconds,
        )
        try:
            print(f"  📁 Saved visualization: {output_path}")
        except Exception:
            pass


def run_benchmark(
    config: Config,
    dataset: Union[SegmentationDataset, Path] = None,
    dataset_path: Optional[Path] = None,
    dataset_class: Optional[type] = None,
    output_dir: Optional[Path] = None,
    num_samples: Optional[int] = None,
    save_visualizations: bool = False,
    verbose: bool = True,
    use_smart_k: bool = False,
    model_cache: Optional[dict] = None,
) -> BenchmarkResults:
    """
    Convenience function to run benchmark.

    Args:
        config: Segmentation configuration
        dataset: Dataset instance OR path to dataset
        dataset_path: (Deprecated) Path to dataset - use dataset parameter instead
        dataset_class: Dataset class to instantiate (default: ISPRSPotsdamDataset)
        output_dir: Optional output directory for results
        num_samples: Number of samples to evaluate (None = all)
        save_visualizations: Whether to save visualization images
        verbose: Whether to print progress
        use_smart_k: If True, set K to match the number of classes in each image's ground truth
        model_cache: Optional dict to cache models across runs

    Returns:
        BenchmarkResults
    """
    # Handle backward compatibility
    if dataset_path is not None:
        dataset = dataset_path

    # Load dataset if path provided
    if isinstance(dataset, Path):
        if dataset_class is None:
            dataset_class = ISPRSPotsdamDataset
        dataset = dataset_class(dataset)

    # Create and run benchmark
    runner = BenchmarkRunner(
        config=config,
        dataset=dataset,
        output_dir=output_dir,
        save_visualizations=save_visualizations,
        use_smart_k=use_smart_k,
        model_cache=model_cache,
    )

    results = runner.run(num_samples=num_samples, verbose=verbose)

    return results
