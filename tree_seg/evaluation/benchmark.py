"""Benchmark runner for evaluating segmentation methods."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from tree_seg.api import TreeSegmentation
from tree_seg.core.types import Config
from tree_seg.evaluation.datasets import ISPRSPotsdamDataset, SegmentationDataset
from tree_seg.evaluation.metrics import EvaluationResults, evaluate_segmentation


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
    ):
        """
        Initialize benchmark runner.

        Args:
            config: Configuration for segmentation method
            dataset: Dataset to evaluate on (any dataset implementing SegmentationDataset protocol)
            output_dir: Optional directory to save results and visualizations
            save_visualizations: Whether to save visualization images
        """
        self.config = config
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_visualizations = save_visualizations

        # Create segmentation instance
        self.segmenter = TreeSegmentation(config)

        # Create output directories if needed
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if save_visualizations:
                (self.output_dir / "visualizations").mkdir(exist_ok=True)

    def run_single_sample(
        self, idx: int, verbose: bool = True
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

        # Run segmentation with timing
        start_time = time.time()
        results = self.segmenter.segment_image(image)
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
            self._save_visualization(image_id, image, pred_labels_resized, gt_labels, eval_results)

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

        # Run on all samples
        sample_results = []
        for idx in range(start_idx, end_idx):
            sample_result, _ = self.run_single_sample(idx, verbose=verbose)
            sample_results.append(sample_result)

        # Compute aggregated metrics
        mean_miou = np.mean([s.miou for s in sample_results])
        mean_pixel_acc = np.mean([s.pixel_accuracy for s in sample_results])
        mean_runtime = np.mean([s.runtime_seconds for s in sample_results])

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
    ):
        """Save visualization comparing prediction and ground truth."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Predicted segmentation
        axes[1].imshow(pred_labels, cmap="tab20")
        axes[1].set_title(f"Prediction (K={eval_results.num_predicted_clusters})")
        axes[1].axis("off")

        # Ground truth
        # Mask out ignored pixels
        gt_vis = gt_labels.copy()
        gt_vis[gt_labels == self.dataset.IGNORE_INDEX] = self.dataset.NUM_CLASSES
        axes[2].imshow(gt_vis, cmap="tab20")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        # Add metrics as title
        fig.suptitle(
            f"{image_id} | mIoU: {eval_results.miou:.3f} | Pixel Acc: {eval_results.pixel_accuracy:.3f}",
            fontsize=12,
        )

        plt.tight_layout()
        save_path = self.output_dir / "visualizations" / f"{image_id}_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def run_benchmark(
    config: Config,
    dataset: Union[SegmentationDataset, Path] = None,
    dataset_path: Optional[Path] = None,
    dataset_class: Optional[type] = None,
    output_dir: Optional[Path] = None,
    num_samples: Optional[int] = None,
    save_visualizations: bool = False,
    verbose: bool = True,
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
    )

    results = runner.run(num_samples=num_samples, verbose=verbose)

    return results
