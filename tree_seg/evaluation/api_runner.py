"""Shared helpers to construct segmenters and run process_image for API/benchmark."""

from __future__ import annotations

import time

import numpy as np
import torch

from tree_seg.core.segmentation import process_image
from tree_seg.core.types import Config, SegmentationResults
from tree_seg.core.output_manager import OutputManager
from tree_seg.models import get_preprocess, initialize_model
from tree_seg.models.mask2former import Mask2FormerSegmentor


def select_device(force_gpu: bool = False) -> torch.device:
    """Choose a device based on FORCE_GPU flag and availability."""
    if force_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def init_model_and_preprocess(config: Config, device: torch.device):
    model = initialize_model(config.stride, config.model_display_name, device)
    preprocess = get_preprocess(config.image_size)
    return model, preprocess


def run_segmentation_numpy(
    *,
    image_np: np.ndarray,
    config: Config,
    device: torch.device,
    model,
    preprocess,
) -> SegmentationResults:
    """Run process_image on an in-memory numpy image."""
    result = process_image(  # type: ignore
        image_np,
        model,
        preprocess,
        n_clusters=config.n_clusters,
        stride=config.stride,
        device=device,
        auto_k=config.auto_k,
        k_range=config.k_range,
        elbow_threshold=config.elbow_threshold / 100.0,
        use_pca=config.use_pca,
        pca_dim=config.pca_dim,
        refine=config.refine,
        refine_slic_compactness=config.refine_slic_compactness,
        refine_slic_sigma=config.refine_slic_sigma,
        collect_metrics=True,
        model_name=config.model_display_name,
        output_dir=config.output_dir,
        verbose=config.verbose,
        pipeline=config.version,
        apply_vegetation_filter=config.apply_vegetation_filter,
        exg_threshold=config.exg_threshold,
        use_tiling=config.use_tiling,
        tile_size=config.tile_size,
        tile_overlap=config.tile_overlap,
        tile_threshold=config.tile_threshold,
        downsample_before_tiling=config.downsample_before_tiling,
        clustering_method=config.clustering_method,
        use_multi_layer=config.use_multi_layer,
        layer_indices=config.layer_indices,
        feature_aggregation=config.feature_aggregation,
        use_pyramid=config.use_pyramid,
        pyramid_scales=config.pyramid_scales,
        pyramid_aggregation=config.pyramid_aggregation,
        feature_upsample_factor=config.feature_upsample_factor,
        use_soft_refine=config.use_soft_refine,
        soft_refine_temperature=config.soft_refine_temperature,
        soft_refine_iterations=config.soft_refine_iterations,
        soft_refine_spatial_alpha=config.soft_refine_spatial_alpha,
        use_attention_features=config.use_attention_features,
    )

    if result and len(result) == 3:
        _image_np, labels_resized, metrics = result
        return SegmentationResults(
            image_np=image_np,
            labels_resized=labels_resized,
            n_clusters_used=metrics.get("n_clusters", config.n_clusters),
            image_path="<numpy_array>",
            processing_stats=metrics,
            n_clusters_requested=config.n_clusters,
        )
    return SegmentationResults(
        image_np=image_np,
        labels_resized=result[1] if result else None,
        n_clusters_used=config.n_clusters,
        image_path="<numpy_array>",
        processing_stats={},
        n_clusters_requested=config.n_clusters,
    )


def run_mask2former_numpy(
    *,
    image_np: np.ndarray,
    device: torch.device,
    config: Config,
    mask2former_segmentor: Mask2FormerSegmentor,
) -> SegmentationResults:
    start_time = time.time()
    labels = mask2former_segmentor.predict(image_np)
    runtime = time.time() - start_time
    import numpy as np

    n_segments = int(np.unique(labels).size)
    stats = {
        "original_size": image_np.shape[:2],
        "labels_shape": labels.shape,
        "model": "dinov3_vit7b16",
        "decoder": "mask2former_m2f",
        "runtime_s": runtime,
        "num_decoder_classes": mask2former_segmentor.cfg.num_classes,
    }

    return SegmentationResults(
        image_np=image_np,
        labels_resized=labels,
        n_clusters_used=n_segments,
        image_path="<numpy_array>",
        processing_stats=stats,
        n_clusters_requested=None,
    )


def create_output_manager(cfg: Config) -> OutputManager:
    return OutputManager(cfg)
