"""
Utilities for running the pretrained DINOv3 + Mask2Former segmentation head.

This wraps the upstream ``dinov3_vit7b16_ms`` helper so we can reuse it inside
the existing TreeSeg pipeline (e.g., for the V4 supervised baseline).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from dinov3.eval.segmentation.inference import make_inference
from dinov3.hub.backbones import Weights as BackboneWeights
from dinov3.hub.segmentors import SegmentorWeights, dinov3_vit7b16_ms


@dataclass
class Mask2FormerConfig:
    """Runtime configuration for the pretrained Mask2Former head."""

    image_size: int = 896  # Crop/stride resolution for sliding inference
    num_classes: int = 150  # ADE20K label space
    use_tta: bool = False  # Placeholder for future extensions
    weights: Optional[str] = None  # Path/URL to the Mask2Former head weights
    backbone_weights: Optional[str] = None  # Path/URL to the backbone weights
    check_hash: bool = True


class Mask2FormerSegmentor:
    """Thin wrapper around the upstream Mask2Former head for zero-shot inference."""

    def __init__(
        self, device: torch.device, cfg: Optional[Mask2FormerConfig] = None
    ) -> None:
        self.device = device
        self.cfg = cfg or Mask2FormerConfig()

        autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        # Load the segmentor (backbone + Mask2Former head) with pretrained weights.
        self.segmentor = dinov3_vit7b16_ms(
            weights=self.cfg.weights or SegmentorWeights.ADE20K,
            backbone_weights=self.cfg.backbone_weights or BackboneWeights.LVD1689M,
            check_hash=self.cfg.check_hash,
            autocast_dtype=autocast_dtype,
        )

        # Disable autocast when running on CPU so that the upstream helper does not
        # attempt to enter a CUDA context.
        if device.type != "cuda":
            self._disable_autocast(self.segmentor)

        self.segmentor.to(device)
        self.segmentor.eval()

        self.transform = T.Compose(
            [
                T.Resize(
                    (self.cfg.image_size, self.cfg.image_size),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @staticmethod
    def _disable_autocast(module: torch.nn.Module) -> None:
        """Swap the autocast context for a CPU-friendly no-op."""
        module.autocast_ctx = contextlib.nullcontext
        # The FeatureDecoder stores backbone/head modules in ``segmentation_model``.
        if hasattr(module, "segmentation_model"):
            for submodule in module.segmentation_model:
                if hasattr(submodule, "autocast_ctx"):
                    submodule.autocast_ctx = contextlib.nullcontext

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run zero-shot segmentation on an RGB image.

        Args:
            image: RGB array of shape (H, W, 3) in ``uint8`` format.

        Returns:
            Integer mask of shape (H, W) containing ADE20K class indices.
        """
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        rescale_size = (image.shape[0], image.shape[1])  # (H, W)

        autocast_manager = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )

        with torch.inference_mode(), autocast_manager:
            scores = make_inference(
                tensor,
                self.segmentor,
                inference_mode="slide",
                decoder_head_type="m2f",
                rescale_to=rescale_size,
                n_output_channels=self.cfg.num_classes,
                crop_size=(self.cfg.image_size, self.cfg.image_size),
                stride=(self.cfg.image_size, self.cfg.image_size),
                output_activation=partial(F.softmax, dim=1),
            )

        labels = scores.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        return labels


__all__ = ["Mask2FormerSegmentor", "Mask2FormerConfig"]
