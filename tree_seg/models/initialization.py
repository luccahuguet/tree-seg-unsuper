"""
Model initialization and GPU utilities for tree segmentation.
"""

import os
import torch

from .dinov3_adapter import create_dinov3_model


def print_gpu_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        gpu_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        total_mem = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {total_mem:.2f} GB")
    else:
        print("No CUDA-compatible GPU found.")


def setup_segmentation(output_dir):
    """Setup segmentation environment and return device."""
    print_gpu_info()
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def initialize_model(stride, model_name, device):
    """Initialize and configure the DINOv3 model."""
    model = create_dinov3_model(
        model_name=model_name,
        stride=stride,
        device=device,
        dtype=torch.float16
    )
    model.eval()
    return model 