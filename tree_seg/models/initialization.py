"""
Model initialization and GPU utilities for tree segmentation.
"""

import os
import torch

from ..core.upsampler import HighResDV2
from ..utils.transform import (
    get_shift_transforms, get_flip_transforms, get_rotation_transforms, 
    combine_transforms, iden_partial
)


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
    """Initialize and configure the HighResDV2 model with transforms."""
    model = HighResDV2(model_name, stride=stride, dtype=torch.float16).to(device)
    model.eval()
    
    # Setup transforms
    shift_transforms, shift_inv_transforms = get_shift_transforms(dists=[1], pattern="Moore")
    flip_transforms, flip_inv_transforms = get_flip_transforms()
    rot_transforms, rot_inv_transforms = get_rotation_transforms()
    all_fwd_transforms, all_inv_transforms = combine_transforms(
        shift_transforms, flip_transforms, shift_inv_transforms, flip_inv_transforms
    )
    all_transforms = [t for t in all_fwd_transforms if t != iden_partial]
    all_inv_transforms = [t for t in all_inv_transforms if t != iden_partial]
    model.set_transforms(all_transforms, all_inv_transforms)
    return model 