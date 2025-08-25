"""
Model initialization and GPU utilities for tree segmentation.
"""

import os
import torch
import logging

# Import the latest DINOv3 adapter implementation
try:
    from .dinov3_adapter_final import create_dinov3_model, print_model_info, list_available_models
except ImportError:
    # Fallback to original adapter
    from .dinov3_adapter import create_dinov3_model
    print_model_info = None
    list_available_models = None

logger = logging.getLogger(__name__)


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
    """Initialize and configure the DINOv3 model with improved error handling."""
    try:
        logger.info(f"Initializing DINOv3 model: {model_name}")
        
        # Use float32 for stability (from our debugging)
        model = create_dinov3_model(
            model_name=model_name,
            stride=stride,
            device=device,
            dtype=torch.float32
        )
        model.eval()
        
        logger.info("✅ Model initialization successful")
        return model
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise


def list_models():
    """List available DINOv3 models."""
    if list_available_models:
        return list_available_models()
    else:
        # Fallback list for original adapter
        return {
            "dinov3_vits16": "Small model (21M params)",
            "dinov3_vitb16": "Base model (86M params)", 
            "dinov3_vitl16": "Large model (304M params)",
            "dinov3_vith16plus": "Huge+ model (1.1B params)",
        }


def show_model_info():
    """Show information about available models."""
    if print_model_info:
        print_model_info()
    else:
        models = list_models()
        print("🔍 Available DINOv3 Models:")
        print("=" * 40)
        for name, desc in models.items():
            print(f"📦 {name}: {desc}")