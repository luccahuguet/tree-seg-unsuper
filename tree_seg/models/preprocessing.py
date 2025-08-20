"""
Image preprocessing utilities for tree segmentation.
"""

from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def get_preprocess():
    """Get the standard preprocessing pipeline for DINOv3 tree segmentation."""
    # DINOv3 uses standard ImageNet normalization
    return Compose([
        Resize((518, 518)),  # Keep same size for compatibility
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def init_model_and_preprocess(model_name, stride, device):
    """
    Initialize DINOv3 model with preprocessing (legacy function).
    Note: This is a simplified version. Use initialize_model() for full functionality.
    """
    from .dinov3_adapter import create_dinov3_model
    import torch
    
    model = create_dinov3_model(
        model_name=model_name,
        stride=stride,
        device=device,
        dtype=torch.float16
    )
    model.eval()
    preprocess = get_preprocess()
    return model, preprocess 