"""
Image preprocessing utilities for tree segmentation.
"""

from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def get_preprocess(image_size: int = 518):
    """Get the preprocessing pipeline for DINOv3 tree segmentation.

    Args:
        image_size: Square resize dimension (e.g., 518, 896, 1024)
    """
    # DINOv3 uses standard ImageNet normalization
    return Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def init_model_and_preprocess(model_name, stride, device, image_size: int = 518):
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
    preprocess = get_preprocess(image_size=image_size)
    return model, preprocess 
