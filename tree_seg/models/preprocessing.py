"""
Image preprocessing utilities for tree segmentation.
"""

from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def get_preprocess():
    """Get the standard preprocessing pipeline for tree segmentation."""
    return Compose([
        Resize((518, 518)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def init_model_and_preprocess(model_name, stride, device):
    """
    Initialize model with simple preprocessing (legacy function).
    Note: This is a simplified version. Use initialize_model() for full functionality.
    """
    from ..core.upsampler import HighResDV2
    import torch
    
    model = HighResDV2(model_name, stride=stride, dtype=torch.float16).to(device)
    model.eval()
    model.set_transforms([lambda x: x], [lambda x: x])  # Identity transforms
    preprocess = get_preprocess()
    return model, preprocess 