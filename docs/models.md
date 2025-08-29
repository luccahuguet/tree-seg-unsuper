# DINOv3 Models and Weights

## Overview

This document provides comprehensive information about DINOv3 model variants, their Hugging Face weights, and how they map to our project's configuration.

## Quick Model Selection

```python
config = Config(
    model_name="base",        # Recommended default
    # Options:
    # "small"  → ViT-S/16 (21M params)    - Fast
    # "base"   → ViT-B/16 (86M params)    - Balanced ⭐
    # "large"  → ViT-L/16 (300M params)   - High quality
    # "giant"  → ViT-H+/16 (840M params)  - Maximum quality
    # "mega"   → ViT-7B/16 (6.7B params)  - Satellite optimized
)
```

## Model Access Requirements

⚠️ **Important**: DINOv3 models require manual access approval from Meta AI:
- Visit [Meta AI Model Access Form](https://ai.meta.com/llama/)
- Fill out the access request form 
- Wait for approval email (usually within a few hours)
- Once approved, you can download weights from Hugging Face

## Available Models

### Essential Models (Priority 1)

| Project Name | DINOv3 Model | Parameters | Hugging Face ID | Description |
|--------------|--------------|------------|-----------------|-------------|
| `base` | `dinov3_vitb16` | 86M | `facebook/dinov3-vitb16-pretrain-lvd1689m` | **Recommended** - Best balance of quality vs speed |
| `small` | `dinov3_vits16` | 21M | `facebook/dinov3-vits16-pretrain-lvd1689m` | Fast model for testing and development |

### Enhanced "Plus" Models (Priority 2)

| Project Name | DINOv3 Model | Parameters | Hugging Face ID | Description |
|--------------|--------------|------------|-----------------|-------------|
| `small` | `dinov3_vits16plus` | 29M | `facebook/dinov3-vits16plus-pretrain-lvd1689m` | **Enhanced small** - Uses SwiGLU FFN, better performance |
| `giant` | `dinov3_vith16plus` | 840M | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | **Enhanced huge** - Maximum quality for critical applications |

### High-End Models (Priority 3)

| Project Name | DINOv3 Model | Parameters | Hugging Face ID | Description |
|--------------|--------------|------------|-----------------|-------------|
| `large` | `dinov3_vitl16` | 300M | `facebook/dinov3-vitl16-pretrain-lvd1689m` | High quality, slower processing |
| `mega` | `dinov3_vit7b16` | 6.7B | `facebook/dinov3-vit7b16-pretrain-lvd1689m` | **Satellite-optimized** - Ultimate quality for aerial imagery |

### Satellite-Optimized Variants

| Model | Parameters | Hugging Face ID | Description |
|-------|------------|-----------------|-------------|
| `dinov3_vitl16` (SAT) | 300M | `facebook/dinov3-vitl16-pretrain-sat493m` | Large model trained on satellite data |
| `dinov3_vit7b16` (SAT) | 6.7B | `facebook/dinov3-vit7b16-pretrain-sat493m` | 7B model trained on satellite data |

### ConvNext Models (Not Used)

These models are available but not used in our Vision Transformer-based pipeline:
- `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` (28M params)
- `facebook/dinov3-convnext-small-pretrain-lvd1689m` (50M params)  
- `facebook/dinov3-convnext-base-pretrain-lvd1689m` (89M params)
- `facebook/dinov3-convnext-large-pretrain-lvd1689m` (198M params)

## Architecture Differences

### Standard vs "Plus" Variants

**Standard Models** (vits16, vitb16, vitl16):
- Traditional MLP FFN (Feed-Forward Network)
- Standard parameter counts
- Good baseline performance

**"Plus" Models** (vits16plus, vith16plus):
- **SwiGLU FFN** - More sophisticated feed-forward network
- **More parameters** (e.g., ViT-S+: 29M vs ViT-S: 21M)
- **Better performance** (e.g., 68.8% vs 60.4% on ImageNet-R)

### Satellite-Optimized Models

Models with `-sat493m` suffix are specifically trained on satellite imagery:
- Optimized for aerial/satellite data
- Better performance on remote sensing tasks
- Ideal for tree segmentation from drone imagery

## Usage in Project

### Configuration Mapping

```python
from tree_seg import Config

# Map project names to actual models
config = Config(
    model_name="base",      # Uses dinov3_vitb16
    model_name="small",     # Uses dinov3_vits16  
    model_name="large",     # Uses dinov3_vitl16
    model_name="giant",     # Uses dinov3_vith16plus
    model_name="mega",      # Uses dinov3_vit7b16
)
```

### Automatic Model Selection

The project automatically selects the best available model:
1. Tries "plus" variant first (if available)
2. Falls back to standard variant
3. Uses random initialization if weights unavailable

### Performance Characteristics

| Model Size | Speed | Quality | Memory | Use Case |
|------------|-------|---------|--------|----------|
| Small | ⚡⚡⚡ | ⭐⭐ | 🔋🔋🔋 | Development, testing |
| Small+ | ⚡⚡ | ⭐⭐⭐ | 🔋🔋 | Production (light) |
| Base | ⚡⚡ | ⭐⭐⭐⭐ | 🔋🔋 | **Recommended** |
| Large | ⚡ | ⭐⭐⭐⭐⭐ | 🔋 | High quality needs |
| Giant+ | 🐌 | ⭐⭐⭐⭐⭐ | 💾💾 | Critical applications |
| Mega | 🐌🐌 | ⭐⭐⭐⭐⭐⭐ | 💾💾💾 | Ultimate quality |

## Recommendations

### For Development
Start with: `facebook/dinov3-vitb16-pretrain-lvd1689m` (base model)

### For Production
- **Balanced**: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- **Fast**: `facebook/dinov3-vits16plus-pretrain-lvd1689m` 
- **Quality**: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- **Ultimate**: `facebook/dinov3-vit7b16-pretrain-lvd1689m`

### For Aerial Imagery
Prefer satellite-optimized variants when available:
- `facebook/dinov3-vitl16-pretrain-sat493m`
- `facebook/dinov3-vit7b16-pretrain-sat493m`

## Download Instructions

1. **Get access**: Fill out Meta AI access form
2. **Install dependencies**: `pip install huggingface_hub`
3. **Login**: `huggingface-cli login`
4. **Download**: Models download automatically on first use

## Model Storage

Downloaded models are cached in:
- `~/.cache/torch/hub/checkpoints/` (PyTorch Hub)
- `~/.cache/huggingface/transformers/` (Hugging Face)

## Troubleshooting

### HTTP 403 Forbidden
- **Cause**: No access approval from Meta AI
- **Solution**: Fill out access form and wait for approval

### CUDA Out of Memory
- **Solution**: Use smaller model or reduce stride parameter
- **Order**: small → base → large → giant → mega

### Slow Performance
- **Solution**: Use smaller model or increase stride parameter
- **Fast models**: small, small+, base

## References

- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)
- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [Meta AI Model Access](https://ai.meta.com/llama/)
- [Hugging Face Collection](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009)