---
layout: default
title: "Technical Implementation"
nav_order: 5
---

{% include navbar.html %}
{% include navbar-styles.html %}

# Technical Implementation: Deep Dive into DINOv3 Tree Segmentation

## Configuration Profiles

### Development Profile (Fast Iteration)
```python
config = Config(
    model_name="small",        # Maps to dinov3_vits16 (21M params)
    stride=4,
    elbow_threshold=5.0,
    refine=None,
    image_size=896
)
```
- **Time**: ~15 seconds
- **Use**: Rapid prototyping, parameter tuning

### Production Profile (Balanced)
```python
config = Config(
    model_name="base",         # Maps to dinov3_vitb16 (86M params)
    stride=4,
    elbow_threshold=5.0,
    refine="slic",
    image_size=1024
)
```
- **Time**: ~35 seconds  
- **Use**: Standard forestry analysis

### Research Profile (Maximum Quality)
```python
config = Config(
    model_name="giant",        # Maps to dinov3_vith16plus (1.1B params)
    stride=2, 
    elbow_threshold=5.0,
    refine="slic",
    image_size=1280
)
```
- **Time**: ~100 seconds
- **Use**: Publication-quality results

---

## Reproducing Results

To generate all parameter analysis configurations:

```bash
python generate_docs_images.py input/forest2.jpeg
```

This script:
1. **Runs 13 configurations** systematically with different parameters
2. **Generates web-optimized images** for GitHub Pages
3. **Organizes results** into the documentation structure
4. **Creates consistent naming** for easy reference

### Sweep Configuration Summary

| Configuration | Model | Stride | Threshold | Refinement | Purpose |
|--------------|-------|--------|-----------|------------|---------|
| basic_example | base | 4 | 5.0% | slic | Methodology baseline |
| stride_comparison_str2 | giant | 2 | 5.0% | slic | Quality comparison |
| stride_comparison_str4 | giant | 4 | 5.0% | slic | Speed comparison |
| model_comparison_small | small | 2 | 5.0% | slic | Model size study |
| model_comparison_base | base | 2 | 5.0% | slic | Model size study |
| model_comparison_large | large | 2 | 5.0% | slic | Model size study |
| model_comparison_giant | giant | 2 | 5.0% | slic | Model size study |
| elbow_threshold_2_5 | giant | 2 | 2.5% | slic | Sensitivity study |
| elbow_threshold_5_0 | giant | 2 | 5.0% | slic | Sensitivity study |
| elbow_threshold_10_0 | giant | 2 | 10.0% | slic | Sensitivity study |
| elbow_threshold_20_0 | giant | 2 | 20.0% | slic | Sensitivity study |
| refine_with_slic | giant | 2 | 5.0% | slic | Refinement study |
| refine_none | giant | 2 | 5.0% | none | Refinement study |

All configurations use DINOv3 models with automatic K-selection and web-optimized output for GitHub Pages display.

---

## DINOv3 Feature Extraction Pipeline

### Core Feature Extraction

```python
# Core feature extraction process
with torch.no_grad():
    # DINOv3 uses attention features for v3 (equivalent to v1.5)
    attn_choice = "none" if version == "v1" else "o"
    features_out = model.forward_sequential(image_tensor, attn_choice=attn_choice)
    
    # DINOv3 adapter returns a dictionary with patch features
    if isinstance(features_out, dict):
        patch_features = features_out["x_norm_patchtokens"]
        attn_features = features_out.get("x_patchattn", None)
        # DINOv3 features are already in spatial format (H, W, D)
```

### Model Architecture Comparison

| Component | DINOv2 (Legacy) | DINOv3 (Current) |
|-----------|-----------------|------------------|
| **Architecture** | ViT-B/14, ViT-L/14 | ViT-S/16, ViT-B/16, ViT-L/16, ViT-H+/16 |
| **Feature Dim** | 768D (Base) | 768D (Base), 1280D (Giant) |
| **Patch Size** | 14x14 pixels | 16x16 pixels |
| **Training Data** | ImageNet-22k | LVD-142M (much larger) |
| **Loading** | Direct hub access | HuggingFace + fallback strategies |
| **Robustness** | Single loading path | Multi-strategy with error recovery |

### Key Improvements in v3

1. **Robust Model Loading**: Multi-strategy loading (Meta hub → HuggingFace → random weights)
2. **NaN Prevention**: Automatic LinearKMaskedBias initialization fix
3. **Better Features**: DINOv3 trained on 142M images vs ImageNet-22k
4. **Consistent K-Selection**: Elbow method eliminates stride-dependent paradoxes
5. **Production Ready**: Comprehensive error handling and logging

---

## Elbow Method Implementation

### Fixed Algorithm

```python
def find_optimal_k_elbow(features_flat, k_range=(3, 10), elbow_threshold=5.0):
    """
    Find optimal K using percentage-based diminishing returns.
    
    Args:
        elbow_threshold: Percentage threshold (e.g., 5.0 = 5.0%)
    """
    wcss_values = []
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(features_flat)
        wcss_values.append(kmeans.inertia_)
    
    # Calculate percentage decrease between consecutive K values
    pct_decrease = []
    for i in range(1, len(wcss_values)):
        pct_change = ((wcss_values[i-1] - wcss_values[i]) / wcss_values[i-1]) * 100
        pct_decrease.append(pct_change)
    
    # Find elbow point where improvement drops below threshold
    threshold_idx = len(pct_decrease) - 1  # Default to last K if none found
    for i, pct in enumerate(pct_decrease):
        if pct < elbow_threshold:  # Direct percentage comparison
            threshold_idx = i
            break
    
    # Fix index mapping: pct_decrease[i] represents transition from k_values[i] to k_values[i+1]
    elbow_idx = threshold_idx + 1
    optimal_k = k_range[0] + elbow_idx
    
    return optimal_k, wcss_values
```

 

## Model Loading Strategy

The DINOv3 adapter uses a robust multi-strategy approach:

```python
# Strategy 1: Original Meta hub (often blocked by 403 errors)
try:
    backbone = dinov3_backbones.dinov3_vitb16(pretrained=True)
except:
    # Strategy 2: HuggingFace safetensors with parameter mapping
    weight_loader = HuggingFaceWeightLoader("facebook/dinov3-vitb16-pretrain-lvd1689m")
    backbone = dinov3_backbones.dinov3_vitb16(pretrained=False)
    weight_loader.apply_weights_to_model(backbone)
    # Strategy 3: Random weights (fallback for testing)
```

This ensures the pipeline works even when Meta's servers are inaccessible, which is common in production environments.

### Implementation vs Documentation

| Component | Previous Docs | Actual Implementation |
|-----------|---------------|----------------------|
| **Model Architecture** | DINOv2 (dinov2_vitb14) | DINOv3 (dinov3_vitb16) |
| **Feature Dimensions** | 768D (Base) | 768D (Base), 1280D (Giant) |
| **Parameter Counts** | ~86M (Base) | 86M (Base), 1.1B (Giant) |
| **Patch Size** | 14x14 pixels | 16x16 pixels |
| **Version** | v1.5 | v3 |
| **Elbow Threshold** | 0.05 (decimal) | 5.0 (percentage) |
| **Model Loading** | Single strategy | Multi-strategy with fallbacks |
| **Error Handling** | Basic | Comprehensive with NaN prevention |

The documentation has been updated to reflect the actual DINOv3 implementation with correct model names, parameter counts, and configuration values.
