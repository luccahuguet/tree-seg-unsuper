---
layout: default
title: "Troubleshooting"
nav_order: 5
---

# Troubleshooting Guide

## Common Issues

### Processing Too Fast (No Real Computation)
**Symptoms:**
- Processing completes in 0.06-0.14s (should be 1-2s)
- No GPU fan noise or high memory usage
- Peak GPU memory only ~94MB (should be 200MB+)

**Cause:** Model not loading properly, falling back to random weights

**Solution:**
```bash
# Check if model loads correctly
uv run python -c "
from tree_seg.models import initialize_model
import torch
model = initialize_model(4, 'base', torch.device('cpu'))
print('✅ Model loaded successfully')
"
```

### Identical Results for All Images
**Symptoms:**
- WCSS values identical (e.g., 0.78) for all images
- All extracted features are NaN values
- Clustering produces meaningless noise

**Cause:** NaN propagation in feature extraction

**Solution:**
1. Check feature extraction:
```bash
# Enable verbose mode to see feature shapes
uv run python run_segmentation.py input/ base output --verbose
```

2. Look for NaN warnings in output
3. If NaN detected, the model will auto-fix LinearKMaskedBias layers

### Import Errors
**Symptoms:**
```
TypeError: non-default argument 'image_path' follows default argument
```

**Solution:**
```bash
# Test basic imports
uv run python -c "import tree_seg; print('✅ Works')"
```

### CUDA Out of Memory
**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Use smaller model
uv run python run_segmentation.py input/ small output

# Reduce image size
uv run python run_segmentation.py input/ base output --image-size 896

# Use PCA to reduce feature dimensions
uv run python run_segmentation.py input/ base output --pca-dim 128
```

### Model Access Issues
**Symptoms:**
```
HTTP 403 Forbidden
```

**Solution:**
1. Get Meta AI access approval at https://ai.meta.com/llama/
2. Login to Hugging Face: `huggingface-cli login`
3. Wait for approval (usually within hours)

## Performance Indicators

### Healthy Processing
- **Time**: 1-2 seconds per image
- **GPU Memory**: 200-400MB peak usage
- **Features**: Real value ranges (e.g., -43.4 to 44.3)
- **WCSS**: Different values per image (e.g., 197205 → 173518)

### Problematic Processing  
- **Time**: <0.2 seconds (too fast)
- **GPU Memory**: <100MB (too low)
- **Features**: All NaN or identical values
- **WCSS**: Identical values (e.g., 0.78 for all images)

## Quick Fixes

### Reset Everything
```bash
# Clear cache and reinstall
rm -rf ~/.cache/torch/hub/
rm -rf ~/.cache/huggingface/
uv sync --reinstall
```

### Test Installation
```bash
# Basic test
uv run python -c "import tree_seg; print('✅ Works')"

# Full pipeline test
uv run python run_segmentation.py input/ small output --verbose
```

### Debug Mode
```bash
# Enable all debugging output
uv run python run_segmentation.py input/ base output --verbose --metrics
```

## Getting Help

If issues persist:
1. Check the verbose output for specific error messages
2. Verify GPU memory and processing time indicators
3. Test with the smallest model first (`small`)
4. Ensure HuggingFace access is properly configured

Most issues are resolved by proper model access setup and using appropriate model sizes for available hardware.