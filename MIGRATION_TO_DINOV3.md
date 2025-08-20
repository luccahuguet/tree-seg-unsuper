# DINOv3 Integration Reference

## 🚀 Project Status

This project uses **DINOv3** (Meta AI, 2025) for state-of-the-art tree segmentation performance.

### Key Benefits
- **70% reduction** in tree canopy height measurement error vs DINOv2
- **Satellite-optimized** training on 493M satellite images
- **5 model sizes** from 21M to 6.7B parameters

---

## 🔧 Model Sizes

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

---

## 📋 Installation

```bash
# Initialize submodules
git submodule update --init --recursive

# Install dependencies
pip install -e .
```

---

## 🧪 Quick Test

```bash
# Test installation
python3 test_imports_only.py

# Test basic functionality
python3 -c "from tree_seg import Config; print('✅ DINOv3 ready')"
```

---

## 📊 Architecture Notes

- **DINOv3 submodule**: `dinov3/` (read-only reference)
- **Adapter pattern**: `tree_seg/models/dinov3_adapter.py`
- **No API changes**: Same public interface as before
- **Version**: v3 (was v1.5 with DINOv2)