---
layout: default
title: "Tree Segmentation Research"
nav_order: 1
---

{% include navbar.html %}
{% include navbar-styles.html %}

# Tree Segmentation with DINOv3

## Quick Start

```bash
# Clone and install
git clone https://github.com/luccahuguet/tree-seg-unsuper.git
cd tree-seg-unsuper
uv sync

# Run segmentation
uv run python run_segmentation.py input/forest2.jpeg base output --verbose

# Test installation
uv run python -c "import tree_seg; print('✅ Works')"
```

## Overview

This research presents a systematic study of unsupervised tree segmentation using DINOv3 Vision Transformers for aerial drone imagery. Our approach eliminates the need for manual annotations while achieving high-quality tree boundary detection through intelligent clustering of self-supervised features.

## Research Objectives

1. **Evaluate DINOv3 effectiveness** for forestry applications across multiple model sizes (V1.5)
2. **Develop automatic K-selection** using elbow method for optimal cluster determination (V1.5)
3. **Advance beyond K-means** with unified unsupervised segmentation methods (V2)
4. **Implement dynamic fusion** techniques for adaptive segmentation (V3)
5. **Enable multispectral analysis** for enhanced forest monitoring (V4)
6. **Establish performance benchmarks** across architectural versions and methods
7. **Create extensible pipeline** supporting modular segmentation approaches

## Key Contributions

- **🔬 Empirical Analysis**: Systematic comparison of Small, Base, Large, and Giant DINOv3 models
- **📊 Elbow Method Optimization**: Automatic K-selection with forest-specific thresholds
- **⚖️ Performance Trade-offs**: Quantified model size vs. quality relationships
- **🛠️ Production Pipeline**: Ready-to-use framework with web optimization

## Technical Innovation

**Self-Supervised Features**: DINOv3 Vision Transformers eliminate manual annotation requirements  
**Intelligent Clustering**: Elbow method with forest-optimized thresholds (5.0% default)  
**Multi-Scale Analysis**: Comprehensive evaluation across four model architectures  
**Professional Visualization**: Web-optimized outputs with systematic naming conventions

## Research Impact

This work demonstrates how modern self-supervised learning can advance forestry remote sensing by:
- **Reducing annotation costs** through unsupervised methodology
- **Scaling to large datasets** with automatic processing pipelines  
- **Providing quantitative guidance** for model selection in resource-constrained environments
- **Establishing baseline performance** for future algorithmic development

The systematic model comparison reveals optimal configurations for different use cases, with automatic K-selection ensuring consistent results across varying forest imagery.

## Development Roadmap

### **Current: V1.5 (DINOv3 + K-means)**
- **Baseline**: Solid foundation with state-of-the-art features
- **Architecture**: Clean, extensible design ready for advanced methods

### **Next: V2 (U2Seg)**
**Target**: `tree_seg/clustering/u2seg.py`
- Advanced unsupervised segmentation beyond K-means
- Integration point: `core/segmentation.py` routing logic
- **Paper**: [U2Seg: Unsupervised Universal Image Segmentation](https://arxiv.org/abs/2312.17243)

### **Future: V3 (DynaSeg) + V4 (Multispectral)**
**Target**: `tree_seg/clustering/dynaseg.py` + `tree_seg/models/multispectral_adapter.py`
- Dynamic fusion methods + multi-band imagery support
- Architecture supports both through modular design
- **Paper**: [DynaSeg: A Deep Dynamic Fusion Method for Unsupervised Image Segmentation](https://arxiv.org/abs/2405.05477)

---

## Documentation Structure

- **[Methodology]({{ '/methodology' | relative_url }})**: Technical pipeline and algorithm details
- **[Complete Example]({{ '/complete_example' | relative_url }})**: Full workflow demonstration with all outputs  
- **[Parameter Analysis]({{ '/parameter_analysis' | relative_url }})**: Comprehensive study of all 13 configurations, model comparisons, and performance benchmarks
- **[Technical Implementation]({{ '/technical_implementation' | relative_url }})**: Configuration profiles, code implementation details, and reproduction instructions

