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
# Install and run
uv sync
uv run python run_segmentation.py input/forest2.jpeg base output --verbose

# Test installation
uv run python -c "import tree_seg; print('✅ Works')"
```

## Overview

This research presents a systematic study of unsupervised tree segmentation using DINOv3 Vision Transformers for aerial drone imagery. Our approach eliminates the need for manual annotations while achieving high-quality tree boundary detection through intelligent clustering of self-supervised features.

## Research Objectives

1. **Evaluate DINOv3 effectiveness** for forestry applications across multiple model sizes
2. **Develop automatic K-selection** using elbow method for optimal cluster determination  
3. **Establish performance benchmarks** across computational vs. quality trade-offs
4. **Create reproducible pipeline** for systematic tree segmentation analysis

## Key Contributions

- **🔬 Empirical Analysis**: Systematic comparison of Small, Base, Large, and Giant DINOv3 models
- **📊 Elbow Method Optimization**: Automatic K-selection with forest-specific thresholds
- **⚖️ Performance Trade-offs**: Quantified model size vs. quality relationships
- **🛠️ Production Pipeline**: Ready-to-use framework with web optimization

## Technical Innovation

**Self-Supervised Features**: DINOv3 Vision Transformers eliminate manual annotation requirements  
**Intelligent Clustering**: Elbow method with forest-optimized thresholds (3.5% default)  
**Multi-Scale Analysis**: Comprehensive evaluation across four model architectures  
**Professional Visualization**: Web-optimized outputs with systematic naming conventions

## Research Impact

This work demonstrates how modern self-supervised learning can advance forestry remote sensing by:
- **Reducing annotation costs** through unsupervised methodology
- **Scaling to large datasets** with automatic processing pipelines  
- **Providing quantitative guidance** for model selection in resource-constrained environments
- **Establishing baseline performance** for future algorithmic development

The systematic model comparison reveals optimal configurations for different use cases, with automatic K-selection ensuring consistent results across varying forest imagery.

---

## Documentation Structure

- **[Methodology]({{ '/methodology' | relative_url }})**: Technical pipeline and algorithm details
- **[Complete Example]({{ '/complete_example' | relative_url }})**: Full workflow demonstration with all outputs  
- **[Parameter Analysis]({{ '/parameter_analysis' | relative_url }})**: Comprehensive study of all 12 configurations, model comparisons, and performance benchmarks


