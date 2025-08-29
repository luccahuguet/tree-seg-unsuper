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
uv run python run_segmentation.py input/ base output --verbose

# Test installation
uv run python -c "import tree_seg; print('✅ Works')"
```

## Overview

This research presents a systematic study of unsupervised tree segmentation using DINOv3 Vision Transformers for aerial drone imagery. Our approach eliminates the need for manual annotations while achieving high-quality tree boundary detection through intelligent clustering of self-supervised features.

## Research Objectives

1. **Evaluate DINOv2 effectiveness** for forestry applications across multiple model sizes
2. **Develop automatic K-selection** methodology for optimal cluster number determination  
3. **Establish performance benchmarks** across computational vs. quality trade-offs
4. **Create reproducible pipeline** for systematic tree segmentation analysis

## Key Contributions

- **🔬 Empirical Analysis**: Systematic comparison of Small, Base, Large, and Giant DINOv2 models
- **📊 K-Selection Discovery**: Higher-dimensional features lead to more granular clustering (K=4→5→5→6)
- **⚖️ Performance Trade-offs**: Quantified diminishing returns beyond Base model for most applications
- **🛠️ Methodological Framework**: Production-ready pipeline with automatic parameter optimization

## Technical Innovation

**Self-Supervised Features**: DINOv3 Vision Transformers eliminate manual annotation requirements  
**Intelligent Clustering**: Elbow method with optimized thresholds for forest imagery  
**Multi-Scale Analysis**: Comprehensive evaluation across four model architectures  
**Professional Visualization**: Publication-ready outputs with systematic naming conventions

## Research Impact

This work demonstrates how modern self-supervised learning can advance forestry remote sensing by:
- **Reducing annotation costs** through unsupervised methodology
- **Scaling to large datasets** with automatic processing pipelines  
- **Providing quantitative guidance** for model selection in resource-constrained environments
- **Establishing baseline performance** for future algorithmic development

The systematic model comparison reveals that Base DINOv3 provides optimal balance for most forestry applications, while larger models offer marginal improvements at significant computational cost.


