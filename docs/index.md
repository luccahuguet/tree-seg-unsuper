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

# Run segmentation (processes every image in `data/inputs/` → `data/outputs/`)
uv run python main.py --verbose

# Test installation
uv run python -c "import tree_seg; print('✅ Works')"
```

## Overview

This research presents a systematic study of unsupervised tree segmentation using DINOv3 Vision Transformers for aerial drone imagery. Our approach eliminates the need for manual annotations while achieving high-quality tree boundary detection through intelligent clustering of self-supervised features.

## Research Objectives

1. **Evaluate DINOv3 effectiveness** for forestry applications across multiple model sizes (V1.5)
2. **Develop automatic K-selection** using elbow method for optimal cluster determination (V1.5)
3. **Advance beyond K-means** with feature-space refinement methods (V2) and clustering algorithm exploration (V6)
4. **Implement tree-focused filtering** with vegetation gating and instance segmentation (V3)
5. **Enable multispectral analysis** for enhanced forest monitoring (V5)
6. **Establish performance benchmarks** across unsupervised methods vs supervised baselines (V4)
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

- **V1.5 — Baseline (Active):** Frozen reference built on DINOv3 + K-means (optional SLIC), PCA/overlay artifacts, Hungarian-aligned metrics.
- **V2 — DINO Head Refinement:** Lightweight soft/EM refinement in feature space (DINOv3 embeddings) complementary to SLIC (image space); must beat V1.5 on both mIoU and edge-F before adoption.
- **V3 — Tree Focus (RGB):** Vegetation-gated clustering, shape/area filters, and instance masks (DT + watershed) targeting higher tree precision/recall without edge regressions.
- **V4 — Supervised Baseline (Mask2Former):** DINOv3 ViT-7B/16 + Mask2Former pretrained on ADE20K. Comparison baseline only (requires >40 GB RAM). Not SAM—SAM is future work.
- **V5 — Multispectral:** NDVI/GNDVI/NDRE gating (V5a) plus late fusion of MSI indices with DINO tokens (V5b) evaluated against V3/V4.
- **V6 — K-means Successors (Spike):** Explore clustering algorithms (spherical k-means, soft k-means as clustering choice, DP-means auto-K); only uplifted if they outscore V2 with similar runtime/VRAM. V6 outputs can feed into V2 refinement.

Full gate-driven details and pipeline composition live in `docs/text/version_roadmap.md`.

---

## Documentation Structure

- **[Methodology]({{ '/methodology' | relative_url }})**: Technical pipeline and algorithm details
- **[Complete Example]({{ '/complete_example' | relative_url }})**: Full workflow demonstration with all outputs  
- **[Parameter Analysis]({{ '/parameter_analysis' | relative_url }})**: Comprehensive study of all 13 configurations, model comparisons, and performance benchmarks
- **[Technical Implementation]({{ '/technical_implementation' | relative_url }})**: Configuration profiles, code implementation details, and reproduction instructions
