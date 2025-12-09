# Documentation Index

## Getting Started

### [CLI Usage](cli_usage.md)
Quick start guide for running tree segmentation via command line.
- Installation (uv)
- Common workflows
- Configuration options
- Results/metadata CLI

### [Benchmarking Guide](benchmarking.md)
Dataset setup and evaluation workflows.
- Kaggle authentication
- ISPRS Potsdam download
- Running semantic segmentation benchmarks

## Architecture & Design

### [Architecture](architecture.md)
Complete technical overview of the codebase.
- Module structure
- Data flow
- Design patterns
- API layers

## Methods & Versions

### [Version Roadmap](version_roadmap.md)
Master roadmap defining all versions and their relationships.
- V1.5: Baseline (DINOv3 + K-means + SLIC) ✅
- V2: Soft EM refinement ✅ **IMPLEMENTED**
- V3: Species clustering (vegetation filtering) ✅
- V4: Mask2Former baseline
- V5: Multispectral
- V6: Clustering variants

### [V2 Soft EM Refinement](w2_head_refine_plan.md) ✅ **IMPLEMENTED**
Feature space refinement using iterative soft EM.
- Temperature-scaled softmax assignments
- Iterative cluster center updates
- Optional spatial blending
- Usage: `--refine soft-em` or `--refine soft-em+slic`

### [V3 Species Clustering](v3_species_clustering.md)
Complete V3 documentation: background, implementation, usage.
- Why species clustering (not instance detection)
- Vegetation filtering approach
- Validation results

### [Tiling Implementation](tiling_implementation.md) ✨ NEW
Tile-based processing for ultra-high-resolution imagery.
- Problem: Spatial detail loss from downsampling
- Solution: Overlapping tiles with weighted feature stitching
- Performance: 41.5% pixel accuracy on FORTRESS (vs 38.9% without tiling)
- Usage: Automatic for images >2048px

### [Paper Timeline](paper_timeline.md)
Project timeline and version descriptions for paper.
- Weekly schedule
- Version goals

## Research & Analysis

### [Improvement Experiments](experiments.md)
Systematic testing of clustering and refinement methods.
- V2 soft EM implementation status
- Clustering algorithm comparisons (GMM, spectral, HDBSCAN)
- Refinement method tests (SLIC, bilateral, CRF)
- Multi-scale feature experiments
- Composable CLI usage guide

### [DINOv3 Vegetation Analysis](dinov3_vegetation_analysis.md)
Key finding: DINOv3 naturally encodes vegetation (0.95+ correlation).
- Methodology
- Results
- Implications for V3
- Dataset requirements

### [Week 1 Results](week1_results.md)
Benchmark data and results from initial experiments (Oct 2024).
- V1.5 baseline metrics
- Configuration comparisons
- Performance data

## Datasets

### [OAM-TCD Integration](oam_tcd_integration.md)
OAM-TCD dataset setup and usage.
- Dataset structure
- Download instructions
- Integration with V3
- Limitations and ideal dataset requirements

### [Dataset Search Context](dataset_search_context.md)
Dataset requirements and search criteria.
- Ideal dataset specifications (species-level semantic regions)
- Search keywords and platforms
- Evaluation criteria
- Known candidate datasets

## Planning & Future Work

### [V2 Head Refinement Plan](w2_head_refine_plan.md) ✅ **COMPLETED**
Original planning document for V2 implementation (now complete).
- Soft/EM refinement approach
- Integration strategy
- Implementation: `tree_seg/clustering/head_refine.py`

---

## Quick Reference

**User guides:** `cli_usage.md`, `benchmarking.md`

**Technical docs:** `architecture.md`, `version_roadmap.md`, `tiling_implementation.md`

**Implemented methods:**
- V1.5: Baseline (default config)
- V2: `w2_head_refine_plan.md` - Soft EM refinement ✅
- V3: `v3_species_clustering.md` - Species clustering ✅

**Experiments:** `experiments.md` (CLI guide + results)

**Research:** `dinov3_vegetation_analysis.md`, `week1_results.md` (k-means successor ideas folded into experiments)

**Datasets:** `oam_tcd_integration.md`, `dataset_search_context.md`

**Planning:** `paper_timeline.md`

---

## 🎛️ Composable CLI

**New unified interface** (Dec 2024):
```bash
# Each flag controls a different aspect
tree-seg eval data/datasets/fortress \
  --clustering kmeans|gmm|spectral|spherical|dpmeans|potts \
  --refine none|slic|soft-em|bilateral|soft-em+slic \
  --vegetation-filter \
  --supervised
```

See `experiments.md` for detailed usage examples.
