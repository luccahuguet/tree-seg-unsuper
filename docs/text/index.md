# Documentation Index

## Getting Started

### [CLI Usage](cli_usage.md)
Quick start guide for running tree segmentation via command line.
- Installation
- Common workflows
- Configuration options

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
- V1.5: Baseline (DINOv3 + K-means)
- V2: Soft/EM refinement
- V3: Species clustering (vegetation filtering)
- V4: Mask2Former baseline
- V5: Multispectral
- V6: Clustering variants

### [V3 Species Clustering](v3_species_clustering.md)
Complete V3 documentation: background, implementation, usage.
- Why species clustering (not instance detection)
- Vegetation filtering approach
- Validation results

### [Paper Timeline](paper_timeline.md)
Project timeline and version descriptions for paper.
- Weekly schedule
- Version goals

## Research & Analysis

### [DINOv3 Vegetation Analysis](dinov3_vegetation_analysis.md)
Key finding: DINOv3 naturally encodes vegetation (0.95+ correlation).
- Methodology
- Results
- Implications for V3
- Dataset requirements

### [K-means Successors](kmeans_successors.md)
Alternative clustering algorithms exploration.
- Spherical K-means
- DP-means
- Soft K-means
- Implementation notes

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

### [V2 Head Refinement Plan](w2_head_refine_plan.md)
Planning document for V2 implementation.
- Soft/EM refinement approach
- Integration strategy
- Expected improvements

---

## Quick Reference

**User guides:** `cli_usage.md`, `benchmarking.md`

**Technical docs:** `architecture.md`, `version_roadmap.md`

**V3 docs:** `v3_species_clustering.md`, `dinov3_vegetation_analysis.md`

**Research:** `dinov3_vegetation_analysis.md`, `kmeans_successors.md`, `week1_results.md`

**Datasets:** `oam_tcd_integration.md`, `dataset_search_context.md`

**Planning:** `w2_head_refine_plan.md`, `paper_timeline.md`
