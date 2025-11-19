# Claude Development Learnings

## Project Context
Species-level vegetation segmentation using DINOv3 for aerial imagery. Goal: Cluster vegetation by species/type, separate from non-vegetation.

## Critical Findings (V3.1 Feature Analysis)

**DINOv3 naturally encodes vegetation with 0.95+ correlation!**
- Tested on OAM-TCD aerial imagery (10cm GSD)
- K-means clustering on DINOv3 features automatically separates vegetation from non-vegetation
- Simple cluster-level ExG filter (threshold = 0.10) is sufficient
- **No complex multi-index fusion needed** - DINOv3 already learned vegetation patterns

**Implication**: V3.1 requires minimal filtering (~20 lines of code), not complex feature engineering.

See `docs/text/dinov3_vegetation_analysis.md` for detailed analysis.

## Version Roadmap Summary

**Current status**: V3.1 (species clustering) - Feature analysis complete
**Updated Sequencing**: V1.5 → V3.1 (species clustering) → V2 (optional refinement) → V5 (multispectral)

### Version Roles (Updated)
- **V1.5** (Baseline): DINOv3 + K-means + SLIC. Frozen reference point.
- **V3** (DEPRECATED): Instance segmentation via watershed - wrong approach for species clustering.
- **V3.1** (Active): Species-level semantic clustering via vegetation filtering (cluster-level ExG).
- **V2** (Future): Soft/EM refinement in feature space. May not be needed if V3.1 works well.
- **V4** (Supervised Baseline): Mask2Former (NOT SAM). Requires >40 GB RAM. Comparison point only.
- **V5** (Multispectral): NDVI/GNDVI/NDRE for better species distinction (needs NIR imagery).
- **V6** (Clustering Variants): Research spike exploring alternatives to K-means.

### Key Clarifications
- **V3 vs V3.1**: V3 did instance segmentation (wrong), V3.1 does semantic species clustering (correct).
- **Dataset mismatch**: OAM-TCD has "group of trees" labels (semantic), not individual instances. V3 metrics were meaningless.
- **Goal clarification**: We want species-level regions (pines, firs, grass), not individual tree crowns.
- **DINOv3 power**: Pre-trained features capture vegetation/texture/color patterns without fine-tuning.

## Key Architecture Decisions

### 1. Configuration Management
- **Dataclass Config**: Centralized, type-safe parameters with validation
- **Property methods**: Smart getters like `model_display_name` for name mapping
- **Validation**: Early error detection with clear messages
- **Quality Presets**: `tree_seg/presets.py` - PRESETS dict (quality/balanced/speed)
- **Research Grids**: `tree_seg/evaluation/grids.py` - GRIDS dict (ofat/smart/full)

### 2. Result Objects  
- **SegmentationResults**: Structured returns instead of tuples
- **OutputPaths**: File path management with helper methods
- **ElbowAnalysisResults**: Type-safe K-selection data

### 3. File Management
- **OutputManager**: Centralized filename generation and file discovery
- **Config-based naming**: `{hash}_{version}_{model}_{stride}_{clustering}_type.png`
- **Collision prevention**: SHA1 hash of source filename

### 4. API Design
- **TreeSegmentation**: Full-featured class for advanced usage
- **segment_trees()**: Convenience function for quick usage
- **Clean interfaces**: No parameter explosion (eliminated 15+ parameter functions)

## Technical Patterns

### Error Handling
- Validation at config creation, not runtime
- Graceful fallbacks for file operations
- Clear error messages with context

### File Naming Strategy
```python
# Smart filename: a3f7_v3_base_str4_et3-5_k5_segmentation_legend.png
# Components: hash_version_model_stride_method_clusters_type.png
```

### Modern Python Practices
- Type hints throughout
- Dataclasses for structured data
- Context managers for resources
- Pathlib for file operations

## Elbow Threshold Guidelines
- `0.05-0.1`: Sensitive (finds more clusters)
- `0.1-0.2`: Balanced (recommended)
- `0.2-0.3`: Conservative (fewer clusters)
- `3.0`: Very conservative (legacy default)

## Performance Considerations
- Lazy model initialization
- File cleanup utilities to prevent storage bloat
- Efficient numpy operations for image processing
- Memory-conscious matplotlib figure management

## Development Notes
- Kaggle-optimized workflow (no local execution focus)
- No legacy support needed - full control environment
- Modern visualization with config-based overlays
- Automatic file discovery for notebook display

## Documentation Guidelines
- **Always verify paper links**: Use WebFetch to validate arXiv URLs before adding to docs
- **Search for correct papers**: Use WebSearch with proper keywords to find actual paper references
- **Never assume arXiv IDs**: Paper IDs change and papers may not exist at guessed URLs

## Usage Patterns
```python
# Quick usage
results = segment_trees("image.jpg", model="base", auto_k=True)

# Advanced usage
config = Config(model_name="base", elbow_threshold=5.0)
segmenter = TreeSegmentation(config)
results, paths = segmenter.process_and_visualize("image.jpg")

# Using quality presets
from tree_seg import PRESETS
config = Config(**PRESETS['balanced'])  # or 'quality', 'speed'
segmenter = TreeSegmentation(config)

# Satellite-optimized for maximum accuracy
config = Config(model_name="mega", elbow_threshold=0.05)  # ViT-7B/16
```
