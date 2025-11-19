# Claude Development Learnings

## Project Context
Tree segmentation using DINOv3 for aerial drone imagery. Modern v3.0 architecture with clean API.

## Version Roadmap Summary

**Current status**: V1.5 (baseline locked)
**Sequencing**: V1.5 → V2 → V3 → V4 → V5 → (V6 parallel research)

### Version Roles
- **V1.5** (Baseline): DINOv3 + K-means + SLIC. Frozen reference point.
- **V2** (Refinement): Soft/EM refinement in feature space. Complementary to SLIC (image space).
- **V3** (Tree Logic): Vegetation filtering, IoU-based cluster selection, instance segmentation.
- **V4** (Supervised Baseline): Mask2Former (NOT SAM). Requires >40 GB RAM. Comparison point only.
- **V5** (Multispectral): NDVI/GNDVI/NDRE gating and late fusion with DINOv3 tokens.
- **V6** (Clustering Variants): Research spike exploring spherical/soft/DP-means as K-means replacements.

### Key Clarifications
- **SLIC vs V2**: SLIC operates in image space (RGB edges), V2 in feature space (DINOv3 embeddings). Both can be combined.
- **Soft k-means**: Lives in V6 as a clustering algorithm, NOT in V2 as refinement.
- **V4 is Mask2Former**: V4 implements DINOv3 + Mask2Former supervised baseline for comparison. SAM is future work (not implemented).
- **V2 before V3**: V2 improves general clustering quality; V3 applies tree-specific logic. V3's gate compares against V2.

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
