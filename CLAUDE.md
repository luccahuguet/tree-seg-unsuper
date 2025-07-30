# Claude Development Learnings

## Project Context
Tree segmentation using DINOv2 for aerial drone imagery. Modern v2.0 architecture with clean API.

## Key Architecture Decisions

### 1. Configuration Management
- **Dataclass Config**: Centralized, type-safe parameters with validation
- **Property methods**: Smart getters like `model_display_name` for name mapping
- **Validation**: Early error detection with clear messages

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
# Smart filename: a3f7_v1-5_base_str4_et3-5_k5_segmentation_legend.png
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

## Usage Patterns
```python
# Quick usage
results = segment_trees("image.jpg", model="base", auto_k=True)

# Advanced usage  
config = Config(model_name="base", elbow_threshold=0.1)
segmenter = TreeSegmentation(config)
results, paths = segmenter.process_and_visualize("image.jpg")
```