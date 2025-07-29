---
layout: default
title: "Methodology"
nav_order: 2
---

<nav class="tree-seg-navbar">
  <div class="navbar-container">
    <a href="{{ '/' | relative_url }}" class="navbar-home">🌳 Tree Segmentation</a>
    <div class="navbar-links">
      <a href="{{ '/methodology' | relative_url }}">Methodology</a>
      <a href="{{ '/results' | relative_url }}">Results</a>
      <a href="{{ '/complete_example' | relative_url }}">Example</a>
      <a href="{{ '/parameter_comparison' | relative_url }}">Comparison</a>
      <a href="{{ '/analysis' | relative_url }}">Analysis</a>
    </div>
  </div>
</nav>

<style>
.tree-seg-navbar {
  background-color: #1e1e1e;
  border-bottom: 2px solid #00ff00;
  padding: 0.5rem 0;
  margin-bottom: 2rem;
}

.navbar-container {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 1rem;
}

.navbar-home {
  color: #00ff00 !important;
  text-decoration: none !important;
  font-weight: bold;
  font-size: 1.2rem;
}

.navbar-links {
  display: flex;
  gap: 1.5rem;
}

.navbar-links a {
  color: #fff !important;
  text-decoration: none !important;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  transition: background-color 0.3s ease;
}

.navbar-links a:hover {
  background-color: #333;
  color: #00ff00 !important;
}

@media (max-width: 768px) {
  .navbar-container {
    flex-direction: column;
    gap: 1rem;
  }
  
  .navbar-links {
    flex-wrap: wrap;
    justify-content: center;
    gap: 1rem;
  }
}
</style>

# Methodology

## Pipeline Overview

Our tree segmentation pipeline consists of four main stages:

1. **Feature Extraction** - DINOv2 Vision Transformer
2. **Clustering** - K-means with automatic K selection
3. **Post-processing** - Upsampling and refinement
4. **Visualization** - Multi-format output generation

## Technical Details

### 1. Feature Extraction

**Model**: DINOv2 Vision Transformer (v1.5)
- **Variant**: Base model (dinov2_vitb14)
- **Features**: Patch embeddings + attention features
- **Dimensionality**: 780-dimensional feature space
- **Stride**: 4 (balance of quality vs. speed)

### 2. Automatic K-Selection

**Elbow Method** with intelligent thresholding:
- **K Range**: 3-10 clusters (optimized for tree imagery)
- **Threshold**: 0.1 (sensitive detection for subtle elbows)
- **Metric**: Within-Cluster Sum of Squares (WCSS)
- **Optimization**: Percentage decrease analysis

### 3. Clustering Algorithm

**K-means Configuration**:
- **Initialization**: k-means++ for stable results
- **Iterations**: Auto-convergence
- **Random State**: 42 (reproducible results)

### 4. Post-processing

**Upsampling Strategy**:
- **Method**: High-resolution feature upsampling
- **Target**: Original image resolution
- **Interpolation**: Bilinear for smooth boundaries

## Configuration Parameters

```python
config = Config(
    model_name="base",          # DINOv2 variant
    version="v1.5",             # Algorithm version
    stride=4,                   # Feature resolution
    auto_k=True,                # Automatic K selection
    elbow_threshold=0.1,        # Sensitivity parameter
    k_range=(3, 10),            # Search range
    use_pca=False,              # Keep full feature space
    edge_width=2,               # Visualization parameter
    use_hatching=True           # Visual distinction
)
```

## Innovation Points

1. **Modern Architecture**: Type-safe configuration with validation
2. **Smart File Management**: Config-based naming prevents collisions
3. **Flexible API**: Both quick-use and advanced interfaces
4. **Professional Visualization**: Publication-ready outputs

## Implementation Quality

- **Type Safety**: Full type hints and dataclass validation
- **Error Handling**: Graceful fallbacks and clear error messages
- **Performance**: Optimized numpy operations and memory management
- **Maintainability**: Clean separation of concerns and modular design

---

[← Home](index.html) | [Results →](results.html)