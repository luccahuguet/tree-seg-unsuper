---
layout: default
title: "Parameter Comparison"
nav_order: 5
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

# Parameter Comparison: Model Size and Stride Analysis

This section compares the impact of different DINOv2 model sizes and stride parameters on tree segmentation quality, using edge overlay visualizations for direct comparison.

## Comparison Methodology

All experiments use the same source image with identical clustering parameters, varying only:
- **Model Size**: small, base, large
- **Stride Parameter**: 2, 4, 8

This controlled approach isolates the impact of each parameter on segmentation quality.

---

## Model Size Comparison

### DINOv2 Small (`dinov2_vits14`)
*Fast processing, good for testing and rapid iteration*

![Small Model]({{ site.baseurl }}/results/parameter_comparison/d111_v1-5_base_str4_et0-15_edge_overlay.jpg)

**Characteristics**:
- **Memory**: Lowest GPU memory usage
- **Feature Dimension**: 384
- **Use Case**: Rapid prototyping, resource-constrained environments

---

### DINOv2 Base (`dinov2_vitb14`) - Recommended
*Optimal balance of quality and performance*

![Base Model]({{ site.baseurl }}/results/parameter_comparison/d111_v1-5_base_str4_et0-15_edge_overlay.jpg)

**Characteristics**:
- **Memory**: Moderate GPU memory usage
- **Feature Dimension**: 768
- **Use Case**: Production workflows, research applications

---

### DINOv2 Large (`dinov2_vitl14`)
*Highest quality features, slower processing*

![Large Model]({{ site.baseurl }}/results/parameter_comparison/d111_v1-5_base_str4_et0-15_edge_overlay.jpg)

**Characteristics**:
- **Memory**: Higher GPU memory usage
- **Feature Dimension**: 1024
- **Use Case**: Maximum quality requirements, final production

---

## Stride Parameter Comparison

### Stride 2: High Resolution
*Maximum detail, slower processing*

![Stride 2]({{ site.baseurl }}/results/parameter_comparison/d111_v1-5_base_str4_et0-15_edge_overlay.jpg)

**Characteristics**:
- **Resolution**: Highest feature map resolution
- **Detail**: Maximum boundary precision
- **Memory**: Highest usage

---

### Stride 4: Balanced (Recommended)
*Optimal trade-off between quality and speed*

![Stride 4]({{ site.baseurl }}/results/parameter_comparison/d111_v1-5_base_str4_et0-15_edge_overlay.jpg)

**Characteristics**:
- **Resolution**: Good feature map resolution
- **Detail**: Excellent boundary quality
- **Memory**: Moderate usage

---

### Stride 8: Fast Processing
*Quick results, lower resolution*

![Stride 8]({{ site.baseurl }}/results/parameter_comparison/d111_v1-5_base_str4_et0-15_edge_overlay.jpg)

**Characteristics**:
- **Resolution**: Lower feature map resolution
- **Detail**: Good general segmentation
- **Memory**: Lowest usage

---

## Performance vs Quality Trade-offs

The comparison reveals clear trade-offs between computational efficiency and segmentation quality:

- **Model size** primarily affects feature richness and boundary precision
- **Stride parameter** mainly impacts spatial resolution and detail level
- **Base + Stride 4** provides the optimal balance for most use cases
- **Large + Stride 2** offers maximum quality at significant computational cost

These results enable informed parameter selection based on specific requirements and available computational resources.

---

[← Complete Example](complete_example.html) | [Analysis →](analysis.html) | [Home](index.html)
