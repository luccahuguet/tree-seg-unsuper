---
layout: default
title: "Model Comparison"
nav_order: 6
---

{% include navbar.html %}
{% include navbar-styles.html %}

# DINOv3 Model Comparison

## Available Models

Our pipeline supports four DINOv3 Vision Transformer variants, each offering different trade-offs between computational cost and segmentation quality.

### Model Specifications

| Model | Parameters | Feature Dim | Patch Size | Recommended Use |
|-------|------------|-------------|------------|-----------------|
| **Small** | 22M | 384 | 14×14 | Fast prototyping, resource-constrained |
| **Base** | 86M | 768 | 14×14 | **Recommended default** - best balance |
| **Large** | 307M | 1024 | 14×14 | High-quality applications |
| **Giant** | 1.1B | 1536 | 14×14 | Maximum quality, research use |

## Performance Characteristics

### Computational Requirements

**Processing Time** (approximate, for 1024×1024 image):
- **Small**: ~2-3 seconds
- **Base**: ~4-6 seconds  
- **Large**: ~12-15 seconds
- **Giant**: ~25-35 seconds

**Memory Usage**:
- **Small**: ~2GB VRAM
- **Base**: ~4GB VRAM
- **Large**: ~8GB VRAM
- **Giant**: ~12GB VRAM

### Segmentation Quality

**Typical K-Selection Results** (automatic elbow method):
- **Small**: K=4-6 clusters
- **Base**: K=5-7 clusters
- **Large**: K=6-8 clusters  
- **Giant**: K=7-9 clusters

*Higher-dimensional feature spaces tend to discover more granular tree species distinctions*

## Model Selection Guidelines

### 🚀 **Small Model** - Fast Prototyping
**Best for:**
- Initial testing and development
- Resource-constrained environments
- Real-time applications
- Educational demonstrations

**Limitations:**
- Lower feature dimensionality may miss subtle tree distinctions
- Less robust to varying lighting conditions

### ⭐ **Base Model** - Recommended Default
**Best for:**
- Most production applications
- Balanced quality/performance needs
- Standard forestry analysis
- General-purpose tree segmentation

**Advantages:**
- Excellent quality-to-cost ratio
- Robust across different forest types
- Reasonable computational requirements

### 🎯 **Large Model** - High Quality
**Best for:**
- High-precision forestry research
- Detailed species classification needs
- Applications where quality is critical
- Sufficient computational resources available

**Trade-offs:**
- 3x computational cost vs Base
- Marginal quality improvements in many cases

### 🔬 **Giant Model** - Maximum Quality
**Best for:**
- Research applications
- Benchmark comparisons
- Maximum possible segmentation quality
- Computational resources not a constraint

**Considerations:**
- 6-8x computational cost vs Base
- Diminishing returns for many practical applications
- Requires significant GPU memory

## Configuration Examples

### Quick Start (Base Model)
```bash
python run_segmentation.py input/forest.jpg base output --verbose
```

### High Quality (Giant Model)
```bash
python run_segmentation.py input/forest.jpg giant output --stride 2 --verbose
```

### Fast Processing (Small Model)
```bash
python run_segmentation.py input/forest.jpg small output --stride 4 --verbose
```

### Research Comparison (All Models)
```bash
# Use sweep configuration for systematic comparison
python run_segmentation.py input/forest.jpg base output --sweep model_comparison.json
```

## Technical Implementation

### Feature Extraction
All models use the same DINOv3 architecture with different scales:
- **Patch Embeddings**: 14×14 pixel patches
- **Attention Features**: Multi-head self-attention outputs
- **Feature Fusion**: Concatenated patch + attention representations

### Clustering Adaptation
The elbow method automatically adapts to different feature dimensionalities:
- **Threshold**: 3.5% (consistent across models)
- **K Range**: 3-10 clusters (forest-optimized)
- **Safety Bounds**: Prevents unrealistic cluster counts

### Performance Optimization
- **Stride Parameter**: Adjustable feature resolution (2 or 4)
- **Web Optimization**: Automatic JPEG conversion for documentation
- **Memory Management**: Efficient batch processing for large images

## Recommendations by Use Case

### 📊 **Research & Benchmarking**
- **Primary**: Giant model (stride 2)
- **Comparison**: All models for systematic evaluation
- **Focus**: Maximum quality and comprehensive analysis

### 🏭 **Production Applications**
- **Primary**: Base model (stride 4)
- **Alternative**: Large model if quality critical
- **Focus**: Reliable performance with reasonable costs

### 🎓 **Educational & Demos**
- **Primary**: Small model (stride 4)
- **Benefits**: Fast results, low resource requirements
- **Focus**: Understanding methodology over maximum quality

### 🌲 **Forestry Operations**
- **Primary**: Base or Large model (stride 4)
- **Considerations**: Balance accuracy needs with processing time
- **Focus**: Practical tree boundary detection

---

*Model selection should consider your specific quality requirements, computational constraints, and processing time needs. The Base model provides the best starting point for most applications.*