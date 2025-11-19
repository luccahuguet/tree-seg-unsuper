# Tree Segmentation Architecture

## 🎯 Project Overview

**Modern unsupervised tree segmentation** using DINOv3 Vision Transformers with K-means clustering for aerial drone imagery. Built with clean, type-safe architecture optimized for research and development.

**Key Philosophy**: Professional software patterns + Research flexibility + Zero legacy cruft

---

## 🏗️ High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │    │   DINOv3 Model   │    │   Clustering    │
│   (Image Path)  │───▶│   (Features)     │───▶│   (K-means)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Output    │◀───│  Visualization   │◀───│   Segmentation  │
│   (JPG/PNG)     │    │   (Plotting)     │    │   (Labels)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

**Data Flow**: `Image → DINOv3 Features → K-means Clustering → Visualization → Web-Optimized Output`

---

## 📁 Module Structure

### **🎪 `tree_seg/api.py` - Public Interface**
**Purpose**: Clean, user-facing API with modern patterns
```python
# Two interfaces:
TreeSegmentation(config)           # Class-based for advanced usage
segment_trees(path, model="base")  # Function for quick usage
```
**Responsibilities**: 
- Model initialization coordination
- Pipeline orchestration (process → visualize → optimize)
- Error handling and user feedback

### **🏗️ `tree_seg/core/` - Core Algorithms**

#### **`types.py` - Type System**
**Purpose**: Centralized, type-safe data structures
```python
@dataclass
class Config:           # Configuration with validation
class SegmentationResults:  # Structured return values  
class OutputPaths:      # File path management
```

#### **`segmentation.py` - Processing Engine**
**Purpose**: DINOv3 feature extraction + clustering pipeline
```python
def process_image() -> (image_np, labels_resized)
```
**Key Logic**:
- DINOv3 feature extraction (patch + attention)
- Automatic K-selection via elbow method
- K-means clustering with validation
- Spatial reshaping and upsampling

#### **`output_manager.py` - File Management**
**Purpose**: Smart filename generation and file discovery
```python
class OutputManager:
    generate_filename_prefix()     # Config-based naming
    generate_output_paths()        # All output file paths
    optimize_all_outputs()         # Web optimization
```
**Naming Strategy**: `{hash}_{version}_{model}_{stride}_{clustering}_type.png`

### **🤖 `tree_seg/models/` - DINOv3 Integration**

#### **`dinov3_adapter.py` - Model Adapter**
**Purpose**: Clean DINOv3 interface matching legacy API
```python
class DINOv3Adapter:
    forward_sequential()           # Same interface as old HighResDV2
    _load_backbone()              # Hub-based model loading
```
**Key Features**:
- Multiple model sizes (21M → 6.7B params)
- Satellite-optimized weights (SAT493M when available)
- Memory-efficient float16 processing

#### **`initialization.py` - Model Factory**
**Purpose**: Model creation and device management
```python
def initialize_model(stride, model_name, device) -> DINOv3Adapter
```

### **📊 `tree_seg/analysis/` - K-Selection**

#### **`elbow_method.py` - Automatic K Selection**
**Purpose**: Tree-specific optimal cluster count detection
```python
def find_optimal_k_elbow() -> (optimal_k, analysis_results)
```
**Algorithm**: Percentage decrease threshold with tree-specific bounds (3-8 clusters)

### **🎨 `tree_seg/visualization/` - Output Generation**

#### **`plotting.py` - Modern Visualization**
**Purpose**: High-quality scientific visualizations
```python
def generate_visualizations(results, config, output_paths)
```
**Outputs**:
- Segmentation legend (colored map + legend)
- Edge overlay (original + colored borders ± hatching)
- Side-by-side comparison (original vs segmentation)

---

## 🔄 Data Flow Deep Dive

### **1. Input Processing**
```python
Image (PIL) → Tensor (518x518) → DINOv3 Preprocessing
```

### **2. Feature Extraction** 
```python
DINOv3.forward_features() → Patch Tokens (H×W×D) → Combined Features (patch+attention)
```

### **3. Clustering Pipeline**
```python
Features → K-Selection (elbow) → K-means → Labels (H×W) → Upsampled Labels (orig_H×orig_W)
```

### **4. Visualization Generation**
```python
(Image, Labels) → 3 Visualizations → PNG Output → Web Optimization (JPG)
```

---

## 🎯 Adding New Features

### **🆕 Head Refinements & Tree-Focused Stages (V2/V3)**
**Locations**: `tree_seg/clustering/`, `tree_seg/core/segmentation.py`, `tree_seg/utils/`
```python
# V2 soft/EM head refinement
tree_seg/clustering/head_refine.py

# V3 vegetation + instance helpers
tree_seg/utils/vegetation_masks.py
tree_seg/clustering/tree_focus.py

# Route via segmentation pipeline
if config.clustering_method == "head_refine":
    return head_refine.segment(features, config)
elif config.clustering_method == "tree_focus":
    return tree_focus.segment(features, metadata, config)
```

Key notes:
- Leverage existing K-means output as the initializer for the V2 refinement stage.
- Keep shared utilities (normalization, SLIC adjacency) in `tree_seg/utils/` to minimize duplication across V2/V3.
- Ensure new modules return the same structured results (labels, diagnostics) for downstream visualization.

### **🔧 New Model Architectures**
**Location**: `tree_seg/models/`
```python
# Add new adapter
tree_seg/models/new_model_adapter.py

# Update initialization.py factory
def initialize_model():
    if "dinov3" in model_name:
        return DINOv3Adapter()
    elif "new_model" in model_name:
        return NewModelAdapter()
```

### **📊 New Analysis Methods**
**Location**: `tree_seg/analysis/`
```python
# Add analysis modules
tree_seg/analysis/silhouette_method.py
tree_seg/analysis/gap_statistic.py

# Update core to use analysis factory
```

### **🎨 New Visualization Types**
**Location**: `tree_seg/visualization/`
```python
# Add visualization functions
def generate_3d_visualization()
def generate_animation()

# Update plotting.py to include new types
```

---

## 🧪 Testing Strategy

### **🏗️ Structural Tests**
```python
test_imports_only.py              # Syntax, file structure, imports
```

### **🔧 Unit Tests** (Future)
```python
tests/test_config.py              # Config validation
tests/test_clustering.py          # K-means, elbow method
tests/test_dinov3_adapter.py      # Model interface
tests/test_output_manager.py      # File naming, paths
```

### **🎯 Integration Tests** (Future)
```python
tests/test_full_pipeline.py       # End-to-end processing
tests/test_model_variants.py      # All model sizes
```

---

## 📦 Dependencies & Integration

### **🤖 DINOv3 Submodule**
```bash
dinov3/                           # Read-only reference
├── hubconf.py                    # Model loading functions
└── dinov3/hub/backbones.py      # Available models
```
**Integration**: Path injection + hub loading in `dinov3_adapter.py`

### **📊 Key Dependencies**
```python
torch/torchvision                 # Deep learning
scikit-learn                      # K-means clustering  
matplotlib                       # Visualization
omegaconf                        # DINOv3 configuration
opencv-python                    # Image processing
```

---

## 🚀 Development Workflow

### **🎯 Feature Development Pattern**
1. **Update Config**: Add new parameters to `types.Config`
2. **Core Logic**: Implement in appropriate `core/` module  
3. **Integration**: Wire through `api.py` pipeline
4. **Visualization**: Add outputs in `visualization/`
5. **Testing**: Verify with `test_imports_only.py`

### **🏗️ Architecture Principles**
- **Dataclass configs** over parameter explosion
- **Type hints** throughout for clarity
- **Structured returns** over tuples
- **Factory patterns** for model creation
- **Clean separation** between research and engineering code

---

## 🎓 Research Roadmap Integration

### **📍 Current: V1.5 (Baseline)**
- ✅ **Status**: Frozen reference with DINOv3 + K-means (optional SLIC), PCA/overlay artifacts, and locked metrics.

### **🎯 V2: DINO Head Refinement**
- **Target**: `tree_seg/clustering/head_refine.py`
- **Focus**: Soft/EM refinement over K-means initialization plus single spatial blend (α). Operates in feature space (DINOv3 embeddings), complementary to SLIC which operates in image space (RGB).
- **Gate**: Must improve both mIoU and edge-F relative to V1.5 without major runtime/VRAM increases. Test with and without SLIC to find best configuration.

### **🌳 V3: Tree Focus (RGB)**
- **Target**: `tree_seg/clustering/tree_focus.py` + vegetation utilities.
- **Focus**: Vegetation gating (ExG/CIVE), cluster selection heuristics, shape/area filters, DT + watershed instances.
- **Gate**: Higher tree precision/recall vs V2 with stable edge metrics.

### **✨ V4: Supervised Baseline (Mask2Former)**
- **Target**: `tree_seg/models/mask2former.py` (already implemented)
- **Focus**: DINOv3 ViT-7B/16 + Mask2Former head pretrained on ADE20K for supervised comparison baseline.
- **Gate**: Document performance vs V1.5-V3 for paper discussion. No improvement gate—serves as reference point.
- **Note**: Requires >40 GB RAM. Only ViT-7B/16 has pretrained weights.

### **🌈 V5: Multispectral Expansion**
- **Targets**: `tree_seg/utils/msi_indices.py`, fusion hooks in head refine.
- **Focus**: NDVI/GNDVI/NDRE gating (V5a) plus late fusion of MSI indices with DINO tokens (V5b).
- **Gate**: Improved tree precision/recall or species purity while keeping edge-F stable.

### **🧪 V6: K-Means Successors (Spike)**
- **Targets**: `tree_seg/clustering/experimental/`
- **Focus**: Explore alternative clustering algorithms: (1) Spherical k-means (cosine metric), (2) Soft k-means (as clustering algorithm, not refinement), (3) DP-means (automatic K selection).
- **Clarification**: Soft k-means here is a clustering algorithm choice. This is distinct from V2's soft/EM refinement which operates on K-means output. V6 outputs can feed into V2 refinement.
- **Gate**: Adopt only if outperforming V2 on mIoU/edge-F at similar compute cost; otherwise archive results.

---

## 💡 Key Design Decisions

### **🎯 Why This Architecture?**
1. **Research Velocity**: Clean interfaces make feature development fast
2. **Type Safety**: Catch errors early with dataclasses and type hints  
3. **Modularity**: Each component has single responsibility
4. **Extensibility**: Factory patterns allow easy model/clustering swaps
5. **Professional Quality**: Production-ready patterns for academic code

### **🔧 Technology Choices**
- **DINOv3**: State-of-the-art satellite-optimized features
- **Submodule**: Clean integration without dependency hell
- **Adapter Pattern**: Maintain API consistency across model changes
- **Dataclasses**: Modern Python, type-safe configuration
- **Web Optimization**: Automatic JPG conversion for GitHub Pages

**Result**: A codebase that's both **research-friendly** and **engineering-solid**. 🎯
