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

### **🆕 New Clustering Methods (V2: U2Seg, V3: DynaSeg)**
**Location**: `tree_seg/clustering/`
```python
# Create new module
tree_seg/clustering/u2seg.py
tree_seg/clustering/dynaseg.py

# Update segmentation.py to route based on config
if config.clustering_method == "u2seg":
    return u2seg_cluster(features)
```

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

### **📍 Current: V1.5 (DINOv3 + K-means)**
- ✅ **Baseline**: Solid foundation with state-of-the-art features
- ✅ **Architecture**: Clean, extensible design ready for advanced methods

### **🎯 Next: V2 (U2Seg)**
**Target**: `tree_seg/clustering/u2seg.py`
- Advanced unsupervised segmentation beyond K-means
- Integration point: `core/segmentation.py` routing logic

### **🚀 Future: V3 (DynaSeg) + V4 (Multispectral)**
**Target**: `tree_seg/clustering/dynaseg.py` + `tree_seg/models/multispectral_adapter.py`
- Dynamic fusion methods + multi-band imagery support
- Architecture supports both through modular design

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