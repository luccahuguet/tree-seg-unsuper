# Tree Segmentation with DINOv2

Modern unsupervised tree segmentation using DINOv2 Vision Transformers and K-means clustering for aerial drone imagery.

## 🚀 What's New - Modern Architecture

**Clean API with professional patterns:**
- 🏗️ **Dataclass Configuration** - Type-safe, validated config objects
- 📦 **Result Objects** - Structured returns instead of tuples  
- 🎯 **OutputManager** - Intelligent file naming and management
- 🧹 **Clean Interface** - Simple API for both quick and advanced usage
- ✅ **Pure Modern Code** - No legacy cruft

## 🌳 Project Overview

This project implements an unsupervised tree segmentation pipeline that:
- Uses DINOv2 Vision Transformers to extract deep features from aerial images
- Applies K-means clustering for segmentation
- Supports v1.5 (patch + attention features) with automatic K-selection
- Generates high-quality visualization outputs with config-based naming

## 📁 Project Structure

```
final-paper/
├── src/                       # Core modules
│   ├── __init__.py
│   ├── kmeans_segmentation.py
│   ├── patch.py
│   ├── transform.py
│   └── upsampler.py
│
├── notebooks/                 # Cloud-ready notebooks
│   ├── tree_seg_kaggle.ipynb  # Kaggle notebook
│   └── tree_seg_colab.ipynb   # Google Colab notebook
│
├── deployment/                # Cloud deployment package
│   ├── tree_seg_deployment.zip
│   ├── run.sh                 # Cloud execution script
│   ├── requirements.txt       # Dependencies
│   ├── tree_seg_local.py      # Main script
│   ├── config.yaml           # Configuration
│   ├── README.md             # Deployment instructions
│   └── src/                  # Source code copy
│
├── input/                     # Input images (local runs)
├── output/                    # Output results
├── tree_seg_local.py          # Main script for local execution
├── deploy_to_cloud.py         # Deployment automation
├── config.yaml               # Configuration
├── pyproject.toml            # Project metadata
└── README.md                 # This file
```

## 🚀 Quick Start

### API Usage

**Simple one-liner:**
```python
from tree_seg import segment_trees

# Process single image with smart defaults
results = segment_trees(
    "/kaggle/input/drone-imagery/image.jpg",
    model="base",
    auto_k=True
)
```

**Advanced usage with full control:**
```python
from tree_seg import TreeSegmentation, Config

# Create configuration
config = Config(
    model_name="base",      # small, base, large, giant
    auto_k=True,           # Automatic K selection
    elbow_threshold=0.1,   # Sensitive elbow detection
    use_hatching=True,     # Visual distinction
    edge_width=2
)

# Process with modern API
segmenter = TreeSegmentation(config)
results, paths = segmenter.process_and_visualize("image.jpg")

print(f"Used K = {results.n_clusters_used}")
print(f"Files: {paths.all_paths()}")
```


### Cloud GPU Execution

#### Option 1: Kaggle
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Upload `notebooks/tree_seg_kaggle.ipynb`
3. Attach your dataset or upload images
4. Run on Kaggle GPU

#### Option 2: Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/tree_seg_colab.ipynb`
3. Upload your images when prompted
4. Run on Colab GPU

#### Option 3: Any Cloud GPU VM
1. Upload `deployment/tree_seg_deployment.zip` to your VM
2. Unzip and run:
   ```bash
   unzip tree_seg_deployment.zip
   chmod +x run.sh
   ./run.sh
   ```

## ⚙️ Configuration

### Configuration

```python
from tree_seg import Config

config = Config(
    # Input/Output
    input_dir="/kaggle/input/drone-imagery",
    output_dir="/kaggle/working/output", 
    filename=None,              # Process all images in directory
    
    # Model settings
    model_name="base",          # small/base/large/giant
    version="v1.5",             # Current version
    stride=4,                   # Balance of speed vs quality
    
    # Clustering (automatic K-selection recommended)
    auto_k=True,                # Let elbow method choose K
    elbow_threshold=0.1,        # Sensitivity (0.05-0.3)
    k_range=(3, 10),            # K search range
    n_clusters=6,               # Only used when auto_k=False
    
    # Visualization
    overlay_ratio=4,            # Transparency (1=opaque, 10=transparent)
    edge_width=2,               # Border thickness
    use_hatching=True,          # Pattern distinction
    use_pca=False               # Keep full feature space
)

# Automatic validation
config.validate()  # Raises ValueError if invalid
```

### Configuration Tips

**Elbow Threshold Values:**
- `0.05-0.1`: Sensitive (finds more clusters)
- `0.1-0.2`: Balanced (recommended)  
- `0.2-0.3`: Conservative (fewer clusters)

**Model Sizes:**
- `small`: Fast, good for testing
- `base`: Best balance (recommended)
- `large`: Higher quality, slower
- `giant`: Maximum quality, very slow

## 📊 Output Files

### Smart Filename Generation

Files now use **config-based naming** for easy identification:

```
{hash}_{version}_{model}_{stride}_{clustering}_type.png
```

**Examples:**
- `a3f7_v1-5_base_str4_et0-1_segmentation_legend.png`
- `a3f7_v1-5_base_str4_et0-1_edge_overlay.png`
- `a3f7_v1-5_base_str4_et0-1_side_by_side.png`
- `a3f7_v1-5_base_str4_et0-1_elbow_analysis.png`

**Filename Components:**
- `a3f7`: 4-char hash of original filename (prevents collisions)
- `v1-5`: Version used
- `base`: Model size
- `str4`: Stride value
- `et0-1`: Elbow threshold (or `k6` for fixed K)

**File Types:**
- `segmentation_legend.png`: Colored map with cluster legend
- `edge_overlay.png`: Original image with colored boundaries
- `side_by_side.png`: Original vs segmentation comparison
- `elbow_analysis.png`: K-selection analysis (when auto_k=True)

## 🔧 Dependencies

- **PyTorch** >= 2.0.0
- **torchvision** >= 0.15.0
- **timm** >= 0.9.0 (for DINOv2 models)
- **opencv-python** >= 4.8.0
- **matplotlib** >= 3.7.0
- **scikit-learn** >= 1.3.0
- **Pillow** >= 10.0.0
- **numpy** >= 1.24.0
- **xformers** >= 0.0.20 (optional, for memory efficiency)

## 🎯 Model Variants

- **v1**: Uses only patch features from DINOv2
- **v1.5**: Uses both patch features and attention features (recommended)

## 🖼️ Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tif, .tiff)

## 🚀 Deployment Automation

Use the deployment script to generate cloud-ready files:

```bash
python deploy_to_cloud.py
```

This creates:
- Kaggle notebook in `notebooks/`
- Colab notebook in `notebooks/`
- Deployment package in `deployment/`

## 📝 Usage Examples

### Process a single image:
```python
from tree_seg_local import tree_seg

tree_seg(
    input_dir="input",
    output_dir="output", 
    filename="drone_image.jpg",
    n_clusters=6,
    version="v1.5"
)
```

### Process all images in directory:
```python
tree_seg(
    input_dir="input",
    output_dir="output",
    filename=None,  # Process all images
    n_clusters=8,
    stride=2  # Higher resolution
)
```

## 🔍 Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce `stride` or use smaller images
2. **Import errors**: Ensure all dependencies are installed
3. **No GPU detected**: Script will automatically use CPU (slower)

### Performance Tips:

- Use `stride=2` for higher resolution (slower)
- Use `stride=8` for faster processing (lower resolution)
- Adjust `n_clusters` based on image complexity
- Use `version="v1.5"` for better results

## 📄 License

This project is part of a research paper on unsupervised tree segmentation.

## 🤝 Contributing

This is a research project. For questions or issues, please refer to the paper or contact the authors.

## 🗺️ Roadmap

- **V1:** Patch features only, K-Means clustering.
- **V1.5 (Current):** Patch + attention features, K-Means clustering, PCA cluster visualization.
- **V2:** U2Seg (advanced unsupervised segmentation).
- **V3:** DynaSeg (dynamic fusion segmentation).
- **V4:** Multispectral extension.

## 🔜 Next Steps

- Test with diverse NEON AOP imagery for robustness.
- Explore `transform.py` for advanced feature augmentation (e.g., rotations, flips).
- Evaluate segmentation quality using ground-truth data (metrics: Pixel Accuracy, mIoU).
- Extend to V2 (U2Seg) or V3 (DynaSeg) pipelines for improved segmentation.

## 📚 References

- Docherty et al. (2024). Upsampling DINOv2 features. https://doi.org/10.48550/ARXIV.2410.19836
- HR-Dv2: https://github.com/tldr-group/HR-Dv2
- NEON AOP: https://data.neonscience.org/data-products/DP3.30010.001
- DINOv2: https://github.com/facebookresearch/dinov2

