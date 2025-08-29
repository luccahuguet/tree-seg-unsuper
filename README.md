# Tree Segmentation with DINOv3

Modern unsupervised tree segmentation using DINOv3 Vision Transformers and K-means clustering for aerial drone imagery.

## 🚀 What's New - Modern Architecture

**Clean API with professional patterns:**
- 🏗️ **Dataclass Configuration** - Type-safe, validated config objects
- 📦 **Result Objects** - Structured returns instead of tuples  
- 🎯 **OutputManager** - Intelligent file naming and management
- 🧹 **Clean Interface** - Simple API for both quick and advanced usage
- ✅ **Pure Modern Code** - No legacy cruft

## 🌳 Project Overview

This project implements an unsupervised tree segmentation pipeline that:
- Uses DINOv3 Vision Transformers to extract deep features from aerial images  
- Applies K-means clustering for segmentation
- Supports v3 (satellite-optimized features) with automatic K-selection
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

### Run (uv + CLI)

Using the simple CLI wrapper with uv:

```bash
uv run python run_segmentation.py input base output
```

Defaults now favor high quality:
- Model: `base` (ViT-B/16)
- Image size: 1024×1024
- Feature upsample factor: 2 (clusters on a 128×128 grid)

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
    # Model
    model_name="base",      # small/base/large/giant/mega or full DINOv3 name
    version="v3",

    # Quality & performance
    image_size=1024,         # Resize (square). 518/896/1024 are good picks
    feature_upsample_factor=2, # Upsample feature grid before K-Means
    pca_dim=None,            # Optional PCA target dim (e.g., 128) or None

    # Clustering
    auto_k=True,
    elbow_threshold=0.1,
    k_range=(3, 10),
    n_clusters=6,            # Used only when auto_k=False

    # Visualization
    use_hatching=True,
    edge_width=2,
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
    filename=None,

    # Model settings
    model_name="base",
    version="v3",
    stride=4,
    image_size=1024,
    feature_upsample_factor=2,
    pca_dim=None,

    # Clustering (automatic K-selection recommended)
    auto_k=True,
    elbow_threshold=0.1,
    k_range=(3, 10),
    n_clusters=6,

    # Visualization
    overlay_ratio=4,
    edge_width=2,
    use_hatching=True,
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
- `small`: ViT-S/16 (21M params) - Fast, good for testing
- `base`: ViT-B/16 (86M params) - Best balance (recommended)
- `large`: ViT-L/16 (300M params) - Higher quality, slower  
- `giant`: ViT-H+/16 (840M params) - Maximum quality, very slow
- `mega`: ViT-7B/16 (6.7B params) - Satellite-optimized, ultimate quality

## 📊 Output Files

### Smart Filename Generation

Files now use **config-based naming** for easy identification:

```
{hash}_{version}_{model}_{stride}_{clustering}_type.png
```

**Examples:**
- `a3f7_v3_base_str4_et0-1_segmentation_legend.png`
- `a3f7_v3_base_str4_et0-1_edge_overlay.png`
- `a3f7_v3_base_str4_et0-1_side_by_side.png`
- `a3f7_v3_base_str4_et0-1_elbow_analysis.png`

**Filename Components:**
- `a3f7`: 4-char hash of original filename (prevents collisions)
- `v3`: DINOv3 version used
- `base`: Model size (small/base/large/giant/mega)
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
- **omegaconf** (for DINOv3 configuration)
- **opencv-python** >= 4.8.0
- **matplotlib** >= 3.7.0
- **scikit-learn** >= 1.3.0
- **Pillow** >= 10.0.0
- **numpy** >= 1.24.0
- **ftfy**, **regex**, **torchmetrics** (DINOv3 dependencies)

## 🎯 Model Variants

- **v3 (default)**: Uses DINOv3 features with enhanced performance
- Previous: v1 (patch only), v1.5 (patch + attention)

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

- Lower memory/CPU: reduce `image_size` (e.g., 896 or 518)
- Faster clustering: set `feature_upsample_factor=1`
- Large models: ensure GPU VRAM is sufficient; consider `pca_dim=128`
- Adjust `n_clusters`/`k_range` based on image complexity

## 📄 License

This project is part of a research paper on unsupervised tree segmentation.

## 🤝 Contributing

This is a research project. For questions or issues, please refer to the paper or contact the authors.

## 🗺️ Roadmap

- **V1.5 (Current):** DINOv3 satellite features, K-Means clustering, PCA cluster visualization.
- **V2:** U2Seg (advanced unsupervised segmentation).
- **V3:** DynaSeg (dynamic fusion segmentation).
- **V4:** Multispectral extension.
- **Previous:** V1 (patch only), V1.5 (DINOv2 patch + attention) - now superseded

## 🔜 Next Steps

- Test with diverse NEON AOP imagery for robustness.
- Explore `transform.py` for advanced feature augmentation (e.g., rotations, flips).
- Evaluate segmentation quality using ground-truth data (metrics: Pixel Accuracy, mIoU).
- Extend to V2 (U2Seg) or V3 (DynaSeg) pipelines for improved segmentation.

## 📚 References

- DINOv3: Meta AI (2025). Self-supervised learning for vision at unprecedented scale
- DINOv3 GitHub: https://github.com/facebookresearch/dinov3
- Satellite optimization: WRI forestry applications with 70% accuracy improvement
- NEON AOP: https://data.neonscience.org/data-products/DP3.30010.001
- Previous work: DINOv2 - https://github.com/facebookresearch/dinov2
