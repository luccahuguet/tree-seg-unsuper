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
- Supports automatic K-selection using elbow method
- Generates high-quality visualization outputs with config-based naming

## 📁 Project Structure

```
tree-seg-unsuper/
├── tree_seg/                  # Core package
│   ├── core/                  # Core segmentation logic
│   ├── visualization/         # Plotting and visualization
│   ├── utils/                 # Utilities and helpers
│   └── api.py                 # Main API interface
│
├── docs/                      # Documentation (GitHub Pages)
├── sweeps/                    # Sweep configurations
├── input/                     # Input images
├── output/                    # Output results
├── run_segmentation.py        # Main CLI script
├── generate_docs_images.py    # Documentation image generation
├── pyproject.toml            # Project metadata
└── README.md                 # This file
```

## 🚀 Quick Start

### Run (uv + CLI)

Using the simple CLI wrapper with uv:

```bash
uv run python run_segmentation.py input base output
```

Defaults favor high quality:
- Model: `giant` (ViT-H+/16)
- Image size: 1024×1024
- Feature upsample factor: 2 (clusters on a 128×128 grid)
- Edge refinement: SLIC superpixels (aligns boundaries to image edges)

CLI flags allow tuning quality/performance:

```bash
# Higher quality (even smoother):
uv run python run_segmentation.py input giant output \
  --image-size 1280 --feature-upsample 2

# Faster / lower memory:
uv run python run_segmentation.py input base output \
  --image-size 896 --feature-upsample 1

# Apply PCA to speed up clustering:
uv run python run_segmentation.py input base output \
  --pca-dim 128

# Process a single image (giant default):
uv run python run_segmentation.py input/forest.jpg giant output

# Edge-aware refinement (SLIC superpixels):
uv run python run_segmentation.py input base output \
  --refine slic --refine-slic-compactness 12 --refine-slic-sigma 1.5
```

Flags:
- `--image-size INT`: square resize used before feature extraction (default 1024)
- `--feature-upsample INT`: bilinear upsample of H×W feature grid before K-Means (default 2)
- `--pca-dim INT`: optional PCA target dimension (e.g., 128) for faster clustering
- `--refine {none,slic}`: edge-aware smoothing (default: slic)
- `--refine-slic-compactness FLOAT`: SLIC compactness (higher=smoother, lower=hugs edges). Default 10.0
- `--refine-slic-sigma FLOAT`: Gaussian smoothing for SLIC pre-processing. Default 1.0
- `--metrics`: print timing and VRAM stats (also included in result stats)
 - `--elbow-threshold FLOAT`: elbow method percentage threshold (e.g., 5.0). Default 5.0
 - `--sweep PATH`: run multiple configurations from a JSON/YAML list; outputs saved under `output/<sweeps>/<name>`
 - `--sweep-prefix STR`: subfolder under `output/` to place sweep runs (default: `sweeps`)

Profiles (set sensible defaults; flags still override; default profile is balanced):

```bash
# Highest quality
uv run python run_segmentation.py input base output --profile quality

# Balanced (default)
uv run python run_segmentation.py input base output --profile balanced

# Fast(er) runs / lower memory
uv run python run_segmentation.py input base output --profile speed
```

- `quality`: `image_size=1280`, `feature_upsample=2`, `pca_dim=None`, `refine=slic`, `compactness=12`, `sigma=1.5`
- `balanced`: `image_size=1024`, `feature_upsample=2`, `pca_dim=None`, `refine=slic`, `compactness=10`, `sigma=1.0`
- `speed`: `image_size=896`, `feature_upsample=1`, `pca_dim=128`, `refine=slic`, `compactness=20`, `sigma=1.0`

### API Usage

**Simple one-liner:**
```python
from tree_seg import segment_trees

# Process single image with smart defaults
results = segment_trees(
    "input/forest2.jpeg",
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
    elbow_threshold=5.0,
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


### Generate Documentation Images

To generate all documentation images with comprehensive parameter analysis:

```bash
python generate_docs_images.py input/forest2.jpeg
```

This runs 12 different configurations and organizes results for the GitHub Pages documentation.

## ⚙️ Configuration

### Configuration

```python
from tree_seg import Config

config = Config(
    # Input/Output
    input_dir="input",
    output_dir="output",
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
    elbow_threshold=5.0,
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

**Elbow Threshold Values (percent):**
- `2.5%`: Sensitive — more clusters
- `5.0%`: Balanced — recommended default  
- `10.0%`: Conservative — fewer clusters
- `20.0%`: Very conservative — broad regions

**Model Sizes:**
- `small`: DINOv3 Small (21M params) - Fast, good for testing
- `base`: DINOv3 Base (86M params) - Best balance (recommended)
- `large`: DINOv3 Large (304M params) - Higher quality, slower  
- `giant`: DINOv3 Giant (1.1B params) - Maximum quality, very slow
- `mega`: DINOv3 Mega (6.7B params) - Satellite-optimized, ultimate quality

## 📊 Output Files

### Smart Filename Generation

Files now use **config-based naming** for easy identification:

```
{hash}_{version}_{model}_{stride}_{clustering}_type.png
```

**Examples:**
- `a3f7_v3_base_str4_et5-0_segmentation_legend.png`
- `a3f7_v3_base_str4_et5-0_edge_overlay.png`
- `a3f7_v3_base_str4_et5-0_side_by_side.png`
- `a3f7_v3_base_str4_et5-0_elbow_analysis.png`

**Filename Components:**
- `a3f7`: 4-char hash of original filename (prevents collisions)
- `v3`: DINOv3 version used
- `base`: Model size (small/base/large/giant/mega)
- `str4`: Stride value
- `et5-0`: Elbow threshold (or `k6` for fixed K)

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

- **v3 (current)**: Uses DINOv3 features with elbow method K-selection
- Previous: v1 (patch only), v1.5 (DINOv2 patch + attention), v2 (curvature method)

## 🖼️ Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tif, .tiff)

## 📊 Documentation

Comprehensive documentation is available at the GitHub Pages site, including:
- Methodology and pipeline details
- Complete workflow examples
- Parameter analysis and comparisons
- Performance benchmarks

## 📝 Usage Examples

### Process a single image:
```python
from tree_seg import TreeSegmentation, Config

config = Config(
    model_name="base",
    auto_k=True,
    elbow_threshold=5.0
)

segmenter = TreeSegmentation(config)
results, paths = segmenter.process_and_visualize("input/forest2.jpeg")
```

### Process with CLI:
```bash
# Basic usage
uv run python run_segmentation.py input/forest2.jpeg base output

# With custom parameters
uv run python run_segmentation.py input/forest2.jpeg giant output \
  --stride 2 --elbow-threshold 5.0 --profile quality
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

- **V3 (Current):** DINOv3 features with elbow method K-selection
- **V4:** U2Seg (advanced unsupervised segmentation)
- **V5:** DynaSeg (dynamic fusion segmentation)
- **V6:** Multispectral extension
- **Previous:** V1 (patch only), V1.5 (DINOv2), V2 (curvature method)

## 🔜 Next Steps

- Test with diverse NEON AOP imagery for robustness.
- Explore `transform.py` for advanced feature augmentation (e.g., rotations, flips).
- Evaluate segmentation quality using ground-truth data (metrics: Pixel Accuracy, mIoU).
- Extend to V2 (U2Seg) or V3 (DynaSeg) pipelines for improved segmentation.

## 📚 References

- DINOv3: Meta AI (2024). DINOv3: Scaling Self-Supervised Learning for Computer Vision
- DINOv3 GitHub: https://github.com/facebookresearch/dinov3
- NEON AOP: https://data.neonscience.org/data-products/DP3.30010.001
# Sweeps

Provide a JSON or YAML array of config overrides to iterate. Each item can override any CLI flag (e.g., `model`, `image_size`, `feature_upsample_factor`, `pca_dim`, `refine`, `refine_slic_*`, `profile`). Optionally include `name` to name the run folder.

Example `sweep.yaml`:

```yaml
- name: q1280
  profile: quality
  model: large
- name: b1024
  profile: balanced
  model: base
- name: s896
  profile: speed
  model: small
  refine: none
```

Run the sweep:

```bash
uv run python run_segmentation.py input giant output --sweep sweep.yaml --metrics
```
