# Tree Segmentation with DINOv2

Unsupervised tree segmentation using DINOv2 Vision Transformers and K-means clustering for aerial drone imagery.

## 🌳 Project Overview

This project implements an unsupervised tree segmentation pipeline that:
- Uses DINOv2 Vision Transformers to extract deep features from aerial images
- Applies K-means clustering for segmentation
- Supports multiple versions (v1: patch features only, v1.5: patch + attention features)
- Generates high-quality visualization outputs

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

### ⚠️ Important: Local Execution Limitations

**Currently, the main pipeline is designed to run on cloud GPU platforms (Kaggle/Colab) rather than local machines.**

**Local Execution Issues:**
- The script starts but may hang or fail to complete on Linux systems
- This appears to be related to environment differences, dependency versions, or resource management
- **Note:** The same code runs successfully on Windows 11 on the same laptop (different SSD), suggesting it's a Linux-specific environment issue

**Recommended Workflow:**
1. **Edit and test** your notebook locally in Jupyter/VS Code
2. **Run the full pipeline** on Kaggle or Colab for reliable execution
3. **Use the cloud notebooks** in `notebooks/` directory for production runs

### Local Execution (Experimental)

1. **Install dependencies:**
   ```bash
   pip install torch torchvision timm opencv-python matplotlib scikit-learn Pillow numpy xformers
   ```

2. **Add images to input directory:**
   ```bash
   mkdir -p input
   # Copy your drone images to input/
   ```

3. **Run segmentation:**
   ```bash
   python tree_seg_local.py
   ```

**Note:** Local execution may fail or hang. For reliable results, use cloud platforms.

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

Edit `config.yaml` or modify the config in the scripts:

```yaml
input_dir: "input"           # Input image directory
output_dir: "output"         # Output directory
n_clusters: 6               # Number of segmentation clusters
overlay_ratio: 4            # Overlay transparency (1-10)
stride: 4                   # Feature resolution (2-8)
model_name: "dinov2_vits14" # DINOv2 model variant
filename: null              # Specific file or null for all
version: "v1.5"            # v1 or v1.5
```

## 📊 Output Files

For each processed image, the pipeline generates:
- `{filename}_segmentation_legend.png` - Segmentation map with legend
- `{filename}_overlay.png` - Original image with segmentation overlay
- `{filename}_side_by_side.png` - Original and segmentation comparison

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

