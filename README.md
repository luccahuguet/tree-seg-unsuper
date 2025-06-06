# Final Paper: Unsupervised Tree Species Segmentation from Drone Imagery

### Overview
This project implements the **V1.5 pipeline** for unsupervised tree species segmentation using high-resolution RGB drone imagery, aiming to differentiate tree species without labeled data. The pipeline integrates DINOv2 for deep feature extraction, HighResDV2 (HR-Dv2) for upsampled feature extraction, and K-Means clustering for segmenting tree crowns. It further enhances the baseline by concatenating attention features and providing PCA-based cluster visualizations.  
The research question is: *How can unsupervised machine learning methods, leveraging deep visual feature extraction and clustering, accurately segment and differentiate tree species in high-resolution RGB drone imagery without labeled training data?*

Developed in a WSL2 Ubuntu 24.04 environment using Nushell, Python 3.10.17 (managed by `uv`), and an NVIDIA RTX 3050 for GPU acceleration, the project employs tools like `black` (code formatting), `ruff` (linting), and `ty` (static type checking).

---

### Project Structure
- `main.py`: Orchestrates the pipeline, reading configuration from `config.yaml` and invoking `kmeans_segmentation.py`.
- `config.yaml`: Configuration file specifying input/output directories, number of clusters, and overlay ratio.
- `kmeans_segmentation.py`: Implements K-Means clustering on DINOv2 features (patch + attention), produces segmented images, overlays, legends, and PCA cluster visualizations.
- `upsampler.py`: Implements HighResDV2 for upsampled DINOv2 feature extraction, adapted from HR-Dv2's `high_res.py`.
- `patch.py`: Modifies DINOv2's Vision Transformer to support overlapping patches and attention visualization.
- `transform.py`: Provides image transformation utilities for feature enhancement (copied from HR-Dv2).
- `pyproject.toml`: Defines project metadata and `ruff` configuration.
- `README.md`: This documentation.
- `uv.lock`: Dependency lockfile generated by `uv`.

---

### Prerequisites
- **System**: WSL2 with Ubuntu 24.04 on Windows 11.
- **Hardware**: NVIDIA RTX 3050 (6GB VRAM) for CUDA acceleration.
- **Software**:
  - Nushell (for shell commands).
  - Git (for repository cloning).
  - NVIDIA drivers and CUDA toolkit (for WSL2 GPU support).
  - **Rust (for `ty`)**: Install via `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.

---

### Setup Instructions
1. **Install WSL2 and Ubuntu**
   - In PowerShell (as Administrator):
     ```powershell
     wsl --install
     ```
   - Install Ubuntu 24.04 from the Microsoft Store, then update:
     ```bash
     sudo apt update && sudo apt upgrade -y
     sudo apt install -y build-essential curl libssl-dev zlib1g-dev libbz2-dev \
     libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils libffi-dev liblzma-dev
     ```

2. **Install `uv`**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv --version
   ```

3. **Clone and Set Up the Project**
   ```bash
   git clone <your-repo-url> final-paper
   cd final-paper
   uv venv --python ~/.cache/uv/python/cpython-3.10.17-linux-x86_64-gnu/bin/python
   overlay use .venv/bin/activate.nu
   ```

4. **Install Dependencies**
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   uv pip install timm scikit-learn numpy opencv-python matplotlib pyyaml
   uv add --dev ruff git+https://github.com/astral-sh/ty
   uv sync
   uv lock
   ```

5. **Copy HR-Dv2 Files**
   - Clone HR-Dv2 and copy required files:
     ```bash
     git clone https://github.com/tldr-group/HR-Dv2 ~/pjs/HR-Dv2
     cd ~/pjs/final-paper
     cp ../HR-Dv2/hr_dv2/high_res.py ./upsampler.py
     cp ../HR-Dv2/hr_dv2/patch.py ./patch.py
     cp ../HR-Dv2/hr_dv2/transform.py ./transform.py
     ```

6. **Configure `ruff`**
   - Ensure `pyproject.toml` includes:
     ```toml
     [tool.ruff]
     line-length = 88
     exclude = [".venv", ".git"]
     ```

---

### Usage
1. **Prepare Input Images**
   - Download RGB drone imagery (e.g., from NEON AOP: https://data.neonscience.org/data-products/DP3.30010.001).
   - Place images in the `input` directory (e.g., `input/drone_image.jpg`).

2. **Configure the Pipeline**
   - Edit `config.yaml` to set:
     - `input_dir`: Directory containing input images (default: `input`).
     - `output_dir`: Directory for output images (default: `output`).
     - `n_clusters`: Number of K-Means clusters (default: 5).
     - `overlay_ratio`: Controls overlay transparency (1=fully original image, 10=fully segmented, default: 5).

3. **Run the Pipeline**
   ```bash
   python main.py
   ```
   - For each image in `input_dir`, generates:
     - `<filename>_segmentation_legend.png`: Segmentation map with a colorbar legend.
     - `<filename>_overlay.png`: Original image overlaid with segmentation (transparency set by `overlay_ratio`).
     - `<filename>_pca_scatter.png`: PCA scatter plot of features colored by cluster.
   - Outputs are saved in `output_dir`.

4. **Format, Lint, and Type Check**
   ```bash
   black .
   ruff check . --fix
   ty check main.py kmeans_segmentation.py upsampler.py patch.py
   ```

---

### Technical Details

**Pipeline (V1.5):**
- **Feature Extraction:** Uses HighResDV2 (`upsampler.py`) with `dinov2_vits14` (stride=4, float16 for RTX 3050) to extract upsampled DINOv2 features. Modifications in `patch.py` enable overlapping patches and attention visualization.
- **Attention Features:** Concatenates attention features to patch features for each patch, providing richer context for clustering.
- **Clustering:** Applies K-Means clustering (`kmeans_segmentation.py`) with `n_clusters` (default: 5) to the concatenated feature vectors, reshaping labels to match the original image size.
- **Visualization:** Generates segmentation maps with colormaps (`tab10`, `tab20`, or `gist_ncar` based on `n_clusters`), overlays them on the original image using `overlay_ratio` for transparency, and outputs PCA scatter plots of features colored by cluster.
- **Configuration:** Managed via `config.yaml`, allowing flexible input/output paths and clustering parameters.

**Optimizations:**
- **Float16:** Reduces VRAM usage for RTX 3050 (4-6GB).
- **Stride=4:** Balances resolution and memory for high-res drone imagery.
- **Memory-Efficient Attention:** Patched in `patch.py` using xFormers for scalable attention computation.

**Dependencies:**
- `torch`, `torchvision`: GPU-accelerated tensor operations (CUDA 12.1).
- `timm`: DINOv2 model loading (`vit_base_patch14_dinov2.lvd142m`).
- `scikit-learn`: K-Means clustering and PCA.
- `numpy`, `opencv-python`, `matplotlib`: Image processing and visualization.
- `pyyaml`: Configuration file parsing.
- `black`, `ruff`: Code formatting and linting.
- `ty`: Static type checking (pre-alpha, focuses on type annotations).
- `uv`: Dependency management with cross-platform lockfile generation.

---

### Roadmap

- **V1:** Patch features only, K-Means clustering.
- **V1.5 (Current):** Patch + attention features, K-Means clustering, PCA cluster visualization.
- **V2:** U2Seg (advanced unsupervised segmentation).
- **V3:** DynaSeg (dynamic fusion segmentation).
- **V4:** Multispectral extension.

---

### Troubleshooting
- **CUDA Errors:**
  ```bash
  nvidia-smi
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
  - Verify WSL2 GPU support: https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute.
- **VRAM Issues:**
  - In `config.yaml`, increase stride (e.g., `stride=8`) or use a smaller model (e.g., `dinov2_vits14`).
  - Ensure `float16` is enabled in `upsampler.py`.
- **Dependency Issues:**
  ```bash
  uv pip install <package>
  uv lock
  ```

---

### Next Steps
- Test with diverse NEON AOP imagery for robustness.
- Explore `transform.py` for advanced feature augmentation (e.g., rotations, flips).
- Evaluate segmentation quality using ground-truth data (metrics: Pixel Accuracy, mIoU).
- Extend to V2 (U2Seg) or V3 (DynaSeg) pipelines for improved segmentation.

---

### References
- Docherty et al. (2024). Upsampling DINOv2 features. https://doi.org/10.48550/ARXIV.2410.19836
- HR-Dv2: https://github.com/tldr-group/HR-Dv2
- NEON AOP: https://data.neonscience.org/data-products/DP3.30010.001
- DINOv2: https://github.com/facebookresearch/dinov2

---

### License
MIT

