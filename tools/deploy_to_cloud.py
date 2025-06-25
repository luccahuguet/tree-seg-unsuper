#!/usr/bin/env python3
"""
Cloud GPU Deployment Script for Tree Segmentation
Supports Kaggle, Google Colab, and other cloud platforms
"""

import os
import json
import zipfile
import subprocess
import sys
from pathlib import Path

def create_kaggle_notebook():
    """Create a Kaggle-compatible notebook from the local script."""

    # Read the local script
    with open('tree_seg_local.py', 'r') as f:
        local_code = f.read()

    # Convert to Kaggle notebook format
    kaggle_notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Tree Segmentation with DINOv2\n",
                    "This notebook performs unsupervised tree segmentation using DINOv2 Vision Transformers."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install dependencies\n",
                    "!pip install timm\n",
                    "!pip install xformers --index-url https://download.pytorch.org/whl/cu124"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Clone the repository\n",
                    "!git clone https://github.com/luccahuguet/tree-seg-unsuper /kaggle/working/project\n",
                    "%cd /kaggle/working/project"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Upload your images here\n",
                    "# You can upload images to /kaggle/input/your-dataset-name/\n",
                    "# Then update the config below"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Configuration\n",
                    "config = {\n",
                    "    \"input_dir\": \"/kaggle/input/your-dataset-name\",  # Update this path\n",
                    "    \"output_dir\": \"/kaggle/working/output\",\n",
                    "    \"n_clusters\": 6,\n",
                    "    \"overlay_ratio\": 4,\n",
                    "    \"stride\": 4,\n",
                    "    \"model_name\": \"dinov2_vits14\",\n",
                    "    \"filename\": None,  # Set to specific filename or None for all\n",
                    "    \"version\": \"v1.5\"\n",
                    "}\n",
                    "\n",
                    "print(\"Configuration:\")\n",
                    "for key, value in config.items():\n",
                    "    print(f\"  {key}: {value}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import the tree segmentation code\n",
                    "import sys\n",
                    "sys.path.append('/kaggle/working/project/src')\n",
                    "\n",
                    "# Copy the main functions from tree_seg_local.py\n",
                    "# (The full code will be pasted here)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run the segmentation\n",
                    "tree_seg(**config)\n",
                    "print(\"\\nSegmentation completed! Check the output directory.\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Display results\n",
                    "import matplotlib.pyplot as plt\n",
                    "from IPython.display import Image, display\n",
                    "\n",
                    "# List output files\n",
                    "import os\n",
                    "output_files = os.listdir(config['output_dir'])\n",
                    "print(\"Generated files:\")\n",
                    "for file in output_files:\n",
                    "    print(f\"  {file}\")\n",
                    "\n",
                    "# Display a sample result\n",
                    "if output_files:\n",
                    "    sample_file = os.path.join(config['output_dir'], output_files[0])\n",
                    "    if sample_file.endswith('.png'):\n",
                    "        display(Image(filename=sample_file))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Save the notebook
    with open('tree_seg_kaggle.ipynb', 'w') as f:
        json.dump(kaggle_notebook, f, indent=2)

    print("✅ Created Kaggle notebook: tree_seg_kaggle.ipynb")
    return 'tree_seg_kaggle.ipynb'

def create_colab_notebook():
    """Create a Google Colab compatible notebook."""

    colab_notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Tree Segmentation with DINOv2 (Google Colab)\n",
                    "This notebook performs unsupervised tree segmentation using DINOv2 Vision Transformers."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Mount Google Drive (optional)\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install dependencies\n",
                    "!pip install timm\n",
                    "!pip install xformers --index-url https://download.pytorch.org/whl/cu124"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Clone the repository\n",
                    "!git clone https://github.com/luccahuguet/tree-seg-unsuper /content/project\n",
                    "%cd /content/project"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Upload your images\n",
                    "from google.colab import files\n",
                    "uploaded = files.upload()\n",
                    "\n",
                    "# Create input directory\n",
                    "!mkdir -p input\n",
                    "!mv *.jpg *.png *.jpeg input/ 2>/dev/null || true"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Configuration\n",
                    "config = {\n",
                    "    \"input_dir\": \"/content/project/input\",\n",
                    "    \"output_dir\": \"/content/project/output\",\n",
                    "    \"n_clusters\": 6,\n",
                    "    \"overlay_ratio\": 4,\n",
                    "    \"stride\": 4,\n",
                    "    \"model_name\": \"dinov2_vits14\",\n",
                    "    \"filename\": None,\n",
                    "    \"version\": \"v1.5\"\n",
                    "}\n",
                    "\n",
                    "print(\"Configuration:\")\n",
                    "for key, value in config.items():\n",
                    "    print(f\"  {key}: {value}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import and run the segmentation\n",
                    "import sys\n",
                    "sys.path.append('/content/project/src')\n",
                    "\n",
                    "# The tree_seg_local.py code will be pasted here\n",
                    "# ...\n",
                    "\n",
                    "# Run segmentation\n",
                    "tree_seg(**config)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Download results\n",
                    "!zip -r results.zip output/\n",
                    "files.download('results.zip')"
                ]
            }
        ],
        "metadata": {
            "colab": {
                "name": "Tree Segmentation with DINOv2",
                "private_outputs": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    with open('tree_seg_colab.ipynb', 'w') as f:
        json.dump(colab_notebook, f, indent=2)

    print("✅ Created Colab notebook: tree_seg_colab.ipynb")
    return 'tree_seg_colab.ipynb'

def create_deployment_package():
    """Create a deployment package with all necessary files."""

    # Create a deployment directory
    deploy_dir = Path("deployment")
    deploy_dir.mkdir(exist_ok=True)

    # Copy necessary files
    files_to_copy = [
        "tree_seg_local.py",
        "src/",
        "config.yaml",
        "README.md"
    ]

    for file_path in files_to_copy:
        src = Path(file_path)
        dst = deploy_dir / src.name

        if src.is_file():
            import shutil
            shutil.copy2(src, dst)
        elif src.is_dir():
            import shutil
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # Create a requirements.txt
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "xformers>=0.0.20"
    ]

    with open(deploy_dir / "requirements.txt", "w") as f:
        f.write("\n".join(requirements))

    # Create a run script
    run_script = """#!/bin/bash
# Cloud GPU Run Script

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running tree segmentation..."
python tree_seg_local.py

echo "Done! Check the output directory for results."
"""

    with open(deploy_dir / "run.sh", "w") as f:
        f.write(run_script)

    # Make it executable
    os.chmod(deploy_dir / "run.sh", 0o755)

    # Create zip file
    zip_path = "tree_seg_deployment.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in deploy_dir.rglob('*'):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(deploy_dir))

    print(f"✅ Created deployment package: {zip_path}")
    return zip_path

def main():
    """Main deployment function."""
    print("🌳 Tree Segmentation Cloud Deployment")
    print("=" * 50)

    print("\nCreating all deployment options...")

    # Create all deployment options
    create_kaggle_notebook()
    create_colab_notebook()
    create_deployment_package()

    print("\n🎉 All deployment files created!")
    print("\nNext steps:")
    print("• For Kaggle: Upload tree_seg_kaggle.ipynb to Kaggle Notebooks")
    print("• For Colab: Upload tree_seg_colab.ipynb to Google Colab")
    print("• For other clouds: Upload tree_seg_deployment.zip and run run.sh")
    print("\n📁 Files created:")
    print("  - tree_seg_kaggle.ipynb (Kaggle notebook)")
    print("  - tree_seg_colab.ipynb (Google Colab notebook)")
    print("  - tree_seg_deployment.zip (Universal deployment package)")

if __name__ == "__main__":
    main()