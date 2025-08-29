#!/usr/bin/env python3
"""
Simple CLI for tree segmentation.

Usage:
  python run_segmentation.py [image_path] [model] [output_dir] [options]

Options:
  --image-size INT           Preprocess resize (square). Default: 1024
  --feature-upsample INT     Upsample HxW feature grid before K-Means. Default: 2
  --pca-dim INT              Optional PCA target dim (e.g., 128). Default: None
"""

import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Tree segmentation CLI")
    parser.add_argument("image_path", nargs="?", default="input/", help="Path to image or directory")
    parser.add_argument("model", nargs="?", default="base", help="Model size: small/base/large/giant/mega or full name")
    parser.add_argument("output_dir", nargs="?", default="output", help="Output directory")
    parser.add_argument("--image-size", type=int, default=1024, dest="image_size", help="Preprocess resize (square)")
    parser.add_argument("--feature-upsample", type=int, default=2, dest="feature_upsample_factor", help="Upsample feature grid before K-Means")
    parser.add_argument("--pca-dim", type=int, default=None, dest="pca_dim", help="Optional PCA target dimension (e.g., 128)")
    parser.add_argument("--refine", choices=["none", "slic"], default="slic", help="Edge-aware refinement mode (default: slic)")
    parser.add_argument("--refine-slic-compactness", type=float, default=10.0, dest="refine_slic_compactness", help="SLIC compactness (higher=smoother, lower=edges)")
    parser.add_argument("--refine-slic-sigma", type=float, default=1.0, dest="refine_slic_sigma", help="SLIC Gaussian smoothing sigma")

    args = parser.parse_args()
    image_path = args.image_path
    model = args.model
    output_dir = args.output_dir
    
    # Import after loading env vars
    from tree_seg import segment_trees
    import shutil
    
    # Clear output directory before processing
    if os.path.exists(output_dir):
        # Count existing files
        existing_files = []
        for root, dirs, files in os.walk(output_dir):
            existing_files.extend([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if existing_files:
            print(f"🗂️  Found {len(existing_files)} existing output file(s) in {output_dir}")
            print(f"🧹 Clearing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory ready: {output_dir}")
    print()
    
    # Check if image_path is a directory or single file
    if os.path.isdir(image_path):
        # Process all images in directory
        import glob
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            image_files.extend(glob.glob(os.path.join(image_path, ext)))
            image_files.extend(glob.glob(os.path.join(image_path, ext.upper())))
        
        if not image_files:
            print(f"❌ No image files found in {image_path}")
            return
        
        print(f"🖼️ Found {len(image_files)} image(s) in {image_path}")
        for img_path in sorted(image_files):
            print(f"\n🚀 Processing: {os.path.basename(img_path)}")
            try:
                results = segment_trees(
                    img_path,
                    model=model,
                    auto_k=True,
                    output_dir=output_dir,
                    image_size=args.image_size,
                    feature_upsample_factor=args.feature_upsample_factor,
                    pca_dim=args.pca_dim,
                    refine=(None if args.refine == "none" else args.refine),
                    refine_slic_compactness=args.refine_slic_compactness,
                    refine_slic_sigma=args.refine_slic_sigma,
                )
                print(f"✅ Completed: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"❌ Failed: {os.path.basename(img_path)} - {e}")
    else:
        # Process single image
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"🚀 Processing: {os.path.basename(image_path)}")
        try:
            results = segment_trees(
                image_path,
                model=model,
                auto_k=True,
                output_dir=output_dir,
                image_size=args.image_size,
                feature_upsample_factor=args.feature_upsample_factor,
                pca_dim=args.pca_dim,
                refine=(None if args.refine == "none" else args.refine),
                refine_slic_compactness=args.refine_slic_compactness,
                refine_slic_sigma=args.refine_slic_sigma,
            )
            print(f"✅ Tree segmentation completed!")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
