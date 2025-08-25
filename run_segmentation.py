#!/usr/bin/env python3
"""
Simple CLI for tree segmentation.
Usage: python run_segmentation.py [image_path] [model] [output_dir]
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Default values
    image_path = "input/"
    model = "small"
    output_dir = "output"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 2:
        model = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
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
                    output_dir=output_dir
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
                output_dir=output_dir
            )
            print(f"✅ Tree segmentation completed!")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()