#!/usr/bin/env python3
"""
Generate Documentation Images

Single script that runs the complete documentation image generation sweep
and organizes the results into the proper docs structure.

Usage:
    uv run python scripts/generate_docs_images.py [input_image] [--clean]

Direct execution remains available:
    python scripts/generate_docs_images.py [input_image] [--clean]
    
Options:
    input_image: Path to input image (default: input/forest.jpg)
    --clean: Clean output directories before generation
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import argparse

def create_sweep_config():
    """Create the streamlined sweep configuration for documentation images."""
    return [
        # Basic example (used for both methodology and complete example docs)
        {
            "name": "basic_example",
            "model": "base",
            "profile": "balanced",
            "stride": 4,
            "web_optimize": True,
            "verbose": True
        },
        
        # Stride comparison (Giant model only)
        {
            "name": "stride_comparison_str2",
            "model": "giant",
            "profile": "quality",
            "stride": 2,
            "web_optimize": True,
            "verbose": True
        },
        {
            "name": "stride_comparison_str4",
            "model": "giant",
            "profile": "quality", 
            "stride": 4,
            "web_optimize": True,
            "verbose": True
        },
        
        # Model size comparison (Best stride = 2)
        {
            "name": "model_comparison_small",
            "model": "small",
            "profile": "quality",
            "stride": 2,
            "web_optimize": True,
            "verbose": True
        },
        {
            "name": "model_comparison_base",
            "model": "base",
            "profile": "quality",
            "stride": 2,
            "web_optimize": True,
            "verbose": True
        },
        {
            "name": "model_comparison_large",
            "model": "large",
            "profile": "quality",
            "stride": 2,
            "web_optimize": True,
            "verbose": True
        },
        {
            "name": "model_comparison_giant",
            "model": "giant",
            "profile": "quality",
            "stride": 2,
            "web_optimize": True,
            "verbose": True
        },
        
        # Elbow threshold comparison (Giant model, stride 2) - 2x2 grid
        {
            "name": "elbow_threshold_2_5",
            "model": "giant",
            "profile": "quality",
            "stride": 2,
            "elbow_threshold": 2.5,
            "web_optimize": True,
            "verbose": True
        },
        {
            "name": "elbow_threshold_5_0",
            "model": "giant", 
            "profile": "quality",
            "stride": 2,
            "elbow_threshold": 5.0,
            "web_optimize": True,
            "verbose": True
        },
        {
            "name": "elbow_threshold_10_0",
            "model": "giant",
            "profile": "quality",
            "stride": 2,
            "elbow_threshold": 10.0,
            "web_optimize": True,
            "verbose": True
        },
        {
            "name": "elbow_threshold_20_0",
            "model": "giant",
            "profile": "quality",
            "stride": 2,
            "elbow_threshold": 20.0,
            "web_optimize": True,
            "verbose": True
        },
        
        # Refinement comparison (Giant model, stride 2, default elbow)
        {
            "name": "refine_with_slic",
            "model": "giant",
            "profile": "quality",
            "stride": 2,
            "refine": "slic",
            "web_optimize": True,
            "verbose": True
        },
        {
            "name": "refine_none",
            "model": "giant",
            "profile": "quality", 
            "stride": 2,
            "refine": "none",
            "web_optimize": True,
            "verbose": True
        }
    ]

def run_sweep(input_image):
    """Run the documentation image generation sweep."""
    
    # Create temporary sweep config
    sweep_config = create_sweep_config()
    temp_sweep_file = "temp_docs_sweep.json"
    
    try:
        # Write temporary sweep config
        with open(temp_sweep_file, 'w') as f:
            json.dump(sweep_config, f, indent=2)
        
        print("🚀 Starting documentation image generation sweep...")
        print(f"📸 Input image: {input_image}")
        print(f"🔧 Configurations: {len(sweep_config)}")
        print()
        
        # Build command (always clean for consistent results)
        run_script = Path(__file__).parent / "run_segmentation.py"
        cmd = [
            "python", str(run_script),
            input_image,
            "base",  # Default model (overridden by sweep)
            "output",
            "--sweep", temp_sweep_file,
            "--clean-output",
            "--verbose"
        ]
        
        # Run the sweep
        subprocess.run(cmd, check=True)
        
        print("\n✅ Sweep generation completed!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Sweep failed with error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_sweep_file):
            os.remove(temp_sweep_file)
    
    return True

def organize_images():
    """Organize generated images into documentation structure."""
    
    print("\n🗂️  Organizing documentation images...")
    
    # Base paths
    sweep_base = Path("output/sweeps")
    docs_results = Path("docs/results")
    
    # Clean existing docs results for fresh start
    if docs_results.exists():
        print("🧹 Cleaning existing documentation images...")
        shutil.rmtree(docs_results)
    
    # Ensure docs results directories exist
    (docs_results / "methodology").mkdir(parents=True, exist_ok=True)
    (docs_results / "complete_example").mkdir(parents=True, exist_ok=True)
    (docs_results / "parameter_comparison" / "stride").mkdir(parents=True, exist_ok=True)
    (docs_results / "parameter_comparison" / "model_size").mkdir(parents=True, exist_ok=True)
    (docs_results / "parameter_comparison" / "elbow_threshold").mkdir(parents=True, exist_ok=True)
    (docs_results / "parameter_comparison" / "refinement").mkdir(parents=True, exist_ok=True)
    
    # Image organization mapping with expected documentation filenames
    mappings = [
        # Basic example images (shared by methodology and complete example docs)
        {
            "source": "basic_example/web",
            "target": "methodology",
            "files": {
                "*_segmentation_legend.jpg": "basic_example_segmentation_legend.jpg",
                "*_edge_overlay.jpg": "basic_example_edge_overlay.jpg",
                "*_side_by_side.jpg": "basic_example_side_by_side.jpg",
                "*_elbow_analysis.jpg": "basic_example_elbow_analysis.jpg"
            }
        },
        {
            "source": "basic_example/web",
            "target": "complete_example", 
            "files": {
                "*_segmentation_legend.jpg": "basic_example_segmentation_legend.jpg",
                "*_edge_overlay.jpg": "basic_example_edge_overlay.jpg",
                "*_side_by_side.jpg": "basic_example_side_by_side.jpg",
                "*_elbow_analysis.jpg": "basic_example_elbow_analysis.jpg"
            }
        },
        
        # Stride comparison (web-optimized)
        {
            "source": "stride_comparison_str2/web",
            "target": "parameter_comparison/stride",
            "files": {
                "*_edge_overlay.jpg": "stride_comparison_str2_edge_overlay.jpg"
            }
        },
        {
            "source": "stride_comparison_str4/web",
            "target": "parameter_comparison/stride",
            "files": {
                "*_edge_overlay.jpg": "stride_comparison_str4_edge_overlay.jpg"
            }
        },
        
        # Model size comparison (web-optimized)
        {
            "source": "model_comparison_small/web",
            "target": "parameter_comparison/model_size",
            "files": {
                "*_edge_overlay.jpg": "model_comparison_small_edge_overlay.jpg"
            }
        },
        {
            "source": "model_comparison_base/web", 
            "target": "parameter_comparison/model_size",
            "files": {
                "*_edge_overlay.jpg": "model_comparison_base_edge_overlay.jpg"
            }
        },
        {
            "source": "model_comparison_large/web",
            "target": "parameter_comparison/model_size", 
            "files": {
                "*_edge_overlay.jpg": "model_comparison_large_edge_overlay.jpg"
            }
        },
        {
            "source": "model_comparison_giant/web",
            "target": "parameter_comparison/model_size",
            "files": {
                "*_edge_overlay.jpg": "model_comparison_giant_edge_overlay.jpg"
            }
        },
        
        # Elbow threshold comparison (web-optimized) - 2x2 grid
        {
            "source": "elbow_threshold_2_5/web",
            "target": "parameter_comparison/elbow_threshold",
            "files": {
                "*_edge_overlay.jpg": "elbow_threshold_2_5_edge_overlay.jpg"
            }
        },
        {
            "source": "elbow_threshold_5_0/web",
            "target": "parameter_comparison/elbow_threshold",
            "files": {
                "*_edge_overlay.jpg": "elbow_threshold_5_0_edge_overlay.jpg",
                "*_elbow_analysis.jpg": "elbow_threshold_5_0_elbow_analysis.jpg"
            }
        },
        {
            "source": "elbow_threshold_10_0/web",
            "target": "parameter_comparison/elbow_threshold",
            "files": {
                "*_edge_overlay.jpg": "elbow_threshold_10_0_edge_overlay.jpg"
            }
        },
        {
            "source": "elbow_threshold_20_0/web",
            "target": "parameter_comparison/elbow_threshold",
            "files": {
                "*_edge_overlay.jpg": "elbow_threshold_20_0_edge_overlay.jpg"
            }
        },
        
        # Refinement comparison (web-optimized)
        {
            "source": "refine_with_slic/web",
            "target": "parameter_comparison/refinement",
            "files": {
                "*_edge_overlay.jpg": "refine_with_slic_edge_overlay.jpg"
            }
        },
        {
            "source": "refine_none/web",
            "target": "parameter_comparison/refinement",
            "files": {
                "*_edge_overlay.jpg": "refine_none_edge_overlay.jpg"
            }
        }
    ]
    
    total_copied = 0
    
    for mapping in mappings:
        source_dir = sweep_base / mapping["source"]
        target_dir = docs_results / mapping["target"]
        
        if not source_dir.exists():
            print(f"⚠️  Source directory not found: {source_dir}")
            continue
            
        print(f"📁 Processing {mapping['source']} -> {mapping['target']}")
        
        # Find and copy matching files with renaming
        for pattern, target_name in mapping["files"].items():
            import glob
            matches = glob.glob(str(source_dir / pattern))
            
            for match in matches:
                source_file = Path(match)
                target_file = target_dir / target_name
                
                try:
                    shutil.copy2(source_file, target_file)
                    print(f"   ✅ Copied: {source_file.name} -> {target_name}")
                    total_copied += 1
                    break  # Only copy the first match for each pattern
                except Exception as e:
                    print(f"   ❌ Failed to copy {source_file.name}: {e}")
    
    print("\n🎯 Image organization complete!")
    print(f"📊 Total images copied: {total_copied}")
    print(f"📍 Images organized in: {docs_results}")
    
    return total_copied > 0

def main():
    parser = argparse.ArgumentParser(description="Generate documentation images")
    parser.add_argument("input_image", nargs="?", default="input/forest2.jpeg", 
                       help="Path to input image (default: input/forest2.jpeg)")
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.input_image):
        print(f"❌ Input image not found: {args.input_image}")
        print("Available images in input/:")
        if os.path.exists("input"):
            for f in os.listdir("input"):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    print(f"  - input/{f}")
        return 1
    
    print("🎨 Documentation Image Generation")
    print("=" * 50)
    
    # Step 1: Run the sweep (always clean for consistent results)
    if not run_sweep(args.input_image):
        print("❌ Failed to generate images")
        return 1
    
    # Step 2: Organize images
    if not organize_images():
        print("❌ Failed to organize images")
        return 1
    
    print("\n🎉 Documentation image generation complete!")
    print("\nGenerated images for:")
    print("  📖 Methodology - Pipeline demonstration")
    print("  📋 Complete Example - Full workflow showcase") 
    print("  🔄 Stride Comparison - Giant model at stride 2 vs 4")
    print("  🎯 Model Size Comparison - Small/Base/Large/Giant at stride 4")
    print("  📊 Elbow Threshold Comparison - 2.5%, 5.0%, 10.0%, 20.0% thresholds (2x2 grid)")
    print("  🔧 Refinement Comparison - With/without SLIC refinement")
    print("\n💡 Web-optimized images ready for Jekyll documentation site!")
    print("📦 Only lightweight .jpg files added to git (no heavy .png files)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
