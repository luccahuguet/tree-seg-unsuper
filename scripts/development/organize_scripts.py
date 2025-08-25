#!/usr/bin/env python3
"""
Organize development and test scripts into proper directories.
"""

import shutil
from pathlib import Path

def organize_scripts():
    """Move test and development scripts to appropriate directories."""
    
    project_root = Path(".")
    
    # Create directories
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    
    dev_dir = scripts_dir / "development"
    dev_dir.mkdir(exist_ok=True)
    
    utils_dir = scripts_dir / "utils"
    utils_dir.mkdir(exist_ok=True)
    
    print("📁 Organizing scripts into directories")
    print("="*50)
    
    # Files to move to scripts/development (testing/one-time use)
    dev_files = [
        "benchmark_all_models.py",
        "compare_auth_results.py", 
        "test_auth.py",
        "test_detailed_logging.py",
        "performance_analysis.py",  # Overlaps with view_benchmarks.py
    ]
    
    # Files to move to scripts/utils (ongoing utilities)
    util_files = [
        "view_benchmarks.py",
        "convert_jsonl.py",
    ]
    
    # Files to keep in root (core functionality)
    keep_in_root = [
        "manage_model_weights.py",  # Essential for model management
        "restore_models.py",        # Auto-generated, users expect in root
        "run_segmentation.py",      # Main execution script
        "tree_seg_notebook.py",     # Kaggle notebook
    ]
    
    # Move development files
    print("\n🧪 Moving development/test files:")
    for filename in dev_files:
        src = project_root / filename
        if src.exists():
            dst = dev_dir / filename
            shutil.move(str(src), str(dst))
            print(f"  ✅ {filename} → scripts/development/")
    
    # Move utility files
    print("\n🛠️ Moving utility scripts:")
    for filename in util_files:
        src = project_root / filename
        if src.exists():
            dst = utils_dir / filename
            shutil.move(str(src), str(dst))
            print(f"  ✅ {filename} → scripts/utils/")
    
    # Report what's staying in root
    print("\n📌 Keeping in root:")
    for filename in keep_in_root:
        if (project_root / filename).exists():
            print(f"  📄 {filename}")
    
    # Create README files
    create_readme_files(scripts_dir, dev_dir, utils_dir)
    
    print(f"\n🎉 Organization complete!")
    print(f"📁 Development scripts: scripts/development/")
    print(f"🛠️ Utility scripts: scripts/utils/")
    print(f"📌 Core scripts remain in root")

def create_readme_files(scripts_dir, dev_dir, utils_dir):
    """Create README files for the script directories."""
    
    # Main scripts README
    scripts_readme = scripts_dir / "README.md"
    with open(scripts_readme, 'w') as f:
        f.write("""# Project Scripts

This directory contains various scripts organized by purpose.

## 📁 Directory Structure

- `development/` - Testing, benchmarking, and one-time analysis scripts
- `utils/` - Ongoing utility scripts for data viewing and conversion

## 🚀 Quick Access

Most commonly used scripts remain in the project root:
- `manage_model_weights.py` - Model weight management
- `restore_models.py` - Restore cached model weights  
- `run_segmentation.py` - Main segmentation execution
- `tree_seg_notebook.py` - Kaggle notebook version
""")
    
    # Development README
    dev_readme = dev_dir / "README.md"
    with open(dev_readme, 'w') as f:
        f.write("""# Development Scripts

Scripts used for testing, benchmarking, and one-time analysis tasks.

## 🧪 Testing Scripts

- `test_auth.py` - Test HuggingFace authentication
- `test_detailed_logging.py` - Test detailed model loading diagnostics

## 📊 Benchmarking Scripts

- `benchmark_all_models.py` - Run comprehensive model benchmarks
- `compare_auth_results.py` - Compare pre/post authentication results
- `performance_analysis.py` - Analyze performance scaling metrics

## 💡 Usage

These scripts were primarily used during development and testing phases.
They remain available for debugging or re-analysis but are not part of
the core workflow.
""")
    
    # Utils README  
    utils_readme = utils_dir / "README.md"
    with open(utils_readme, 'w') as f:
        f.write("""# Utility Scripts

Ongoing utility scripts for data management and analysis.

## 📊 Data Analysis

- `view_benchmarks.py` - Elegant benchmark data viewer with multiple display modes
  ```bash
  python scripts/utils/view_benchmarks.py --mode summary
  python scripts/utils/view_benchmarks.py --mode analysis
  python scripts/utils/view_benchmarks.py --latest 5
  ```

## 🔄 Data Conversion

- `convert_jsonl.py` - Convert between pretty-printed JSON and regular JSONL formats
  ```bash
  python scripts/utils/convert_jsonl.py performance_log.jsonl --to-jsonl
  python scripts/utils/convert_jsonl.py performance_log.jsonl --to-pretty
  ```

## 💡 Usage

These utilities provide ongoing value for data analysis and format conversion.
""")
    
    print(f"📝 Created README files for script directories")

if __name__ == "__main__":
    organize_scripts()