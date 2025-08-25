#!/usr/bin/env python3
"""
Test script to verify detailed logging during model loading.
"""

import logging
import sys
from tree_seg.api import segment_trees

# Configure logging to show all details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set specific loggers to INFO level
logging.getLogger('tree_seg.models.dinov3.adapter').setLevel(logging.INFO)
logging.getLogger('tree_seg.models.initialization').setLevel(logging.INFO)
logging.getLogger('tree_seg.models.dinov3.weight_loader').setLevel(logging.INFO)

def test_model_loading():
    """Test model loading with detailed logging."""
    
    print("🔍 Testing DINOv3 model loading with detailed logging")
    print("="*80)
    
    # Test Small model first (should work reliably)
    print("\n📦 Testing Small model (21M parameters):")
    print("-"*50)
    
    try:
        results = segment_trees(
            input_path="input/forest.jpg",
            output_dir="output",
            model="dinov3_vits16",
            auto_k=True,
            elbow_threshold=0.15
        )
        print("✅ Small model completed successfully")
    except Exception as e:
        print(f"❌ Small model failed: {e}")
    
    print("\n" + "="*80)
    
    # Test Base model
    print("\n📦 Testing Base model (86M parameters):")
    print("-"*50)
    
    try:
        results = segment_trees(
            input_path="input/forest.jpg",
            output_dir="output", 
            model="dinov3_vitb16",
            auto_k=True,
            elbow_threshold=0.15
        )
        print("✅ Base model completed successfully")
    except Exception as e:
        print(f"❌ Base model failed: {e}")

if __name__ == "__main__":
    test_model_loading()