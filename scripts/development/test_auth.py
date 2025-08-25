#!/usr/bin/env python3
"""
Test HuggingFace authentication and access to gated models.
"""

import os
import logging
import sys
from tree_seg.utils.env_loader import load_environment
from tree_seg.api import segment_trees

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.getLogger('tree_seg.models.dinov3.weight_loader').setLevel(logging.INFO)

def test_auth():
    """Test authentication with HuggingFace."""
    
    print("🔐 Testing HuggingFace Authentication")
    print("="*60)
    
    # Check if token is loaded
    token = os.getenv('HF_TOKEN')
    if token:
        print(f"✅ HF_TOKEN found: {token[:8]}...{token[-8:]}")
    else:
        print("❌ No HF_TOKEN found")
        return
    
    print(f"\n🧪 Testing model access with authentication")
    print("-"*50)
    
    # Test with Large model (304M) - this should need authentication
    print("📦 Testing Large model (304M parameters) - requires auth:")
    
    try:
        results = segment_trees(
            input_path="input/forest.jpg",
            output_dir="output",
            model="dinov3_vitl16",  # Large model - gated
            auto_k=True,
            elbow_threshold=0.15
        )
        print("✅ Large model with auth completed successfully!")
        
    except Exception as e:
        print(f"❌ Large model failed: {e}")
        
    print(f"\n📊 Results should show 'HuggingFace' strategy (not 'random weights')")

if __name__ == "__main__":
    test_auth()