#!/usr/bin/env python3
"""
Benchmark all available DINOv3 models for performance comparison.
"""

from tree_seg.api import segment_trees

# All available model variants
MODEL_VARIANTS = [
    "dinov3_vits16",    # 21M params - Small
    "dinov3_vitb16",    # 86M params - Base  
    "dinov3_vitl16",    # 300M params - Large
    "dinov3_vith16plus", # 1.1B params - Huge+
    "dinov3_vit7b16",   # 7B params - Mega
]

def run_model_benchmark(model_name: str, image_path: str = "data/input/forest.jpg"):
    """Run benchmark for a single model."""
    print(f"\n{'='*60}")
    print(f"🚀 Running benchmark for {model_name}")
    print(f"{'='*60}")
    
    try:
        # Run segmentation with benchmarking enabled
        segment_trees(
            input_path=image_path,
            output_dir="data/output",
            model=model_name,
            auto_k=True,
            elbow_threshold=0.15
        )
        
        print(f"✅ {model_name} completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} failed: {str(e)}")
        return False

def main():
    """Run benchmarks for all model variants."""
    print("🏁 Starting comprehensive DINOv3 model benchmarking")
    print("="*80)
    
    results = {}
    
    for model_name in MODEL_VARIANTS:
        success = run_model_benchmark(model_name)
        results[model_name] = success
    
    print(f"\n{'='*80}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model_name:<20} {status}")
    
    print("\n📁 Results saved to: data/output/performance_log.jsonl")
    print("🔍 Use 'cat data/output/performance_log.jsonl' to view detailed timings")

if __name__ == "__main__":
    main()
