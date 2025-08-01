"""
Modern, clean API for tree segmentation.
"""

import os
import torch
from typing import Optional, List, Dict, Any
from pathlib import Path

from .core.types import Config, SegmentationResults, OutputPaths
from .core.output_manager import OutputManager
from .models import initialize_model, get_preprocess, print_gpu_info
from .core.segmentation import process_image
from .visualization.plotting import generate_visualizations


class TreeSegmentation:
    """
    Modern tree segmentation API with clean interface.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize tree segmentation with configuration.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or Config()
        self.config.validate()
        
        self.output_manager = OutputManager(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        
        print(f"🌳 TreeSegmentation initialized")
        print(f"📱 Device: {self.device}")
        print(f"🔧 Model: {self.config.model_display_name}")
        print(f"📁 Output: {self.config.output_dir}")
    
    def initialize_model(self) -> None:
        """Initialize the model and preprocessing pipeline."""
        if self.model is None:
            print("🔄 Initializing model...")
            self.model = initialize_model(
                self.config.stride, 
                self.config.model_display_name, 
                self.device
            )
            self.preprocess = get_preprocess()
            print("✅ Model initialized")
    
    def process_single_image(self, image_path: str) -> SegmentationResults:
        """
        Process a single image for tree segmentation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            SegmentationResults with processed data
        """
        self.initialize_model()
        
        print(f"🖼️ Processing: {os.path.basename(image_path)}")
        
        # Process with the updated signature
        result = process_image(
            image_path=image_path,
            model=self.model,
            preprocess=self.preprocess,
            n_clusters=self.config.n_clusters,
            stride=self.config.stride,
            version=self.config.version,
            device=self.device,
            auto_k=self.config.auto_k,
            k_range=self.config.k_range,
            elbow_threshold=self.config.elbow_threshold_decimal,
            use_pca=self.config.use_pca,
            model_name=self.config.model_display_name
        )
        
        if result[0] is None:
            raise RuntimeError(f"Failed to process image: {image_path}")
        
        image_np, labels_resized = result
        n_clusters_used = len(torch.unique(torch.from_numpy(labels_resized)))
        
        return SegmentationResults(
            image_np=image_np,
            labels_resized=labels_resized,
            n_clusters_used=n_clusters_used,
            image_path=image_path,
            processing_stats={
                'original_size': image_np.shape[:2],
                'labels_shape': labels_resized.shape,
                'auto_k_used': self.config.auto_k,
                'elbow_threshold': self.config.elbow_threshold if self.config.auto_k else None
            }
        )
    
    def generate_visualizations(self, results: SegmentationResults) -> OutputPaths:
        """
        Generate visualization outputs for segmentation results.
        
        Args:
            results: SegmentationResults from processing
            
        Returns:
            OutputPaths with generated file paths
        """
        output_paths = self.output_manager.generate_output_paths(
            results.image_path, 
            results.n_clusters_used,
            include_elbow=self.config.auto_k
        )
        
        # Generate visualizations using OutputManager paths
        generate_visualizations(
            results=results,
            config=self.config,
            output_paths=output_paths
        )
        
        # Auto-optimize for web if enabled (now enabled by default)
        if self.config.web_optimize:
            output_paths = self.output_manager.optimize_all_outputs(output_paths)
        
        return output_paths
    
    def process_and_visualize(self, image_path: str) -> tuple[SegmentationResults, OutputPaths]:
        """
        Complete pipeline: process image and generate visualizations.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (SegmentationResults, OutputPaths)
        """
        results = self.process_single_image(image_path)
        output_paths = self.generate_visualizations(results)
        
        print(f"✅ Completed: {os.path.basename(image_path)}")
        print(f"🎯 Used K = {results.n_clusters_used}")
        
        if self.config.auto_k:
            print(f"📊 Method: elbow (threshold: {self.config.elbow_threshold})")
        
        return results, output_paths
    
    def process_directory(self, directory_path: Optional[str] = None) -> List[tuple[SegmentationResults, OutputPaths]]:
        """
        Process all images in a directory.
        
        Args:
            directory_path: Directory path. If None, uses config.input_dir
            
        Returns:
            List of (SegmentationResults, OutputPaths) tuples
        """
        input_dir = directory_path or self.config.input_dir
        
        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        
        image_files = []
        for ext in extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No supported images found in {input_dir}")
        
        print(f"📁 Found {len(image_files)} images to process")
        
        results = []
        for image_path in image_files:
            try:
                result = self.process_and_visualize(str(image_path))
                results.append(result)
            except Exception as e:
                print(f"❌ Failed to process {image_path.name}: {e}")
                continue
        
        print(f"🎉 Completed processing {len(results)}/{len(image_files)} images")
        return results
    
    def find_latest_outputs(self) -> Optional[OutputPaths]:
        """Find the most recently generated output files."""
        return self.output_manager.find_latest_outputs()
    
    def cleanup_old_outputs(self, keep_latest: int = 5) -> None:
        """Clean up old output files."""
        self.output_manager.cleanup_old_outputs(keep_latest)


# Convenience function for quick usage
def segment_trees(
    input_path: str,
    output_dir: str = "/kaggle/working/output", 
    model: str = "base",
    auto_k: bool = True,
    web_optimize: bool = False,
    **kwargs
) -> List[tuple[SegmentationResults, OutputPaths]]:
    """
    Convenience function for quick tree segmentation.
    
    Args:
        input_path: Path to image file or directory
        output_dir: Output directory path
        model: Model size ("small", "base", "large", "giant")
        auto_k: Whether to use automatic K selection
        web_optimize: Whether to auto-optimize images for web display
        **kwargs: Additional config parameters
        
    Returns:
        List of (SegmentationResults, OutputPaths) tuples
    """
    config = Config(
        output_dir=output_dir,
        model_name=model,
        auto_k=auto_k,
        web_optimize=web_optimize,
        **kwargs
    )
    
    segmenter = TreeSegmentation(config)
    
    if os.path.isfile(input_path):
        # Single file
        config.filename = os.path.basename(input_path)
        config.input_dir = os.path.dirname(input_path)
        result = segmenter.process_and_visualize(input_path)
        return [result]
    else:
        # Directory
        config.input_dir = input_path
        return segmenter.process_directory()