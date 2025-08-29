"""
Output file management for tree segmentation.
"""

import os
import hashlib
import glob
from pathlib import Path
from typing import Optional, List
from PIL import Image

from .types import Config, OutputPaths
from ..utils.config import parse_model_info


class OutputManager:
    """
    Manages output file naming, paths, and discovery.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.png_dir = self.output_dir / "png"
        self.web_dir = self.output_dir / "web"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.web_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_filename_prefix(self, image_path: str, n_clusters_used: int, requested_k: int | None = None) -> str:
        """
        Generate config-based filename prefix.
        
        Args:
            image_path: Path to source image
            n_clusters_used: Number of clusters used
            requested_k: Optional K requested (e.g., from elbow); appended if differs
            
        Returns:
            Config-based filename prefix (e.g., "a3f7_v1-5_base_str4_et3-5_k5")
        """
        filename = os.path.basename(image_path)
        
        # Generate 4-character hash from filename
        file_hash = hashlib.sha1(filename.encode()).hexdigest()[:4]
        
        # Parse model info
        _, nickname, _ = parse_model_info(self.config.model_display_name)
        model_nick = nickname.lower()
        
        # Format version (replace dots with hyphens)
        version_str = self.config.version.replace(".", "-")
        
        # Build filename components
        components = [
            file_hash,
            version_str,
            model_nick,
            f"str{self.config.stride}"
        ]
        
        # Add clustering method info
        if self.config.auto_k:
            # Format elbow threshold without float artifacts (e.g., 3.5)
            et_clean = (f"{self.config.elbow_threshold:.2f}".rstrip('0').rstrip('.')).replace('.', '-')
            et_str = f"et{et_clean}"
            components.append(et_str)
        
        # Always add the actual K value used; append requested if different
        if requested_k is not None and requested_k != n_clusters_used:
            components.append(f"k{n_clusters_used}r{requested_k}")
        else:
            components.append(f"k{n_clusters_used}")
        
        return "_".join(components)
    
    def generate_output_paths(self, image_path: str, n_clusters: int,
                              requested_k: int | None = None,
                              include_elbow: bool = False) -> OutputPaths:
        """
        Generate all output file paths.
        
        Args:
            image_path: Path to source image
            n_clusters: Number of clusters used
            include_elbow: Whether to include elbow analysis path
            
        Returns:
            OutputPaths with all file paths
        """
        prefix = self.generate_filename_prefix(image_path, n_clusters, requested_k)
        
        # Generate PNG paths (stored in png/ subfolder)
        paths = OutputPaths(
            segmentation_legend=str(self.png_dir / f"{prefix}_segmentation_legend.png"),
            edge_overlay=str(self.png_dir / f"{prefix}_edge_overlay.png"),
            side_by_side=str(self.png_dir / f"{prefix}_side_by_side.png"),
            elbow_analysis=str(self.png_dir / f"{prefix}_elbow_analysis.png") if include_elbow else None
        )
        
        return paths
    
    def find_latest_outputs(self) -> Optional[OutputPaths]:
        """
        Find the most recently generated output files.
        Prefers web-optimized outputs, falls back to PNG if not available.
        
        Returns:
            OutputPaths with latest files, or None if no files found
        """
        patterns = {
            'segmentation_legend': "*_segmentation_legend",
            'edge_overlay': "*_edge_overlay", 
            'side_by_side': "*_side_by_side",
            'elbow_analysis': "*_elbow_analysis"
        }
        
        latest_files = {}
        
        for key, pattern in patterns.items():
            # First try web-optimized (JPG) files
            web_files = glob.glob(str(self.web_dir / f"{pattern}.jpg"))
            png_files = glob.glob(str(self.png_dir / f"{pattern}.png"))
            
            all_files = web_files + png_files
            if all_files:
                latest_files[key] = max(all_files, key=os.path.getmtime)
            else:
                latest_files[key] = None
        
        # Return None if no core files found
        if not any([latest_files['segmentation_legend'], 
                   latest_files['edge_overlay'], 
                   latest_files['side_by_side']]):
            return None
        
        return OutputPaths(
            segmentation_legend=latest_files['segmentation_legend'],
            edge_overlay=latest_files['edge_overlay'],
            side_by_side=latest_files['side_by_side'],
            elbow_analysis=latest_files['elbow_analysis']
        )
    
    def list_all_outputs(self) -> List[str]:
        """
        List all output files in both PNG and web directories.
        
        Returns:
            List of output file paths
        """
        patterns = [
            "*_segmentation_legend",
            "*_edge_overlay", 
            "*_side_by_side",
            "*_elbow_analysis"
        ]
        
        all_files = []
        for pattern in patterns:
            # Add PNG files
            png_files = glob.glob(str(self.png_dir / f"{pattern}.png"))
            all_files.extend(png_files)
            
            # Add web-optimized files
            web_files = glob.glob(str(self.web_dir / f"{pattern}.jpg"))
            all_files.extend(web_files)
        
        return sorted(all_files, key=os.path.getmtime, reverse=True)
    
    def cleanup_old_outputs(self, keep_latest: int = 5) -> None:
        """
        Clean up old output files, keeping only the most recent ones.
        
        Args:
            keep_latest: Number of latest files to keep for each type
        """
        patterns = [
            "*_segmentation_legend",
            "*_edge_overlay", 
            "*_side_by_side",
            "*_elbow_analysis"
        ]
        
        for pattern in patterns:
            # Clean PNG files
            png_files = glob.glob(str(self.png_dir / f"{pattern}.png"))
            if len(png_files) > keep_latest:
                files_by_time = sorted(png_files, key=os.path.getmtime, reverse=True)
                files_to_delete = files_by_time[keep_latest:]
                
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        print(f"🗑️ Cleaned up PNG: {os.path.basename(file_path)}")
                    except OSError:
                        pass
            
            # Clean web files
            web_files = glob.glob(str(self.web_dir / f"{pattern}.jpg"))
            if len(web_files) > keep_latest:
                files_by_time = sorted(web_files, key=os.path.getmtime, reverse=True)
                files_to_delete = files_by_time[keep_latest:]
                
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        print(f"🗑️ Cleaned up web: {os.path.basename(file_path)}")
                    except OSError:
                        pass
    
    def optimize_image_for_web(self, input_path: str) -> Optional[str]:
        """
        Optimize a single image for web display.
        
        Args:
            input_path: Path to original PNG image
            
        Returns:
            Path to optimized JPEG image, or None if optimization failed
        """
        if not self.config.web_optimize:
            return None
            
        try:
            # Determine output path in web/ folder
            input_file = Path(input_path)
            filename_stem = input_file.stem
            output_file = self.web_dir / f"{filename_stem}.jpg"
            
            with Image.open(input_path) as img:
                # Convert to RGB if necessary (handles PNG with transparency)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too wide (maintains aspect ratio)
                if img.width > self.config.web_max_width:
                    ratio = self.config.web_max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((self.config.web_max_width, new_height), Image.Resampling.LANCZOS)
                
                # Save as optimized JPEG
                img.save(output_file, 'JPEG', quality=self.config.web_quality, optimize=True)
                
                # Get file sizes for reporting
                original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
                optimized_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                
                print(f"🌐 Web optimized: {input_file.name}")
                print(f"   {original_size:.1f}MB → {optimized_size:.1f}MB ({optimized_size/original_size*100:.0f}%)")
                
                return str(output_file)
                
        except Exception as e:
            print(f"⚠️ Web optimization failed for {input_path}: {e}")
            return None
    
    def optimize_all_outputs(self, output_paths: OutputPaths) -> OutputPaths:
        """
        Optimize all output images for web display.
        
        Args:
            output_paths: Original output paths
            
        Returns:
            OutputPaths with optimized image paths (or original if optimization disabled)
        """
        if not self.config.web_optimize:
            return output_paths
        
        print("🌐 Optimizing images for web...")
        
        optimized = OutputPaths(
            segmentation_legend=self.optimize_image_for_web(output_paths.segmentation_legend) or output_paths.segmentation_legend,
            edge_overlay=self.optimize_image_for_web(output_paths.edge_overlay) or output_paths.edge_overlay,
            side_by_side=self.optimize_image_for_web(output_paths.side_by_side) or output_paths.side_by_side,
            elbow_analysis=self.optimize_image_for_web(output_paths.elbow_analysis) if output_paths.elbow_analysis else None
        )
        
        return optimized
