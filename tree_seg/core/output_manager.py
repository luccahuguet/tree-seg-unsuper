"""
Output file management for tree segmentation.
"""

import os
import hashlib
import glob
from pathlib import Path
from typing import Optional, List

from .types import Config, OutputPaths
from ..utils.config import parse_model_info


class OutputManager:
    """
    Manages output file naming, paths, and discovery.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_filename_prefix(self, image_path: str, n_clusters: int) -> str:
        """
        Generate config-based filename prefix.
        
        Args:
            image_path: Path to source image
            n_clusters: Number of clusters used
            
        Returns:
            Config-based filename prefix (e.g., "a3f7_v1-5_base_str4_et3-0")
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
        
        # Add clustering info
        if self.config.auto_k:
            et_str = f"et{str(self.config.elbow_threshold).replace('.', '-')}"
            components.append(et_str)
        else:
            components.append(f"k{n_clusters}")
        
        return "_".join(components)
    
    def generate_output_paths(self, image_path: str, n_clusters: int, 
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
        prefix = self.generate_filename_prefix(image_path, n_clusters)
        
        paths = OutputPaths(
            segmentation_legend=str(self.output_dir / f"{prefix}_segmentation_legend.png"),
            edge_overlay=str(self.output_dir / f"{prefix}_edge_overlay.png"),
            side_by_side=str(self.output_dir / f"{prefix}_side_by_side.png"),
            elbow_analysis=str(self.output_dir / f"{prefix}_elbow_analysis.png") if include_elbow else None
        )
        
        return paths
    
    def find_latest_outputs(self) -> Optional[OutputPaths]:
        """
        Find the most recently generated output files.
        
        Returns:
            OutputPaths with latest files, or None if no files found
        """
        patterns = {
            'segmentation_legend': "*_segmentation_legend.png",
            'edge_overlay': "*_edge_overlay.png", 
            'side_by_side': "*_side_by_side.png",
            'elbow_analysis': "*_elbow_analysis.png"
        }
        
        latest_files = {}
        
        for key, pattern in patterns.items():
            files = glob.glob(str(self.output_dir / pattern))
            if files:
                latest_files[key] = max(files, key=os.path.getmtime)
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
        List all output files in the output directory.
        
        Returns:
            List of output file paths
        """
        patterns = [
            "*_segmentation_legend.png",
            "*_edge_overlay.png", 
            "*_side_by_side.png",
            "*_elbow_analysis.png"
        ]
        
        all_files = []
        for pattern in patterns:
            files = glob.glob(str(self.output_dir / pattern))
            all_files.extend(files)
        
        return sorted(all_files, key=os.path.getmtime, reverse=True)
    
    def cleanup_old_outputs(self, keep_latest: int = 5) -> None:
        """
        Clean up old output files, keeping only the most recent ones.
        
        Args:
            keep_latest: Number of latest files to keep for each type
        """
        patterns = [
            "*_segmentation_legend.png",
            "*_edge_overlay.png", 
            "*_side_by_side.png",
            "*_elbow_analysis.png"
        ]
        
        for pattern in patterns:
            files = glob.glob(str(self.output_dir / pattern))
            if len(files) > keep_latest:
                # Sort by modification time, keep latest
                files_by_time = sorted(files, key=os.path.getmtime, reverse=True)
                files_to_delete = files_by_time[keep_latest:]
                
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        print(f"🗑️ Cleaned up: {os.path.basename(file_path)}")
                    except OSError:
                        pass