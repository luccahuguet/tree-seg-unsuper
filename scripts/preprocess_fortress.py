#!/usr/bin/env python3
"""
Convert FORTRESS shapefile annotations to raster segmentation masks.

The FORTRESS dataset provides:
- RGB orthomosaics (*.tif) in orthomosaic/
- Vector polygon annotations (*.shp) in shapefile/
- Each polygon has 'species' (text) and 'species_ID' (integer) attributes

This script rasterizes the shapefiles to create pixel-level semantic masks
that match the orthomosaic dimensions.
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from tqdm import tqdm


def get_species_mapping():
    """Return the species ID to class index mapping based on dataset inspection."""
    # Based on the paper and shapefile inspection
    # We'll collect all unique species_IDs across all files first
    return {
        # Will be populated dynamically from the data
    }


def rasterize_shapefile_to_mask(ortho_path, shapefile_path, output_path):
    """
    Rasterize a shapefile to a semantic mask matching the orthomosaic.
    
    Args:
        ortho_path: Path to orthomosaic TIF
        shapefile_path: Path to polygon shapefile
        output_path: Path to save rasterized mask
    """
    # Read orthomosaic metadata (to get transform, crs, dimensions)
    with rasterio.open(ortho_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs
    
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Ensure CRS matches
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    
    # Create list of (geometry, species_ID) pairs
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf['species_ID'])]
    
    # Rasterize
    mask = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=255,  # Background value (IGNORE_INDEX)
        dtype=np.uint8
    )
    
    # Save mask
    meta.update({
        'count': 1,
        'dtype': 'uint8',
        'compress': 'lzw'
    })
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mask, 1)
    
    return mask, gdf


def collect_all_species_ids(shapefile_dir):
    """Collect all unique species IDs and names from all shapefiles."""
    species_map = {}
    
    shapefiles = sorted(shapefile_dir.glob("poly_*.shp"))
    
    print(f"Scanning {len(shapefiles)} shapefiles for species codes...")
    
    for shp_path in tqdm(shapefiles, desc="Collecting species"):
        try:
            gdf = gpd.read_file(shp_path)
            for species_name, species_id in zip(gdf['species'], gdf['species_ID']):
                if species_id not in species_map:
                    species_map[species_id] = species_name
        except Exception as e:
            print(f"Warning: Could not read {shp_path.name}: {e}")
    
    return species_map


def process_fortress_dataset(data_dir, output_dir):
    """
    Process entire FORTRESS dataset.
    
    Creates:
    - images/ directory with symbolic links to orthomosaics
    - labels/ directory with rasterized semantic masks
    - species_mapping.txt with ID to species name mapping
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    ortho_dir = data_dir / "orthomosaic"
    shapefile_dir = data_dir / "shapefile"
    
    # Create output structure
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect species mapping
    species_map = collect_all_species_ids(shapefile_dir)
    
    # Save species mapping
    mapping_file = output_dir / "species_mapping.txt"
    with open(mapping_file, 'w') as f:
        f.write("# FORTRESS Species Mapping\n")
        f.write("# species_ID -> species_name\n\n")
        for species_id in sorted(species_map.keys()):
            f.write(f"{species_id}: {species_map[species_id]}\n")
    
    print(f"\nFound {len(species_map)} unique species/classes:")
    for species_id in sorted(species_map.keys()):
        print(f"  {species_id}: {species_map[species_id]}")
    
    # Find orthomosaic-shapefile pairs
    ortho_files = sorted(ortho_dir.glob("*_ortho.tif"))
    
    print(f"\nProcessing {len(ortho_files)} orthomosaic-shapefile pairs...")
    
    processed = 0
    for ortho_path in tqdm(ortho_files, desc="Rasterizing"):
        # Extract site ID (e.g., CFB003 from CFB003_ortho.tif)
        site_id = ortho_path.stem.replace("_ortho", "")
        
        # Find corresponding shapefile
        shapefile_path = shapefile_dir / f"poly_{site_id}.shp"
        
        if not shapefile_path.exists():
            print(f"Warning: No shapefile found for {site_id}")
            continue
        
        # Create symlink to orthomosaic
        image_link = images_dir / f"{site_id}.tif"
        if not image_link.exists():
            image_link.symlink_to(ortho_path.resolve())
        
        # Rasterize shapefile to mask
        label_path = labels_dir / f"{site_id}_label.tif"
        
        try:
            mask, gdf = rasterize_shapefile_to_mask(ortho_path, shapefile_path, label_path)
            processed += 1
        except Exception as e:
            print(f"Error processing {site_id}: {e}")
    
    print(f"\nSuccessfully processed {processed}/{len(ortho_files)} samples")
    print(f"Output directory: {output_dir}")
    print("  - images/: Symbolic links to orthomosaics")
    print("  - labels/: Rasterized semantic masks")
    print("  - species_mapping.txt: Species ID mapping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess FORTRESS dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data/fortress/10.35097-538/data/dataset",
        help="Path to FORTRESS dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/fortress_processed",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--extract-ortho",
        action="store_true",
        help="Extract orthomosaic.zip if not already extracted"
    )
    
    args = parser.parse_args()
    
    # Extract orthomosaic if requested
    if args.extract_ortho:
        import zipfile
        ortho_zip = args.data_dir / "orthomosaic.zip"
        if ortho_zip.exists():
            print(f"Extracting {ortho_zip}...")
            with zipfile.ZipFile(ortho_zip, 'r') as zip_ref:
                zip_ref.extractall(args.data_dir)
            print("Extraction complete")
        else:
            print(f"Warning: {ortho_zip} not found")
    
    process_fortress_dataset(args.data_dir, args.output_dir)
