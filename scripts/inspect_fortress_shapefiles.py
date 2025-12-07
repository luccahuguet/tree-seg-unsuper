#!/usr/bin/env python3
"""Inspect FORTRESS shapefile attributes to understand species labeling."""

import sys
from pathlib import Path

# Try ogr first, fall back to fiona/geopandas
try:
    from osgeo import ogr

    USE_OGR = True
except ImportError:
    try:
        import fiona

        USE_OGR = False
    except ImportError:
        print("Error: Neither GDAL/OGR nor fiona available.")
        print("Install with: pip install gdal OR pip install fiona")
        sys.exit(1)


def inspect_shapefile_ogr(shp_path):
    """Inspect shapefile using OGR."""
    ds = ogr.Open(str(shp_path))
    if not ds:
        print(f"Failed to open {shp_path}")
        return

    layer = ds.GetLayer()
    print(f"\nFile: {shp_path.name}")
    print(f"Features: {layer.GetFeatureCount()}")

    # Get field names
    layer_defn = layer.GetLayerDefn()
    field_names = [
        layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())
    ]
    print(f"Fields: {field_names}")

    # Show first feature
    feat = layer.GetNextFeature()
    if feat:
        print("\nFirst feature attributes:")
        for field in field_names:
            print(f"  {field}: {feat.GetField(field)}")

    # Collect unique values for key fields
    if "species" in [f.lower() for f in field_names]:
        species_field = [f for f in field_names if f.lower() == "species"][0]
        layer.ResetReading()
        species_set = set()
        for f in layer:
            val = f.GetField(species_field)
            if val:
                species_set.add(val)
        print(f"\nUnique species codes ({len(species_set)}):")
        for sp in sorted(species_set):
            print(f"  {sp}")

    ds = None


def inspect_shapefile_fiona(shp_path):
    """Inspect shapefile using fiona."""
    with fiona.open(str(shp_path)) as src:
        print(f"\nFile: {shp_path.name}")
        print(f"Features: {len(src)}")
        print(f"Schema: {src.schema}")

        # Show first feature
        first = next(iter(src))
        print("\nFirst feature properties:")
        for k, v in first["properties"].items():
            print(f"  {k}: {v}")

        # Collect species if present
        if "species" in [k.lower() for k in first["properties"].keys()]:
            species_key = [
                k for k in first["properties"].keys() if k.lower() == "species"
            ][0]
            species_set = {
                feat["properties"][species_key]
                for feat in src
                if feat["properties"].get(species_key)
            }
            print(f"\nUnique species codes ({len(species_set)}):")
            for sp in sorted(species_set):
                print(f"  {sp}")


if __name__ == "__main__":
    shapefile_dir = Path("data/fortress/10.35097-538/data/dataset/shapefile")

    # Find a few sample shapefiles
    shapefiles = sorted(shapefile_dir.glob("poly_CFB*.shp"))[:3]

    if not shapefiles:
        print(f"No shapefiles found in {shapefile_dir}")
        sys.exit(1)

    print(f"Inspecting {len(shapefiles)} sample shapefiles...")

    for shp_path in shapefiles:
        try:
            if USE_OGR:
                inspect_shapefile_ogr(shp_path)
            else:
                inspect_shapefile_fiona(shp_path)
        except Exception as e:
            print(f"Error inspecting {shp_path.name}: {e}")
