#!/usr/bin/env python3
"""Inspect OAM-TCD dataset structure and visualize samples."""

import json
from pathlib import Path
from datasets import load_from_disk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def inspect_dataset(data_dir: str = "data/datasets/oam_tcd"):
    """Inspect OAM-TCD dataset structure and show examples."""
    data_path = Path(data_dir)

    print("=" * 80)
    print("OAM-TCD Dataset Inspection")
    print("=" * 80)

    # Load metadata
    metadata_path = data_path / "dataset_info.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            info = json.load(f)
        print("\nDataset Info:")
        print(f"  Name: {info['name']}")
        print(f"  Source: {info['source']}")
        print(f"  Description: {info['description']}")
        print(f"  Resolution: {info['resolution_cm']} cm/pixel")
        print(f"  License: {info['license']}")
        print(f"  Splits: {info['splits']}")

    # Load train split
    train_path = data_path / "train"
    if not train_path.exists():
        print(f"\nError: Train split not found at {train_path}")
        return

    print(f"\nLoading train split from {train_path}...")
    train_data = load_from_disk(str(train_path))

    print(f"\nTrain split loaded: {len(train_data)} samples")
    print(f"Features: {list(train_data.features.keys())}")

    # Inspect first sample
    print("\n" + "=" * 80)
    print("First Sample Inspection")
    print("=" * 80)

    sample = train_data[0]

    print("\nBasic Info:")
    print(f"  Image ID: {sample['image_id']}")
    print(f"  Dimensions: {sample['width']} x {sample['height']}")
    print(f"  License: {sample['license']}")
    print(f"  Biome: {sample['biome_name']} (code: {sample['biome']})")
    print(f"  Location: ({sample['lat']:.4f}, {sample['lon']:.4f})")
    print(f"  CRS: {sample['crs']}")
    print(f"  Bounds: {sample['bounds']}")

    # Parse COCO annotations
    print("\n\nCOCO Annotations:")
    coco_data = json.loads(sample["coco_annotations"])
    print(f"  Type: {type(coco_data)}")

    # Handle both list and dict formats
    if isinstance(coco_data, dict):
        print(f"  Keys: {list(coco_data.keys())}")
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])
    elif isinstance(coco_data, list):
        # coco_data is directly a list of annotations
        annotations = coco_data
        categories = []
        print("  Direct annotation list (no categories)")
    else:
        annotations = []
        categories = []

    if annotations:
        print(f"  Number of annotations: {len(annotations)}")

        # Count by category
        category_counts = {}
        for ann in annotations:
            cat_id = ann.get("category_id", 0)
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

        print("\n  Annotations by category:")
        for cat_id, count in sorted(category_counts.items()):
            # Find category name if categories exist
            if categories:
                cat_name = next(
                    (c["name"] for c in categories if c["id"] == cat_id), "unknown"
                )
            else:
                # Common tree dataset categories
                cat_names = {0: "background", 1: "tree", 2: "canopy"}
                cat_name = cat_names.get(cat_id, f"class_{cat_id}")
            print(f"    Category {cat_id} ({cat_name}): {count} instances")

        # Show first annotation
        if annotations:
            print("\n  First annotation example:")
            ann = annotations[0]
            print(f"    Keys: {list(ann.keys())}")
            print(f"    ID: {ann.get('id', 'N/A')}")
            print(f"    Category: {ann.get('category_id', 'N/A')}")
            print(f"    Area: {ann.get('area', 'N/A')}")
            print(f"    BBox: {ann.get('bbox', 'N/A')}")
            print(f"    Segmentation type: {type(ann.get('segmentation'))}")
            if "segmentation" in ann and ann["segmentation"]:
                seg = ann["segmentation"]
                if isinstance(seg, dict) and "counts" in seg:
                    print("      Format: RLE (Run-Length Encoding)")
                    print(f"      RLE size: {len(seg.get('counts', []))} bytes")
                elif isinstance(seg, list):
                    print(f"      Format: Polygon ({len(seg)} polygons)")
                    if seg and isinstance(seg[0], list):
                        print(f"      First polygon: {len(seg[0])} points")

    # Parse segments (alternative format)
    print("\n\nSegments field:")
    segments = json.loads(sample["segments"])
    print(f"  Type: {type(segments)}")
    print(f"  Keys: {list(segments.keys()) if isinstance(segments, dict) else 'N/A'}")

    # Parse meta
    print("\n\nMeta field:")
    try:
        if sample["meta"]:
            meta = json.loads(sample["meta"])
            if isinstance(meta, dict):
                print(f"  Keys: {list(meta.keys())}")
                for key, value in meta.items():
                    print(f"    {key}: {value}")
            else:
                print(f"  Type: {type(meta)}")
                print(f"  Value: {meta}")
        else:
            print("  Empty")
    except json.JSONDecodeError:
        print("  Invalid JSON or empty")
    except Exception as e:
        print(f"  Error parsing: {e}")

    return train_data, sample


def visualize_sample(sample, output_path: str = None):
    """Visualize a sample with annotations."""
    print("\n" + "=" * 80)
    print("Visualizing Sample")
    print("=" * 80)

    # Get image and annotation
    image = sample["image"]
    annotation = sample["annotation"]

    # Parse COCO annotations
    coco_data = json.loads(sample["coco_annotations"])

    # Handle both list and dict formats
    if isinstance(coco_data, list):
        annotations = coco_data
        categories = []
    else:
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])

    # Count categories
    category_counts = {}
    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\n{sample['width']}x{sample['height']} @ 10cm/px")
    axes[0].axis("off")

    # Annotation mask
    axes[1].imshow(annotation, cmap="tab20")
    axes[1].set_title(f"Annotation Mask\n{len(annotations)} instances")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(annotation, alpha=0.5, cmap="tab20")
    axes[2].set_title(f"Overlay\nBiome: {sample['biome_name']}")
    axes[2].axis("off")

    # Add legend for categories
    if categories:
        legend_patches = [
            mpatches.Patch(color=f"C{i}", label=cat["name"])
            for i, cat in enumerate(categories)
        ]
        axes[2].legend(handles=legend_patches, loc="upper right", fontsize=8)
    else:
        # Manual categories for tree dataset
        cat_names = {1: "tree", 2: "canopy"}
        legend_patches = [
            mpatches.Patch(
                color=f"C{cat_id}",
                label=f"{cat_names.get(cat_id, f'cat_{cat_id}')} ({count})",
            )
            for cat_id, count in sorted(category_counts.items())
        ]
        if legend_patches:
            axes[2].legend(handles=legend_patches, loc="upper right", fontsize=8)

    plt.suptitle(
        f"OAM-TCD Sample {sample['image_id']} - {sample['license']}", fontsize=12
    )
    plt.tight_layout()

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"\n✓ Saved visualization to {output_file}")
    else:
        plt.savefig(
            "data/datasets/oam_tcd/sample_visualization.png",
            dpi=150,
            bbox_inches="tight",
        )
        print(
            "\n✓ Saved visualization to data/datasets/oam_tcd/sample_visualization.png"
        )

    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect OAM-TCD dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/datasets/oam_tcd",
        help="Dataset directory",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Create visualization of first sample"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for visualization"
    )

    args = parser.parse_args()

    train_data, sample = inspect_dataset(args.data_dir)

    if args.visualize and train_data:
        visualize_sample(sample, args.output)
