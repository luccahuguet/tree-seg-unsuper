#!/usr/bin/env python3
"""
Simple 2-Panel V3 Visualization

Creates a clean visualization similar to V1.5:
- Left: Original image
- Right: V3 segmentation (vegetation clusters + filtered background as cluster 0)

Includes config legend showing settings used.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from tree_seg import TreeSegmentation, Config
from tree_seg.constants import HATCH_PATTERNS


def visualize_v3_simple(
    image_path: str,
    exg_threshold: float = 0.10,
    overlay_style: str = "hatching",  # "hatching", "color", or "both"
    auto_k: bool = True,
    elbow_threshold: float = 5.0,
    n_clusters: int = 20,
    output_dir: str = "data/output/v3_simple"
):
    """
    Create simple 2-panel V3 visualization.

    Args:
        image_path: Path to input image
        exg_threshold: ExG threshold for vegetation filtering
        overlay_style: Overlay style - "hatching", "color", or "both"
        auto_k: Use elbow method for automatic K selection
        elbow_threshold: Elbow threshold (if auto_k=True)
        n_clusters: Fixed K value (if auto_k=False)
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem

    print("=" * 80)
    print(f"V3 Simple Visualization: {image_name}")
    print("=" * 80)
    print()

    # Load image
    image_np = cv2.imread(image_path)
    if image_np is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Run V3
    print(f"Running V3 (ExG threshold={exg_threshold})...")
    config = Config(
        pipeline="v3",
        auto_k=auto_k,
        elbow_threshold=elbow_threshold,
        n_clusters=n_clusters,
        v3_exg_threshold=exg_threshold,
        verbose=True
    )

    seg = TreeSegmentation(config)
    results = seg.process_single_image(image_path)

    print(f"  V3: {results.n_clusters_used} vegetation clusters (+ background)")
    print()

    # Create 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')

    # Right: V3 clusters overlay
    axes[1].imshow(image_np)

    # Color palette (bright colors)
    def get_cluster_color(cluster_id, n_clusters):
        bright_colors = [
            (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (1.0, 0.0, 0.5),
            (0.0, 1.0, 0.0), (1.0, 0.3, 0.7)
        ]

        if cluster_id < len(bright_colors):
            return bright_colors[cluster_id]
        else:
            # Generate color from viridis for clusters beyond bright palette
            cmap = plt.get_cmap("viridis")
            return cmap((cluster_id - len(bright_colors)) / max(1, n_clusters - len(bright_colors)))[:3]

    # Draw clusters (0 = filtered background, 1-N = vegetation clusters)
    legend_elements = []

    for cluster_id in range(results.n_clusters_used + 1):  # +1 for background (cluster 0)
        cluster_mask = (results.labels_resized == cluster_id)

        if not cluster_mask.any():
            continue

        if cluster_id == 0:
            # Background (filtered regions) - gray color
            cluster_color = (0.5, 0.5, 0.5)
            cluster_label = "Filtered (non-vegetation)"
        else:
            # Vegetation clusters - bright colors
            cluster_color = get_cluster_color(cluster_id - 1, results.n_clusters_used)
            cluster_label = f"Veg Cluster {cluster_id}"

        # Determine hatching pattern
        if cluster_id == 0:
            # Filtered background gets distinct hatching
            hatch_pattern = 'xxx'
        else:
            hatch_pattern = HATCH_PATTERNS[(cluster_id - 1) % len(HATCH_PATTERNS)]

        # Apply overlay based on style
        if overlay_style in ["hatching", "both"]:
            # Draw borders
            axes[1].contour(cluster_mask.astype(int), levels=[0.5], colors=[cluster_color],
                           linewidths=2.5, alpha=0.9)
            axes[1].contour(cluster_mask.astype(int), levels=[0.5], colors=[cluster_color],
                           linewidths=3.5, alpha=0.6)
            axes[1].contour(cluster_mask.astype(int), levels=[0.5], colors='white',
                           linewidths=1, alpha=0.3)

            # Add hatching
            cs = axes[1].contourf(cluster_mask.astype(int), levels=[0.5, 1.5], colors='none',
                                 hatches=[hatch_pattern])

            # Handle both old and new matplotlib API
            collections = getattr(cs, 'collections', [])
            for collection in collections:
                collection.set_facecolor('none')
                collection.set_edgecolor(cluster_color)
                collection.set_alpha(0.7)
                collection.set_linewidth(0.)

            # Legend with hatch pattern
            legend_elements.append(
                mpatches.Patch(facecolor='none', edgecolor=cluster_color,
                              hatch=hatch_pattern, label=f'{cluster_label} {hatch_pattern}')
            )

        elif overlay_style == "color":
            # Solid color fill
            axes[1].contourf(cluster_mask.astype(int), levels=[0.5, 1.5],
                            colors=[cluster_color], alpha=0.5)

            # Draw borders
            axes[1].contour(cluster_mask.astype(int), levels=[0.5], colors=[cluster_color],
                           linewidths=2, alpha=0.9)

            # Legend without hatch
            legend_elements.append(
                mpatches.Patch(facecolor=cluster_color, edgecolor=cluster_color,
                              alpha=0.5, label=cluster_label)
            )

    axes[1].set_title(f"V3 Species Clustering ({results.n_clusters_used} vegetation clusters)",
                     fontsize=14)
    axes[1].axis('off')

    # Add legend
    legend = axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8,
                           framealpha=0.9, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')

    # Calculate filtering statistics
    background_mask = results.labels_resized == 0
    veg_mask = results.labels_resized > 0
    total_pixels = results.labels_resized.size
    background_pixels = background_mask.sum()
    veg_pixels = veg_mask.sum()
    filtered_pct = (background_pixels / total_pixels) * 100

    # Add config text overlay
    if auto_k:
        config_text = (
            f"V3 Species Clustering\n"
            f"Image: {image_name}\n"
            f"Auto K (elbow={elbow_threshold}) → {results.n_clusters_used} veg clusters\n"
            f"ExG threshold: {exg_threshold}\n"
            f"Model: {config.model_display_name}\n"
            f"Stride: {config.stride}"
        )
    else:
        config_text = (
            f"V3 Species Clustering\n"
            f"Image: {image_name}\n"
            f"K={n_clusters} → {results.n_clusters_used} veg clusters\n"
            f"ExG threshold: {exg_threshold}\n"
            f"Model: {config.model_display_name}\n"
            f"Stride: {config.stride}"
        )

    axes[1].text(
        0.02, 0.98, config_text,
        transform=axes[1].transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.5')
    )

    # Add filtering statistics annotation (bottom left)
    if background_pixels > 0:
        # Show filtered regions info
        filter_text = (
            f"Filtered Regions (gray/hatched):\n"
            f"  {filtered_pct:.1f}% of image removed\n"
            f"  ({background_pixels:,} px)"
        )
        text_color = 'darkred'
    else:
        # No filtering occurred
        filter_text = (
            "No Filtering:\n"
            "  All regions are vegetation\n"
            "  (0% removed)"
        )
        text_color = 'darkgreen'

    axes[1].text(
        0.02, 0.02, filter_text,
        transform=axes[1].transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='left',
        color=text_color, weight='bold',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor=text_color, boxstyle='round,pad=0.5')
    )

    plt.tight_layout()

    # Save
    save_path = output_path / f"{image_name}_v3_simple.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization to: {save_path}")
    print()

    # Print statistics
    background_mask = results.labels_resized == 0
    veg_mask = results.labels_resized > 0

    total_pixels = results.labels_resized.size
    background_pixels = background_mask.sum()
    veg_pixels = veg_mask.sum()

    print("Summary:")
    print(f"  Image: {image_name}")
    print(f"  Config: {'Auto K' if auto_k else f'K={n_clusters}'}")
    if auto_k:
        print(f"  Elbow threshold: {elbow_threshold}")
    print(f"  ExG threshold: {exg_threshold}")
    print()
    print(f"  Vegetation clusters: {results.n_clusters_used}")
    print(f"  Vegetation pixels: {veg_pixels:,} ({100*veg_pixels/total_pixels:.1f}%)")
    print(f"  Filtered pixels: {background_pixels:,} ({100*background_pixels/total_pixels:.1f}%)")
    print()


if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="Simple 2-panel V3 visualization")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image or OAM-TCD image ID (e.g., '3828')")
    parser.add_argument("--threshold", type=float, default=0.10,
                       help="ExG threshold for vegetation")
    parser.add_argument("--style", type=str, default="hatching",
                       choices=["hatching", "color", "both"],
                       help="Overlay style: hatching (default), color, or both")
    parser.add_argument("--auto-k", action="store_true", default=True,
                       help="Use elbow method (default)")
    parser.add_argument("--elbow", type=float, default=5.0,
                       help="Elbow threshold (default: 5.0)")
    parser.add_argument("--k", type=int, default=20,
                       help="Fixed K value (only if --no-auto-k)")
    parser.add_argument("--no-auto-k", action="store_true",
                       help="Use fixed K instead of auto K")
    parser.add_argument("--output", type=str, default="data/output/v3_simple",
                       help="Output directory")

    args = parser.parse_args()

    # Handle OAM-TCD image ID shorthand
    image_path = args.image
    temp_file = None

    if args.image.isdigit():
        # Assume it's an OAM-TCD image ID - load from dataset
        try:
            from datasets import load_from_disk

            print(f"Loading OAM-TCD image {args.image}...")
            dataset = load_from_disk("data/oam_tcd/test")

            # Find image by ID
            image_id = int(args.image)
            sample = None
            for item in dataset:
                if item['image_id'] == image_id:
                    sample = item
                    break

            if sample is None:
                print(f"Error: Image ID {image_id} not found in OAM-TCD test set")
                exit(1)

            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            sample['image'].save(temp_file.name)
            image_path = temp_file.name
            print(f"Loaded OAM-TCD image {image_id}")

        except Exception as e:
            print(f"Error loading OAM-TCD dataset: {e}")
            print(f"Treating '{args.image}' as a file path instead")

    visualize_v3_simple(
        image_path=image_path,
        exg_threshold=args.threshold,
        overlay_style=args.style,
        auto_k=not args.no_auto_k,
        elbow_threshold=args.elbow,
        n_clusters=args.k,
        output_dir=args.output
    )

    # Cleanup temp file
    if temp_file:
        import os
        os.unlink(temp_file.name)
