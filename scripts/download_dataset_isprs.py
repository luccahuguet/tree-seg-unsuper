#!/usr/bin/env python3
"""
Download and prepare ISPRS Potsdam dataset from Kaggle.

This script:
1. Downloads the dataset from Kaggle
2. Extracts the archives
3. Organizes files into images/ and labels/ directories
4. Cleans up temporary files

Requirements:
- Kaggle API credentials in ~/.kaggle/kaggle.json
- Get your API token from: https://www.kaggle.com/settings (click "Create New Token")
"""

import argparse
import shutil
import zipfile
from pathlib import Path
import subprocess
import sys


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_config.exists():
        print("❌ Kaggle API credentials not found!")
        print("\nTo set up Kaggle API:")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token' (downloads kaggle.json)")
        print("4. Run these commands:")
        print("   mkdir -p ~/.kaggle")
        print("   mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("   chmod 600 ~/.kaggle/kaggle.json")
        print("\nThen run this script again.\n")
        return False

    print("✓ Kaggle API credentials found")
    return True


def download_dataset(dataset_id: str, download_path: Path):
    """Download dataset from Kaggle."""
    print(f"\n📥 Downloading dataset: {dataset_id}")
    print(f"   Destination: {download_path}")

    download_path.mkdir(parents=True, exist_ok=True)

    try:
        # Use subprocess to call kaggle CLI
        result = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_id,
                "-p",
                str(download_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        print("✓ Download complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Kaggle CLI not found. This shouldn't happen since we installed it.")
        print("   Try running: uv add kaggle")
        return False


def extract_dataset(download_path: Path):
    """Extract all zip files in the download directory."""
    print("\n📦 Extracting archives...")

    zip_files = list(download_path.glob("*.zip"))

    if not zip_files:
        print("⚠ No zip files found to extract")
        return False

    for zip_path in zip_files:
        print(f"   Extracting: {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(download_path)
            print(f"   ✓ Extracted {zip_path.name}")
        except Exception as e:
            print(f"   ❌ Failed to extract {zip_path.name}: {e}")
            return False

    print("✓ Extraction complete")
    return True


def organize_files(download_path: Path):
    """
    Organize files into images/ and labels/ directories.

    ISPRS Potsdam naming convention:
    - Images: top_potsdam_X_Y_RGB.tif (or similar)
    - Labels: top_potsdam_X_Y_label.tif (or *_label_noBoundary.tif)
    """
    print("\n📁 Organizing files...")

    images_dir = download_path / "images"
    labels_dir = download_path / "labels"

    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Find all files in download directory (and subdirectories)
    all_files = []
    for pattern in ["**/*.tif", "**/*.tiff", "**/*.png", "**/*.jpg"]:
        all_files.extend(download_path.glob(pattern))

    # Filter out files already in images/ or labels/
    all_files = [
        f for f in all_files if "images" not in f.parts and "labels" not in f.parts
    ]

    if not all_files:
        print("⚠ No image files found to organize")
        return False

    image_count = 0
    label_count = 0

    for file_path in all_files:
        filename = file_path.name.lower()

        # Determine if this is a label or image
        if "label" in filename:
            # This is a label file
            dest = labels_dir / file_path.name
            label_count += 1
        elif "rgb" in filename or "irrg" in filename or "dsm" in filename:
            # This is an image file
            # Skip DSM (Digital Surface Model) files if they exist
            if "dsm" in filename:
                continue
            dest = images_dir / file_path.name
            image_count += 1
        else:
            # Unknown file type, skip
            continue

        # Move or copy the file
        if not dest.exists():
            shutil.copy2(file_path, dest)
            print(f"   ✓ {file_path.name} -> {dest.parent.name}/")

    print(f"\n✓ Organized {image_count} images and {label_count} labels")

    return image_count > 0 and label_count > 0


def cleanup_temp_files(download_path: Path, keep_zips: bool = False):
    """Remove temporary files and directories."""
    print("\n🧹 Cleaning up temporary files...")

    # Remove zip files unless requested to keep
    if not keep_zips:
        for zip_path in download_path.glob("*.zip"):
            print(f"   Removing: {zip_path.name}")
            zip_path.unlink()

    # Remove common extracted directory structures
    common_dirs = ["ISPRS_Potsdam", "potsdam", "Potsdam", "data"]
    for dir_name in common_dirs:
        dir_path = download_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"   Removing directory: {dir_name}/")
            shutil.rmtree(dir_path)

    print("✓ Cleanup complete")


def verify_dataset(dataset_path: Path):
    """Verify the dataset is properly organized."""
    print("\n✅ Verifying dataset structure...")

    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.exists():
        print("❌ images/ directory not found")
        return False

    if not labels_dir.exists():
        print("❌ labels/ directory not found")
        return False

    image_files = list(images_dir.glob("*.tif")) + list(images_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.tif")) + list(labels_dir.glob("*.png"))

    print(f"   Images: {len(image_files)} files")
    print(f"   Labels: {len(label_files)} files")

    if len(image_files) == 0:
        print("❌ No image files found")
        return False

    if len(label_files) == 0:
        print("❌ No label files found")
        return False

    print("\n✓ Dataset is ready!")
    print(f"   Location: {dataset_path.absolute()}")
    print("\nNext steps:")
    print("   1. Run tests: uv run python scripts/test_benchmark.py")
    print(
        "   2. Run benchmark: uv run python scripts/run_benchmark.py --dataset data/datasets/isprs_potsdam --num-samples 5"
    )

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and prepare ISPRS Potsdam dataset from Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset-id",
        type=str,
        default="jahidhasan66/isprs-potsdam",
        help="Kaggle dataset ID (default: jahidhasan66/isprs-potsdam)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets/isprs_potsdam"),
        help="Output directory (default: data/datasets/isprs_potsdam)",
    )

    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep downloaded zip files after extraction",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (useful if already downloaded)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ISPRS Potsdam Dataset Downloader")
    print("=" * 60)

    # Step 1: Check Kaggle credentials
    if not args.skip_download and not check_kaggle_credentials():
        return 1

    # Step 2: Download dataset
    if not args.skip_download:
        if not download_dataset(args.dataset_id, args.output_dir):
            return 1
    else:
        print("\n⏭️  Skipping download step")

    # Step 3: Extract archives
    if not extract_dataset(args.output_dir):
        return 1

    # Step 4: Organize files
    if not organize_files(args.output_dir):
        print("\n⚠ File organization may have failed. Check the directory structure.")
        print(f"   Location: {args.output_dir.absolute()}")
        return 1

    # Step 5: Cleanup
    cleanup_temp_files(args.output_dir, keep_zips=args.keep_zips)

    # Step 6: Verify
    if not verify_dataset(args.output_dir):
        print(
            "\n⚠ Dataset verification failed. You may need to manually organize the files."
        )
        print(f"   See: {args.output_dir / 'README.md'}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
