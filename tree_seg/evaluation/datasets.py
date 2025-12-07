"""Dataset loaders for benchmark evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np
from PIL import Image


class SegmentationDataset(Protocol):
    """
    Protocol defining the interface for segmentation datasets.

    All dataset classes should implement this interface to work with
    the generic BenchmarkRunner.
    """

    NUM_CLASSES: int
    IGNORE_INDEX: int
    dataset_path: Path

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label, image_id)
            - image: RGB image array (H, W, 3) in range [0, 255]
            - label: Class index array (H, W) with class indices
            - image_id: Unique identifier for the image
        """
        ...


@dataclass
class DatasetSample:
    """Single sample from a dataset."""

    image_path: Path
    label_path: Path
    image_id: str


class FortressDataset:
    """
    FORTRESS (Forest Tree Species Segmentation) dataset loader.

    Source: Schiefer et al. 2020 (KIT)
    Contains tree species annotations as rasterized semantic masks.

    The dataset has been preprocessed using scripts/preprocess_fortress.py
    to convert vector polygons into pixel-level semantic masks.
    """

    # Class mapping based on species_mapping.txt from preprocessing
    # Note: species_ID values are not contiguous (gaps in numbering)
    CLASS_NAMES = {
        0: "Picea abies (Spruce)",
        2: "Acer pseudoplatanus (Maple)",
        4: "Fagus sylvatica (Beech)",
        5: "Fraxinus excelsior (Ash)",
        6: "Quercus spec. (Oak)",
        8: "Deadwood",
        9: "Forest floor",
        10: "Abies alba (Fir)",
        11: "Larix decidua (Larch)",
        12: "Picea abies (duplicate - verify)",  # Appears twice in mapping
        13: "Pinus sylvestris (Pine)",
        14: "Pseudotsuga menziesii (Douglas Fir)",
        15: "Betula pendula (Birch)",
    }

    NUM_CLASSES = 13  # Unique classes (excluding duplicate ID 12)
    IGNORE_INDEX = 255

    def __init__(self, dataset_path: Path, split: Optional[str] = None):
        """
        Initialize FORTRESS dataset.

        Args:
            dataset_path: Path to preprocessed FORTRESS dataset
                         (e.g., data/fortress_processed/)
            split: Optional split name (not used, kept for API consistency)
        """
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images"
        self.labels_path = self.dataset_path / "labels"

        if not self.images_path.exists():
            raise ValueError(f"Images directory not found: {self.images_path}")
        if not self.labels_path.exists():
            raise ValueError(f"Labels directory not found: {self.labels_path}")

        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.dataset_path}")

    def _find_samples(self) -> List[DatasetSample]:
        """Find all image-label pairs in the preprocessed dataset."""
        samples = []

        # Find all images (symbolic links to orthomosaics)
        image_files = sorted(self.images_path.glob("*.tif")) + sorted(
            self.images_path.glob("*.tiff")
        )

        for image_path in image_files:
            # Corresponding label has _label suffix
            stem = image_path.stem
            label_path = self.labels_path / f"{stem}_label.tif"

            if not label_path.exists():
                print(f"Warning: No label found for {image_path.name}")
                continue

            samples.append(
                DatasetSample(
                    image_path=image_path, label_path=label_path, image_id=stem
                )
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label, image_id)
            - image: RGB image array (H, W, 3) in range [0, 255]
            - label: Class index array (H, W) with species_ID values
            - image_id: Unique identifier for the image
        """
        sample = self.samples[idx]

        # Load image (RGB orthomosaic)
        image = np.array(Image.open(sample.image_path).convert("RGB"))

        # Load label (already rasterized as uint8 with species_ID values)
        label = np.array(Image.open(sample.label_path))

        # Ensure single channel
        if len(label.shape) > 2:
            label = label[:, :, 0]

        return image, label.astype(np.int64), sample.image_id

    def get_sample_paths(self, idx: int) -> DatasetSample:
        """Get file paths for a sample."""
        return self.samples[idx]


class ISPRSPotsdamDataset:
    """
    ISPRS Potsdam dataset loader.

    The dataset contains aerial imagery with semantic segmentation labels.
    Labels are RGB-encoded and need to be converted to class indices.
    """

    # RGB to class index mapping
    LABEL_COLORS = {
        (255, 255, 255): 0,  # Impervious surfaces
        (0, 0, 255): 1,  # Building
        (0, 255, 255): 2,  # Low vegetation
        (0, 255, 0): 3,  # Tree
        (255, 255, 0): 4,  # Car
        (255, 0, 0): 5,  # Clutter/background
    }

    CLASS_NAMES = {
        0: "Impervious surfaces",
        1: "Building",
        2: "Low vegetation",
        3: "Tree",
        4: "Car",
        5: "Clutter",
    }

    NUM_CLASSES = 6
    IGNORE_INDEX = -1  # For unlabeled pixels (0, 0, 0)

    def __init__(self, dataset_path: Path, split: Optional[str] = None):
        """
        Initialize dataset.

        Args:
            dataset_path: Path to dataset root (containing images/ and labels/)
            split: Optional split name (not used for ISPRS, kept for API consistency)
        """
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images"
        self.labels_path = self.dataset_path / "labels"

        # Validate paths
        if not self.images_path.exists():
            raise ValueError(f"Images directory not found: {self.images_path}")
        if not self.labels_path.exists():
            raise ValueError(f"Labels directory not found: {self.labels_path}")

        # Find all image files
        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.dataset_path}")

    def _find_samples(self) -> List[DatasetSample]:
        """Find all image-label pairs in the dataset."""
        samples = []

        # Find all RGB image files
        image_files = sorted(self.images_path.glob("*.tif")) + sorted(
            self.images_path.glob("*.tiff")
        )
        image_files += sorted(self.images_path.glob("*.png")) + sorted(
            self.images_path.glob("*.jpg")
        )

        for image_path in image_files:
            # Infer label path
            image_stem = image_path.stem

            # Try different naming conventions
            label_candidates = []

            # Convention 1: ISPRS official (top_potsdam_2_10_RGB.tif -> top_potsdam_2_10_label.tif)
            if "_RGB" in image_stem or "_IRRG" in image_stem:
                label_candidates.append(
                    image_stem.replace("_RGB", "_label").replace("_IRRG", "_label")
                )

            # Convention 2: Image_N -> Label_N (Kaggle dataset)
            if image_stem.startswith("Image_"):
                label_candidates.append(image_stem.replace("Image_", "Label_"))

            # Convention 3: Just add _label suffix
            label_candidates.append(f"{image_stem}_label")

            # Try all candidates with different extensions
            label_path = None
            for label_stem in label_candidates:
                for ext in [".tif", ".tiff", ".png"]:
                    potential_path = self.labels_path / f"{label_stem}{ext}"
                    if potential_path.exists():
                        label_path = potential_path
                        break
                if label_path:
                    break

            if label_path is None:
                print(f"Warning: No label found for {image_path.name}, skipping")
                continue

            samples.append(
                DatasetSample(
                    image_path=image_path, label_path=label_path, image_id=image_stem
                )
            )

        return samples

    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label, image_id)
            - image: RGB image array (H, W, 3) in range [0, 255]
            - label: Class index array (H, W) with values 0 to NUM_CLASSES-1
            - image_id: Unique identifier for the image
        """
        sample = self.samples[idx]

        # Load image
        image = np.array(Image.open(sample.image_path).convert("RGB"))

        # Load and convert label
        label_rgb = np.array(Image.open(sample.label_path).convert("RGB"))
        label = self._rgb_to_class_indices(label_rgb)

        return image, label, sample.image_id

    def _rgb_to_class_indices(self, label_rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB-encoded labels to class indices.

        Args:
            label_rgb: RGB label array (H, W, 3)

        Returns:
            Class index array (H, W) with values 0 to NUM_CLASSES-1
            Unlabeled pixels are set to IGNORE_INDEX
        """
        h, w = label_rgb.shape[:2]
        label_indices = np.full((h, w), self.IGNORE_INDEX, dtype=np.int32)

        # Convert each color to class index
        for rgb, class_idx in self.LABEL_COLORS.items():
            mask = np.all(label_rgb == rgb, axis=-1)
            label_indices[mask] = class_idx

        return label_indices

    def get_sample_paths(self, idx: int) -> DatasetSample:
        """Get file paths for a sample."""
        return self.samples[idx]


def load_isprs_potsdam(dataset_path: Path) -> ISPRSPotsdamDataset:
    """
    Convenience function to load ISPRS Potsdam dataset.

    Args:
        dataset_path: Path to dataset root

    Returns:
        ISPRSPotsdamDataset instance
    """
    return ISPRSPotsdamDataset(dataset_path)
