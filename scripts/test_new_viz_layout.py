#!/usr/bin/env python3
"""Test new 2×2 visualization layout."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage import segmentation

# Load existing visualization to get data
viz_path = Path("data/output/results/viz_tile/visualizations/CFB003_comparison.png")

# For now, create a simple test
# This would be integrated into the actual plotting function

print("Testing new 2×2 layout...")
print("Creating mock visualization...")

# Create figure with 2×2 grid + legend row
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[10, 10, 2], hspace=0.25, wspace=0.1)

# Create axes
ax_original = fig.add_subplot(gs[0, 0])
ax_gt = fig.add_subplot(gs[0, 1])
ax_pred = fig.add_subplot(gs[1, 0])
ax_overlay = fig.add_subplot(gs[1, 1])
ax_config = fig.add_subplot(gs[2, 0])
ax_classes = fig.add_subplot(gs[2, 1])

# Mock data for testing layout
mock_img = np.random.rand(100, 100, 3)
mock_labels = np.random.randint(0, 6, (100, 100))

# Plot images
ax_original.imshow(mock_img)
ax_original.set_title("Original Image", fontsize=12, fontweight='bold')
ax_original.axis('off')

ax_gt.imshow(mock_labels, cmap='tab20')
ax_gt.set_title("Ground Truth", fontsize=12, fontweight='bold')
ax_gt.axis('off')

ax_pred.imshow(mock_labels, cmap='tab20')
ax_pred.set_title("Prediction (K=4)", fontsize=12, fontweight='bold')
ax_pred.axis('off')

# Edge overlay
boundaries = segmentation.find_boundaries(mock_labels, mode='thick')
overlay = mock_img.copy()
overlay[boundaries] = [1, 0, 0]
ax_overlay.imshow(overlay)
ax_overlay.set_title("Prediction Overlay", fontsize=12, fontweight='bold')
ax_overlay.axis('off')

# Config card
config_text = """Configuration:
• Model: dinov3_vitb16
• Method: v1.5  |  Stride: 4  |  Tiling: Yes
• Refine: slic
• K: 4 (Auto=True)  |  GT: 6 classes"""

ax_config.text(0.5, 0.5, config_text,
               ha='center', va='center', fontsize=10,
               bbox=dict(facecolor='wheat', alpha=0.9, edgecolor='black',
                        boxstyle='round,pad=1', linewidth=2))
ax_config.axis('off')

# Class legend
legend_patches = []
for i in range(6):
    patch = mpatches.Patch(color=plt.cm.tab20(i / 20), label=f"{i}: Class {i}")
    legend_patches.append(patch)

legend = ax_classes.legend(handles=legend_patches, loc='center',
                           ncol=2, fontsize=9, frameon=True,
                           fancybox=True, shadow=True,
                           title="Ground Truth Classes",
                           title_fontsize=10)
legend.get_frame().set_facecolor('lightgray')
legend.get_frame().set_alpha(0.9)
ax_classes.axis('off')

# Main title
fig.suptitle("CFB003  |  mIoU: 0.089  |  Pixel Accuracy: 0.415",
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('data/output/test_new_layout.png', dpi=120, bbox_inches='tight', facecolor='white')
print("✅ Test layout saved to: data/output/test_new_layout.png")
plt.close()
