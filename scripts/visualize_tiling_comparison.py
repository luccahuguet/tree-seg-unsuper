#!/usr/bin/env python3
"""
Generate side-by-side comparison of tiling vs no-tiling results.
"""
import matplotlib.pyplot as plt

# Results from evaluation
results = {
    "No Tiling (1024×1024)": {
        "base_s4": {"miou": 0.090, "pixel_acc": 0.389, "time": 55, "k": 5},
    },
    "With Tiling (9372×9372)": {
        "base_s4": {"miou": 0.089, "pixel_acc": 0.415, "time": 209, "k": 4},
    }
}

# Create comparison figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Tiling vs No Tiling Comparison (FORTRESS CFB003)", fontsize=16, fontweight='bold')

configs = ["No Tiling (1024×1024)", "With Tiling (9372×9372)"]
colors = ['#FF6B6B', '#4ECDC4']

for idx, (config_name, ax) in enumerate(zip(configs, axes)):
    data = results[config_name]["base_s4"]

    # Bar chart of metrics
    metrics = ['mIoU\n(%)', 'Pixel Acc\n(%)', 'Time\n(s)', 'K']
    values = [data['miou'] * 100, data['pixel_acc'] * 100, data['time'], data['k']]

    bars = ax.bar(metrics, values, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_title(config_name, fontsize=14, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(values) * 1.2)

plt.tight_layout()

# Add summary text
summary_text = """
Key Findings:
• Pixel Accuracy: 38.9% → 41.5% (+6.7% improvement)
• Processing Time: 55s → 209s (4× slower)
• Clusters: K=5 → K=4 (tiling selects fewer clusters)
• Trade-off: Better spatial detail at cost of compute time
"""

fig.text(0.5, -0.05, summary_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig('data/output/tiling_comparison_metrics.png', dpi=150, bbox_inches='tight')
print("✅ Saved comparison chart to: data/output/tiling_comparison_metrics.png")
plt.close()
