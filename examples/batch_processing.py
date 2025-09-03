# %%
# Batch Processing Example
# Process multiple images automatically

import sys
import os
from collections import defaultdict
sys.path.append("/kaggle/working/project")

from tree_seg import TreeSegmentation, Config

# %%
# Setup configuration for batch processing
config = Config(
    input_dir="/kaggle/input/drone-10-best",
    output_dir="/kaggle/working/output",
    model_name="base",
    auto_k=True,
    elbow_threshold=5.0,
    web_optimize=True  # Auto-optimize for web
)

segmenter = TreeSegmentation(config)

# %%
# Process all images in directory
print("📁 Batch processing all images:")

batch_results = segmenter.process_directory()

print(f"✅ Processed {len(batch_results)} images")
for result, paths in batch_results:
    filename = os.path.basename(result.image_path)
    print(f"  • {filename}: K={result.n_clusters_used}")

# %%
# Show summary of all processed files
all_outputs = segmenter.output_manager.list_all_outputs()
print(f"\n📊 Total output files generated: {len(all_outputs)}")

# Group by image hash for summary
by_image = defaultdict(list)

for file_path in all_outputs:
    filename = os.path.basename(file_path)
    # Extract hash (first part before underscore)
    hash_part = filename.split('_')[0]
    by_image[hash_part].append(filename)

print(f"📸 Processed {len(by_image)} unique images:")
for image_hash, files in by_image.items():
    print(f"  • {image_hash}: {len(files)} files")
