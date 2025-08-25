# Model Weights Directory

This directory can be used to store local model weights for faster loading.

## Current Caching Strategy

The system uses HuggingFace's built-in cache at `~/.cache/huggingface/hub/` which automatically:
- Caches downloaded models locally
- Reuses cached files for subsequent runs
- Handles version management and updates

## Manual Weight Storage

You can also manually download and store weights here for:
- Offline usage
- Custom model variants
- Faster access in production environments

## Usage

The DINOv3 adapter automatically:
1. Tries local HuggingFace cache first (`local_files_only=True`)
2. Downloads from HuggingFace if not cached
3. Falls back to random weights if download fails

This ensures minimal network usage and fast startup times.