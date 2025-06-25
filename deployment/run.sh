#!/bin/bash
# Cloud GPU Run Script

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running tree segmentation..."
python tree_seg_local.py

echo "Done! Check the output directory for results."
