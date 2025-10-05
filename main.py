#!/usr/bin/env python3
"""Convenience entry point for the tree segmentation CLI.

Running ``uv run python main.py`` (or ``python main.py`` directly) executes the
standard segmentation pipeline with the same defaults exposed by
``scripts/run_segmentation.py``.
"""

from scripts.run_segmentation import main as run_segmentation_cli


if __name__ == "__main__":
    run_segmentation_cli()
