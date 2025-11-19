# Tree Species Segmentation with DINOv3

Unsupervised species-level semantic segmentation for aerial vegetation imagery powered by DINOv3 Vision Transformers.

## Goal
Segment aerial imagery into **species-level regions** where visually similar vegetation (same species/type) is grouped together:
- Separate vegetation from non-vegetation (soil, roads, buildings)
- Distinguish different tree/vegetation species based on visual features (texture, color, pattern)
- Output semantic maps where each region represents a distinct species/vegetation type

**Not instance segmentation**: We don't separate individual tree crowns. Instead, we cluster regions by species (e.g., "this area is pines", "this area is firs").

## Quick Start
- Install dependencies: `uv sync`
- Run the CLI: `UV_CACHE_DIR=.uv_cache uv run python main.py --help`
- Default inputs live in `data/input/`; results are written to `data/output/`

For full setup instructions, walkthroughs, and benchmarks, jump to the dedicated docs below.

## Documentation Map
- Project overview & quick-start guide – `docs/index.md`
- Methodology & pipeline details – `docs/methodology.md`
- Complete end-to-end example – `docs/complete_example.md`
- Parameter sweeps & analysis – `docs/parameter_analysis.md`
- CLI usage reference – `docs/text/cli_usage.md`
- Dataset benchmarks & evaluation plans – `docs/text/benchmarking.md`
- Technical deep dive – `docs/technical_implementation.md`
- Architecture notes – `docs/text/ARCHITECTURE.md`
- Week 1 research report – `docs/text/week1_results.md`
- Kaggle-specific setup & download script guide – `docs/text/benchmarking.md#1-kaggle-access--dataset-download`
- Research planning notes – `docs/text/ideation.md`, `docs/text/paper_timeline.md`
- Documentation sweep instructions – `examples/sweeps/README.md`

## Data Layout
- `data/input/` – source imagery used by the CLI and examples (ignored from version control)
- `data/output/` – generated artifacts, benchmarks, and documentation assets

## Reproducible Workflows
- CLI usage & automation scripts live under `scripts/`
- Notebook-style examples and sweeps reside in `examples/`
- Model weights and adapters are stored under `weights/` and `tree_seg/`

See the linked docs for detailed commands, configuration options, and research findings.
