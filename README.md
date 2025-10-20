# Tree Segmentation with DINOv3

Modern unsupervised tree segmentation for aerial imagery powered by DINOv3 Vision Transformers and configuration-driven workflows.

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
