"""Main CLI entry point for tree-seg."""

import typer
from rich.console import Console

from tree_seg.cli.evaluate import evaluate_command
from tree_seg.cli.eval_super import eval_super_command
from tree_seg.cli.results import results_command
from tree_seg.cli.segment import segment_command

app = typer.Typer(
    name="tree-seg",
    help="Tree segmentation toolkit using DINOv3 for aerial imagery",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.callback()
def main():
    """Tree segmentation using DINOv3 self-supervised features."""
    pass


# Register commands
app.command(name="segment", help="Segment trees in aerial imagery")(segment_command)
app.command(name="eval", help="Evaluate segmentation methods on labeled datasets")(evaluate_command)
app.command(name="eval-super", help="Evaluate supervised baseline (shortcut)")(eval_super_command)
app.command(name="results", help="Query stored experiment metadata")(results_command)


def cli_main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli_main()
