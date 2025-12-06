"""Shortcut command for supervised evaluation."""

from pathlib import Path
from typing import Literal, Optional

import typer

from tree_seg.cli.evaluate import evaluate_command


def eval_super_command(
    dataset: Path = typer.Argument(
        ...,
        help="Path to dataset directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    head: Literal["linear", "sklearn", "mlp"] = typer.Option(
        "linear",
        "--head",
        "-h",
        help="Supervised head: linear (PyTorch), sklearn logistic, or sklearn MLP",
    ),
    model: Literal["small", "base", "large", "mega"] = typer.Option(
        "base",
        "--model",
        "-m",
        help="DINOv3 model size",
    ),
    stride: int = typer.Option(
        4,
        "--stride",
        "-s",
        help="Feature extraction stride",
    ),
    epochs: int = typer.Option(
        100,
        "--epochs",
        "-e",
        help="Epochs for the linear head (ignored for sklearn; early stopping applies)",
    ),
    max_patches: int = typer.Option(
        1_000_000,
        "--max-patches",
        "-p",
        help="Max patches for training (linear head)",
    ),
    lr: float = typer.Option(
        1e-3,
        "--lr",
        help="Learning rate for the linear head (ignored for sklearn)",
    ),
    val_split: float = typer.Option(
        0.1,
        "--val-split",
        help="Validation split for early stopping (linear head). 0 disables.",
    ),
    patience: int = typer.Option(
        5,
        "--patience",
        help="Patience for early stopping on val loss (linear head). 0 disables.",
    ),
    ignore_index: Optional[int] = typer.Option(
        None,
        "--ignore-index",
        help="Ignore index for supervised training/eval (set to 255 if masks use 255 as unlabeled; default=None keeps all labels)",
    ),
    hidden_dim: int = typer.Option(
        1024,
        "--hidden-dim",
        help="Hidden dimension for the linear head MLP",
    ),
    dropout: float = typer.Option(
        0.1,
        "--dropout",
        help="Dropout for the linear head MLP",
    ),
    mlp_use_xy: bool = typer.Option(
        False,
        "--mlp-use-xy",
        help="Append normalized XY coords to patch features for sklearn MLP head",
    ),
    use_xy: bool = typer.Option(
        False,
        "--use-xy",
        "-x",
        help="Append normalized XY coords to patch features for torch linear head",
    ),
    train_ratio: float = typer.Option(
        1.0,
        "--train-ratio",
        help="Fraction of samples to use for supervised training (rest used for holdout)",
    ),
    num_samples: Optional[int] = typer.Option(
        None,
        "--num-samples",
        "-n",
        help="Number of samples to evaluate (default: all)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress output",
    ),
):
    """
    Run supervised baseline (sklearn or linear head) with concise flags.
    """
    return evaluate_command(
        dataset=dataset,
        supervised=True,
        supervised_head=head,
        model=model,
        stride=stride,
        num_samples=num_samples,
        quiet=quiet,
        supervised_epochs=epochs,
        supervised_max_patches=max_patches,
        supervised_val_split=val_split,
        supervised_patience=patience,
        supervised_lr=lr,
        supervised_ignore_index=ignore_index,
        supervised_hidden_dim=hidden_dim,
        supervised_dropout=dropout,
        supervised_mlp_use_xy=mlp_use_xy,
        supervised_use_xy=use_xy,
        supervised_train_ratio=train_ratio,
    )
