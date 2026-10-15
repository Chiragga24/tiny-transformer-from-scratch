from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Keep Matplotlib's generated font cache with the project. This avoids relying
# on a writable user-profile directory in terminals, containers, or CI.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot losses and learning rate from metrics.jsonl.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoothing-window", type=int, default=50)
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    # Padding preserves one plotted value per training step and avoids shifting
    # the curve horizontally by half the smoothing window.
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    args = parse_args()
    if args.smoothing_window <= 0:
        raise ValueError("smoothing window must be positive")

    with args.metrics.open(encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    if not records:
        raise ValueError(f"no metrics found in {args.metrics}")

    steps = np.asarray([record["step"] for record in records])
    train_losses = np.asarray([record["train_loss"] for record in records])
    learning_rates = np.asarray([record["learning_rate"] for record in records])
    validation = [
        (record["step"], record["validation_loss"])
        for record in records
        if record["validation_loss"] is not None
    ]

    figure, loss_axis = plt.subplots(figsize=(10, 6))
    loss_axis.plot(steps, train_losses, color="lightsteelblue", alpha=0.45, label="training loss")
    loss_axis.plot(
        steps,
        moving_average(train_losses, args.smoothing_window),
        color="royalblue",
        linewidth=2,
        label=f"training loss ({args.smoothing_window}-step mean)",
    )
    if validation:
        validation_steps, validation_losses = zip(*validation)
        loss_axis.plot(
            validation_steps,
            validation_losses,
            color="darkorange",
            marker="o",
            linewidth=2,
            label="validation loss",
        )
    loss_axis.set_xlabel("Training step")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.grid(alpha=0.2)

    lr_axis = loss_axis.twinx()
    lr_axis.plot(steps, learning_rates, color="seagreen", linestyle="--", alpha=0.7, label="learning rate")
    lr_axis.set_ylabel("Learning rate")

    handles1, labels1 = loss_axis.get_legend_handles_labels()
    handles2, labels2 = lr_axis.get_legend_handles_labels()
    loss_axis.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    figure.suptitle("TinyStories training run")
    figure.tight_layout()

    output_path = args.output or args.metrics.with_name("training_curves.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
