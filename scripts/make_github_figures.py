from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "figures"
COLORS = {
    "blue": "#2563EB",
    "orange": "#F97316",
    "green": "#16A34A",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "gray": "#64748B",
}


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#0F172A",
            "axes.titlecolor": "#0F172A",
            "text.color": "#0F172A",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.45,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "figure.titlesize": 18,
            "figure.titleweight": "bold",
        }
    )


def finish(figure: plt.Figure, filename: str) -> None:
    figure.tight_layout()
    destination = OUTPUT_DIR / filename
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {destination}")


def add_value_labels(axis: plt.Axes, bars: object, decimals: int = 3) -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            f"{value:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_protocol_comparison() -> None:
    fixed_names = ["small", "medium", "large", "xlarge"]
    fixed_losses = [2.196275, 2.392747, 2.591513, 2.946195]
    equal = read_json(ROOT / "checkpoints" / "equal_steps" / "comparison.json")["results"]
    equal_losses = [float(result["validation_loss"]) for result in equal]
    equal_compute = [float(result["approx_training_flops_6nt"]) / 1e15 for result in equal]
    parameters = [float(result["parameters"]) / 1e6 for result in equal]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(fixed_names))
    width = 0.36
    left = axes[0].bar(
        x - width / 2,
        fixed_losses,
        width,
        color=COLORS["orange"],
        label="Fixed parameter-token budget",
    )
    right = axes[0].bar(
        x + width / 2,
        equal_losses,
        width,
        color=COLORS["blue"],
        label="Equal 5,000 steps / 10.24M tokens",
    )
    axes[0].set_xticks(x, [name.title() for name in fixed_names])
    axes[0].set_ylabel("Full validation loss")
    axes[0].set_title("The protocol changes the conclusion")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="y")
    add_value_labels(axes[0], left)
    add_value_labels(axes[0], right)

    axes[1].plot(parameters, equal_losses, marker="o", linewidth=2.5, color=COLORS["blue"])
    for name, x_value, y_value, compute in zip(
        fixed_names, parameters, equal_losses, equal_compute, strict=True
    ):
        offset = (7, -24) if name == "small" else (7, 4)
        axes[1].annotate(
            f"{name}\n{compute:.2f} PF",
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )
    axes[1].set_xlabel("Parameters (millions)")
    axes[1].set_ylabel("Full validation loss")
    axes[1].set_title("More parameters help when compute also grows")
    axes[1].grid()
    figure.suptitle("TinyStories scaling: fixed compute versus equal training tokens")
    finish(figure, "01_protocol_comparison.png")


def plot_data_scaling() -> None:
    results = read_json(ROOT / "checkpoints" / "data_scaling" / "comparison.json")[
        "results"
    ]
    sizes = [float(result["source_megabytes_decimal"]) for result in results]
    losses = [float(result["validation_loss"]) for result in results]
    accuracies = [100 * float(result["validation_accuracy"]) for result in results]
    labels = [str(result["condition"]) for result in results]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, values, title, ylabel, color in [
        (axes[0], losses, "Validation loss", "Cross-entropy (lower is better)", COLORS["blue"]),
        (axes[1], accuracies, "Next-token accuracy", "Accuracy (%)", COLORS["green"]),
    ]:
        axis.plot(sizes, values, marker="o", linewidth=2.5, markersize=8, color=color)
        axis.set_xscale("log")
        axis.set_xticks(sizes, labels)
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Available TinyStories text")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid()
        for x_value, y_value in zip(sizes, values, strict=True):
            axis.annotate(
                f"{y_value:.3f}",
                (x_value, y_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    figure.suptitle("Dataset scaling at a fixed 10.24M sampled training tokens")
    finish(figure, "02_data_scaling.png")


def plot_controls() -> None:
    original = read_json(
        ROOT / "checkpoints" / "scaling_bf16" / "xlarge" / "full_validation_ctx256.json"
    )
    controlled = read_json(
        ROOT
        / "checkpoints"
        / "context_control"
        / "xlarge_ctx256"
        / "full_validation_ctx256.json"
    )
    models = ["Small", "Medium", "Large", "Xlarge"]
    sampled = [2.238129, 2.380224, 2.555178, 2.928469]
    exhaustive = [2.196275, 2.392747, 2.591513, 2.946195]

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    context_bars = axes[0].bar(
        ["Context 512\nBatch 4", "Context 256\nBatch 8"],
        [float(original["validation_loss"]), float(controlled["validation_loss"])],
        color=[COLORS["orange"], COLORS["blue"]],
        width=0.62,
    )
    axes[0].set_ylabel("Full validation loss")
    axes[0].set_title("Xlarge context-length control")
    axes[0].grid(axis="y")
    add_value_labels(axes[0], context_bars, decimals=3)

    x = np.arange(len(models))
    axes[1].plot(x, sampled, marker="o", linewidth=2.3, color=COLORS["orange"], label="20 batches")
    axes[1].plot(
        x,
        exhaustive,
        marker="s",
        linewidth=2.3,
        color=COLORS["blue"],
        label="544,256 tokens",
    )
    axes[1].set_xticks(x, models)
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title("Sampled validation preserved the ranking")
    axes[1].legend(frameon=False)
    axes[1].grid()
    figure.suptitle("Follow-up controls for context shape and evaluation noise")
    finish(figure, "03_experimental_controls.png")


def plot_schedule_sweep() -> None:
    results = read_json(ROOT / "checkpoints" / "hparam_sweep" / "comparison.json")[
        "results"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for warmup, marker, color in [
        (98, "o", COLORS["orange"]),
        (196, "s", COLORS["blue"]),
    ]:
        subset = [result for result in results if int(result["warmup_steps"]) == warmup]
        rates = [float(result["max_learning_rate"]) for result in subset]
        losses = [float(result["validation_loss"]) for result in subset]
        accuracies = [100 * float(result["validation_accuracy"]) for result in subset]
        label = f"{100 * warmup / 1955:.0f}% warmup"
        axes[0].plot(rates, losses, marker=marker, linewidth=2.5, color=color, label=label)
        axes[1].plot(rates, accuracies, marker=marker, linewidth=2.5, color=color, label=label)
    for axis, title, ylabel in [
        (axes[0], "Validation loss", "Cross-entropy (lower is better)"),
        (axes[1], "Next-token accuracy", "Accuracy (%)"),
    ]:
        axis.set_xscale("log", base=2)
        axis.set_xticks([1.5e-4, 3e-4, 6e-4], ["1.5e-4", "3e-4", "6e-4"])
        axis.set_xlabel("Peak learning rate")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid()
        axis.legend(frameon=False)
    figure.suptitle("Medium-model schedule sweep at fixed 4.00M training tokens")
    finish(figure, "04_schedule_sweep.png")


def plot_gpu_memory() -> None:
    benchmark = read_json(ROOT / "benchmarks" / "gpu_memory.json")
    results = benchmark["results"]
    cap = float(benchmark["total_vram_mib"]) * float(benchmark["memory_fraction_cap"])
    labels = [str(result["name"]).replace("_", "\n") for result in results]
    values = [
        float(result.get("peak_allocated_mib", cap)) / 1024
        for result in results
    ]
    colors = [
        COLORS["red"] if result["status"] == "out_of_memory" else COLORS["blue"]
        for result in results
    ]

    figure, axis = plt.subplots(figsize=(11, 4.8))
    bars = axis.bar(labels, values, color=colors, width=0.7)
    bars[-1].set_hatch("///")
    axis.axhline(cap / 1024, color=COLORS["red"], linestyle="--", linewidth=2,
                 label=f"80% safety cap ({cap / 1024:.2f} GiB)")
    axis.set_ylabel("Peak allocated GPU memory (GiB)")
    axis.set_title("RTX 4050 laptop GPU memory envelope")
    axis.grid(axis="y")
    axis.legend(frameon=False)
    for bar, result, value in zip(bars, results, values, strict=True):
        label = "OOM" if result["status"] == "out_of_memory" else f"{value:.2f}"
        axis.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    figure.suptitle("Controlled BF16 memory benchmark", y=1.02)
    finish(figure, "05_gpu_memory.png")


def plot_generation_diagnostics() -> None:
    data = read_json(ROOT / "checkpoints" / "generation_suite" / "results.json")
    metrics = data["automatic_metrics"]
    model_keys = ["small", "medium", "medium_tuned", "large", "xlarge"]
    labels = ["Small", "Medium", "Tuned\nmedium", "Large", "Xlarge"]
    sampled = [metrics[f"{key}:sampled"] for key in model_keys]
    eos = [100 * float(metric["eos_rate"]) for metric in sampled]
    distinct = [float(metric["distinct_3"]) for metric in sampled]
    repetition = [100 * float(metric["mean_within_sample_repetition_3"]) for metric in sampled]

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    panels = [
        (axes[0], eos, "EOS rate", "% of samples", COLORS["green"]),
        (axes[1], distinct, "Distinct-3", "Unique / total trigrams", COLORS["blue"]),
        (axes[2], repetition, "Repeated trigrams", "Mean within-sample rate (%)", COLORS["purple"]),
    ]
    for axis, values, title, ylabel, color in panels:
        bars = axis.bar(labels, values, color=color, width=0.68)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y")
        add_value_labels(axis, bars, decimals=2)
    figure.suptitle("Sampled-generation diagnostics across 12 continuations per model")
    finish(figure, "06_generation_diagnostics.png")


def main() -> None:
    style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_protocol_comparison()
    plot_data_scaling()
    plot_controls()
    plot_schedule_sweep()
    plot_gpu_memory()
    plot_generation_diagnostics()


if __name__ == "__main__":
    main()
