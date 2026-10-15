from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt


RUNS = {
    "small": Path("checkpoints/scaling_bf16/small"),
    "medium": Path("checkpoints/equal_steps/medium"),
    "large": Path("checkpoints/equal_steps/large"),
    "xlarge": Path("checkpoints/equal_steps/xlarge"),
}


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def final_metric(run_dir: Path) -> dict[str, object]:
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    return records[-1]


def main() -> None:
    results = []
    for name, run_dir in RUNS.items():
        config = read_json(run_dir / "config.json")
        validation = read_json(run_dir / "full_validation_ctx256.json")
        metric = final_metric(run_dir)
        runtime_path = run_dir / "runtime_summary.json"
        runtime = read_json(runtime_path) if runtime_path.exists() else None

        parameters = int(validation["parameters"])
        tokens_seen = int(metric["tokens_seen"])
        elapsed_seconds = float(
            runtime["total_seconds"] if runtime is not None else metric["elapsed_seconds"]
        )
        results.append(
            {
                "run": name,
                "parameters": parameters,
                "steps": int(metric["step"]),
                "tokens_seen": tokens_seen,
                "approx_training_flops_6nt": 6 * parameters * tokens_seen,
                "elapsed_seconds": elapsed_seconds,
                "training_seconds": (
                    float(runtime["training_seconds"]) if runtime is not None else None
                ),
                "training_tokens_per_second": (
                    float(runtime["training_tokens_per_second"])
                    if runtime is not None
                    else None
                ),
                "peak_allocated_mib": (
                    float(runtime["peak_allocated_mib"]) if runtime is not None else None
                ),
                "validation_loss": float(validation["validation_loss"]),
                "perplexity": float(validation["perplexity"]),
                "validation_accuracy": float(validation["validation_accuracy"]),
                "context_length": int(config["context_length"]),
                "batch_size": int(config["batch_size"]),
            }
        )

    output_dir = Path("checkpoints/equal_steps")
    output_path = output_dir / "comparison.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump({"results": results}, output_file, indent=2)

    parameters_m = [int(result["parameters"]) / 1e6 for result in results]
    flops_pf = [int(result["approx_training_flops_6nt"]) / 1e15 for result in results]
    elapsed_minutes = [float(result["elapsed_seconds"]) / 60 for result in results]
    losses = [float(result["validation_loss"]) for result in results]
    names = [str(result["run"]) for result in results]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x_series = (parameters_m, flops_pf, elapsed_minutes)
    x_labels = ("Parameters (millions)", "Approx. training compute (PFLOPs, 6NT)", "Elapsed minutes")
    for axis, x_values, x_label in zip(axes, x_series, x_labels, strict=True):
        axis.plot(x_values, losses, marker="o", linewidth=2, color="royalblue")
        for name, x_value, loss in zip(names, x_values, losses, strict=True):
            axis.annotate(name, (x_value, loss), xytext=(4, 5), textcoords="offset points")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Full validation cross-entropy")
        axis.grid(alpha=0.25)
    figure.suptitle("TinyStories equal-step scaling: 5,000 steps and 10.24M tokens per model")
    figure.tight_layout()

    plot_path = output_dir / "comparison.png"
    figure.savefig(plot_path, dpi=160)
    print(f"Saved summary: {output_path}")
    print(f"Saved plot: {plot_path}")
    for result in results:
        print(
            f"{result['run']:>7}: loss {result['validation_loss']:.4f} | "
            f"ppl {result['perplexity']:.2f} | accuracy {100 * result['validation_accuracy']:.2f}% | "
            f"{result['elapsed_seconds'] / 60:.1f} min"
        )


if __name__ == "__main__":
    main()
