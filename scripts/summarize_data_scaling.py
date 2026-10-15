from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "checkpoints" / "data_scaling"
CONDITIONS = [
    (
        "20 MB",
        ROOT / "data" / "tinystories_laptop" / "dataset.json",
        ROOT / "checkpoints" / "scaling_bf16" / "small" / "full_validation_ctx256.json",
    ),
    (
        "100 MB",
        ROOT / "data" / "tinystories_100mb" / "dataset.json",
        OUTPUT_DIR / "100mb" / "full_validation_ctx256.json",
    ),
    (
        "500 MB",
        ROOT / "data" / "tinystories_500mb" / "dataset.json",
        OUTPUT_DIR / "500mb" / "full_validation_ctx256.json",
    ),
]
TRAINING_TOKENS = 5_000 * 8 * 256


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def main() -> None:
    results: list[dict[str, object]] = []
    for label, dataset_path, evaluation_path in CONDITIONS:
        dataset = load_json(dataset_path)
        evaluation = load_json(evaluation_path)
        train = dataset["train"]
        assert isinstance(train, dict)
        source_bytes = int(train["source_bytes"])
        available_tokens = int(train["tokens"])
        results.append(
            {
                "condition": label,
                "source_megabytes_decimal": source_bytes / 1_000_000,
                "available_training_tokens": available_tokens,
                "sampled_training_tokens": TRAINING_TOKENS,
                "sampled_to_available_token_ratio": TRAINING_TOKENS / available_tokens,
                "validation_loss": evaluation["validation_loss"],
                "perplexity": evaluation["perplexity"],
                "validation_accuracy": evaluation["validation_accuracy"],
                "validation_tokens": evaluation["validation_tokens"],
                "checkpoint": evaluation["checkpoint"],
            }
        )

    baseline_loss = float(results[0]["validation_loss"])
    for result in results:
        result["loss_improvement_vs_20mb"] = baseline_loss - float(result["validation_loss"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT_DIR / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "controlled_variables": {
                    "parameters": 6_139_136,
                    "steps": 5_000,
                    "batch_size": 8,
                    "context_length": 256,
                    "sampled_training_tokens": TRAINING_TOKENS,
                    "seed": 42,
                    "precision": "bf16",
                    "validation_protocol": "all non-overlapping 256-token windows",
                },
                "results": results,
            },
            output_file,
            indent=2,
        )

    sizes = [float(result["source_megabytes_decimal"]) for result in results]
    losses = [float(result["validation_loss"]) for result in results]
    perplexities = [float(result["perplexity"]) for result in results]
    accuracies = [100 * float(result["validation_accuracy"]) for result in results]

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [
        (axes[0], losses, "Validation loss", "lower is better"),
        (axes[1], perplexities, "Perplexity", "lower is better"),
        (axes[2], accuracies, "Token accuracy (%)", "higher is better"),
    ]
    for axis, values, title, subtitle in panels:
        axis.plot(sizes, values, marker="o", linewidth=2)
        axis.set_xscale("log")
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(result["condition"]) for result in results])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_title(f"{title}\n{subtitle}", fontsize=11)
        axis.set_xlabel("Available training text")
        axis.grid(alpha=0.25)
        for x_value, y_value in zip(sizes, values, strict=True):
            axis.annotate(f"{y_value:.3f}", (x_value, y_value), xytext=(0, 7),
                          textcoords="offset points", ha="center", fontsize=8)

    figure.suptitle("Dataset scaling at fixed model size and 10.24M sampled tokens")
    figure.tight_layout()
    plot_path = OUTPUT_DIR / "comparison.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    print(f"Saved {comparison_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
