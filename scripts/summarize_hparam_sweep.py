from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "checkpoints" / "hparam_sweep"
CONDITIONS = [
    (
        "1.5e-4, 10% warmup",
        1.5e-4,
        196,
        OUTPUT_DIR / "lr_1p5e4_warmup_196" / "full_validation_ctx256.json",
    ),
    (
        "3e-4, 5% warmup",
        3e-4,
        98,
        OUTPUT_DIR / "lr_3e4_warmup_98" / "full_validation_ctx256.json",
    ),
    (
        "3e-4, 10% warmup (baseline)",
        3e-4,
        196,
        ROOT / "checkpoints" / "scaling_bf16" / "medium" / "full_validation_ctx256.json",
    ),
    (
        "6e-4, 5% warmup",
        6e-4,
        98,
        OUTPUT_DIR / "lr_6e4_warmup_98" / "full_validation_ctx256.json",
    ),
    (
        "6e-4, 10% warmup",
        6e-4,
        196,
        OUTPUT_DIR / "lr_6e4_warmup_196" / "full_validation_ctx256.json",
    ),
]


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def main() -> None:
    results: list[dict[str, object]] = []
    for label, max_lr, warmup_steps, evaluation_path in CONDITIONS:
        evaluation = load_json(evaluation_path)
        results.append(
            {
                "condition": label,
                "max_learning_rate": max_lr,
                "min_learning_rate": max_lr / 10,
                "warmup_steps": warmup_steps,
                "warmup_fraction": warmup_steps / 1_955,
                "validation_loss": evaluation["validation_loss"],
                "perplexity": evaluation["perplexity"],
                "validation_accuracy": evaluation["validation_accuracy"],
                "validation_tokens": evaluation["validation_tokens"],
                "checkpoint": evaluation["checkpoint"],
            }
        )

    baseline_loss = float(results[2]["validation_loss"])
    for result in results:
        result["loss_improvement_vs_baseline"] = baseline_loss - float(
            result["validation_loss"]
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT_DIR / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "controlled_variables": {
                    "parameters": 15_698_304,
                    "steps": 1_955,
                    "batch_size": 8,
                    "context_length": 256,
                    "sampled_training_tokens": 1_955 * 8 * 256,
                    "seed": 42,
                    "precision": "bf16",
                    "validation_protocol": "all non-overlapping 256-token windows",
                },
                "results": results,
            },
            output_file,
            indent=2,
        )

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for warmup_steps, marker, color in [(98, "o", "tab:orange"), (196, "s", "tab:blue")]:
        subset = [result for result in results if result["warmup_steps"] == warmup_steps]
        rates = [float(result["max_learning_rate"]) for result in subset]
        losses = [float(result["validation_loss"]) for result in subset]
        accuracies = [100 * float(result["validation_accuracy"]) for result in subset]
        label = f"{100 * warmup_steps / 1_955:.0f}% warmup"
        axes[0].plot(rates, losses, marker=marker, color=color, linewidth=2, label=label)
        axes[1].plot(rates, accuracies, marker=marker, color=color, linewidth=2, label=label)

    panels = [
        (axes[0], "Validation loss\nlower is better"),
        (axes[1], "Token accuracy (%)\nhigher is better"),
    ]
    for axis, title in panels:
        axis.set_xscale("log", base=2)
        axis.set_xticks([1.5e-4, 3e-4, 6e-4], ["1.5e-4", "3e-4", "6e-4"])
        axis.set_xlabel("Peak learning rate")
        axis.set_title(title, fontsize=11)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("Medium-model schedule sweep at fixed 4.00M training tokens")
    figure.tight_layout()
    plot_path = OUTPUT_DIR / "comparison.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    print(f"Saved {comparison_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
