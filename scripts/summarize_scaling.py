from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

# Keep Matplotlib's cache in the project so the script also works in restricted
# terminals and CI environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fixed-compute scaling runs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("checkpoints/scaling_bf16"))
    return parser.parse_args()


def parameter_count(config: dict[str, object]) -> int:
    """Count this project's TransformerLM parameters from its dimensions."""
    vocab_size = int(config["vocab_size"])
    d_model = int(config["d_model"])
    d_ff = int(config["d_ff"])
    num_layers = int(config["num_layers"])

    embeddings_and_head = 2 * vocab_size * d_model
    transformer_layers = num_layers * (
        2 * d_model  # two RMSNorm scale vectors
        + 4 * d_model * d_model  # Q, K, V, and output projections
        + 3 * d_model * d_ff  # SwiGLU's W1, W2, and W3
    )
    final_norm = d_model
    return embeddings_and_head + transformer_layers + final_norm


def final_validation_record(metrics_path: Path) -> dict[str, object]:
    with metrics_path.open(encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    validation_records = [record for record in records if record["validation_loss"] is not None]
    if not validation_records:
        raise ValueError(f"no validation records found in {metrics_path}")
    return validation_records[-1]


def main() -> None:
    args = parse_args()
    results: list[dict[str, object]] = []

    for run_dir in sorted(path for path in args.runs_dir.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.jsonl"
        if not config_path.exists() or not metrics_path.exists():
            continue

        with config_path.open(encoding="utf-8") as input_file:
            config = json.load(input_file)
        record = final_validation_record(metrics_path)
        parameters = parameter_count(config)
        tokens_seen = int(record["tokens_seen"])
        validation_loss = float(record["validation_loss"])

        dataset_path = Path(config["dataset"])
        with dataset_path.open(encoding="utf-8") as input_file:
            dataset = json.load(input_file)
        training_tokens = int(dataset["train"]["tokens"])

        results.append(
            {
                "run": run_dir.name,
                "parameters": parameters,
                "steps": int(record["step"]),
                "context_length": int(config["context_length"]),
                "batch_size": int(config["batch_size"]),
                "tokens_seen": tokens_seen,
                "dataset_equivalents": tokens_seen / training_tokens,
                "parameter_token_budget": parameters * tokens_seen,
                "validation_loss": validation_loss,
                "perplexity": math.exp(validation_loss),
                "validation_accuracy": float(record["validation_accuracy"]),
                "elapsed_seconds": float(record["elapsed_seconds"]),
                "precision": config["precision"],
            }
        )

    if not results:
        raise ValueError(f"no completed runs found under {args.runs_dir}")
    results.sort(key=lambda result: int(result["parameters"]))

    output_json = args.runs_dir / "comparison.json"
    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump({"results": results}, output_file, indent=2)

    names = [str(result["run"]) for result in results]
    parameters_m = [int(result["parameters"]) / 1e6 for result in results]
    validation_losses = [float(result["validation_loss"]) for result in results]
    accuracies = [100 * float(result["validation_accuracy"]) for result in results]

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 5))
    loss_axis.plot(parameters_m, validation_losses, marker="o", linewidth=2, color="royalblue")
    accuracy_axis.plot(parameters_m, accuracies, marker="o", linewidth=2, color="darkorange")
    for name, x_value, loss, accuracy in zip(
        names, parameters_m, validation_losses, accuracies, strict=True
    ):
        loss_axis.annotate(name, (x_value, loss), xytext=(5, 5), textcoords="offset points")
        accuracy_axis.annotate(name, (x_value, accuracy), xytext=(5, 5), textcoords="offset points")

    loss_axis.set(xlabel="Parameters (millions)", ylabel="Validation cross-entropy")
    accuracy_axis.set(xlabel="Parameters (millions)", ylabel="Validation token accuracy (%)")
    for axis in (loss_axis, accuracy_axis):
        axis.grid(alpha=0.25)
    figure.suptitle("TinyStories: model size at approximately fixed parameter-token compute")
    figure.tight_layout()

    output_plot = args.runs_dir / "comparison.png"
    figure.savefig(output_plot, dpi=160)

    print(f"Saved summary: {output_json}")
    print(f"Saved plot: {output_plot}")
    for result in results:
        print(
            f"{result['run']:>7}: {int(result['parameters']) / 1e6:6.2f}M params | "
            f"loss {float(result['validation_loss']):.4f} | "
            f"ppl {float(result['perplexity']):.2f} | "
            f"accuracy {100 * float(result['validation_accuracy']):.2f}%"
        )


if __name__ == "__main__":
    main()
