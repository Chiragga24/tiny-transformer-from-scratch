from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tiny_transformer.model import TransformerLM
from tiny_transformer.nn_utils import cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically evaluate TinyStories checkpoints on sequential validation tokens."
    )
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--eval-context-length",
        type=int,
        default=256,
        help="Common evaluation context; must not exceed a checkpoint's trained maximum.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_final_sampled_metrics(run_dir: Path) -> dict[str, float | int] | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    with metrics_path.open(encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    validation_records = [record for record in records if record["validation_loss"] is not None]
    if not validation_records:
        return None
    final = validation_records[-1]
    return {
        "step": int(final["step"]),
        "validation_loss": float(final["validation_loss"]),
        "validation_accuracy": float(final["validation_accuracy"]),
    }


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    eval_context_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    run_dir = checkpoint_path.parent
    with (run_dir / "config.json").open(encoding="utf-8") as input_file:
        config = json.load(input_file)

    model_context_length = int(config["context_length"])
    if eval_context_length > model_context_length:
        raise ValueError(
            f"evaluation context {eval_context_length} exceeds {checkpoint_path}'s "
            f"model context {model_context_length}"
        )

    dataset_path = Path(config["dataset"])
    with dataset_path.open(encoding="utf-8") as input_file:
        metadata = json.load(input_file)
    validation_data = np.memmap(
        dataset_path.parent / metadata["valid"]["path"],
        dtype=np.dtype(metadata["dtype"]),
        mode="r",
    )

    # Consecutive, non-overlapping windows make evaluation deterministic and
    # cover every validation target except a short tail smaller than one window.
    usable_tokens = ((len(validation_data) - 1) // eval_context_length) * eval_context_length
    inputs = np.asarray(validation_data[:usable_tokens], dtype=np.int64).reshape(
        -1, eval_context_length
    )
    targets = np.asarray(validation_data[1 : usable_tokens + 1], dtype=np.int64).reshape(
        -1, eval_context_length
    )

    model = TransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=model_context_length,
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    precision = str(config.get("precision", "fp32"))
    total_loss = 0.0
    correct_tokens = 0
    token_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    started_at = time.perf_counter()

    for index in range(0, len(inputs), batch_size):
        input_batch = torch.tensor(inputs[index : index + batch_size], device=device)
        target_batch = torch.tensor(targets[index : index + batch_size], device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16" and device.type == "cuda",
        ):
            logits = model(input_batch)
            loss = cross_entropy(
                logits.float().reshape(-1, int(config["vocab_size"])),
                target_batch.reshape(-1),
            )
        batch_tokens = target_batch.numel()
        total_loss += loss.item() * batch_tokens
        correct_tokens += (logits.argmax(dim=-1) == target_batch).sum().item()
        token_count += batch_tokens

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started_at
    validation_loss = total_loss / token_count
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["iteration"]),
        "parameters": parameter_count,
        "precision": precision,
        "model_context_length": model_context_length,
        "evaluation_context_length": eval_context_length,
        "validation_tokens": token_count,
        "validation_loss": validation_loss,
        "perplexity": math.exp(validation_loss),
        "validation_accuracy": correct_tokens / token_count,
        "evaluation_seconds": elapsed_seconds,
        "sampled_final_metrics": load_final_sampled_metrics(run_dir),
    }


def main() -> None:
    args = parse_args()
    if args.eval_context_length <= 0 or args.batch_size <= 0:
        raise ValueError("evaluation context length and batch size must be positive")

    device = choose_device()
    for checkpoint_path in args.checkpoint:
        result = evaluate_checkpoint(
            checkpoint_path,
            args.eval_context_length,
            args.batch_size,
            device,
        )
        output_path = checkpoint_path.parent / (
            f"full_validation_ctx{args.eval_context_length}.json"
        )
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)
        print(
            f"{checkpoint_path.parent.name:>7}: loss {result['validation_loss']:.4f} | "
            f"ppl {result['perplexity']:.2f} | "
            f"accuracy {result['validation_accuracy']:.2%} | "
            f"{result['validation_tokens']:,} tokens | {result['evaluation_seconds']:.1f}s"
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
