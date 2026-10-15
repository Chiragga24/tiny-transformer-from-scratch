from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from tiny_transformer.data import get_batch
from tiny_transformer.model import TransformerLM
from tiny_transformer.nn_utils import clip_gradients, cross_entropy
from tiny_transformer.optimizer import AdamW, get_lr_cosine_schedule
from tiny_transformer.serialization import load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small decoder-only LM on TinyStories.")
    parser.add_argument("--dataset", type=Path, default=Path("data/tinystories_laptop/dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/tinystories_laptop"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another PyTorch device")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")

    # These deliberately small defaults exercise the complete workflow on a
    # laptop. They can be raised later when CUDA-enabled PyTorch is installed.
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=384)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation cannot use CUDA")
    return device


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    vocab_size: int,
    device: torch.device,
    precision: str,
) -> tuple[float, float]:
    model.eval()
    losses = []
    correct_tokens = 0
    token_count = 0
    for _ in range(eval_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            logits = model(inputs)
            loss = cross_entropy(logits.float().reshape(-1, vocab_size), targets.reshape(-1))
        losses.append(loss.item())
        correct_tokens += (logits.argmax(dim=-1) == targets).sum().item()
        token_count += targets.numel()
    model.train()
    return float(np.mean(losses)), correct_tokens / token_count


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    if args.context_length <= 0 or args.d_model <= 0 or args.d_ff <= 0:
        raise ValueError("model dimensions must be positive")
    if args.num_layers <= 0 or args.num_heads <= 0:
        raise ValueError("the numbers of layers and attention heads must be positive")
    if args.eval_interval <= 0 or args.eval_batches <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("evaluation and checkpoint intervals/counts must be positive")
    if args.warmup_steps > args.steps:
        raise ValueError("warmup steps cannot exceed total steps")
    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    with args.dataset.open(encoding="utf-8") as input_file:
        metadata = json.load(input_file)
    if metadata.get("format_version") != 1:
        raise ValueError("unsupported dataset metadata format")
    dataset_dir = args.dataset.parent
    dtype = np.dtype(metadata["dtype"])
    train_data = np.memmap(dataset_dir / metadata["train"]["path"], dtype=dtype, mode="r")
    valid_data = np.memmap(dataset_dir / metadata["valid"]["path"], dtype=dtype, mode="r")
    vocab_size = int(metadata["vocab_size"])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if args.precision == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("BF16 training requires a CUDA GPU with native BF16 support")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
    )
    start_step = load_checkpoint(args.resume, model, optimizer) if args.resume else 0
    if start_step >= args.steps:
        raise ValueError(f"checkpoint is already at step {start_step}, but --steps is {args.steps}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config["dataset"] = str(args.dataset)
    run_config["output_dir"] = str(args.output_dir)
    run_config["resume"] = str(args.resume) if args.resume else None
    run_config["device_used"] = str(device)
    run_config["vocab_size"] = vocab_size
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as output_file:
        json.dump(run_config, output_file, indent=2)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}; parameters: {parameter_count:,}; starting at step {start_step}")
    model.train()
    started_at = time.perf_counter()
    training_seconds = 0.0
    evaluation_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # One JSON object per step is easy to inspect, append after resuming, and
    # consume from plotting tools without a tracking service or online account.
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_mode = "a" if args.resume and metrics_path.exists() else "w"
    metrics_file = metrics_path.open(metrics_mode, encoding="utf-8")

    # tqdm gives immediate feedback during longer runs. `initial` makes a
    # resumed checkpoint begin at its saved step instead of visually at zero.
    progress = tqdm(
        range(start_step, args.steps),
        initial=start_step,
        total=args.steps,
        desc="training",
        unit="step",
    )
    for step in progress:
        training_step_started_at = time.perf_counter()
        lr = get_lr_cosine_schedule(
            step,
            args.max_lr,
            args.min_lr,
            args.warmup_steps,
            args.steps,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = lr

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.precision == "bf16",
        ):
            logits = model(inputs)
            loss = cross_entropy(logits.float().reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        clip_gradients(model.parameters(), args.grad_clip)
        optimizer.step()

        # loss.item() synchronizes CUDA, which makes this elapsed measurement
        # include the completed GPU work rather than only asynchronous dispatch.
        train_loss = loss.item()
        training_seconds += time.perf_counter() - training_step_started_at

        completed_step = step + 1
        validation_loss = None
        validation_accuracy = None
        progress.set_postfix(train_loss=f"{train_loss:.4f}", lr=f"{lr:.2e}")
        should_evaluate = (
            completed_step == 1
            or completed_step % args.eval_interval == 0
            or completed_step == args.steps
        )
        if should_evaluate:
            evaluation_started_at = time.perf_counter()
            validation_loss, validation_accuracy = evaluate(
                model,
                valid_data,
                args.batch_size,
                args.context_length,
                args.eval_batches,
                vocab_size,
                device,
                args.precision,
            )
            evaluation_seconds += time.perf_counter() - evaluation_started_at
            elapsed = time.perf_counter() - started_at
            tqdm.write(
                f"step {completed_step:>5}/{args.steps} | "
                f"train {train_loss:.4f} | valid {validation_loss:.4f} | "
                f"accuracy {validation_accuracy:.2%} | "
                f"lr {lr:.2e} | {elapsed:.1f}s"
            )
            progress.set_postfix(
                train_loss=f"{loss.item():.4f}",
                valid_loss=f"{validation_loss:.4f}",
                lr=f"{lr:.2e}",
            )

        elapsed = time.perf_counter() - started_at
        peak_allocated_mib = (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        )
        metric = {
            "step": completed_step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": lr,
            "elapsed_seconds": elapsed,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "tokens_seen": completed_step * args.batch_size * args.context_length,
            "training_tokens_per_second": (
                completed_step * args.batch_size * args.context_length / training_seconds
            ),
            "peak_allocated_mib": peak_allocated_mib,
        }
        metrics_file.write(json.dumps(metric) + "\n")
        metrics_file.flush()

        should_checkpoint = (
            completed_step % args.checkpoint_interval == 0 or completed_step == args.steps
        )
        if should_checkpoint:
            checkpoint_path = args.output_dir / f"step_{completed_step:06d}.pt"
            save_checkpoint(model, optimizer, completed_step, checkpoint_path)
            tqdm.write(f"Saved checkpoint: {checkpoint_path}")

    metrics_file.close()
    total_seconds = time.perf_counter() - started_at
    tokens_seen = args.steps * args.batch_size * args.context_length
    runtime_summary = {
        "parameters": parameter_count,
        "steps": args.steps,
        "tokens_seen": tokens_seen,
        "parameter_token_budget": parameter_count * tokens_seen,
        # 6*N*T is the standard dense-Transformer training-FLOP proxy: about
        # 2*N*T for the forward pass and twice that again for backpropagation.
        "approx_training_flops_6nt": 6 * parameter_count * tokens_seen,
        "total_seconds": total_seconds,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "other_seconds": total_seconds - training_seconds - evaluation_seconds,
        "training_tokens_per_second": tokens_seen / training_seconds,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        ),
    }
    with (args.output_dir / "runtime_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(runtime_summary, output_file, indent=2)


if __name__ == "__main__":
    main()
