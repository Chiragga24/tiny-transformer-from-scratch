from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


MIB = 1024**2


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    context_length: int
    batch_size: int


CASES = [
    BenchmarkCase("baseline", 128, 4, 4, 384, 128, 8),
    BenchmarkCase("small", 256, 6, 4, 768, 256, 8),
    BenchmarkCase("medium", 384, 8, 6, 1024, 256, 8),
    BenchmarkCase("large", 512, 8, 8, 1344, 256, 8),
    BenchmarkCase("xlarge", 640, 10, 10, 1728, 512, 4),
    BenchmarkCase("stress", 768, 12, 12, 2048, 512, 4),
    BenchmarkCase("guarded_limit", 1024, 12, 16, 2752, 512, 4),
    BenchmarkCase("extreme", 1152, 14, 18, 3072, 512, 4),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure full-training-step CUDA memory safely.")
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.80,
        help="Maximum fraction of total VRAM available to this benchmark process.",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/gpu_memory.json"))
    return parser.parse_args()


def benchmark_case(
    case: BenchmarkCase,
    vocab_size: int,
    steps: int,
    device: torch.device,
    precision: str,
) -> dict[str, int | float | str]:
    torch.cuda.empty_cache()

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=case.context_length,
        d_model=case.d_model,
        num_layers=case.num_layers,
        num_heads=case.num_heads,
        d_ff=case.d_ff,
        rope_theta=10_000.0,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    inputs = torch.randint(
        vocab_size,
        (case.batch_size, case.context_length),
        device=device,
    )
    targets = torch.randint(
        vocab_size,
        (case.batch_size, case.context_length),
        device=device,
    )

    def training_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            logits = model(inputs)
            # Keep the numerically sensitive loss reduction in FP32.
            loss = cross_entropy(logits.float().reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()

    # One untimed step initializes CUDA kernels and AdamW's moment tensors.
    # This makes throughput numbers comparable across differently sized cases.
    training_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    started_at = time.perf_counter()
    for _ in range(steps):
        training_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        **asdict(case),
        "status": "passed",
        "parameters": parameter_count,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / MIB, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / MIB, 1),
        "seconds_per_step": round(elapsed / steps, 4),
        "tokens_per_second": round(case.batch_size * case.context_length * steps / elapsed),
        "precision": precision,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("memory fraction must be in (0, 1]")

    device = torch.device("cuda", torch.cuda.current_device())
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this GPU does not support native BF16")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    properties = torch.cuda.get_device_properties(device)
    print(
        f"GPU: {properties.name}; total VRAM: {properties.total_memory / MIB:.0f} MiB; "
        f"benchmark cap: {args.memory_fraction:.0%}"
    )

    results = []
    for case in CASES:
        print(f"Testing {case.name}: d={case.d_model}, layers={case.num_layers}, context={case.context_length}...")
        try:
            result = benchmark_case(case, args.vocab_size, args.steps, device, args.precision)
            results.append(result)
            print(
                f"  passed | {result['parameters']:,} params | "
                f"peak {result['peak_allocated_mib']:.1f} MiB | "
                f"{result['tokens_per_second']:,} tokens/s"
            )
        except torch.OutOfMemoryError as error:
            result = {**asdict(case), "status": "out_of_memory", "error": str(error)}
            results.append(result)
            print("  stopped: reached the guarded CUDA memory limit")
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    report = {
        "gpu": properties.name,
        "total_vram_mib": round(properties.total_memory / MIB, 1),
        "memory_fraction_cap": args.memory_fraction,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": args.precision,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
    print(f"Saved benchmark report: {args.output}")


if __name__ == "__main__":
    main()
