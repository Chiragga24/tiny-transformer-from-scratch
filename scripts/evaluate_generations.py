from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from tiny_transformer.generation import generate_token_ids
from tiny_transformer.model import TransformerLM
from tiny_transformer.tokenizer import Tokenizer
from tiny_transformer.tokenizer_io import load_tokenizer


DEFAULT_CHECKPOINTS = {
    "small": Path("checkpoints/scaling_bf16/small/step_005000.pt"),
    "medium": Path("checkpoints/scaling_bf16/medium/step_001955.pt"),
    "medium_tuned": Path("checkpoints/hparam_sweep/lr_6e4_warmup_196/step_001955.pt"),
    "large": Path("checkpoints/scaling_bf16/large/step_001139.pt"),
    "xlarge": Path("checkpoints/scaling_bf16/xlarge/step_000589.pt"),
}
PROMPTS = [
    "Once upon a time",
    "Lily found a tiny dragon",
    "The little robot was afraid because",
    "Mia opened the old wooden door and",
    "The puppy could not find its way home, so",
    "Noah wanted to share his toy, but",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a repeatable multi-prompt qualitative TinyStories evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="May be repeated; defaults to the fixed-compute models plus tuned medium.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/generation_suite"))
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    return device


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    if not values:
        return DEFAULT_CHECKPOINTS
    checkpoints: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"checkpoint must be NAME=PATH, received {value!r}")
        name, path = value.split("=", 1)
        if not name or name in checkpoints:
            raise ValueError(f"checkpoint name is empty or duplicated: {name!r}")
        checkpoints[name] = Path(path)
    return checkpoints


def load_model_and_tokenizer(
    checkpoint_path: Path, device: torch.device
) -> tuple[TransformerLM, Tokenizer, dict[str, object], int]:
    with (checkpoint_path.parent / "config.json").open(encoding="utf-8") as input_file:
        config = json.load(input_file)
    dataset_path = Path(str(config["dataset"]))
    with dataset_path.open(encoding="utf-8") as input_file:
        dataset = json.load(input_file)
    tokenizer = load_tokenizer(dataset_path.parent / str(dataset["tokenizer"]))

    model = TransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
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
    return model, tokenizer, config, int(checkpoint["iteration"])


def ngrams(token_ids: list[int], order: int) -> list[tuple[int, ...]]:
    return [tuple(token_ids[index : index + order]) for index in range(len(token_ids) - order + 1)]


def summarize_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    generated_counts = [int(sample["generated_tokens"]) for sample in samples]
    summaries: dict[str, object] = {
        "samples": len(samples),
        "mean_generated_tokens": sum(generated_counts) / len(generated_counts),
        "eos_rate": sum(bool(sample["ended_with_eos"]) for sample in samples) / len(samples),
        "exact_duplicate_fraction": 1
        - len({str(sample["continuation"]) for sample in samples}) / len(samples),
    }
    for order in (1, 2, 3):
        all_ngrams: list[tuple[int, ...]] = []
        repetition_rates: list[float] = []
        for sample in samples:
            sample_ngrams = ngrams(list(sample["continuation_token_ids"]), order)
            all_ngrams.extend(sample_ngrams)
            if sample_ngrams:
                repetition_rates.append(1 - len(set(sample_ngrams)) / len(sample_ngrams))
        summaries[f"distinct_{order}"] = (
            len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0
        )
        summaries[f"mean_within_sample_repetition_{order}"] = (
            sum(repetition_rates) / len(repetition_rates) if repetition_rates else 0.0
        )
    return summaries


def write_blind_review(samples: list[dict[str, object]], output_dir: Path) -> None:
    shuffled = list(samples)
    random.Random(336).shuffle(shuffled)
    review_path = output_dir / "blind_review.csv"
    key_path = output_dir / "blind_review_key.json"
    key: list[dict[str, object]] = []
    with review_path.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = [
            "sample_id",
            "prompt",
            "text",
            "coherence_1_to_5",
            "grammar_1_to_5",
            "story_progress_1_to_5",
            "notes",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample in enumerate(shuffled, start=1):
            sample_id = f"S{index:03d}"
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "prompt": sample["prompt"],
                    "text": sample["text"],
                    "coherence_1_to_5": "",
                    "grammar_1_to_5": "",
                    "story_progress_1_to_5": "",
                    "notes": "",
                }
            )
            key.append(
                {
                    "sample_id": sample_id,
                    "model": sample["model"],
                    "decoding": sample["decoding"],
                    "seed": sample["seed"],
                }
            )
    with key_path.open("w", encoding="utf-8") as output_file:
        json.dump(key, output_file, indent=2)


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    checkpoints = parse_checkpoints(args.checkpoint)
    sample_seeds = args.sample_seed or [42, 43]
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict[str, object]] = []
    model_metadata: dict[str, dict[str, object]] = {}
    for model_name, checkpoint_path in checkpoints.items():
        print(f"Loading {model_name}: {checkpoint_path}")
        model, tokenizer, config, checkpoint_step = load_model_and_tokenizer(
            checkpoint_path, device
        )
        eos_token_id = tokenizer.special_to_id.get("<|endoftext|>")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model_metadata[model_name] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "parameters": parameter_count,
            "training_tokens": checkpoint_step
            * int(config["batch_size"])
            * int(config["context_length"]),
        }

        decoding_conditions: list[tuple[str, int | None, float]] = [("greedy", None, 0.0)]
        decoding_conditions.extend(("sampled", seed, args.temperature) for seed in sample_seeds)
        for decoding, seed, temperature in decoding_conditions:
            for prompt_index, prompt in enumerate(PROMPTS):
                generator = None
                if seed is not None:
                    # Separate deterministic streams avoid dependence on whether
                    # an earlier prompt happened to emit EOS quickly.
                    generator = torch.Generator(device=device).manual_seed(seed + prompt_index)
                prompt_ids = tokenizer.encode(prompt)
                output_ids = generate_token_ids(
                    model,
                    prompt_ids,
                    args.max_new_tokens,
                    temperature=temperature,
                    top_k=args.top_k,
                    eos_token_id=eos_token_id,
                    generator=generator,
                    precision=str(config.get("precision", "fp32")),
                )
                continuation_ids = output_ids[len(prompt_ids) :]
                all_samples.append(
                    {
                        "model": model_name,
                        "prompt": prompt,
                        "decoding": decoding,
                        "seed": seed,
                        "text": tokenizer.decode(output_ids),
                        "continuation": tokenizer.decode(continuation_ids),
                        "continuation_token_ids": continuation_ids,
                        "generated_tokens": len(continuation_ids),
                        "ended_with_eos": bool(
                            eos_token_id is not None
                            and continuation_ids
                            and continuation_ids[-1] == eos_token_id
                        ),
                    }
                )
        print(f"Finished {model_name}: {len(PROMPTS) * len(decoding_conditions)} samples")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in all_samples:
        grouped[str(sample["model"])].append(sample)
        grouped[f"{sample['model']}:{sample['decoding']}"].append(sample)
    metrics = {name: summarize_samples(samples) for name, samples in grouped.items()}

    payload = {
        "prompts": PROMPTS,
        "sample_seeds": sample_seeds,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "models": model_metadata,
        "automatic_metrics": metrics,
        "metric_caveat": (
            "Diversity, repetition, and EOS rates are diagnostics, not measures of semantic quality. "
            "Use blind_review.csv for human evaluation."
        ),
        "samples": all_samples,
    }
    results_path = args.output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
    write_blind_review(all_samples, args.output_dir)
    print(f"Saved {results_path}")
    print(f"Saved {args.output_dir / 'blind_review.csv'}")
    print(f"Saved {args.output_dir / 'blind_review_key.json'}")


if __name__ == "__main__":
    main()
