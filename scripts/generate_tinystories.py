from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tiny_transformer.generation import generate_token_ids
from tiny_transformer.model import TransformerLM
from tiny_transformer.tokenizer_io import load_tokenizer


DEFAULT_PROMPTS = [
    "Once upon a time",
    "Lily went into the forest",
    "The little robot was afraid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a TinyStories checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/tinystories_laptop/step_001000.pt"),
    )
    parser.add_argument("--prompt", action="append", help="May be supplied more than once.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="Defaults to generations.json beside the checkpoint.")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    return device


def main() -> None:
    args = parse_args()
    run_dir = args.checkpoint.parent
    with (run_dir / "config.json").open(encoding="utf-8") as input_file:
        config = json.load(input_file)

    dataset_path = Path(config["dataset"])
    with dataset_path.open(encoding="utf-8") as input_file:
        dataset_metadata = json.load(input_file)
    tokenizer_path = dataset_path.parent / dataset_metadata["tokenizer"]
    tokenizer = load_tokenizer(tokenizer_path)

    device = choose_device(args.device)
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    eos_token_id = tokenizer.special_to_id.get("<|endoftext|>")
    prompts = args.prompt or DEFAULT_PROMPTS
    print(
        f"Checkpoint step {checkpoint['iteration']} on {device}; "
        f"temperature={args.temperature}, top_k={args.top_k}"
    )

    samples = []
    for index, prompt in enumerate(prompts, start=1):
        # Give each prompt an independent deterministic random stream. This
        # makes prompt-by-prompt comparisons fair even if another sample ends
        # early after producing the end-of-text token.
        generator = torch.Generator(device=device).manual_seed(args.seed + index - 1)
        prompt_ids = tokenizer.encode(prompt)
        generated_ids = generate_token_ids(
            model,
            prompt_ids,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=eos_token_id,
            generator=generator,
            precision=config.get("precision", "fp32"),
        )
        generated_text = tokenizer.decode(generated_ids)
        samples.append(
            {
                "prompt": prompt,
                "text": generated_text,
                "generated_tokens": len(generated_ids) - len(prompt_ids),
            }
        )
        print(f"\n--- Sample {index} ---\n{generated_text}")

    output_path = args.output or run_dir / "generations.json"
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["iteration"]),
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "samples": samples,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
    print(f"\nSaved samples: {output_path}")


if __name__ == "__main__":
    main()
