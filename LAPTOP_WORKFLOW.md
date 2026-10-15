# TinyStories laptop workflow

The repository now contains the complete path from raw text to a trainable
decoder-only language model. The defaults are intentionally small enough to
exercise the workflow on a CPU laptop; they are not intended to produce a good
language model.

## 1. Download TinyStories

The two raw files currently live under `data/`. On a fresh checkout they can be
downloaded from PowerShell with:

```powershell
New-Item -ItemType Directory -Force data
curl.exe -L --output data/TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl.exe -L --output data/TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

## 2–3. Train the tokenizer and encode token arrays

From the repository root, run:

```powershell
uv run python scripts/prepare_tinystories.py
```

By default this command:

- learns a 2,000-entry byte-level BPE vocabulary from 5 MB of training text;
- encodes 20 MB of training text and 2 MB of validation text;
- writes compact `uint16` arrays and metadata under `data/tinystories_laptop/`.

The original downloaded files stay unchanged. To use more data later, pass
larger values to `--tokenizer-bytes`, `--train-bytes`, and `--valid-bytes`.

## 4. Train the small model

```powershell
uv run python scripts/train_tinystories.py
```

The default model has four layers, model width 128, four attention heads, a
context length of 128, and about 1.4 million parameters. It trains for 1,000
steps and saves checkpoints under `checkpoints/tinystories_laptop/`.

For a quick end-to-end smoke test:

```powershell
uv run python scripts/train_tinystories.py --steps 1 --warmup-steps 0 --eval-batches 1
```

To continue an interrupted run, keep the same architecture and raise the total
step count:

```powershell
uv run python scripts/train_tinystories.py --steps 1500 --resume checkpoints/tinystories_laptop/step_001000.pt
```

The project is pinned to PyTorch's official CUDA 12.8 wheel index. On this
laptop, `--device auto` detects the RTX 4050 and selects it automatically. Pass
`--device cpu` only when you deliberately want a CPU run.

## Inspect and compare experiments

Training writes one JSON record per step to `metrics.jsonl`. Plot that log with:

```powershell
uv run python scripts/plot_training.py --metrics checkpoints/tinystories_laptop/metrics.jsonl
```

Generate three reproducible sample stories from a checkpoint with:

```powershell
uv run python scripts/generate_tinystories.py --checkpoint checkpoints/tinystories_laptop/step_001000.pt
```

The text is printed and saved as `generations.json` beside the checkpoint. Use
the same prompts, temperature, top-k value, and seed when comparing models.

The controlled GPU benchmark executes complete forward, backward, and AdamW
update steps. It allows its process to use at most 80% of VRAM and stops cleanly
at the first out-of-memory result:

```powershell
uv run python scripts/benchmark_gpu.py
```

Its machine-readable report is written to `benchmarks/gpu_memory.json`.
