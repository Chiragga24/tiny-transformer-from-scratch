# Transformer Language Model from Scratch

This is a from-scratch language-modeling project that implements a small
decoder-only Transformer and its supporting training pipeline in PyTorch, then
studies what can be trained on an RTX 4050 laptop GPU.

The project is not a pretrained-model fine-tuning exercise. The tokenizer,
model layers, optimizer, loss, batching, checkpointing, training loop, and text
generation path are implemented in this repository.

> This is an independent learning and experimentation project.

For a detailed, code-oriented explanation of the complete pipeline, see the
[Tiny Transformer Implementation Guide](IMPLEMENTATION_GUIDE.md).

## What is implemented

- byte-level BPE training and tokenization, including special tokens;
- token embeddings, linear layers, RMSNorm, SwiGLU, and RoPE;
- causal scaled dot-product attention and multi-head self-attention;
- pre-norm decoder blocks and a complete decoder-only language model;
- numerically stable cross-entropy and global gradient clipping;
- AdamW and a warmup-plus-cosine learning-rate schedule;
- memory-mapped data loading, checkpoint save/resume, BF16 training, and JSONL logging;
- deterministic full-validation evaluation, text generation, plotting, and GPU benchmarks;
- Windows-compatible tests and workflow scripts.

The final test result is **46 passed, 2 skipped**. The two skipped tests depend
on Unix resource-limit APIs that Windows does not provide.

## Hardware and experiment setup

Experiments ran locally on an NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM using
PyTorch 2.11, CUDA 12.8, and BF16. The tokenizer has 2,000 entries. Unless a
control changes one of these values, training uses 2,048 tokens per update,
AdamW with weight decay 0.1, gradient clipping at 1.0, and a cosine schedule.

Four model sizes were tested:

| Model | Parameters | Width | Layers | Heads | FFN width |
|---|---:|---:|---:|---:|---:|
| Small | 6.14M | 256 | 6 | 4 | 768 |
| Medium | 15.70M | 384 | 8 | 6 | 1,024 |
| Large | 26.96M | 512 | 8 | 8 | 1,344 |
| Xlarge | 52.14M | 640 | 10 | 10 | 1,728 |

## Main result: the experimental protocol matters

The first experiment held `parameters * training tokens` approximately fixed.
That favored the small model because larger models received fewer updates. A
second experiment gave every model 5,000 steps and 10.24M tokens. That reversed
the ranking, but also increased compute with model size.

![Fixed-compute versus equal-token scaling](docs/figures/01_protocol_comparison.png)

This is the central lesson from the project: "best model" is not meaningful
until the compute constraint and training-token budget are stated precisely.

## Dataset scaling

To test whether the original 20 MB training subset caused excessive repetition,
the same small model was trained with nested 20 MB, 100 MB, and 500 MB corpus
prefixes. Every condition still sampled exactly 10.24M training tokens.

![Dataset scaling](docs/figures/02_data_scaling.png)

More available data helped monotonically, but only slightly: full-validation
loss moved from 2.1963 to 2.1815. At this short training horizon, simply making
the pool larger was less important than optimization and total compute.

## Experimental controls

The xlarge model originally used context 512 and batch 4, while the other models
used context 256 and batch 8. Retraining xlarge with context 256 improved loss
by 0.0765, but did not explain its entire fixed-compute gap. Replacing 20 random
validation batches with all 544,256 non-overlapping validation targets also
confirmed that the original ranking was not a sampling accident.

![Context and validation controls](docs/figures/03_experimental_controls.png)

## Optimizer sensitivity

The medium model was tested at three peak learning rates and two warmup lengths
while its architecture, data, seed, and 4.00M-token budget stayed fixed.

![Learning-rate and warmup sweep](docs/figures/04_schedule_sweep.png)

Peak learning rate mattered much more than changing warmup from 5% to 10%.
Raising the peak from `3e-4` to `6e-4` improved medium-model loss from 2.3927 to
2.1902. The tuned medium model narrowly beat the small model's 2.1963 under the
same approximate parameter-token compute budget. Because both were trained with
only one seed, that 0.006 difference is not yet statistically convincing.

## Laptop GPU limit

The controlled benchmark runs a complete BF16 forward pass, backward pass, and
AdamW update while limiting the process to 80% of VRAM.

![GPU memory benchmark](docs/figures/05_gpu_memory.png)

An 88M-parameter stress model fit below the guard at about 3.91 GiB allocated.
The next 1024-wide configuration exceeded the 4.80 GiB safety cap and ran out of
memory. This benchmark measures one step fitting in memory, not whether a model
is practical to train for hours.

## Generation evaluation

Qualitative evaluation was expanded from three examples to 90 generations:
five checkpoints, six prompts, greedy decoding, and two fixed sampling seeds.
Automatic diagnostics measure termination, diversity, and surface repetition.

![Generation diagnostics](docs/figures/06_generation_diagnostics.png)

These metrics are not semantic-quality scores. Generated stories still contain
identity, role, and causal inconsistencies. A shuffled blind-review CSV is
produced for human ratings of coherence, grammar, and story progression.

## Decisions made during the project

The work evolved in stages rather than jumping directly to a large training run:

1. Implement and test each mathematical component independently.
2. Make the inherited test suite Windows-compatible without weakening Unix checks.
3. Build the smallest complete tokenizer-to-generation workflow first.
4. Benchmark BF16 memory before choosing model sizes.
5. Use an 80% VRAM cap so experiments fail safely instead of destabilizing the laptop.
6. Start with a fixed parameter-token budget using the common `6NT` compute proxy.
7. Record actual time, throughput, and memory after observing that laptop clock and power states vary.
8. Add equal-step, context-length, dataset-size, and validation controls when the first comparison exposed confounds.
9. Sweep optimizer settings after recognizing that one schedule may not be fair to every model size.
10. Replace a few anecdotal generations with fixed prompts, multiple seeds, diagnostics, and blind human review.

The complete rationale, experiment tables, interpretations, and limitations are
in [EXPERIMENT_REPORT.md](./EXPERIMENT_REPORT.md). The compact, tracked
[results snapshot](./docs/results_summary.json) contains the values used by the
figures, while large raw logs and checkpoints remain local.

## Running the project

Install dependencies and run the tests:

```powershell
uv sync
uv run pytest
```

Place the TinyStories train and validation text files in `data/`, then train the
2,000-token BPE and prepare the default laptop subset:

```powershell
uv run python scripts/prepare_tinystories.py
```

Train the default smoke-test model:

```powershell
uv run python scripts/train_tinystories.py
```

Train the 6.14M-parameter BF16 small configuration used in the experiments:

```powershell
uv run python scripts/train_tinystories.py `
  --dataset data/tinystories_laptop/dataset.json `
  --output-dir checkpoints/scaling_bf16/small `
  --precision bf16 --context-length 256 --d-model 256 `
  --num-layers 6 --num-heads 4 --d-ff 768 `
  --batch-size 8 --steps 5000 --warmup-steps 500 `
  --eval-interval 500 --eval-batches 20 --checkpoint-interval 5000
```

Generate text or run deterministic evaluation:

```powershell
uv run python scripts/generate_tinystories.py --checkpoint checkpoints/scaling_bf16/small/step_005000.pt
uv run python scripts/evaluate_tinystories.py --checkpoint checkpoints/scaling_bf16/small/step_005000.pt
```

Regenerate the GitHub figures after running the experiments:

```powershell
uv run python scripts/make_github_figures.py
```

Large corpora, encoded arrays, checkpoints, and raw benchmark logs are ignored
by Git. The summary figures under `docs/figures/` are intentionally tracked.

## Repository guide

| Path | Purpose |
|---|---|
| `IMPLEMENTATION_GUIDE.md` | Detailed explanation of the code, equations, shapes, and workflow |
| `tiny_transformer/model.py` | Transformer layers and complete language model |
| `tiny_transformer/tokenizer.py` | Byte-level BPE tokenizer |
| `tiny_transformer/bpe.py` | BPE training |
| `tiny_transformer/optimizer.py` | AdamW and cosine schedule |
| `tiny_transformer/nn_utils.py` | Cross-entropy and gradient clipping |
| `scripts/prepare_tinystories.py` | Tokenizer training and streaming corpus encoding |
| `scripts/train_tinystories.py` | Training, evaluation, logging, and checkpointing |
| `scripts/evaluate_tinystories.py` | Deterministic full-validation evaluation |
| `scripts/evaluate_generations.py` | Multi-prompt generation suite and blind-review sheet |
| `scripts/benchmark_gpu.py` | Guarded BF16 GPU benchmark |
| `scripts/make_github_figures.py` | Rebuilds the tracked summary figures |

## License

This repository is distributed under the included MIT license.
