# Tiny Transformer: Detailed Implementation Guide

This document explains what the repository implements, how data moves through
it, why the equations have their particular form, and where each idea appears
in code. It is intended to be read beside the source rather than memorized.

The fastest way to use it is:

1. Read the model flow and tensor-shape sections once.
2. Open the linked implementation file for the component you are studying.
3. Trace one small tensor through the equations and the corresponding code.
4. Return to the training and data sections to see how the components interact
   in a complete language-model workflow.

## Repository map

| File | What it represents |
|---|---|
| [`tiny_transformer/model.py`](tiny_transformer/model.py) | Neural-network primitives, attention, RoPE, Transformer blocks, and the complete language model |
| [`tiny_transformer/nn_utils.py`](tiny_transformer/nn_utils.py) | Stable cross-entropy and global gradient clipping |
| [`tiny_transformer/optimizer.py`](tiny_transformer/optimizer.py) | AdamW and the warmup/cosine learning-rate schedule |
| [`tiny_transformer/data.py`](tiny_transformer/data.py) | Random next-token batch sampling from an encoded token array |
| [`tiny_transformer/bpe.py`](tiny_transformer/bpe.py) | Training a byte-level BPE vocabulary and ordered merge list |
| [`tiny_transformer/tokenizer.py`](tiny_transformer/tokenizer.py) | Encoding and decoding text with the learned BPE rules |
| [`tiny_transformer/tokenizer_io.py`](tiny_transformer/tokenizer_io.py) | Portable JSON serialization for arbitrary token bytes |
| [`tiny_transformer/serialization.py`](tiny_transformer/serialization.py) | Model/optimizer checkpoint save and restore |
| [`tiny_transformer/generation.py`](tiny_transformer/generation.py) | Greedy and top-k autoregressive generation |
| [`scripts/prepare_tinystories.py`](scripts/prepare_tinystories.py) | Streaming corpus preparation and compact binary token storage |
| [`scripts/train_tinystories.py`](scripts/train_tinystories.py) | End-to-end training, evaluation, logging, and checkpointing |
| [`scripts/evaluate_tinystories.py`](scripts/evaluate_tinystories.py) | Deterministic evaluation over nearly the complete validation split |
| [`scripts/generate_tinystories.py`](scripts/generate_tinystories.py) | Reconstructing a model from its run configuration and generating samples |
| [`scripts/benchmark_gpu.py`](scripts/benchmark_gpu.py) | Controlled GPU-memory and throughput measurement |
| [`tests/`](tests) | Executable shape, numerical, and behavioral specifications |

The project intentionally implements the learning-critical pieces instead of
wrapping `nn.Linear`, `nn.Embedding`, `nn.MultiheadAttention`,
`torch.optim.AdamW`, or a pretrained language-model class. PyTorch still
provides tensors, automatic differentiation, devices, mixed precision, and the
base `nn.Module`/`Optimizer` machinery.

## 1. Notation and shapes

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `T` | sequence/context length |
| `V` | vocabulary size |
| `D` | model width (`d_model`) |
| `H` | number of attention heads |
| `Dh` | head width, `D / H` |
| `F` | feed-forward width (`d_ff`) |
| `L` | number of Transformer blocks |

The main tensors are:

```text
token IDs       (B, T)
embeddings      (B, T, D)
Q, K, V         (B, H, T, Dh)
attention       (B, H, T, T)
hidden states   (B, T, D)
logits          (B, T, V)
targets         (B, T)
```

Two required divisibility checks:

```text
D % H == 0
Dh % 2 == 0       # RoPE rotates pairs of features
```

### Shape convention used throughout the code

The final axis always contains features. The axis immediately before it is
normally sequence position. Any axes before those are treated as leading batch
dimensions and are preserved by linear layers and normalization. This is why a
single expression such as `x @ weight.T` works for `(D,)`, `(T, D)`, and
`(B, T, D)` inputs.

Attention temporarily inserts the head axis:

```text
ordinary hidden state: (..., T, D)
split attention state: (..., H, T, Dh)
```

Using `...` in the equations means the implementation does not need separate
logic for a single sequence and a batch of sequences.

### The PyTorch concepts the implementation relies on

`nn.Module` is more than a convenient parent class. Calling `super().__init__()`
creates the bookkeeping used to:

- discover nested parameters through `model.parameters()`;
- move a complete model between CPU and GPU with `.to(...)`;
- switch behavior with `.train()` and `.eval()`;
- serialize named tensors through `state_dict()`.

`nn.Parameter` marks a tensor as trainable. A tensor assigned as a normal
attribute does not automatically appear in `model.parameters()`. This matters
for the linear weights, embeddings, and RMSNorm scales.

A registered buffer is model state that should move with the model but should
not receive gradients. The cached RoPE sine and cosine tables are buffers:
they depend on configuration, not training.

Autograd records tensor operations during the forward pass. Calling
`loss.backward()` follows that graph in reverse and accumulates gradients in
each parameter's `.grad`. The optimizer reads those gradients; it does not
differentiate the model itself.

## 2. The complete model in one picture

```text
token IDs
   |
token embedding lookup
   |
   +-------------------------------- residual stream -------------------------+
   |                                                                          |
   |  repeat L times:                                                         |
   |    z = x + causal_attention(RMSNorm(x))                                  |
   |    x = z + SwiGLU(RMSNorm(z))                                            |
   |                                                                          |
   +--------------------------------------------------------------------------+
   |
final RMSNorm
   |
linear projection D -> V
   |
logits (no final softmax during training)
```

This is a decoder-only, causal, pre-norm Transformer. "Pre-norm" means the
normalization occurs before each attention or feed-forward sublayer.

### A concrete forward-pass shape trace

Suppose:

```text
B = 8       batch size
T = 256     tokens per sequence
V = 2,000   vocabulary size
D = 384     model width
H = 6       attention heads
Dh = 64     features per head
F = 1,024   feed-forward width
L = 8       Transformer blocks
```

The forward pass is:

1. `token_ids` begins as `(8, 256)` integer indices.
2. Embedding lookup produces `(8, 256, 384)` floating-point vectors.
3. Q, K, and V projections each preserve `(8, 256, 384)`.
4. Splitting heads changes each to `(8, 6, 256, 64)`.
5. `Q @ K.transpose(-2, -1)` produces `(8, 6, 256, 256)` scores.
6. Multiplying the attention probabilities by V returns `(8, 6, 256, 64)`.
7. Joining heads returns `(8, 256, 384)` so the residual addition is valid.
8. SwiGLU expands two branches to `(8, 256, 1,024)`, gates them
   elementwise, and projects back to `(8, 256, 384)`.
9. Each of the eight blocks repeats steps 3-8 without changing the residual
   stream shape.
10. The vocabulary head maps `(8, 256, 384)` to `(8, 256, 2,000)` logits.
11. Cross-entropy reshapes the logits to `(2,048, 2,000)` and the targets to
    `(2,048)`, producing one average scalar loss.

That final scalar depends on every selected token prediction. Backpropagation
therefore sends gradients through the vocabulary head, every residual block,
and the embedding rows used by the batch.

## 3. Linear layer

Store the weight as:

$$W \in \mathbb{R}^{D_{out} \times D_{in}}$$

Then compute:

$$y = xW^T$$

```text
x:      (..., Din)
W:      (Dout, Din)
output: (..., Dout)
```

This project uses no linear biases. Its initialization is a truncated normal:

$$\sigma = \sqrt{\frac{2}{D_{in}+D_{out}}}$$

Memory cue: a linear layer mixes features; it does not mix batch or sequence
positions.

## 4. Token embeddings

An embedding table is a learned matrix:

$$E \in \mathbb{R}^{V \times D}$$

Token ID `i` selects row $E_i$:

```text
IDs:          (B, T)
E[IDs]:       (B, T, D)
```

The output is not a newly sampled random tensor on every forward pass. The
table starts random, becomes a model parameter, and is learned by gradient
descent.

### What initialization is doing

`torch.empty` allocates memory but does not initialize meaningful values. The
constructor immediately fills that memory with a truncated normal
distribution. Truncation prevents unusually large initial weights from
creating extreme activations before training begins.

The linear-layer scale depends on both input and output widths so activation
and gradient magnitudes remain reasonably stable across projections. The
embedding table uses a fixed standard deviation of one in this implementation.
Those choices affect only the starting point; after the first optimizer step,
they are ordinary learned parameters.

The vocabulary output matrix is separate from the input embedding matrix. This
is called **untied embeddings**. Weight tying could reuse the same matrix and
reduce parameters, but this model deliberately keeps the two roles explicit.

## 5. RMSNorm

For a vector $x \in \mathbb{R}^{D}$:

$$\mathrm{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2 + \epsilon}$$

$$\mathrm{RMSNorm}(x) = g \odot \frac{x}{\mathrm{RMS}(x)}$$

where $g \in \mathbb{R}^{D}$ is a learned scale initialized to ones.

RMSNorm has:

- no mean subtraction;
- no learned additive bias;
- normalization only across the final feature dimension;
- float32 accumulation for stability when activations use BF16/FP16.

Memory cue: LayerNorm recenters and rescales; RMSNorm only rescales.

The implementation casts to float32 only for the square/mean/root calculation,
then converts the normalized result back to the input dtype. This is a common
mixed-precision pattern: keep large tensors in BF16 to save memory, but perform
sensitive reductions in FP32.

## 6. SiLU and SwiGLU

SiLU is:

$$\mathrm{SiLU}(x) = x\,\sigma(x)$$

SwiGLU uses two parallel projections with the same shape:

$$\mathrm{SwiGLU}(x)
= W_2\left(\mathrm{SiLU}(W_1x) \odot W_3x\right)$$

```text
x                    (..., D)
W1(x), W3(x)         (..., F)
elementwise product  (..., F)
W2(...)              (..., D)
```

Memory cue: one branch proposes values, the other gates them, and the last
projection returns to the residual-stream width.

## 7. Scaled dot-product attention

$$\mathrm{Attention}(Q,K,V)
= \mathrm{softmax}\left(\frac{QK^T}{\sqrt{D_h}} + M\right)V$$

For every head:

```text
Q:                   (B, H, T, Dh)
K transpose:         (B, H, Dh, T)
Q @ K transpose:     (B, H, T, T)
softmax over keys:   dim = -1
result @ V:          (B, H, T, Dh)
```

Why divide by $\sqrt{D_h}$? Dot-product variance grows with the number of
features. Scaling keeps logits in a range where softmax still has useful
gradients.

Why softmax over the last dimension? Every query needs a probability
distribution over all keys it may read.

### Stable softmax

$$\mathrm{softmax}(x)_i
= \frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\qquad m=\max_j x_j$$

Subtracting the maximum changes neither the probabilities nor their ordering,
but prevents exponentials from overflowing.

## 8. Causal masking

For next-token prediction, position `i` may read positions `0...i`, but not
future positions:

```text
allowed mask for T = 4

1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Set forbidden attention scores to $-\infty$ before softmax. Their resulting
probabilities become zero.

The targets are shifted by one token, while causal masking prevents the model
from looking directly at those future answers.

## 9. Multi-head self-attention

1. Project the same hidden states into `Q`, `K`, and `V`.
2. Split the final `D` features into `H` heads of width `Dh`.
3. Apply RoPE to `Q` and `K`.
4. Run causal attention independently for each head.
5. Join the heads back into width `D`.
6. Apply the output projection.

```text
(B, T, D)
    -> project Q/K/V
(B, T, D)
    -> reshape and transpose
(B, H, T, Dh)
    -> attention
(B, H, T, Dh)
    -> transpose and reshape
(B, T, D)
```

Heads do not make the total width larger: `H * Dh = D`.

### Why the transpose and reshape order matters

The projection initially lays out features as one contiguous width `D`. The
code first reshapes:

```text
(..., T, D) -> (..., T, H, Dh)
```

and then transposes the head and sequence axes:

```text
(..., T, H, Dh) -> (..., H, T, Dh)
```

Attention treats all leading axes as independent batches, so every head now
runs its own attention calculation. After attention, the inverse transpose is
followed by `reshape`, which safely handles the non-contiguous view created by
`transpose`.

The causal mask has shape `(T, T)`, not `(B, H, T, T)`. PyTorch broadcasting
reuses that same logical mask for every sequence and head. A mask entry is
`True` when reading is allowed; the implementation negates it before
`masked_fill` so forbidden entries receive negative infinity.

## 10. RoPE

RoPE encodes position by rotating adjacent feature pairs in queries and keys.
For pair index $i$ and token position $p$:

$$\omega_i = \theta^{-2i/D_h}$$

$$\phi_{p,i} = p\omega_i$$

For a feature pair $(a,b)$:

$$
\begin{aligned}
a' &= a\cos\phi - b\sin\phi \\
b' &= a\sin\phi + b\cos\phi
\end{aligned}
$$

Feature pairs are adjacent coordinates such as `(0,1)`, `(2,3)`, and so on.

- Position zero has angle zero, so its rotation is the identity.
- Different feature pairs rotate at different frequencies.
- RoPE changes `Q` and `K`, because position should affect where attention
  looks.
- `V` is not rotated, because it contains the information retrieved after
  attention chooses where to look.

Memory cue: RoPE rotates the address (`Q`, `K`), not the payload (`V`).

### How the RoPE cache broadcasts

The constructor precomputes cosine and sine tables of shape `(max_seq_len,
Dh/2)`. At runtime, indexing by `token_positions` selects `(T, Dh/2)`. The
attention input is `(B, H, T, Dh)`, so singleton dimensions are inserted before
the sequence axis until the cache has shape `(1, 1, T, Dh/2)`. Broadcasting
then applies the same position rotations to every batch item and head without
copying the table.

The cache is created in float32 for accurate trigonometry and converted to the
activation dtype only when applied.

## 11. Transformer block

The two pre-norm residual updates are:

$$z = x + \mathrm{MHA}(\mathrm{RMSNorm}_1(x))$$

$$y = z + \mathrm{SwiGLU}(\mathrm{RMSNorm}_2(z))$$

Important details:

- The second residual starts from `z`, not the original `x`.
- A residual addition requires matching shapes.
- Attention and SwiGLU both return `(B, T, D)`.
- There is no post-norm inside this block.

## 12. Language-model output

After all blocks:

$$\text{logits} = W_{vocab}\mathrm{RMSNorm}(x)$$

```text
hidden states: (B, T, D)
lm_head:       D -> V
logits:        (B, T, V)
```

Do not apply softmax before cross-entropy. Cross-entropy consumes logits and
computes the stable normalization internally. Softmax is needed during
generation when sampling a probability distribution.

## 13. Next-token training data

From a token window:

```text
tokens:  [10, 20, 30, 40]
input x: [10, 20, 30]
target y:[20, 30, 40]
```

For batch size `B` and context length `T`:

```text
x: (B, T)
y: (B, T)
```

Every position supplies a training target, so one optimizer step processes:

$$\text{tokens per step} = B \times T$$

## 14. Cross-entropy

For logits $z \in \mathbb{R}^{V}$ and correct token $y$:

$$\mathcal{L}(z,y) = \log\sum_{j=1}^{V}e^{z_j} - z_y$$

Stable log-sum-exp:

$$\log\sum_j e^{z_j}
= m + \log\sum_j e^{z_j-m},\qquad m=\max_j z_j$$

During training, flatten only the batch and sequence axes:

```text
logits: (B, T, V) -> (B*T, V)
targets:(B, T)    -> (B*T)
```

Then average the per-token loss.

Perplexity is:

$$\mathrm{PPL} = e^{\text{cross-entropy}}$$

Lower validation loss and perplexity are better. Compare models using the same
tokenizer and evaluation data.

## 15. Global gradient clipping

Treat all parameter gradients as one long vector:

$$\|g\|_2 = \sqrt{\sum_p\sum_i (g_{p,i})^2}$$

Use one shared scale factor:

$$c = \min\left(1,\frac{M}{\|g\|_2+\epsilon}\right)$$

$$g_p \leftarrow c\,g_p$$

where $M$ is the maximum norm.

The shared factor preserves the overall gradient direction. Clip after
`loss.backward()` and before `optimizer.step()`.

## 16. AdamW

For gradient $g_t$:

$$m_t = \beta_1m_{t-1} + (1-\beta_1)g_t$$

$$v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2$$

Bias correction compensates for moments initialized at zero:

$$\hat m_t = \frac{m_t}{1-\beta_1^t}$$

$$\hat v_t = \frac{v_t}{1-\beta_2^t}$$

The decoupled weight-decay update is:

$$\theta_t = (1-\eta_t\lambda)\theta_{t-1}
- \eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

Memory cues:

- `m` remembers direction: an exponential mean of gradients.
- `v` remembers scale: an exponential mean of squared gradients.
- bias correction fixes the cold start.
- AdamW shrinks weights separately from the gradient update.

## 17. Warmup and cosine learning rate

During warmup:

$$\eta_t = \eta_{max}\frac{t}{T_{warmup}}$$

During cosine decay, define:

$$r = \frac{t-T_{warmup}}{T_{cosine}-T_{warmup}}$$

$$\eta_t = \eta_{min}
+ \frac{1}{2}(1+\cos(\pi r))(\eta_{max}-\eta_{min})$$

After the cosine interval, keep the learning rate at $\eta_{min}$.

```text
learning rate
     /\
    /  \__
   /      \____ minimum
  warmup   cosine decay
```

Warmup prevents unstable large updates before Adam's moments and the model's
activations have settled.

## 18. One training step

```python
lr = schedule(step)
set_optimizer_lr(lr)

optimizer.zero_grad(set_to_none=True)
x, y = get_batch(...)

with autocast_if_using_bf16():
    logits = model(x)                         # (B, T, V)
    loss = cross_entropy(
        logits.reshape(-1, vocab_size),       # (B*T, V)
        y.reshape(-1),                        # (B*T)
    )

loss.backward()
clip_gradients(model.parameters(), max_norm)
optimizer.step()
```

The order to remember is:

```text
zero -> batch -> forward -> loss -> backward -> clip -> step
```

Evaluation uses `model.eval()` and `torch.no_grad()`. Training uses
`model.train()`.

## 19. Byte-level BPE

### Training

1. Start with all 256 individual byte tokens.
2. Add indivisible special tokens.
3. Split text around special tokens.
4. Pre-tokenize ordinary text with a regex.
5. Count adjacent byte-pair frequencies, weighted by pre-token frequency.
6. Merge the most frequent pair.
7. Add the merged bytes to the vocabulary.
8. Repeat until reaching the requested vocabulary size.

One merge creates one new vocabulary entry; 100 merge operations therefore
create 100 new entries, not 100 unrelated pairs per word.

### Encoding

1. Separate special tokens so merges cannot cross them.
2. Pre-tokenize ordinary text.
3. Convert each pre-token to its UTF-8 bytes.
4. Repeatedly apply the currently available pair with the earliest learned
   merge rank.
5. Map the final byte pieces to token IDs.

### Decoding

Concatenate token bytes first, then decode the complete byte string as UTF-8.
Do not decode tokens separately because one Unicode character may span several
tokens.

Memory cue: training learns the merge order; encoding replays that priority.

### Why bytes are the base vocabulary

Starting with all 256 byte values guarantees coverage. Any UTF-8 text can be
expressed even if a character or word never appeared during tokenizer
training. A character such as `é` may initially occupy multiple byte tokens;
frequent byte sequences can later become a single learned BPE token.

This also explains why vocabulary entries are stored as `bytes`, not Python
strings. Some valid token fragments are not independently valid UTF-8 text.

### How `train_bpe` avoids recounting the entire corpus

The trainer stores each unique pre-token once along with its corpus frequency.
For every adjacent pair, it tracks:

- the pair's total frequency, weighted by pre-token frequency;
- the IDs of the unique pre-tokens containing that pair.

When the best pair is merged, only pre-tokens containing that pair can change.
The algorithm subtracts their old pair contributions and adds their new ones.
This inverted index is why each iteration does not need to rescan all corpus
text.

If pair frequencies tie, the implementation uses the lexicographically larger
byte pair. A deterministic tie-break is important: otherwise the learned merge
order, token IDs, encoded dataset, and all later checkpoints could differ
between runs.

### Special-token handling

Special tokens are indivisible. Before ordinary pre-tokenization, the trainer
and encoder split text around their markers. Sorting alternatives by length
prevents an overlapping shorter marker from consuming the prefix of a longer
one.

The TinyStories pipeline uses `<|endoftext|>` as a story boundary. It gives the
model an explicit signal that one document has ended and another may begin.

### Tokenizer JSON format

Arbitrary bytes cannot safely be inserted directly into a UTF-8 JSON string.
[`tokenizer_io.py`](tiny_transformer/tokenizer_io.py) therefore Base64-encodes
each vocabulary entry and both sides of every merge. The file records:

```text
format version
token ID -> Base64 bytes
ordered merge pairs
special-token strings
```

Loading reverses Base64 and reconstructs the exact `Tokenizer` object. The
explicit format version makes future incompatible changes detectable.

## 20. Autoregressive generation

At each step:

1. Keep at most the last `context_length` token IDs.
2. Run the model.
3. Select logits from the final position: `logits[:, -1, :]`.
4. Divide by temperature.
5. Optionally keep only the top `k` logits.
6. Apply softmax.
7. Sample one token and append it.
8. Stop at the end token or the maximum generated length.

Temperature effects:

```text
temperature < 1   sharper, safer, more repetitive
temperature = 1   unchanged distribution
temperature > 1   flatter, more varied, less reliable
```

## 21. Parameter and memory intuition

Ignoring small normalization terms, an untied model has approximately:

$$N \approx 2VD + L(4D^2 + 3DF)$$

- `2VD`: input embeddings plus output vocabulary projection;
- `4D^2`: Q, K, V, and attention output projections per block;
- `3DF`: the three SwiGLU matrices per block.

Attention-score memory grows roughly as:

$$O(BHT^2)$$

This quadratic dependence makes context length particularly expensive. BF16
reduces tensor memory, but it does not change the number of trainable
parameters or the mathematical amount of work.

## 22. Experiment comparisons

Always state what is held constant:

| Protocol | Held constant | What it answers |
|---|---|---|
| Equal steps/tokens | optimizer steps and tokens seen | Which model learns best from the same data exposure? |
| Equal wall time | elapsed training time | Which model is best under a time budget? |
| Approximate fixed compute | parameters multiplied by tokens | How should a rough compute budget be allocated? |
| Equal VRAM | memory ceiling | Which configuration works within the same hardware limit? |

Do not call a comparison "fair" without naming the constraint. Track at least:

- validation cross-entropy and perplexity;
- tokens processed;
- optimizer steps;
- elapsed time;
- peak GPU memory;
- model parameter count;
- several fixed generation prompts.

## 23. Frequent mistakes

- Using `x @ W` when the stored weight is `(Dout, Din)`; use `x @ W.T`.
- Treating embeddings as fresh random vectors instead of learned table rows.
- Applying attention softmax over queries instead of keys.
- Forgetting the $\sqrt{D_h}$ attention scale.
- Using a mask with the wrong orientation or meaning.
- Rotating values with RoPE; rotate only queries and keys.
- Adding residual tensors whose final dimensions do not match.
- Starting the second block residual from `x` instead of `z`.
- Applying softmax before training cross-entropy.
- Clipping each parameter independently instead of using one global norm.
- Applying gradient clipping before `backward()`.
- Implementing L2 regularization inside Adam moments instead of decoupled AdamW
  decay.
- Comparing losses from different tokenizers as though they were directly
  equivalent.
- Reporting only generated samples without deterministic validation loss.

## 24. How to remember the equations

Reconstruct them from their jobs:

```text
RMSNorm:     divide by typical magnitude, then learned scale
attention:   similarity -> probability -> weighted information
RoPE:        rotate each feature pair by a position-dependent angle
SwiGLU:      activated values * gate, then project back
cross entropy: all-class log score - correct-class score
clipping:    shrink only when the global norm is too large
AdamW:       smooth direction / smooth scale + separate shrinkage
schedule:    line up, cosine down, then hold
```

If the purpose, input shape, and output shape are clear, most implementation
details can be derived instead of memorized.

## 25. How raw text becomes a memory-mapped dataset

[`scripts/prepare_tinystories.py`](scripts/prepare_tinystories.py) connects the
tokenizer implementation to a dataset large enough that loading every token as
a Python object would be wasteful.

### Preserve complete story boundaries

`iter_story_chunks` reads binary blocks from the corpus but emits text only up
to the last complete `<|endoftext|>` marker. Any incomplete final story remains
in `pending` and is prefixed to the next disk read.

This solves two problems:

1. A UTF-8 character is never decoded from an incomplete byte sequence.
2. A tokenizer pre-token is never split differently merely because it crossed
   an arbitrary filesystem read boundary.

When a byte limit is requested for a laptop-scale subset, the code may read up
to that budget but still emits only complete stories. When the entire file is
requested, it also emits a final document even if the file lacks a trailing
separator.

### Train once, encode much more data

BPE training is comparatively expensive in pure Python and currently reads its
bounded sample into memory. The workflow therefore learns merges from a small,
complete-story sample and then reuses that tokenizer to stream larger training
and validation subsets.

The optional `tiktoken` encoding backend is an acceleration, not a different
tokenizer. It is constructed from the exact learned vocabulary, regex, merge
ranks, and special-token IDs. Before processing the corpus, several probes
verify that it returns the same IDs as the Python implementation.

### Compact binary storage

Encoded IDs are written directly to `train.bin` and `valid.bin`:

```text
vocabulary <= 65,536 entries -> uint16 tokens
larger vocabulary            -> uint32 tokens
```

The adjacent `dataset.json` records the dtype, paths, vocabulary size,
tokenizer filename, token counts, and source-byte counts. The binary files do
not need headers because the metadata supplies their interpretation.

The training script opens them with `numpy.memmap`. A memory map makes the token
array look indexable while the operating system pages in only the regions that
are actually sampled. This is why the dataset can exceed available RAM.

## 26. What `get_batch` samples

[`tiny_transformer/data.py`](tiny_transformer/data.py) receives a flat token
array. For each of `B` examples, it samples one valid starting index and uses a
broadcasted offset vector to construct all `T` positions:

```text
starts shape:           (B, 1)
offsets shape:          (1, T)
starts + offsets:       (B, T)
inputs at indices:      start ... start+T-1
targets at indices:     start+1 ... start+T
```

The largest valid start must leave room for the extra target token. Converting
to `int64` is necessary because PyTorch embedding indices use `torch.long`,
even though the on-disk representation is compact `uint16` or `uint32`.

Random windows are simple and mix distant portions of the corpus. They can
overlap and repeat, so `steps * B * T` counts processed training tokens, not
necessarily unique corpus tokens.

## 27. Training-script startup

Before the first optimization step,
[`scripts/train_tinystories.py`](scripts/train_tinystories.py) performs several
important consistency checks:

- dimensions, batch size, steps, and intervals must be positive;
- warmup cannot extend beyond the run;
- model width must divide evenly across heads;
- dataset metadata must use a supported format;
- CUDA must actually be available when requested;
- BF16 is accepted only on a CUDA device with native BF16 support.

Python, NumPy, CPU PyTorch, and all CUDA devices receive the same configured
seed. This controls model initialization, random batch starts, and other random
operations as far as the selected backend permits.

The model is constructed in FP32. BF16 is activated through `torch.autocast`
around the forward pass, which lets PyTorch select lower precision for suitable
operations while retaining FP32 where needed. Logits are explicitly converted
to FP32 before the custom cross-entropy reduction.

The run directory receives `config.json` before training. This is not merely a
record for humans: generation and evaluation use it later to reconstruct a
model with exactly the checkpoint's dimensions and locate its dataset and
tokenizer.

## 28. The exact training-loop lifecycle

For every iteration, the script does the following:

1. Compute the scheduled learning rate for the current zero-based step.
2. Write that rate into every optimizer parameter group.
3. Sample an input/target batch and transfer it to the selected device.
4. Clear old gradients with `set_to_none=True`.
5. Run the model under optional BF16 autocast.
6. Reshape logits and targets and compute FP32 cross-entropy.
7. Call `backward()` to populate every used parameter's gradient.
8. Clip the single global gradient norm.
9. Apply AdamW.
10. Synchronize enough GPU work through `loss.item()` to record meaningful
    elapsed training time.
11. Optionally evaluate and checkpoint.
12. Append one JSON object to `metrics.jsonl` and flush it to disk.

`set_to_none=True` avoids filling every gradient tensor with zeros. The next
backward pass allocates or writes fresh gradients, which can be faster and use
less memory.

The logged `tokens_seen` value is:

$$\text{completed steps} \times B \times T$$

The runtime summary also records the rough dense-Transformer training compute
proxy:

$$\text{training FLOPs} \approx 6NT$$

where here $N$ is parameter count and $T$ is the total number of processed
tokens. This is an approximation useful for comparisons, not a profiler's exact
hardware-operation count.

## 29. Training evaluation versus deterministic evaluation

The training loop periodically samples a configured number of random
validation batches. This is fast and useful for monitoring, but two runs may
see slightly different validation windows unless their full random state and
execution sequence match.

[`scripts/evaluate_tinystories.py`](scripts/evaluate_tinystories.py) provides a
stronger comparison:

1. It lays out consecutive, non-overlapping validation windows.
2. It uses every full window and discards only a final tail shorter than one
   context.
3. It batches those fixed windows only for execution efficiency.
4. It weights batch losses by their token counts before computing the final
   mean.

This produces deterministic coverage of nearly every validation target. It
also evaluates different checkpoints at one requested context length, provided
that length does not exceed any model's trained maximum.

Token accuracy asks whether the largest logit corresponds to the correct next
token. It is easy to understand but ignores the rest of the predicted
distribution. Cross-entropy is the primary language-model metric because it
rewards calibrated probability assigned to the correct answer.

## 30. Checkpoints and resuming

[`tiny_transformer/serialization.py`](tiny_transformer/serialization.py) saves
three items:

```text
model.state_dict()
optimizer.state_dict()
completed iteration
```

The model state contains named parameters and persistent buffers. The optimizer
state contains parameter groups plus each parameter's Adam moment estimates and
step counter. Restoring only model weights would allow inference, but resuming
training without optimizer state would discard the accumulated moments and
change the optimization trajectory.

Loading maps tensors to the device of the supplied model, then copies the saved
state into already-constructed model and optimizer objects. Architecture is not
implicitly stored as an executable Python object, which is why `config.json`
must accompany the checkpoint.

`weights_only=True` restricts PyTorch loading to the tensor-oriented checkpoint
format and avoids unpickling arbitrary Python objects.

## 31. The actual generation path

[`scripts/generate_tinystories.py`](scripts/generate_tinystories.py) first reads
the run configuration, dataset metadata, tokenizer, and checkpoint. It then
constructs the same architecture and loads the saved model parameters.

[`tiny_transformer/generation.py`](tiny_transformer/generation.py) performs the
token loop. There is no key/value cache in this educational implementation, so
every new token reruns the model over the retained context window. This is
simple and correct, but slower than production generation.

When the sequence exceeds the model's context length, the oldest tokens are
dropped from the next model input. The returned output still contains the full
prompt and all generated tokens; cropping affects only what the model can see
for its next prediction.

`temperature == 0` selects `argmax` directly. For positive temperature, top-k
filtering finds the kth-largest logit and replaces everything below that
threshold with negative infinity. Softmax then assigns probability only among
the retained candidates before `torch.multinomial` samples one.

Each prompt receives an independent seeded `torch.Generator`. If one prompt
reaches the end token early, it therefore does not consume random draws that
would alter later prompts. That makes prompt-by-prompt comparisons across
checkpoints much cleaner.

## 32. Why `train()`, `eval()`, and `no_grad()` all appear

`model.train()` and `model.eval()` control module behavior. This architecture
currently has no dropout or batch normalization, so they do not change its
numerical forward pass, but using the correct mode keeps the code valid if such
layers are added later.

`torch.no_grad()` is separate: it prevents autograd from constructing a
backward graph. Evaluation and generation need outputs but never call
`backward()`, so disabling graph recording reduces memory and overhead.

Autocast is also separate from both. It controls operation dtypes, not model
behavior or gradient recording.

## 33. How the tests specify the implementation

The tests are useful as executable documentation:

- component tests compare each layer against known reference outputs;
- shape-sensitive tests catch incorrect transposes and head layouts;
- tokenizer tests compare behavior with a known byte-level tokenizer and test
  Unicode, special tokens, iteration, and round trips;
- optimizer tests compare parameter updates after multiple steps;
- checkpoint tests verify that parameters, optimizer state, and iteration are
  restored;
- snapshot files provide fixed numerical targets for deterministic inputs.

Two tokenizer memory-limit tests are skipped on Windows because the Unix
`resource` API is unavailable. The functional tokenizer tests still run. This
is a platform limitation in the measurement mechanism, not a skipped tokenizer
feature.

When changing a component, run its focused test first, then the complete suite:

```powershell
uv run pytest tests/test_model.py -k rope
uv run pytest
```

## 34. End-to-end dependency chain

The entire repository can be understood as one dependency chain:

```text
raw TinyStories text
    -> BPE vocabulary + ordered merges
    -> tokenizer JSON
    -> train.bin / valid.bin + dataset metadata
    -> random shifted token batches
    -> Transformer logits
    -> cross-entropy scalar
    -> autograd gradients
    -> global clipping
    -> scheduled AdamW updates
    -> checkpoints + metrics
    -> deterministic evaluation + generated samples
    -> summary tables and figures
```

Every artifact has a specific consumer. The tokenizer determines the vocabulary
size and encoded IDs. Dataset metadata tells training how to interpret binary
files. Run configuration tells later scripts how to rebuild the architecture.
Checkpoints supply learned state. Metrics and evaluation outputs supply the
values summarized in plots and in
[`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md).

## 35. Deliberate simplifications and possible extensions

This is a complete educational language-model pipeline, but not a production
training system. Important simplifications include:

- no attention key/value cache during generation;
- no dropout;
- no distributed or multi-GPU training;
- no gradient accumulation;
- no fused attention or fused optimizer kernels;
- no dynamic loss scaling for FP16 (the project uses BF16 instead);
- no weight tying between embeddings and the language-model head;
- no padding mask because training windows are fixed-length contiguous text;
- BPE training reads its bounded tokenizer sample into memory;
- checkpoints do not currently save Python/NumPy/PyTorch RNG states.

These are good extension points because each changes a clearly isolated part of
the current design. For example, gradient accumulation belongs around several
forward/backward microbatches before clipping and stepping; a KV cache belongs
inside attention and generation; exact-resume reproducibility requires saving
and restoring every RNG state alongside model and optimizer state.

The current code favors visibility over abstraction. That is the right tradeoff
for understanding how the components work together before introducing the
performance machinery used by large training frameworks.
