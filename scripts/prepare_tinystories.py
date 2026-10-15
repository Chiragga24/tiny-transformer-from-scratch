from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

from tiny_transformer.bpe import train_bpe
from tiny_transformer.tokenizer import PRETOKEN_PATTERN, Tokenizer
from tiny_transformer.tokenizer_io import load_tokenizer, save_tokenizer


END_OF_TEXT = "<|endoftext|>"
END_OF_TEXT_BYTES = END_OF_TEXT.encode("utf-8")


class TextEncoder(Protocol):
    """The small interface shared by our tokenizer and tiktoken.Encoding."""

    def encode(self, text: str, *args: object, **kwargs: object) -> list[int]: ...


def iter_story_chunks(
    path: Path,
    max_bytes: int | None,
    read_size: int = 1024 * 1024,
) -> Iterator[str]:
    """Yield UTF-8 chunks that end at TinyStories document boundaries.

    Cutting only after <|endoftext|> keeps a story intact and ensures a BPE
    pre-token can never be split between calls to Tokenizer.encode().
    """
    bytes_read = 0
    pending = b""

    with path.open("rb") as input_file:
        while max_bytes is None or bytes_read < max_bytes:
            amount = read_size if max_bytes is None else min(read_size, max_bytes - bytes_read)
            if amount == 0:
                break

            block = input_file.read(amount)
            if not block:
                break
            bytes_read += len(block)
            pending += block

            # Keep an unfinished story for the next read. The emitted prefix is
            # valid UTF-8 because the ASCII delimiter ends at a character edge.
            boundary = pending.rfind(END_OF_TEXT_BYTES)
            if boundary >= 0:
                boundary += len(END_OF_TEXT_BYTES)
                yield pending[:boundary].decode("utf-8")
                pending = pending[boundary:]

    # When processing the entire file, do not silently lose a final document
    # in case the corpus does not end with the special token.
    if max_bytes is None and pending:
        yield pending.decode("utf-8")


def write_tokenizer_sample(source: Path, destination: Path, max_bytes: int) -> int:
    """Write a bounded, whole-story text sample for the in-memory BPE trainer."""
    written = 0
    with destination.open("wb") as output_file:
        for chunk in iter_story_chunks(source, max_bytes):
            encoded = chunk.encode("utf-8")
            output_file.write(encoded)
            written += len(encoded)
    return written


def encode_corpus(
    tokenizer: TextEncoder,
    source: Path,
    destination: Path,
    dtype: np.dtype,
    max_bytes: int | None,
    allowed_special: set[str] | None = None,
) -> tuple[int, int]:
    """Stream text through the tokenizer and write compact binary token IDs."""
    token_count = 0
    source_bytes = 0
    with destination.open("wb") as output_file:
        for text_chunk in iter_story_chunks(source, max_bytes):
            if allowed_special is None:
                token_ids = tokenizer.encode(text_chunk)
            else:
                # tiktoken rejects special-token text unless it is explicitly
                # allowed. Our pure-Python tokenizer accepts it by design.
                token_ids = tokenizer.encode(text_chunk, allowed_special=allowed_special)
            np.asarray(token_ids, dtype=dtype).tofile(output_file)
            token_count += len(token_ids)
            source_bytes += len(text_chunk.encode("utf-8"))
    return token_count, source_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small TinyStories BPE tokenizer and encode laptop-sized datasets."
    )
    parser.add_argument("--train-text", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--valid-text", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tinystories_laptop"))
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Reuse an existing tokenizer instead of training a new one.",
    )
    parser.add_argument(
        "--tokenizer-bytes",
        type=int,
        default=5_000_000,
        help="Training-corpus bytes used to learn BPE merges (default: 5 MB).",
    )
    parser.add_argument(
        "--train-bytes",
        type=int,
        default=20_000_000,
        help="Training-corpus bytes encoded for the laptop run (default: 20 MB).",
    )
    parser.add_argument(
        "--valid-bytes",
        type=int,
        default=2_000_000,
        help="Validation-corpus bytes encoded for the laptop run (default: 2 MB).",
    )
    parser.add_argument(
        "--encoding-backend",
        choices=("python", "tiktoken"),
        default="python",
        help=(
            "Tokenizer used only for corpus encoding. 'tiktoken' is much faster "
            "and is checked against our implementation before use."
        ),
    )
    return parser.parse_args()


def make_fast_encoder(tokenizer: Tokenizer) -> TextEncoder:
    """Build a tiktoken encoder with exactly the IDs learned by our BPE."""
    import tiktoken

    special_ids = set(tokenizer.special_to_id.values())
    mergeable_ranks = {
        token_bytes: token_id
        for token_id, token_bytes in tokenizer.vocab.items()
        if token_id not in special_ids
    }
    encoder = tiktoken.Encoding(
        name="tiny_transformer_tinystories",
        pat_str=PRETOKEN_PATTERN.pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=tokenizer.special_to_id,
    )

    # A fast backend is useful only if it preserves the experiment exactly.
    # Check ordinary text, Unicode, whitespace, and the document separator.
    probes = [
        "Once upon a time, there was a little girl.",
        "café — 123\n\n",
        f"story one{END_OF_TEXT}story two{END_OF_TEXT}",
    ]
    for probe in probes:
        expected = tokenizer.encode(probe)
        actual = encoder.encode(probe, allowed_special=set(tokenizer.special_tokens))
        if actual != expected:
            raise RuntimeError("tiktoken backend does not match the Python tokenizer")
    return encoder


def main() -> None:
    args = parse_args()
    if args.tokenizer is None and args.vocab_size < 257:
        raise ValueError("vocab size must fit 256 byte tokens and the end-of-text token")
    if args.tokenizer_bytes <= 0 or args.train_bytes <= 0 or args.valid_bytes <= 0:
        raise ValueError("all byte limits must be positive")
    for path in (args.train_text, args.valid_text):
        if not path.is_file():
            raise FileNotFoundError(f"missing corpus file: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "tokenizer_sample.txt"
    tokenizer_path = args.output_dir / "tokenizer.json"
    train_tokens_path = args.output_dir / "train.bin"
    valid_tokens_path = args.output_dir / "valid.bin"

    if args.tokenizer is not None:
        if not args.tokenizer.is_file():
            raise FileNotFoundError(f"missing tokenizer: {args.tokenizer}")
        tokenizer = load_tokenizer(args.tokenizer)
        print(f"Reusing {len(tokenizer.vocab):,}-token BPE from {args.tokenizer}...")
    else:
        sample_bytes = write_tokenizer_sample(args.train_text, sample_path, args.tokenizer_bytes)
        if sample_bytes == 0:
            raise ValueError("the tokenizer sample did not contain a complete TinyStories document")
        print(f"Training a {args.vocab_size:,}-token BPE on {sample_bytes:,} bytes...")
        vocab, merges = train_bpe(sample_path, args.vocab_size, [END_OF_TEXT])
        tokenizer = Tokenizer(vocab, merges, [END_OF_TEXT])
    save_tokenizer(tokenizer, tokenizer_path)

    # uint16 halves the storage needed by int32 while supporting vocabularies
    # up to 65,536 entries. Larger requested vocabularies switch automatically.
    vocab_size = len(tokenizer.vocab)
    dtype = np.dtype("uint16" if vocab_size <= np.iinfo(np.uint16).max + 1 else "uint32")

    encoder: TextEncoder = tokenizer
    allowed_special: set[str] | None = None
    if args.encoding_backend == "tiktoken":
        encoder = make_fast_encoder(tokenizer)
        allowed_special = set(tokenizer.special_tokens)
        print("Verified tiktoken produces the same token IDs as the Python tokenizer.")

    print("Encoding the training split...")
    train_tokens, train_bytes = encode_corpus(
        encoder,
        args.train_text,
        train_tokens_path,
        dtype,
        args.train_bytes,
        allowed_special,
    )
    print("Encoding the validation split...")
    valid_tokens, valid_bytes = encode_corpus(
        encoder,
        args.valid_text,
        valid_tokens_path,
        dtype,
        args.valid_bytes,
        allowed_special,
    )

    metadata = {
        "format_version": 1,
        "dtype": dtype.name,
        "vocab_size": vocab_size,
        "special_tokens": [END_OF_TEXT],
        "tokenizer": tokenizer_path.name,
        "train": {
            "path": train_tokens_path.name,
            "tokens": train_tokens,
            "source_bytes": train_bytes,
        },
        "valid": {
            "path": valid_tokens_path.name,
            "tokens": valid_tokens,
            "source_bytes": valid_bytes,
        },
    }
    metadata_path = args.output_dir / "dataset.json"
    with metadata_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)

    print(f"Saved tokenizer and dataset metadata to {args.output_dir}")
    print(f"Training tokens: {train_tokens:,}; validation tokens: {valid_tokens:,}")


if __name__ == "__main__":
    main()
