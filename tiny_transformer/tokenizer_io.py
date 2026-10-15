from __future__ import annotations

import base64
import json
import os

from tiny_transformer.tokenizer import Tokenizer


def _bytes_to_text(value: bytes) -> str:
    """Represent arbitrary token bytes safely inside a JSON string."""
    return base64.b64encode(value).decode("ascii")


def _text_to_bytes(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def save_tokenizer(tokenizer: Tokenizer, path: str | os.PathLike) -> None:
    """Save a byte-level BPE tokenizer in a small, portable JSON file."""
    payload = {
        "format_version": 1,
        "vocab": {
            str(token_id): _bytes_to_text(token)
            for token_id, token in sorted(tokenizer.vocab.items())
        },
        "merges": [
            [_bytes_to_text(left), _bytes_to_text(right)]
            for left, right in tokenizer.merges
        ],
        "special_tokens": tokenizer.special_tokens,
    }
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def load_tokenizer(path: str | os.PathLike) -> Tokenizer:
    """Load a tokenizer written by :func:`save_tokenizer`."""
    with open(path, encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if payload.get("format_version") != 1:
        raise ValueError("unsupported tokenizer file format")

    vocab = {
        int(token_id): _text_to_bytes(token)
        for token_id, token in payload["vocab"].items()
    }
    merges = [
        (_text_to_bytes(left), _text_to_bytes(right))
        for left, right in payload["merges"]
    ]
    return Tokenizer(vocab, merges, payload["special_tokens"])
