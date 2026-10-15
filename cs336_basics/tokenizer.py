from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex


# GPT-2-style pre-tokenization keeps common contractions separate, groups
# letters and numbers, and makes leading spaces part of the following token.
PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    """A byte-level BPE tokenizer with optional indivisible special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merges = list(merges)

        # A smaller rank means the pair was learned earlier and therefore has
        # higher priority when several possible merges are present.
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        self.special_tokens = list(special_tokens or [])
        self.special_to_id: dict[str, int] = {}
        for special_token in self.special_tokens:
            encoded = special_token.encode("utf-8")
            if encoded not in self.token_to_id:
                raise ValueError(f"special token {special_token!r} is missing from the vocabulary")
            self.special_to_id[special_token] = self.token_to_id[encoded]

        # Prefer the longest alternative when special tokens overlap, e.g. a
        # doubled end-of-text token versus its single-token prefix.
        if self.special_tokens:
            alternatives = sorted(self.special_tokens, key=len, reverse=True)
            escaped = "|".join(regex.escape(token) for token in alternatives)
            self.special_pattern = regex.compile(f"({escaped})")
        else:
            self.special_pattern = None

    def _apply_merges(self, pretoken: str) -> list[int]:
        # Byte-level BPE begins with one token for each UTF-8 byte, guaranteeing
        # that arbitrary Unicode text can always be represented.
        pieces = [bytes([byte]) for byte in pretoken.encode("utf-8")]

        while len(pieces) > 1:
            # Of the adjacent pairs currently present, apply the pair with the
            # earliest learned merge rank. Pairs absent from the merge table stop.
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None
            for left, right in zip(pieces, pieces[1:]):
                pair = (left, right)
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            # Merge all non-overlapping occurrences of the selected pair in one
            # left-to-right pass, matching the BPE encoding procedure.
            merged: list[bytes] = []
            index = 0
            while index < len(pieces):
                if (
                    index + 1 < len(pieces)
                    and pieces[index] == best_pair[0]
                    and pieces[index + 1] == best_pair[1]
                ):
                    merged.append(pieces[index] + pieces[index + 1])
                    index += 2
                else:
                    merged.append(pieces[index])
                    index += 1
            pieces = merged

        return [self.token_to_id[piece] for piece in pieces]

    def _encode_ordinary_text(self, text: str) -> Iterator[int]:
        # BPE merges never cross pre-token boundaries. finditer avoids building
        # a separate list containing every pre-token in a large input string.
        for match in PRETOKEN_PATTERN.finditer(text):
            yield from self._apply_merges(match.group())

    def encode(self, text: str) -> list[int]:
        """Encode one string into token IDs."""
        if self.special_pattern is None:
            return list(self._encode_ordinary_text(text))

        token_ids: list[int] = []
        for part in self.special_pattern.split(text):
            if not part:
                continue
            special_id = self.special_to_id.get(part)
            if special_id is not None:
                token_ids.append(special_id)
            else:
                token_ids.extend(self._encode_ordinary_text(part))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode text chunks without loading the whole corpus at once."""
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs, replacing any incomplete UTF-8 byte sequences."""
        # Join bytes before decoding: one Unicode character can span several BPE
        # tokens, so decoding individual tokens first would be incorrect.
        encoded = b"".join(self.vocab[token_id] for token_id in token_ids)
        return encoded.decode("utf-8", errors="replace")
