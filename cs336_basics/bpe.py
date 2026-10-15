from __future__ import annotations

import os
from collections import Counter, defaultdict

import regex

from cs336_basics.tokenizer import PRETOKEN_PATTERN


def _adjacent_pair_counts(pieces: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    """Count adjacent pairs in one current pre-token representation."""
    return Counter(zip(pieces, pieces[1:]))


def _merge_pair(
    pieces: tuple[bytes, ...],
    pair_to_merge: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    """Merge all non-overlapping occurrences of one pair from left to right."""
    merged: list[bytes] = []
    index = 0
    while index < len(pieces):
        if (
            index + 1 < len(pieces)
            and pieces[index] == pair_to_merge[0]
            and pieces[index + 1] == pair_to_merge[1]
        ):
            merged.append(pieces[index] + pieces[index + 1])
            index += 2
        else:
            merged.append(pieces[index])
            index += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE vocabulary and ordered merge list."""
    if vocab_size < 256 + len(set(special_tokens)):
        raise ValueError("vocab_size is too small for all bytes and special tokens")

    # IDs 0-255 are the corresponding individual bytes. Special tokens are
    # indivisible vocabulary entries and do not participate in merge training.
    vocab: dict[int, bytes] = {byte: bytes([byte]) for byte in range(256)}
    for special_token in special_tokens:
        encoded = special_token.encode("utf-8")
        if encoded not in vocab.values():
            vocab[len(vocab)] = encoded

    with open(input_path, encoding="utf-8") as corpus_file:
        text = corpus_file.read()

    # Split out special tokens before regex pre-tokenization. This prevents a
    # learned token from ever containing part of a special-token marker.
    if special_tokens:
        alternatives = sorted(set(special_tokens), key=len, reverse=True)
        special_pattern = regex.compile("|".join(regex.escape(token) for token in alternatives))
        ordinary_segments = special_pattern.split(text)
    else:
        ordinary_segments = [text]

    pretoken_frequencies: Counter[tuple[bytes, ...]] = Counter()
    for segment in ordinary_segments:
        for match in PRETOKEN_PATTERN.finditer(segment):
            byte_pieces = tuple(bytes([byte]) for byte in match.group().encode("utf-8"))
            if byte_pieces:
                pretoken_frequencies[byte_pieces] += 1

    # Give each unique pre-token a stable integer ID. Frequencies retain all
    # corpus multiplicity while avoiding duplicate sequence storage.
    words = list(pretoken_frequencies)
    frequencies = [pretoken_frequencies[word] for word in words]

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for word_id, (word, frequency) in enumerate(zip(words, frequencies)):
        for pair, occurrences in _adjacent_pair_counts(word).items():
            pair_counts[pair] += occurrences * frequency
            pair_to_words[pair].add(word_id)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_counts:
        # max() implements both requirements: highest corpus frequency first,
        # then lexicographically greatest byte pair to break frequency ties.
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        affected_word_ids = tuple(pair_to_words[best_pair])

        for word_id in affected_word_ids:
            old_word = words[word_id]
            old_counts = _adjacent_pair_counts(old_word)
            new_word = _merge_pair(old_word, best_pair)
            new_counts = _adjacent_pair_counts(new_word)
            words[word_id] = new_word
            frequency = frequencies[word_id]

            # Only words containing best_pair changed. Adjust their old/new pair
            # contributions rather than recounting pairs across the full corpus.
            for pair in old_counts.keys() | new_counts.keys():
                old_occurrences = old_counts.get(pair, 0)
                new_occurrences = new_counts.get(pair, 0)
                delta = (new_occurrences - old_occurrences) * frequency
                if delta:
                    pair_counts[pair] += delta

                if old_occurrences and not new_occurrences:
                    pair_to_words[pair].discard(word_id)
                elif new_occurrences and not old_occurrences:
                    pair_to_words[pair].add(word_id)

                if pair_counts.get(pair, 0) <= 0:
                    pair_counts.pop(pair, None)
                    pair_to_words.pop(pair, None)

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

    return vocab, merges
