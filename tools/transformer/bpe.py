"""Byte-level BPE tokenizer (GPT-2 style, ASCII-friendly), written from scratch.

* Base vocabulary: the 256 byte values, so *any* string can be encoded.
* Text is first split into "pre-tokens" (words with their leading space, numbers, punctuation,
  whitespace) and merges never cross a pre-token boundary, exactly like GPT-2.
* Merges are learned greedily from pair frequencies; the token created by merge i gets id 256 + i.
* One special token, `<|endoftext|>`, takes the last id.

The TypeScript tokenizer in packages/deep reproduces `encode` / `decode` exactly (parity tests).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Keep this pattern in sync with PRETOKEN_PATTERN in packages/deep/src/transformer/tokenizer.ts
PRETOKEN_PATTERN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
EOT = "<|endoftext|>"


class BPE:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = list(merges or [])
        self._rebuild()

    # ------------------------------------------------------------------ vocabulary
    def _rebuild(self) -> None:
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.token_bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        self.eot_id = len(self.token_bytes)
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return self.eot_id + 1

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts: list[str], n_merges: int) -> "BPE":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(PRETOKEN_PATTERN.findall(text))
        words = {tuple(w.encode("utf-8")): c for w, c in counts.items()}
        merges: list[tuple[int, int]] = []
        for step in range(n_merges):
            pairs: Counter[tuple[int, int]] = Counter()
            for word, c in words.items():
                for pair in zip(word, word[1:]):
                    pairs[pair] += c
            if not pairs:
                break
            # most frequent pair; ties broken by the smallest pair so training is deterministic
            best = min(pairs, key=lambda p: (-pairs[p], p))
            new_id = 256 + step
            merges.append(best)
            merged: dict[tuple[int, ...], int] = {}
            for word, c in words.items():
                out: list[int] = []
                i = 0
                while i < len(word):
                    if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(word[i])
                        i += 1
                merged[tuple(out)] = merged.get(tuple(out), 0) + c
            words = merged
        return cls(merges)

    # ------------------------------------------------------------------ encode / decode
    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                rank = self.ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_rank is None:
                break
            ids[best_i : best_i + 2] = [256 + best_rank]
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        # the special token is recognised literally, like in GPT-2
        for chunk_index, chunk in enumerate(text.split(EOT)):
            if chunk_index > 0:
                out.append(self.eot_id)
            for word in PRETOKEN_PATTERN.findall(chunk):
                out.extend(self._encode_word(word))
        return out

    def decode(self, ids: list[int]) -> str:
        data = bytearray()
        for i in ids:
            data += EOT.encode() if i == self.eot_id else self.token_bytes[i]
        return data.decode("utf-8", errors="replace")

    def token_string(self, i: int) -> str:
        if i == self.eot_id:
            return EOT
        return self.token_bytes[i].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"merges": self.merges, "specialTokens": {EOT: self.eot_id}}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BPE":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]])


if __name__ == "__main__":
    import corpus

    texts = corpus.generate(5000, seed=0)
    bpe = BPE.train(texts, 255)
    sample = "Anna found a ball. She carried it to the park."
    ids = bpe.encode(sample)
    assert bpe.decode(ids) == sample
    print(f"vocab {bpe.vocab_size}, {len(ids)} tokens:", [bpe.token_string(i) for i in ids])
