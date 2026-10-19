"""Writes packages/deep/fixtures/bpe-train.json: merges learned by the reference trainer, for the TypeScript parity test."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bpe import BPE  # noqa: E402

TEXTS = [
    "the cat sat on the mat. the cat ate the rat; the rat ran under the mat.",
    "Tokenizers split text into pieces: 12345 and 2024-05-17, naïve café, tokenization!",
    "def add(a, b):\n    return a + b\n\nprint(add(1, 2))\n",
]
out = {"texts": TEXTS, "cases": []}
for n in (0, 5, 40, 400):
    bpe = BPE.train(TEXTS, n)
    out["cases"].append({
        "merges": n,
        "learned": [list(m) for m in bpe.merges],
        "encoded": bpe.encode(TEXTS[1]),
    })
path = Path(__file__).resolve().parents[2] / "packages" / "deep" / "fixtures" / "bpe-train.json"
path.write_text(json.dumps(out), encoding="utf-8")
print("wrote", path, [len(c["learned"]) for c in out["cases"]])
