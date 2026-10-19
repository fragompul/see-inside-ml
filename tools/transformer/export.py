"""Publish the trained tiny GPT for the TypeScript engine.

    python tools/transformer/export.py

Writes to packages/deep/models/tiny-gpt/:
  weights.bin     all tensors, float32 little-endian, concatenated in manifest order
  model.json      config + tensor manifest (name, shape, byte offset)
  tokenizer.json  BPE merges + special tokens

and packages/deep/fixtures/tiny-gpt-parity.json: reference activations computed in float64
from the *same float32 weights*, so the TypeScript forward pass must match to ~1e-9.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import corpus
from bpe import BPE
from gpt import GPT, GPTConfig

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[1] / "packages" / "engine"
MODEL_DIR = ENGINE / "models" / "tiny-gpt"
FIXTURE = ENGINE / "fixtures" / "tiny-gpt-parity.json"

PROMPTS = [
    "Anna found a ball. She",
    "The small mice play in the forest. They",
    "Tom and Leo went to the park. Tom said hello to",
]

TOKENIZER_CASES = [
    "Anna found a ball. She carried it to the park.",
    "The brave foxes jump in the garden.",
    "Zebra 123 é!",  # unseen word, digits, non-ASCII
    "  leading and trailing   spaces  ",
    "Hello<|endoftext|>World",
    "",
    "A\nB\tC",
    "Don't stop.",
]


def tolist(a: np.ndarray, digits: int = 9):
    return np.round(a.astype(np.float64), digits).tolist()


def main() -> None:
    ckpt = np.load(HERE / "out" / "checkpoint.npz")
    vocab, ctx, d, layers, heads = (int(x) for x in (HERE / "out" / "config.txt").read_text().split())
    cfg = GPTConfig(vocab=vocab, ctx=ctx, d=d, layers=layers, heads=heads)
    bpe = BPE.load(HERE / "out" / "tokenizer.json")
    assert bpe.vocab_size == cfg.vocab

    weights32 = {k: np.ascontiguousarray(ckpt[k], dtype="<f4") for k in ckpt.files}

    # ---- weights + manifest
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tensors, blob, offset = [], bytearray(), 0
    for name, arr in weights32.items():
        tensors.append({"name": name, "shape": list(arr.shape), "offset": offset})
        blob += arr.tobytes()
        offset += arr.nbytes
    (MODEL_DIR / "weights.bin").write_bytes(bytes(blob))
    (MODEL_DIR / "model.json").write_text(
        json.dumps(
            {
                "config": {"vocab": cfg.vocab, "ctx": cfg.ctx, "d": cfg.d, "layers": cfg.layers, "heads": cfg.heads},
                "eotId": bpe.eot_id,
                "dtype": "float32",
                "tensors": tensors,
                "trainedOn": "synthetic mini-stories (tools/transformer/corpus.py)",
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    bpe.save(MODEL_DIR / "tokenizer.json")
    # per-token display strings so the UI never has to guess how to print a lone byte
    print(f"weights.bin {len(blob) / 1024:.0f} kB, {len(tensors)} tensors")

    # ---- parity fixtures, float64 math on the exact float32 weights
    model = GPT(cfg, dtype=np.float64)
    model.p = {k: v.astype(np.float64) for k, v in weights32.items()}

    cases = []
    for prompt in PROMPTS:
        ids = bpe.encode(prompt)
        logits, _, tr = model.forward(np.array([ids]), trace=True)
        cases.append(
            {
                "prompt": prompt,
                "ids": ids,
                "tok": tolist(tr["tok"][0]),
                "pos": tolist(tr["pos"]),  # (T, d): positions are shared across the batch
                "layers": [
                    {
                        "ln1": tolist(L["ln1"][0]),
                        "q": tolist(L["q"][0]),  # (H, T, hd)
                        "k": tolist(L["k"][0]),
                        "v": tolist(L["v"][0]),
                        "probs": tolist(L["probs"][0]),  # (H, T, T)
                        "attnOut": tolist(L["attn_out"][0]),
                        "residMid": tolist(L["resid_mid"][0]),
                        "mlpPre": tolist(L["mlp_pre"][0]),
                        "mlpOut": tolist(L["mlp_out"][0]),
                        "residOut": tolist(L["resid_out"][0]),
                    }
                    for L in tr["layers"]
                ],
                "finalLn": tolist(tr["final_ln"][0]),
                "logits": tolist(tr["logits"][0]),
            }
        )

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(
            {
                "cases": cases,
                "tokenizer": [
                    {"text": t, "ids": bpe.encode(t), "decoded": bpe.decode(bpe.encode(t))} for t in TOKENIZER_CASES
                ],
                "tokenStrings": [bpe.token_string(i) for i in range(bpe.vocab_size)],
            }
        ),
        encoding="utf-8",
    )
    print(f"fixtures {FIXTURE.stat().st_size / 1024:.0f} kB")

    # sanity: model outputs a sensible story
    gen = GPT(cfg)
    gen.p = {k: v.astype(np.float32) for k, v in weights32.items()}
    for prompt in PROMPTS[:2]:
        ids = gen.generate(bpe.encode(prompt), 20, temperature=0.6, seed=3)
        print("  ", ascii(bpe.decode(ids)))


if __name__ == "__main__":
    main()
