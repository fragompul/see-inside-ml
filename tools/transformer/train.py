"""Train the tiny GPT on the synthetic story corpus (CPU, NumPy only).

    python tools/transformer/train.py --steps 6000

Writes a checkpoint to tools/transformer/out/ (git-ignored). Run `export.py` afterwards to publish
the weights, tokenizer and parity fixtures into the web app.
"""

from __future__ import annotations

import os

# These matrices are tiny: BLAS multi-threading only adds overhead (3x slower on 16 cores).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import corpus  # noqa: E402
from bpe import BPE  # noqa: E402
from gpt import GPT, GPTConfig  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def build_data(n_train: int, n_val: int, n_merges: int):
    train_text = corpus.generate(n_train, seed=1)
    val_text = corpus.generate(n_val, seed=2)
    bpe = BPE.train(train_text[:20000], n_merges)

    def encode(stories: list[str]) -> np.ndarray:
        ids: list[int] = []
        for s in stories:
            ids.extend(bpe.encode(s))
            ids.append(bpe.eot_id)
        return np.array(ids, dtype=np.int64)

    return bpe, encode(train_text), encode(val_text)


def batch(data: np.ndarray, B: int, T: int, rng: np.random.Generator):
    starts = rng.integers(0, len(data) - T - 1, size=B)
    x = np.stack([data[s : s + T] for s in starts])
    y = np.stack([data[s + 1 : s + T + 1] for s in starts])
    return x, y


def lr_at(step: int, steps: int, peak: float, warmup: int = 150, floor: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=48)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--merges", type=int, default=255)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bpe, train, val = build_data(60000, 2000, args.merges)
    cfg = GPTConfig(vocab=bpe.vocab_size, ctx=args.ctx, d=args.d, layers=args.layers, heads=args.heads)
    model = GPT(cfg, seed=args.seed)
    print(f"vocab {cfg.vocab} | {len(train):,} train tokens | {model.n_params:,} parameters")

    rng = np.random.default_rng(args.seed)
    m = {k: np.zeros_like(v) for k, v in model.p.items()}
    v2 = {k: np.zeros_like(v) for k, v in model.p.items()}
    b1, b2, eps, wd = 0.9, 0.99, 1e-8, 0.01
    t0 = time.time()

    def evaluate() -> float:
        r = np.random.default_rng(123)
        total = 0.0
        for _ in range(20):
            x, y = batch(val, args.batch, args.ctx, r)
            total += model.loss(x, y)[0]
        return total / 20

    for step in range(args.steps):
        x, y = batch(train, args.batch, args.ctx, rng)
        loss, cache = model.loss(x, y)
        grads = model.backward(cache)

        norm = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))
        clip = min(1.0, 1.0 / (norm + 1e-6))
        lr = lr_at(step, args.steps, args.lr)
        t = step + 1
        for name, param in model.p.items():
            g = grads[name] * clip
            m[name] = b1 * m[name] + (1 - b1) * g
            v2[name] = b2 * v2[name] + (1 - b2) * g * g
            mh = m[name] / (1 - b1**t)
            vh = v2[name] / (1 - b2**t)
            param -= lr * (mh / (np.sqrt(vh) + eps) + (wd * param if name.endswith(".w") else 0))

        if step % 250 == 0 and step > 0:
            OUT.mkdir(exist_ok=True)
            np.savez(OUT / "checkpoint.npz", **model.p)  # usable even if the run is interrupted
            bpe.save(OUT / "tokenizer.json")
            (OUT / "config.txt").write_text(
                f"{cfg.vocab} {cfg.ctx} {cfg.d} {cfg.layers} {cfg.heads}\n", encoding="utf-8"
            )
        if step % 250 == 0 or step == args.steps - 1:
            print(
                f"step {step:5d}  loss {loss:.4f}  val {evaluate():.4f}  lr {lr:.5f}  "
                f"{(time.time() - t0) / (step + 1) * 1000:.0f} ms/step",
                flush=True,
            )

    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "checkpoint.npz", **model.p)
    bpe.save(OUT / "tokenizer.json")
    (OUT / "config.txt").write_text(
        f"{cfg.vocab} {cfg.ctx} {cfg.d} {cfg.layers} {cfg.heads}\n", encoding="utf-8"
    )
    print(f"saved to {OUT} in {time.time() - t0:.0f}s, final val loss {evaluate():.4f}")

    for prompt in ["Anna found a", "The small mice", "Tom and Leo went to the", "Mia"]:
        ids = bpe.encode(prompt)
        out = model.generate(ids, 24, temperature=0.7, seed=1)
        print(f"  {prompt!r} -> {ascii(bpe.decode(out))}")


if __name__ == "__main__":
    main()
