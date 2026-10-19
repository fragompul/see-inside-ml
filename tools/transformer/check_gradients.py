"""Finite-difference check of the hand-written backward pass (float64, tiny model).

Run:  python tools/transformer/check_gradients.py
"""

import numpy as np

from gpt import GPT, GPTConfig


def main() -> None:
    cfg = GPTConfig(vocab=17, ctx=8, d=8, layers=2, heads=2)
    model = GPT(cfg, seed=3, dtype=np.float64)
    rng = np.random.default_rng(0)
    # larger random weights than the training init so every path carries signal
    for name, v in model.p.items():
        if name.endswith(".g"):
            v[:] = 1 + 0.1 * rng.standard_normal(v.shape)
        else:
            v[:] = 0.3 * rng.standard_normal(v.shape)

    idx = rng.integers(0, cfg.vocab, size=(3, 6))
    targets = rng.integers(0, cfg.vocab, size=(3, 6))
    loss, cache = model.loss(idx, targets)
    grads = model.backward(cache)

    eps = 1e-6
    worst = 0.0
    for name, param in model.p.items():
        flat = param.reshape(-1)
        picks = rng.choice(flat.size, size=min(12, flat.size), replace=False)
        for j in picks:
            old = flat[j]
            flat[j] = old + eps
            plus, _ = model.loss(idx, targets)
            flat[j] = old - eps
            minus, _ = model.loss(idx, targets)
            flat[j] = old
            numeric = (plus - minus) / (2 * eps)
            analytic = grads[name].reshape(-1)[j]
            err = abs(numeric - analytic) / max(1e-6, abs(numeric) + abs(analytic))
            worst = max(worst, err)
            if err > 1e-4:
                print(f"MISMATCH {name}[{j}] analytic={analytic:.8f} numeric={numeric:.8f}")
    print(f"loss={loss:.4f}  worst relative error over {len(model.p)} tensors = {worst:.2e}")
    assert worst < 1e-4, "backward pass is wrong"
    print("OK")


if __name__ == "__main__":
    main()
