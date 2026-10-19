"""A GPT-2-style transformer in plain NumPy: forward AND hand-written backward pass.

Architecture (identical to GPT-2, just smaller): learned token + position embeddings, N pre-LN
blocks (LayerNorm -> causal multi-head self-attention -> residual, LayerNorm -> GELU MLP ->
residual), a final LayerNorm and an output projection tied to the token embedding.

Shapes: B batch, T sequence length, D model width, H heads, hd = D / H, V vocabulary.
The TypeScript implementation in packages/deep mirrors `forward` operation for operation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LN_EPS = 1e-5
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


@dataclass(frozen=True)
class GPTConfig:
    vocab: int
    ctx: int
    d: int
    layers: int
    heads: int

    @property
    def head_dim(self) -> int:
        return self.d // self.heads


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x**3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * SQRT_2_OVER_PI * (1.0 + 3 * 0.044715 * x**2)


def layer_norm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + LN_EPS)
    xhat = (x - mu) * inv
    return xhat * g + b, (xhat, inv, g)


def layer_norm_backward(dy, cache):
    xhat, inv, g = cache
    n = xhat.shape[-1]
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    dg = (dy * xhat).reshape(-1, n).sum(0)
    db = dy.reshape(-1, n).sum(0)
    return dx, dg, db


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class GPT:
    def __init__(self, cfg: GPTConfig, seed: int = 0, dtype=np.float32):
        assert cfg.d % cfg.heads == 0
        self.cfg = cfg
        self.dtype = dtype
        rng = np.random.default_rng(seed)
        d, L = cfg.d, cfg.layers
        std = 0.02
        proj_std = std / np.sqrt(2 * L)  # GPT-2: damp the residual-stream writes

        def normal(shape, s=std):
            return (rng.standard_normal(shape) * s).astype(dtype)

        p: dict[str, np.ndarray] = {
            "wte": normal((cfg.vocab, d)),
            "wpe": normal((cfg.ctx, d), 0.01),
            "lnf.g": np.ones(d, dtype),
            "lnf.b": np.zeros(d, dtype),
        }
        for i in range(L):
            p |= {
                f"h{i}.ln1.g": np.ones(d, dtype),
                f"h{i}.ln1.b": np.zeros(d, dtype),
                f"h{i}.attn.w": normal((d, 3 * d)),
                f"h{i}.attn.b": np.zeros(3 * d, dtype),
                f"h{i}.proj.w": normal((d, d), proj_std),
                f"h{i}.proj.b": np.zeros(d, dtype),
                f"h{i}.ln2.g": np.ones(d, dtype),
                f"h{i}.ln2.b": np.zeros(d, dtype),
                f"h{i}.fc.w": normal((d, 4 * d)),
                f"h{i}.fc.b": np.zeros(4 * d, dtype),
                f"h{i}.fc2.w": normal((4 * d, d), proj_std),
                f"h{i}.fc2.b": np.zeros(d, dtype),
            }
        self.p = p

    @property
    def n_params(self) -> int:
        return sum(v.size for v in self.p.values())

    # ------------------------------------------------------------------ forward
    def forward(self, idx: np.ndarray, trace: bool = False):
        """idx: (B, T) int array. Returns logits (B, T, V), cache (and a `trace` dict if asked)."""
        cfg, p = self.cfg, self.p
        B, T = idx.shape
        assert T <= cfg.ctx
        H, hd, d = cfg.heads, cfg.head_dim, cfg.d
        tok = p["wte"][idx]
        pos = p["wpe"][:T]
        x = tok + pos
        cache: dict = {"idx": idx, "layers": []}
        tr: dict = {"tok": tok, "pos": pos, "x0": x, "layers": []} if trace else {}
        mask = np.triu(np.ones((T, T), dtype=bool), 1)

        for i in range(cfg.layers):
            h1, ln1 = layer_norm(x, p[f"h{i}.ln1.g"], p[f"h{i}.ln1.b"])
            qkv = h1 @ p[f"h{i}.attn.w"] + p[f"h{i}.attn.b"]
            q, k, v = np.split(qkv, 3, axis=-1)

            def heads(a):
                return a.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

            q, k, v = heads(q), heads(k), heads(v)
            scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)
            masked = np.where(mask, -np.inf, scores)
            probs = softmax(masked)
            y = probs @ v  # (B,H,T,hd)
            y_merged = y.transpose(0, 2, 1, 3).reshape(B, T, d)
            attn_out = y_merged @ p[f"h{i}.proj.w"] + p[f"h{i}.proj.b"]
            x_mid = x + attn_out

            h2, ln2 = layer_norm(x_mid, p[f"h{i}.ln2.g"], p[f"h{i}.ln2.b"])
            a = h2 @ p[f"h{i}.fc.w"] + p[f"h{i}.fc.b"]
            g = gelu(a)
            mlp_out = g @ p[f"h{i}.fc2.w"] + p[f"h{i}.fc2.b"]
            x_next = x_mid + mlp_out

            cache["layers"].append(
                dict(h1=h1, ln1=ln1, q=q, k=k, v=v, probs=probs, y_merged=y_merged,
                     h2=h2, ln2=ln2, a=a, g=g)
            )
            if trace:
                tr["layers"].append(
                    dict(ln1=h1, q=q, k=k, v=v, scores=scores, probs=probs, heads_out=y_merged,
                         attn_out=attn_out, resid_mid=x_mid, ln2=h2, mlp_pre=a, mlp_act=g,
                         mlp_out=mlp_out, resid_out=x_next)
                )
            x = x_next

        xf, lnf = layer_norm(x, p["lnf.g"], p["lnf.b"])
        logits = xf @ p["wte"].T
        cache |= {"xf": xf, "lnf": lnf}
        if trace:
            tr |= {"final_ln": xf, "logits": logits}
            return logits, cache, tr
        return logits, cache

    def loss(self, idx: np.ndarray, targets: np.ndarray):
        logits, cache = self.forward(idx)
        probs = softmax(logits.astype(np.float64))
        B, T = targets.shape
        nll = -np.log(probs[np.arange(B)[:, None], np.arange(T)[None, :], targets] + 1e-30)
        cache["probs_out"] = probs
        cache["targets"] = targets
        return float(nll.mean()), cache

    # ------------------------------------------------------------------ backward
    def backward(self, cache) -> dict[str, np.ndarray]:
        cfg, p = self.cfg, self.p
        idx, targets = cache["idx"], cache["targets"]
        B, T = idx.shape
        H, hd, d = cfg.heads, cfg.head_dim, cfg.d
        g: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in p.items()}

        dlogits = cache["probs_out"].astype(self.dtype)
        dlogits[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1.0
        dlogits /= B * T

        xf = cache["xf"]
        g["wte"] += dlogits.reshape(-1, cfg.vocab).T @ xf.reshape(-1, d)
        dxf = dlogits @ p["wte"]
        dx, dg, db = layer_norm_backward(dxf, cache["lnf"])
        g["lnf.g"] += dg
        g["lnf.b"] += db

        for i in reversed(range(cfg.layers)):
            c = cache["layers"][i]
            # --- MLP branch (x_next = x_mid + mlp_out)
            dmlp = dx
            g[f"h{i}.fc2.w"] += c["g"].reshape(-1, 4 * d).T @ dmlp.reshape(-1, d)
            g[f"h{i}.fc2.b"] += dmlp.reshape(-1, d).sum(0)
            dgelu = dmlp @ p[f"h{i}.fc2.w"].T
            da = dgelu * gelu_grad(c["a"])
            g[f"h{i}.fc.w"] += c["h2"].reshape(-1, d).T @ da.reshape(-1, 4 * d)
            g[f"h{i}.fc.b"] += da.reshape(-1, 4 * d).sum(0)
            dh2 = da @ p[f"h{i}.fc.w"].T
            dxm, dg2, db2 = layer_norm_backward(dh2, c["ln2"])
            g[f"h{i}.ln2.g"] += dg2
            g[f"h{i}.ln2.b"] += db2
            dx_mid = dx + dxm

            # --- attention branch (x_mid = x + attn_out)
            dattn = dx_mid
            g[f"h{i}.proj.w"] += c["y_merged"].reshape(-1, d).T @ dattn.reshape(-1, d)
            g[f"h{i}.proj.b"] += dattn.reshape(-1, d).sum(0)
            dy = (dattn @ p[f"h{i}.proj.w"].T).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            probs, q, k, v = c["probs"], c["q"], c["k"], c["v"]
            dprobs = dy @ v.transpose(0, 1, 3, 2)
            dv = probs.transpose(0, 1, 3, 2) @ dy
            dscores = probs * (dprobs - (dprobs * probs).sum(-1, keepdims=True))
            dscores = dscores / np.sqrt(hd)
            dq = dscores @ k
            dk = dscores.transpose(0, 1, 3, 2) @ q

            def merge(a):
                return a.transpose(0, 2, 1, 3).reshape(B, T, d)

            dqkv = np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1)
            g[f"h{i}.attn.w"] += c["h1"].reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
            g[f"h{i}.attn.b"] += dqkv.reshape(-1, 3 * d).sum(0)
            dh1 = dqkv @ p[f"h{i}.attn.w"].T
            dxa, dg1, db1 = layer_norm_backward(dh1, c["ln1"])
            g[f"h{i}.ln1.g"] += dg1
            g[f"h{i}.ln1.b"] += db1
            dx = dx_mid + dxa

        g["wpe"][:T] += dx.sum(0)
        np.add.at(g["wte"], idx, dx)
        return g

    # ------------------------------------------------------------------ generation
    def generate(self, ids: list[int], n: int, temperature: float = 1.0, seed: int = 0) -> list[int]:
        rng = np.random.default_rng(seed)
        out = list(ids)
        for _ in range(n):
            window = np.array([out[-self.cfg.ctx :]])
            logits, _ = self.forward(window)
            z = logits[0, -1].astype(np.float64) / max(temperature, 1e-6)
            out.append(int(rng.choice(len(z), p=softmax(z))))
        return out
