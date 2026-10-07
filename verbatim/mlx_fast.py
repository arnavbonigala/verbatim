import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.models.base as base

from verbatim import attn, qmv

MAX_M = 16
ENABLED = True
STRICT = False


def _time(f, n=8):
    for _ in range(2):
        mx.eval(f())
    ts = time.perf_counter()
    for _ in range(n):
        mx.eval(f())
    return (time.perf_counter() - ts) / n


def _tune_linear(m):
    K = m.weight.shape[1] * 8
    stock = nn.QuantizedLinear.__call__
    use = []
    for M in range(1, MAX_M + 1):
        x = mx.random.normal((1, M, K)).astype(m.scales.dtype)
        a = _time(lambda: [stock(m, x) for _ in range(4)])
        b = _time(lambda: [qmv.qmv(x, m._vb, m._sb) for _ in range(4)])
        use.append(b < 0.97 * a)
    return use


def _tune_attention(T=4096, H=28, kvh=4, D=128, dtype=mx.float16):
    k = mx.random.normal((1, kvh, T, D)).astype(dtype)
    v = mx.random.normal((1, kvh, T, D)).astype(dtype)
    use = []
    for L in range(1, MAX_M + 1):
        q = mx.random.normal((1, H, L, D)).astype(dtype)
        a = _time(lambda: [mx.fast.scaled_dot_product_attention(q, k, v, scale=0.088, mask="causal") for _ in range(8)])
        b = _time(lambda: [attn.attention(q, k, v, 0.088) for _ in range(8)])
        use.append(b < 0.97 * a)
    return use


def accelerate(model, profile_path=None, strict=False):
    shapes = {}
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear) and m.bits == 4 and m.group_size == 64 and m.weight.shape[0] % (8 * qmv.RT * qmv.SIMDS) == 0:
            m._vb = qmv.repack(m.weight)
            m._sb = qmv.pack_sb(m.scales, m.biases)
            shapes.setdefault(tuple(m.weight.shape), []).append(m)
    plan = json.load(open(profile_path)) if profile_path and os.path.exists(profile_path) else None
    if plan is None:
        cfg = model.args if hasattr(model, "args") else None
        plan = {"linear": {f"{s[0]}x{s[1]}": _tune_linear(ms[0]) for s, ms in shapes.items()}}
        plan["attention"] = _tune_attention(H=cfg.num_attention_heads, kvh=cfg.num_key_value_heads, D=cfg.hidden_size // cfg.num_attention_heads, dtype=next(m for ms in shapes.values() for m in ms).scales.dtype) if cfg else [False] * MAX_M
        if profile_path:
            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            json.dump(plan, open(profile_path, "w"))
    if strict:
        plan = {"linear": {k: [True] * MAX_M for k in plan["linear"]}, "attention": [True] * MAX_M}
    for s, ms in shapes.items():
        use = plan["linear"][f"{s[0]}x{s[1]}"]
        for m in ms:
            m._use = use
    stock = nn.QuantizedLinear.__call__

    def linear(self, x):
        rows = x.size // x.shape[-1]
        use = getattr(self, "_use", None)
        if use and rows <= MAX_M and (STRICT or (ENABLED and use[rows - 1])):
            y = qmv.gemv(x, self._vb, self._sb) if rows == 1 and STRICT else qmv.qmv(x, self._vb, self._sb)
            return y + self.bias if "bias" in self else y
        return stock(self, x)

    nn.QuantizedLinear.__call__ = linear
    sdpa = base.scaled_dot_product_attention
    use_attn = plan["attention"]

    def attention(queries, keys, values, cache, scale, mask, sinks=None):
        L = queries.shape[2]
        if sinks is None and not hasattr(cache, "bits") and L <= MAX_M and (STRICT or (ENABLED and use_attn[L - 1] and keys.shape[2] >= 1024)):
            return attn.attention(queries, keys, values, scale)
        return sdpa(queries, keys, values, cache, scale, mask, sinks)

    import sys
    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and getattr(mod, "scaled_dot_product_attention", None) is sdpa:
            mod.scaled_dot_product_attention = attention
    return plan
