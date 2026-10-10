import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache
from dflash.model_mlx import _patch_model, _prefill_target, _trim_recent_cache


def verify_costs(model, ctx=512, widths=16):
    cache = make_prompt_cache(model)
    p = mx.array(list(range(1000, 1000 + ctx)))[None]
    model(p, cache=cache)
    mx.eval([c.state for c in cache])
    cost = [0.0]
    for b in range(1, widths + 1):
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            mx.eval(model(p[:, -b:], cache=cache))
            ts.append(time.perf_counter() - t0)
            _trim_recent_cache(cache, b)
        cost.append(sorted(ts)[2])
    return cost


def generate(model, draft, prompt, max_tokens, cost, eos, block=16, adaptive=True, copier=None, k_fixed=None, backoff=8, log=None, copy_max=15, seed=False, warmup=0, draft_ctx=None, recycle=0, state=None, arbitrate=False):
    _patch_model(model, draft.config.target_layer_ids)
    draft.bind(model)
    mask_id = int(draft.config.mask_token_id)
    prompt = mx.array(prompt)
    if state and "cache" in state:
        target_cache, draft_cache, n0 = state["cache"], state["dcache"], state["len"]
        _rewind(target_cache, n0)
        _rewind(draft_cache, n0)
        logits, hidden, _ = _prefill_target(model, prompt[n0:], target_cache, None, 2048)
        hs = [state["hidden"], hidden] if state["hidden"] is not None else [hidden]
    else:
        target_cache = make_prompt_cache(model)
        draft_cache = make_prompt_cache(draft)
        if draft_ctx:
            draft_cache = [RotatingKVCache(max_size=draft_ctx, keep=0) for _ in draft_cache]
        logits, hidden, offset = _prefill_target(model, prompt, target_cache, draft_ctx, 2048)
        for c in draft_cache:
            c.offset = offset
        hs = [hidden]
    tokens = [int(mx.argmax(logits[0, -1]).item())]
    steps = []
    pred, got, skip, idle = 4.0, 4.0, 0, 0
    thr = {"dflash": 1e9, "plain": 1 / cost[1]}
    since_probe, primed, misses = 0, False, 0
    prompt_list = prompt.tolist()
    succ = dict(zip(prompt_list, prompt_list[1:])) if recycle else {}
    rs, rn = 1.0, 2.0
    t0 = time.perf_counter()
    while len(tokens) < max_tokens and tokens[-1] not in eos:
        c = [int(x) for x in copier(np.array(prompt_list + tokens, dtype=np.intc))][:copy_max] if copier is not None else []
        if arbitrate and c and primed and thr["dflash"] > (1 + copier.last_exp) / cost[min(len(c), len(cost) - 2) + 1]:
            c = []
        if not c and copier is not None and skip == 0:
            since_probe += 1
            misses += 1
            if (not primed and misses <= warmup) or (thr["plain"] > thr["dflash"] and since_probe < 16):
                skip = 1
            else:
                since_probe = 0
        rc = []
        if recycle and skip > 0 and not c:
            x, n = tokens[-1], _rlen(rs / rn, cost, recycle)
            while len(rc) < n and x in succ:
                x = succ[x]
                rc.append(x)
        if skip > 0 and not c and not rc:
            ts = time.perf_counter()
            lt = model(mx.array([[tokens[-1]]]), target_cache)
            hs.append(mx.concatenate(model._hidden_states, axis=-1))
            nxt = mx.argmax(lt[0, -1])
            mx.eval(nxt, hs[-1])
            succ[tokens[-1]] = int(nxt.item())
            tokens.append(int(nxt.item()))
            steps.append((0, 0, "plain", time.perf_counter()))
            thr["plain"] = 0.8 * thr["plain"] + 0.2 / (time.perf_counter() - ts)
            skip -= 1
            continue
        if rc or (c and not (seed and len(c) < block - 1)):
            ts, src = time.perf_counter(), "recycle" if rc else "copy-only"
            c = c or rc
            k = min(len(c), max_tokens - len(tokens))
            lt = model(mx.array([[tokens[-1]] + c[:k]]), target_cache)
            new_h = mx.concatenate(model._hidden_states, axis=-1)
            t_tok = mx.argmax(lt[0], axis=-1)
            mx.eval(t_tok, new_h)
            t_tok = t_tok.tolist()
            a = next((i for i in range(k) if c[i] != t_tok[i]), k)
            inp = [tokens[-1]] + c[:k]
            for i in range(a + 1):
                succ[inp[i]] = t_tok[i]
            new = c[:a] + [t_tok[a]]
            stop = next((i for i, t in enumerate(new) if t in eos), None)
            tokens.extend(new[: stop + 1] if stop is not None else new)
            steps.append((k, a, src, time.perf_counter()))
            if rc:
                skip -= 1
                rs, rn = 0.95 * rs + a, 0.95 * rn + min(a + 1, k)
                thr["plain"] = 0.8 * thr["plain"] + 0.2 * (a + 1) / (time.perf_counter() - ts)
            if k - a > 0:
                _trim_recent_cache(target_cache, k - a)
            hs.append(new_h[:, : a + 1, :])
            continue
        ts = time.perf_counter()
        n_before = len(tokens)
        seeds = c[: block - 1] if seed else []
        block_in = mx.array([[tokens[-1]] + seeds + [mask_id] * (block - 1 - len(seeds))])
        hidden = mx.concatenate(hs, axis=1) if len(hs) > 1 else hs[0]
        if draft_ctx and hidden.shape[1] > draft_ctx:
            drop = hidden.shape[1] - draft_ctx
            hidden = hidden[:, drop:]
            for dc in draft_cache:
                dc.offset += drop
        dl = draft(block_in, hidden, draft_cache, logits_start=1)
        trim_n = draft_cache[0].offset - (prompt.size + len(tokens) - 1)
        if trim_n > 0:
            _trim_recent_cache(draft_cache, trim_n)
        probs = mx.softmax(dl[0].astype(mx.float32), axis=-1)
        d_tok = mx.argmax(probs, axis=-1)
        d_p = mx.take_along_axis(probs, d_tok[:, None], axis=-1)[:, 0]
        mx.eval(d_tok, d_p)
        d_tok, d_p = d_tok.tolist(), np.array(d_p.tolist())
        if seeds:
            d_tok[: len(seeds)] = seeds
            d_p[: len(seeds)] = 0.95
        src = "seeded" if seeds else "dflash"
        if k_fixed is not None:
            k = k_fixed
        elif adaptive:
            exp_acc = np.concatenate([[0.0], np.cumsum(np.cumprod(d_p))]) * min(1.0, got / pred)
            rate = [(1 + exp_acc[k]) / cost[k + 1] for k in range(block)]
            k = int(np.argmax(rate))
            idle = idle + 1 if k == 0 else 0
            if backoff and idle >= 2:
                skip, idle = backoff, 0
        else:
            k = block - 1
        k = min(k, max_tokens - len(tokens))
        verify_in = mx.array([[tokens[-1]] + d_tok[:k]])
        lt = model(verify_in, target_cache)
        hidden = mx.concatenate(model._hidden_states, axis=-1)
        t_tok = mx.argmax(lt[0], axis=-1)
        mx.eval(t_tok, hidden)
        t_tok = t_tok.tolist()
        a = next((i for i in range(k) if d_tok[i] != t_tok[i]), k)
        inp = [tokens[-1]] + d_tok[:k]
        for i in range(a + 1):
            succ[inp[i]] = t_tok[i]
        new = d_tok[:a] + [t_tok[a]]
        stop = next((i for i, t in enumerate(new) if t in eos), None)
        if stop is not None:
            new = new[: stop + 1]
        tokens.extend(new)
        steps.append((k, a, src, time.perf_counter()))
        if log is not None:
            top2 = mx.argpartition(-probs, 1, axis=-1)[:, :2]
            log.append((d_tok[:k], t_tok, a, top2.tolist(), mx.take_along_axis(probs, top2, axis=-1).tolist()))
        if adaptive and k_fixed is None and src == "dflash" and k > 0:
            pred, got = 0.95 * pred + exp_acc[k] / min(1.0, got / pred), 0.95 * got + a
        if k - a > 0:
            _trim_recent_cache(target_cache, k - a)
        hs = [hidden[:, : a + 1, :]]
        r = (len(tokens) - n_before) / (time.perf_counter() - ts)
        if primed:
            thr["dflash"] = r if thr["dflash"] > 1e8 else 0.8 * thr["dflash"] + 0.2 * r
        primed = True
    dt = time.perf_counter() - t0
    if state is not None:
        n0, d0 = prompt.size, draft_cache[0].offset
        hidden = mx.concatenate(hs, axis=1) if len(hs) > 1 else hs[0]
        state.update(cache=target_cache, dcache=draft_cache, len=n0, hidden=hidden[:, : n0 - d0] if d0 < n0 else None)
    return tokens[:max_tokens], steps, dt


def _rlen(q, cost, n):
    return max(range(n + 1), key=lambda k: (1 + sum(q ** i for i in range(1, k + 1))) / cost[k + 1])


def _rewind(cache, n):
    k = cache[0].offset - n
    if k > 0:
        _trim_recent_cache(cache, k)


def generate_copy(model, prompt, max_tokens, eos, copier, copy_max=15, recycle=0, state=None, cost=None):
    cache = state.get("cache") or make_prompt_cache(model) if state is not None else make_prompt_cache(model)
    prompt_list = list(prompt)
    p = mx.array(prompt_list)
    start = state.get("len", 0) if state is not None else 0
    _rewind(cache, start)
    while p.size - start > 1:
        n = min(2048, p.size - start - 1)
        model(p[None, start : start + n], cache=cache)
        mx.eval([c.state for c in cache])
        start += n
    logits = model(p[None, start:], cache=cache)
    tokens = [int(mx.argmax(logits[0, -1]).item())]
    steps = []
    succ = dict(zip(prompt_list, prompt_list[1:])) if recycle else {}
    rs, rn = 1.0, 2.0
    t0 = time.perf_counter()
    while len(tokens) < max_tokens and tokens[-1] not in eos:
        c = [int(x) for x in copier(np.array(prompt_list + tokens, dtype=np.intc))][:copy_max]
        src = "copy" if c else "plain"
        if recycle and not c:
            x, n = tokens[-1], _rlen(rs / rn, cost, recycle) if cost else recycle
            while len(c) < n and x in succ:
                x = succ[x]
                c.append(x)
            src = "recycle" if c else "plain"
        k = min(len(c), max_tokens - len(tokens))
        lt = model(mx.array([[tokens[-1]] + c[:k]]), cache=cache)
        t_tok = mx.argmax(lt[0], axis=-1)
        mx.eval(t_tok)
        t_tok = t_tok.tolist()
        a = next((i for i in range(k) if c[i] != t_tok[i]), k)
        inp = [tokens[-1]] + c[:k]
        for i in range(a + 1):
            succ[inp[i]] = t_tok[i]
        new = c[:a] + [t_tok[a]]
        stop = next((i for i, t in enumerate(new) if t in eos), None)
        tokens.extend(new[: stop + 1] if stop is not None else new)
        steps.append((k, a, src, time.perf_counter()))
        if src == "recycle":
            rs, rn = 0.95 * rs + a, 0.95 * rn + min(a + 1, k)
        if k - a > 0:
            _trim_recent_cache(cache, k - a)
    dt = time.perf_counter() - t0
    if state is not None:
        state.update(cache=cache, len=len(prompt_list))
    return tokens[:max_tokens], steps, dt
