import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
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


def generate(model, draft, prompt, max_tokens, cost, eos, block=16, adaptive=True, copier=None, k_fixed=None, backoff=8, log=None, copy_max=15, seed=False):
    _patch_model(model, draft.config.target_layer_ids)
    draft.bind(model)
    mask_id = int(draft.config.mask_token_id)
    target_cache = make_prompt_cache(model)
    draft_cache = make_prompt_cache(draft)
    prompt = mx.array(prompt)
    logits, hidden, offset = _prefill_target(model, prompt, target_cache, None, 2048)
    for c in draft_cache:
        c.offset = offset
    tokens = [int(mx.argmax(logits[0, -1]).item())]
    steps = []
    pred, got, skip, idle = 4.0, 4.0, 0, 0
    thr = {"dflash": 1e9, "plain": 1 / cost[1]}
    since_probe, primed = 0, False
    prompt_list = prompt.tolist()
    t0 = time.perf_counter()
    while len(tokens) < max_tokens and tokens[-1] not in eos:
        c = [int(x) for x in copier(np.array(prompt_list + tokens, dtype=np.intc))][:copy_max] if copier is not None else []
        if not c and copier is not None and skip == 0:
            since_probe += 1
            if thr["plain"] > thr["dflash"] and since_probe < 16:
                skip = 1
            else:
                since_probe = 0
        if skip > 0 and not c:
            ts = time.perf_counter()
            lt = model(mx.array([[tokens[-1]]]), target_cache)
            hidden = mx.concatenate([hidden, mx.concatenate(model._hidden_states, axis=-1)], axis=1)
            nxt = mx.argmax(lt[0, -1])
            mx.eval(nxt, hidden)
            tokens.append(int(nxt.item()))
            steps.append((0, 0, "plain"))
            thr["plain"] = 0.8 * thr["plain"] + 0.2 / (time.perf_counter() - ts)
            skip -= 1
            continue
        if c and not (seed and len(c) < block - 1):
            k = min(len(c), max_tokens - len(tokens))
            lt = model(mx.array([[tokens[-1]] + c[:k]]), target_cache)
            new_h = mx.concatenate(model._hidden_states, axis=-1)
            t_tok = mx.argmax(lt[0], axis=-1)
            mx.eval(t_tok, new_h)
            t_tok = t_tok.tolist()
            a = next((i for i in range(k) if c[i] != t_tok[i]), k)
            new = c[:a] + [t_tok[a]]
            stop = next((i for i, t in enumerate(new) if t in eos), None)
            tokens.extend(new[: stop + 1] if stop is not None else new)
            steps.append((k, a, "copy-only"))
            if k - a > 0:
                _trim_recent_cache(target_cache, k - a)
            hidden = mx.concatenate([hidden, new_h[:, : a + 1, :]], axis=1)
            continue
        ts = time.perf_counter()
        n_before = len(tokens)
        seeds = c[: block - 1] if seed else []
        block_in = mx.array([[tokens[-1]] + seeds + [mask_id] * (block - 1 - len(seeds))])
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
        new = d_tok[:a] + [t_tok[a]]
        stop = next((i for i, t in enumerate(new) if t in eos), None)
        if stop is not None:
            new = new[: stop + 1]
        tokens.extend(new)
        steps.append((k, a, src))
        if log is not None:
            top2 = mx.argpartition(-probs, 1, axis=-1)[:, :2]
            log.append((d_tok[:k], t_tok, a, top2.tolist(), mx.take_along_axis(probs, top2, axis=-1).tolist()))
        if adaptive and k_fixed is None and src == "dflash" and k > 0:
            pred, got = 0.95 * pred + exp_acc[k] / min(1.0, got / pred), 0.95 * got + a
        if k - a > 0:
            _trim_recent_cache(target_cache, k - a)
        hidden = hidden[:, : a + 1, :]
        r = (len(tokens) - n_before) / (time.perf_counter() - ts)
        if primed:
            thr["dflash"] = r if thr["dflash"] > 1e8 else 0.8 * thr["dflash"] + 0.2 * r
        primed = True
    return tokens[:max_tokens], steps, time.perf_counter() - t0


def generate_copy(model, prompt, max_tokens, eos, copier, copy_max=15):
    cache = make_prompt_cache(model)
    prompt_list = list(prompt)
    p = mx.array(prompt_list)
    start = 0
    while p.size - start > 1:
        n = min(2048, p.size - start - 1)
        model(p[None, start : start + n], cache=cache)
        mx.eval([c.state for c in cache])
        start += n
    logits = model(p[None, start:], cache=cache)
    tokens = [int(mx.argmax(logits[0, -1]).item())]
    steps = []
    t0 = time.perf_counter()
    while len(tokens) < max_tokens and tokens[-1] not in eos:
        c = [int(x) for x in copier(np.array(prompt_list + tokens, dtype=np.intc))][:copy_max]
        k = min(len(c), max_tokens - len(tokens))
        lt = model(mx.array([[tokens[-1]] + c[:k]]), cache=cache)
        t_tok = mx.argmax(lt[0], axis=-1)
        mx.eval(t_tok)
        t_tok = t_tok.tolist()
        a = next((i for i in range(k) if c[i] != t_tok[i]), k)
        new = c[:a] + [t_tok[a]]
        stop = next((i for i, t in enumerate(new) if t in eos), None)
        tokens.extend(new[: stop + 1] if stop is not None else new)
        steps.append((k, a, "copy" if k else "plain"))
        if k - a > 0:
            _trim_recent_cache(cache, k - a)
    return tokens[:max_tokens], steps, time.perf_counter() - t0
