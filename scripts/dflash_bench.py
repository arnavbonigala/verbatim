import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
mx.set_cache_limit(1 << 29)
from mlx_lm import stream_generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from dflash.model_mlx import stream_generate as dflash_generate
from dflash.benchmark import load_and_process_dataset, load_mlx_models
from verbatim import mlx_fast
import dflash.model_mlx as dm

_snap = dm.snapshot_download
dm.snapshot_download = lambda repo, **kw: repo if os.path.isdir(repo) else _snap(repo, **kw)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/mlx-qwen3-8b-4bit")
ap.add_argument("--draft", default="models/dflash-qwen3-8b")
ap.add_argument("--datasets", default="gsm8k,humaneval,mt-bench")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--configs", default="base,dflash5,fast5,fast8,fast12,fast16")
ap.add_argument("--out", default="results/dflash_qwen3_8b.jsonl")
ap.add_argument("--stock", action="store_true")
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--copy-max", type=int, default=15)
ap.add_argument("--draft-ctx", type=int, default=None)
ap.add_argument("--repeat", type=int, default=1)
args = ap.parse_args()

model, draft, tok = load_mlx_models(args.model, args.draft, 4)
if not args.stock:
    mlx_fast.accelerate(model, "data/mlx_plan_mlx-qwen3-8b-4bit.json", free=True)
    mlx_fast.accelerate(draft, "data/mlx_plan_dflash-qwen3-8b.json", free=True)


def prompt_ids(text):
    msgs = [{"role": "user", "content": text}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(s, add_special_tokens=False)


from verbatim import spec_mlx
from verbatim.drafter import Verbatim

COSTS = {}
CACHE = f"data/verify_costs_{'stock' if args.stock else 'verbatim'}.json"
cached = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
for en in ((False,) if args.stock else (True, "strict")):
    mlx_fast.ENABLED, mlx_fast.STRICT = en is True, en == "strict"
    if str(en) in cached:
        COSTS[en] = cached[str(en)]
        continue
    COSTS[en] = [spec_mlx.verify_costs(model, ctx=c, widths=16 if en == "strict" else max(16, args.copy_max + 1)) for c in (512, 6000)]
    print("verify cost ms", en, [[round(x * 1000) for x in c[1:]] for c in COSTS[en]], flush=True)
json.dump({str(k): v for k, v in COSTS.items()}, open(CACHE, "w"))
mlx_fast.STRICT = False


def costs_at(en, n):
    a, b = COSTS[en]
    f = min(max((n - 512) / (6000 - 512), 0.0), 1.5)
    return [x + (y - x) * f for x, y in zip(a, b)]
EOS = set(tok.eos_token_ids)
tb = lambda b: tok._tokenizer.encode(b.decode("utf-8", "replace"), add_special_tokens=False)
db = lambda ids: tok._tokenizer.decode(ids).encode()


STEPS = {}


def summarize(cfg, steps):
    by = collections.defaultdict(lambda: [0, 0, 0])
    for k, a, src in steps:
        by[src][0] += 1; by[src][1] += a + 1; by[src][2] += k
    STEPS[cfg] = {s: {"steps": n, "tokens": t, "drafted": d} for s, (n, t, d) in by.items()}


def run(cfg, ids):
    mlx_fast.ENABLED = cfg.startswith("fast") or (cfg.startswith("adapt") and not cfg.endswith("stock"))
    mlx_fast.STRICT = cfg in ("sbase", "sverb", "sverbr") or cfg.startswith(("strict", "sadapt"))
    out, t0 = [], None
    if cfg in ("verb", "sverb", "sverbr"):
        cm = 15 if mlx_fast.STRICT else args.copy_max
        copier = Verbatim(tb, db, costs_at("strict" if mlx_fast.STRICT else mlx_fast.ENABLED, len(ids)) + [9.9] * 64, k_max=cm)
        out, steps, dt = spec_mlx.generate_copy(model, ids, args.max_tokens, EOS, copier, copy_max=cm, recycle=7 if cfg.endswith("r") else 0)
        summarize(cfg, steps)
        return out, dt, len(steps)
    if cfg.startswith("adapt") or cfg.startswith("sadapt"):
        cost = costs_at("strict" if mlx_fast.STRICT else mlx_fast.ENABLED, len(ids))
        cm = 15 if mlx_fast.STRICT else args.copy_max
        copier = Verbatim(tb, db, cost + [9.9] * 64, k_max=cm) if cfg.endswith(("c", "cs")) else None
        out, steps, dt = spec_mlx.generate(model, draft, ids, args.max_tokens, cost, EOS, copier=copier, block=args.block, copy_max=cm, seed=cfg.endswith("cs"), warmup=8 if "w" in cfg else 0, draft_ctx=1024 if "d" in cfg[6:] else args.draft_ctx, recycle=7 if "r" in cfg[6:] else 0, arbitrate="x" in cfg[6:])
        summarize(cfg, steps)
        return out, dt, len(steps)
    if cfg in ("base", "sbase"):
        gen = mlx_generate(model, tok, ids, max_tokens=args.max_tokens, sampler=make_sampler(0.0))
        for r in gen:
            if t0 is None:
                t0 = time.perf_counter()
            out.append(r.token)
        steps = len(out)
    else:
        bs = int(cfg.replace("dflash", "").replace("fast", "").replace("strict", ""))
        steps = 0
        for r in dflash_generate(model, draft, tok, ids, block_size=bs, max_tokens=args.max_tokens):
            if t0 is None:
                t0 = time.perf_counter()
            out.extend(r.tokens)
            steps += 1
    dt = time.perf_counter() - t0
    return out, dt, steps


configs = args.configs.split(",")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
done = set()
if os.path.exists(args.out):
    done = {(r["dataset"], r["i"], r["cfg"]) for r in map(json.loads, open(args.out))}
warm = prompt_ids("Say hello.")
for cfg in configs:
    run(cfg, warm)
with open(args.out, "a") as fh:
    for ds in args.datasets.split(","):
        rows = (lambda w: w[:: max(1, len(w) // args.n)][: args.n])([json.loads(l) for l in open("data/workload.jsonl")]) if ds == "agent" else load_and_process_dataset(ds)[: args.n]
        for i, row in enumerate(rows):
            ids = tok.encode(row["prompt"] + "<think>\n\n</think>\n\n", add_special_tokens=False) if ds == "agent" else prompt_ids(row["turns"][0])
            res = {}
            order = configs[i % len(configs):] + configs[: i % len(configs)]
            for cfg in order:
                if (ds, i, cfg) in done:
                    continue
                out, dt, steps = min((run(cfg, ids) for _ in range(args.repeat)), key=lambda r: r[1])
                mx.clear_cache()
                res[cfg] = (out, dt, steps)
                time.sleep(2)
            ref = lambda c: res.get("sbase" if c in ("sbase", "sverb", "sverbr") or c.startswith(("strict", "sadapt")) else "base", (None,))[0]
            for cfg, (out, dt, steps) in res.items():
                base = ref(cfg)
                n_same = next((j for j in range(min(len(out), len(base or []))) if out[j] != base[j]), min(len(out), len(base or []))) if base else None
                rec = {"dataset": ds, "i": i, "cfg": cfg, "prompt_tokens": len(ids), "n": len(out), "dt": dt, "steps": steps, "identical": base is not None and out == base, "agree_prefix": n_same, "out": out, "sources": STEPS.pop(cfg, None)}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            line = " ".join(f"{c}={len(o)/d:.1f}t/s({len(o)/max(1,s):.1f}/pass){'' if ref(c) is None or o == ref(c) else '!'}" for c, (o, d, s) in res.items())
            print(f"[{ds} {i}] {line}", flush=True)
