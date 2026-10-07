import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
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
args = ap.parse_args()

model, draft, tok = load_mlx_models(args.model, args.draft, 4)
mlx_fast.accelerate(model, "data/mlx_plan_mlx-qwen3-8b-4bit.json")


def prompt_ids(text):
    msgs = [{"role": "user", "content": text}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(s, add_special_tokens=False)


from verbatim import spec_mlx
from verbatim.drafter import Verbatim

COSTS = {}
for en in (False, True, "strict"):
    mlx_fast.ENABLED, mlx_fast.STRICT = en is True, en == "strict"
    COSTS[en] = spec_mlx.verify_costs(model)
mlx_fast.STRICT = False
print("verify cost ms stock", [round(c * 1000) for c in COSTS[False][1:]], "fast", [round(c * 1000) for c in COSTS[True][1:]], flush=True)
EOS = set(tok.eos_token_ids)
tb = lambda b: tok._tokenizer.encode(b.decode("utf-8", "replace"), add_special_tokens=False)
db = lambda ids: tok._tokenizer.decode(ids).encode()


def run(cfg, ids):
    mlx_fast.ENABLED = cfg.startswith("fast") or (cfg.startswith("adapt") and not cfg.endswith("stock"))
    mlx_fast.STRICT = cfg == "sbase" or cfg.startswith(("strict", "sadapt"))
    out, t0 = [], None
    if cfg.startswith("adapt") or cfg.startswith("sadapt"):
        cost = COSTS["strict" if mlx_fast.STRICT else mlx_fast.ENABLED]
        copier = Verbatim(tb, db, cost + [9.9] * 64, k_max=15) if cfg.endswith("c") else None
        out, steps, dt = spec_mlx.generate(model, draft, ids, args.max_tokens, cost, EOS, copier=copier)
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
                out, dt, steps = run(cfg, ids)
                res[cfg] = (out, dt, steps)
                time.sleep(2)
            ref = lambda c: res.get("sbase" if c == "sbase" or c.startswith(("strict", "sadapt")) else "base", (None,))[0]
            for cfg, (out, dt, steps) in res.items():
                base = ref(cfg)
                n_same = next((j for j in range(min(len(out), len(base or []))) if out[j] != base[j]), min(len(out), len(base or []))) if base else None
                rec = {"dataset": ds, "i": i, "cfg": cfg, "prompt_tokens": len(ids), "n": len(out), "dt": dt, "steps": steps, "identical": base is not None and out == base, "agree_prefix": n_same}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            line = " ".join(f"{c}={len(o)/d:.1f}t/s({len(o)/max(1,s):.1f}/pass){'' if ref(c) is None or o == ref(c) else '!'}" for c, (o, d, s) in res.items())
            print(f"[{ds} {i}] {line}", flush=True)
