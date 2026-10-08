import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dflash.model_mlx as dm

_snap = dm.snapshot_download
dm.snapshot_download = lambda repo, **kw: repo if os.path.isdir(repo) else _snap(repo, **kw)

import mlx.core as mx
import pyarrow.parquet as pq
from dflash.benchmark import load_mlx_models
from verbatim import mlx_fast, spec_mlx
from verbatim.drafter import Verbatim

ap = argparse.ArgumentParser()
ap.add_argument("shard")
ap.add_argument("--trajs", type=int, default=2)
ap.add_argument("--turns", type=int, default=12)
ap.add_argument("--max-ctx", type=int, default=12000)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--configs", default="plain,sverb,sadaptc")
ap.add_argument("--out", default="results/session_mlx.jsonl")
args = ap.parse_args()
mx.set_cache_limit(1 << 29)

model, draft, tok = load_mlx_models("models/mlx-qwen3-8b-4bit", "models/dflash-qwen3-8b", 4)
mlx_fast.accelerate(model, "data/mlx_plan_mlx-qwen3-8b-4bit.json", free=True)
mlx_fast.accelerate(draft, "data/mlx_plan_dflash-qwen3-8b.json", free=True)
mlx_fast.ENABLED, mlx_fast.STRICT = False, True
EOS = set(tok.eos_token_ids)
COST = json.load(open("data/verify_costs_verbatim.json"))["strict"]
tb = lambda b: tok._tokenizer.encode(b.decode("utf-8", "replace"), add_special_tokens=False)
db = lambda ids: tok._tokenizer.decode(ids).encode()
T = lambda s: tok._tokenizer.encode(s, add_special_tokens=False)
HEAD = T("<|im_start|>assistant\n<think>\n\n</think>\n\n")
END = T("<|im_end|>\n")


def cost_at(n):
    a, b = COST
    f = min(1.0, max(0.0, (n - 512) / (6000 - 512)))
    return [x + f * (y - x) for x, y in zip(a, b)]


def turns(msgs):
    ctx = []
    for m in msgs:
        if m["role"] == "assistant":
            if ctx and len(ctx) + len(HEAD) < args.max_ctx:
                yield ctx + HEAD
            ctx = ctx + HEAD + T(m["content"]) + END
        else:
            ctx = ctx + T(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")


def run(cfg, prompt, state):
    cost = cost_at(len(prompt))
    if cfg == "plain":
        return spec_mlx.generate_copy(model, prompt, args.max_tokens, EOS, lambda x: [], state=state)
    copier = Verbatim(tb, db, cost + [9.9] * 64, k_max=15)
    if cfg == "sverb":
        return spec_mlx.generate_copy(model, prompt, args.max_tokens, EOS, copier, copy_max=15, state=state)
    return spec_mlx.generate(model, draft, prompt, args.max_tokens, cost, EOS, copier=copier, block=8, copy_max=15, state=state)


rows = [r for r in pq.read_table(args.shard, columns=["model", "messages", "traj_id"]).to_pylist() if r["model"].startswith("claude-3-7")]
rows.sort(key=lambda r: r["traj_id"])
configs = args.configs.split(",")
spec_mlx.generate_copy(model, T("Say hello."), 8, EOS, lambda x: [])
with open(args.out, "a") as fh:
    for ti, r in enumerate(rows[: args.trajs]):
        prompts = list(turns(json.loads(r["messages"])))[: args.turns]
        outs = {}
        for cfg in configs[ti % len(configs):] + configs[: ti % len(configs)]:
            state, outs[cfg] = {}, []
            mx.clear_cache()
            for j, p in enumerate(prompts):
                t = time.perf_counter()
                out, steps, dt = run(cfg, p, state)
                mx.clear_cache()
                outs[cfg].append((out, dt, len(steps), time.perf_counter() - t - dt, len(p)))
        for cfg in configs:
            for j, (out, dt, ns, pre, n) in enumerate(outs[cfg]):
                fh.write(json.dumps({"traj": r["traj_id"], "turn": j, "cfg": cfg, "ctx": n, "n": len(out), "dt": dt, "prefill": pre, "steps": ns, "identical": out == outs["plain"][j][0] if "plain" in outs else None}) + "\n")
        fh.flush()
        line = " ".join(f"{c}={sum(x[0].__len__() for x in outs[c]) / sum(x[1] for x in outs[c]):.1f}t/s" for c in configs)
        same = {c: sum(outs[c][j][0] == outs["plain"][j][0] for j in range(len(prompts))) for c in configs if "plain" in outs}
        print(f"[traj {ti}] turns={len(prompts)} {line} identical={same}", flush=True)
