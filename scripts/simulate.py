import argparse, json, os, random, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from verbatim.drafter import accepted_prefix

MODEL = "claude-3-7-sonnet-20250219"
ap = argparse.ArgumentParser()
ap.add_argument("shard")
ap.add_argument("--trajs", type=int, default=120)
ap.add_argument("--max-ctx", type=int, default=32768)
ap.add_argument("--cost", default="data/cost_qwen2.5-coder-7b-instruct-q4_k_m.json")
ap.add_argument("--out", default="results/sim.jsonl")
ap.add_argument("--procs", type=int, default=7)
ap.add_argument("--tok", default="models/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
ap.add_argument("--format", default="swesmith", choices=["swesmith", "openhands"])
args = ap.parse_args()


def kind(c):
    for k in ("str_replace", "create", "view", "insert", "undo_edit"):
        if f"<parameter=command>{k}</parameter>" in c or f'"command": "{k}"' in c:
            return k
    if "<function=bash>" in c or '"name": "execute_bash"' in c:
        return "bash"
    if "<function=submit>" in c or '"name": "finish"' in c:
        return "submit"
    return "reasoning"


def openhands_messages(traj):
    out = []
    for m in traj:
        c = m.get("content") or ""
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                f = tc["function"]
                try:
                    a = json.loads(f["arguments"])
                except (ValueError, TypeError):
                    a = f["arguments"]
                c += "\n<tool_call>\n" + json.dumps({"name": f["name"], "arguments": a}) + "\n</tool_call>"
            out.append({"role": "assistant", "content": c})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": f"<tool_response>\n{c}\n</tool_response>"})
        else:
            out.append({"role": m["role"], "content": c})
    return out


def init():
    global tok, COST, make
    from llama_cpp import Llama
    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
    from verbatim.drafter import Verbatim
    llm = Llama(args.tok, vocab_only=True, verbose=False)
    tok = llm
    COST = json.load(open(args.cost))
    t = lambda b: llm.tokenize(b, add_bos=False, special=False)
    d = llm.detokenize
    make = {
        "pld": lambda: LlamaPromptLookupDecoding(),
        "pld15": lambda: LlamaPromptLookupDecoding(num_pred_tokens=15),
        "verbatim": lambda: Verbatim(t, d, COST),
        "verbatim-noforms": lambda: Verbatim(t, d, COST, forms=False),
        "verbatim-multi": lambda: Verbatim(t, d, COST, branches=3),
        "verbatim-max": lambda: Verbatim(t, d, None, k_max=64),
    }


def run(row):
    msgs = row["messages"]
    T = lambda s: tok.tokenize(s.encode(), add_bos=False, special=False)
    S = lambda s: tok.tokenize(s.encode(), add_bos=False, special=True)
    end = S("<|im_end|>\n")
    pieces = [S(f"<|im_start|>{m['role']}\n") + T(m["content"]) + end for m in msgs]
    head = S("<|im_start|>assistant\n")
    drafters = {k: f() for k, f in make.items()}
    recs, ctx = [], []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and i >= 2:
            ref = T(m["content"])
            prompt = ctx + head
            if len(prompt) + len(ref) > args.max_ctx:
                break
            seq = np.array(prompt + ref, dtype=np.intc)
            rec = {"traj": row["traj_id"], "turn": i, "kind": kind(m["content"]), "ctx": len(prompt), "n": len(ref), "base_time": len(ref) * COST[1], "cfg": {}}
            for name, dr in drafters.items():
                pos, passes, drafted, accepted, t_proj = len(prompt) + 1, 0, 0, 0, 0.0
                while pos < len(seq):
                    ds = dr.propose(seq[:pos]) if hasattr(dr, "propose") else [[int(x) for x in dr(seq[:pos])]]
                    actual = seq[pos:pos + 128].tolist()
                    a = max([accepted_prefix(d, actual) for d in ds] + [0])
                    n_draft = sum(len(d) for d in ds)
                    passes += 1
                    drafted += n_draft
                    accepted += a
                    t_proj += COST[min(n_draft + 1, len(COST) - 1)]
                    pos += a + 1
                rec["cfg"][name] = {"passes": passes + 1, "drafted": drafted, "accepted": accepted, "time": t_proj + COST[1]}
            recs.append(rec)
        ctx += pieces[i]
    return recs


if __name__ == "__main__":
    import pyarrow.parquet as pq
    if args.format == "swesmith":
        rows = [r for r in pq.read_table(args.shard, columns=["model", "messages", "traj_id"]).to_pylist() if r["model"] == MODEL]
        rows = [{"traj_id": r["traj_id"], "messages": json.loads(r["messages"])} for r in rows]
    else:
        rows = [{"traj_id": f"oh-{i:05d}", "messages": openhands_messages(r["trajectory"])} for i, r in enumerate(pq.read_table(args.shard, columns=["trajectory"]).to_pylist())]
    rows.sort(key=lambda r: r["traj_id"])
    random.Random(1).shuffle(rows)
    rows = [r for r in rows if all(isinstance(m.get("content"), str) for m in r["messages"])][: args.trajs]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0, n = time.time(), 0
    with Pool(args.procs, initializer=init) as pool, open(args.out, "w") as f:
        for recs in pool.imap_unordered(run, rows):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"{n}/{len(rows)} trajectories {time.time()-t0:.0f}s", flush=True)
