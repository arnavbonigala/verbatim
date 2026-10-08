import argparse, json, os, random, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from verbatim.drafter import accepted_prefix
from verbatim.traj import openhands_messages

MODEL = "claude-3-7-sonnet-20250219"
ap = argparse.ArgumentParser()
ap.add_argument("shard")
ap.add_argument("--trajs", type=int, default=120)
ap.add_argument("--max-ctx", type=int, default=32768)
ap.add_argument("--cost", default="data/cost_qwen2.5-coder-7b-instruct-q4_k_m.json")
ap.add_argument("--out", default="results/sim.jsonl")
ap.add_argument("--procs", type=int, default=7)
ap.add_argument("--tok", default="models/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
ap.add_argument("--configs", default="")
ap.add_argument("--corpus", type=int, default=0)
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


def init(corpus=b""):
    global tok, COST, make
    from llama_cpp import Llama
    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
    from verbatim.drafter import Verbatim
    from verbatim.drafter_exp import Verbatim as VX
    from verbatim.suffix_baseline import SuffixDecoding
    llm = Llama(args.tok, vocab_only=True, verbose=False)
    tok = llm
    COST = json.load(open(args.cost))
    t = lambda b: llm.tokenize(b, add_bos=False, special=False)
    d = llm.detokenize
    corpus_ids = t(corpus) if corpus and (not args.configs or "suffix-global" in args.configs) else []
    make = {
        "pld": lambda: LlamaPromptLookupDecoding(),
        "pld15": lambda: LlamaPromptLookupDecoding(num_pred_tokens=15),
        "pld-max": lambda: LlamaPromptLookupDecoding(num_pred_tokens=64),
        "verbatim": lambda: Verbatim(t, d, COST),
        "verbatim-noforms": lambda: Verbatim(t, d, COST, forms=False),
        "verbatim-b4": lambda: Verbatim(t, d, COST, branches=4),
        "verbatim-k15": lambda: Verbatim(t, d, COST, k_max=15),
        "verbatim-k31": lambda: Verbatim(t, d, COST, k_max=31),
        "suffix": lambda: SuffixDecoding(),
        "agspec-forms": lambda: Verbatim(t, d, COST, forms="agspec"),
        "suffix-global": lambda: SuffixDecoding(corpus_ids=corpus_ids),
        "verbatim-corpus": lambda: VX(t, d, COST, corpus=corpus),
        "verbatim-max": lambda: Verbatim(t, d, None, k_max=64),
    }
    for q in (0.5, 0.6, 0.7, 0.8, 0.9):
        make[f"mix-p{q}"] = lambda q=q: (q, Verbatim(t, d, COST, k_max=len(COST) - 2))
        for b in (2, 4, 8):
            make[f"tree-p{q}-b{b}"] = lambda q=q, b=b: ((1 - (1 - q) ** b, b), Verbatim(t, d, COST, k_max=len(COST) - 2))


def run(row):
    msgs = row["messages"]
    T = lambda s: tok.tokenize(s.encode(), add_bos=False, special=False)
    S = lambda s: tok.tokenize(s.encode(), add_bos=False, special=True)
    end = S("<|im_end|>\n")
    pieces = [S(f"<|im_start|>{m['role']}\n") + T(m["content"]) + end for m in msgs]
    head = S("<|im_start|>assistant\n")
    drafters = {k: f() for k, f in make.items() if not args.configs or k in args.configs.split(",")}
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
                q, dr = dr if isinstance(dr, tuple) else (None, dr)
                q, br = q if isinstance(q, tuple) else (q, 1)
                rng = random.Random(pos)
                while pos < len(seq):
                    ds = dr.propose(seq[:pos]) if hasattr(dr, "propose") else [[int(x) for x in dr(seq[:pos])]]
                    actual = seq[pos:pos + 128].tolist()
                    a = max([accepted_prefix(d, actual) for d in ds] + [0])
                    n_draft = sum(len(d) for d in ds)
                    if q is not None and n_draft == 0:
                        depth = max(range((len(COST) - 2) // br + 1), key=lambda k: (1 + sum(q ** i for i in range(1, k + 1))) / COST[min(k * br + 1, len(COST) - 1)])
                        a = next((i for i in range(depth) if rng.random() > q), depth)
                        a = min(a, len(seq) - pos)
                        n_draft = depth * br
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
    rows = [r for r in rows if all(isinstance(m.get("content"), str) for m in r["messages"])]
    corpus = "\x00".join(m["content"] for r in rows[args.trajs : args.trajs + args.corpus] for m in r["messages"] if m["role"] == "assistant").encode()
    print("corpus bytes", len(corpus), flush=True)
    rows = rows[: args.trajs]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0, n = time.time(), 0
    with Pool(args.procs, initializer=init, initargs=(corpus,)) as pool, open(args.out, "w") as f:
        for recs in pool.imap_unordered(run, rows):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"{n}/{len(rows)} trajectories {time.time()-t0:.0f}s", flush=True)
