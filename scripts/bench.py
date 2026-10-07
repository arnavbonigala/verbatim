import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from verbatim.llama_engine import Engine
from verbatim.drafter import Verbatim

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
ap.add_argument("--workload", default="data/workload.jsonl")
ap.add_argument("--out", default="results/q4_k_m")
ap.add_argument("--configs", default="none,pld,pld16,verbatim,verbatim-noforms,verbatim-fixed")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--only", default="")
ap.add_argument("--cooldown", type=float, default=2.0)
args = ap.parse_args()

e = Engine(args.model, n_ctx=8192)
tag = os.path.basename(args.model).replace(".gguf", "")
cost = e.cost_profile(f"data/cost_{tag}.json")
flat = [0.0] + [1.0] * 66


def make(name):
    return {
        "none": lambda: None,
        "pld": lambda: LlamaPromptLookupDecoding(),
        "pld16": lambda: LlamaPromptLookupDecoding(num_pred_tokens=15),
        "verbatim": lambda: Verbatim(e.tokenize, e.detokenize, cost),
        "verbatim-noforms": lambda: Verbatim(e.tokenize, e.detokenize, cost, forms=False),
        "verbatim-fixed": lambda: Verbatim(e.tokenize, e.detokenize, flat, k_max=31),
    }[name]()


turns = [json.loads(l) for l in open(args.workload)]
if args.only:
    turns = [t for t in turns if t["id"] in args.only.split(",")]
if args.limit:
    turns = turns[: args.limit]
configs = args.configs.split(",")
os.makedirs(args.out, exist_ok=True)
for i, t in enumerate(turns):
    path = f"{args.out}/{t['id']}.json"
    if os.path.exists(path):
        continue
    text, ref = e.render(t["messages"]), t["reference"]
    for marker in ("<parameter=new_str>", "<parameter=file_text>"):
        if t["kind"] in ("str_replace", "create") and marker in ref:
            cut = ref.index(marker) + len(marker)
            text, ref = text + ref[:cut], ref[cut:]
            break
    prompt = e.tokenize_prompt(text)
    max_tokens = min(1024, int(1.3 * len(e.tokenize(ref.encode()))) + 32)
    e.cached = []
    ts = time.perf_counter()
    e.prefill(prompt)
    prefill = time.perf_counter() - ts
    order = configs[i % len(configs):] + configs[: i % len(configs)]
    runs = {}
    for name in order:
        time.sleep(args.cooldown)
        out, steps, dt = e.generate(prompt, make(name), max_tokens=max_tokens)
        runs[name] = {"out": out, "steps": steps, "dt": dt}
    base = runs["none"]["out"] if "none" in runs else None
    for name, r in runs.items():
        r["identical"] = base is None or r["out"] == base
    rec = {"id": t["id"], "kind": t["kind"], "prompt_tokens": len(prompt), "prefill_s": prefill, "order": order, "runs": runs}
    json.dump(rec, open(path, "w"))
    line = " ".join(f"{n}={len(r['out'])/r['dt']:.1f}t/s({len(r['out'])/max(1,len(r['steps'])):.1f}/pass){'' if r['identical'] else '!'}" for n, r in runs.items())
    print(f"[{i+1}/{len(turns)}] {t['id']} p={len(prompt)} n={len(base or [])} {line}", flush=True)
