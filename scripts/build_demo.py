import glob
import json
import os
import sys

from llama_cpp import Llama

model, results = sys.argv[1], sys.argv[2:]
tok = Llama(model, vocab_only=True, verbose=False)
piece = lambda t: tok.detokenize([t]).decode("utf-8", "replace")

out = {"runs": {}, "summaries": {}, "sim": {}}
for d in results:
    name = os.path.basename(d.rstrip("/"))
    if os.path.exists(f"{d}/summary.json"):
        out["summaries"][name] = json.load(open(f"{d}/summary.json"))
    turns = []
    for f in sorted(glob.glob(f"{d}/*.json")):
        if f.endswith("summary.json"):
            continue
        r = json.load(open(f))
        runs = {}
        for c, x in r["runs"].items():
            runs[c] = {"steps": [[s["pos"], s["draft"], s["accepted"], round(s["dt"], 4)] for s in x["steps"]], "dt": x["dt"], "n": len(x["out"]), "identical": x["identical"]}
        turns.append({"id": r["id"], "kind": r["kind"], "prompt_tokens": r["prompt_tokens"], "prefill_s": r["prefill_s"], "pieces": [piece(t) for t in r["runs"]["none"]["out"]], "runs": runs})
    out["runs"][name] = turns

for f in glob.glob("results/sim_*.jsonl"):
    recs = [json.loads(l) for l in open(f)]
    agg = {}
    for r in recs:
        for k in (r["kind"], "all"):
            a = agg.setdefault(k, {"turns": 0, "tokens": 0, "base": 0.0, "cfg": {}})
            a["turns"] += 1
            a["tokens"] += r["n"]
            a["base"] += r["base_time"]
            for c, v in r["cfg"].items():
                b = a["cfg"].setdefault(c, {"passes": 0, "time": 0.0})
                b["passes"] += v["passes"]
                b["time"] += v["time"]
    out["sim"][os.path.basename(f)[4:-6]] = agg

os.makedirs("demo", exist_ok=True)
with open("demo/data.js", "w") as fh:
    fh.write("window.VERBATIM = " + json.dumps(out, separators=(",", ":")) + ";\n")
print("wrote demo/data.js", os.path.getsize("demo/data.js") // 1024, "KB")
