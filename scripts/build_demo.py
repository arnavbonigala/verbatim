import collections
import json
import os
import sys

from transformers import AutoTokenizer

stock_f, verb_f = sys.argv[1:3]
tok = AutoTokenizer.from_pretrained("models/mlx-qwen3-8b-4bit")
recs = [json.loads(l) for f in (stock_f, verb_f) for l in open(f)]
ds = recs[0]["dataset"]
work = [json.loads(l) for l in open(f"data/{'workload' if ds == 'agent' else 'clinical_workload'}.jsonl")]
n = len({r["i"] for r in recs})
work = dict(enumerate(work[:: max(1, len(work) // n)][:n]))


def pieces(ids):
    out, prev = [], ""
    for j in range(len(ids)):
        s = tok.decode(ids[: j + 1])
        out.append(s[len(prev):] if s.startswith(prev) else tok.decode([ids[j]]))
        prev = s
    return out


def run(r):
    tr, steps, pos = r["trace"], [], 1 if r["cfg"] != "base" else 0
    t_prev = tr[0][3] - max(0.0, r["dt"] - (tr[-1][3] - tr[0][3]))
    for k, a, src, t in tr:
        if pos >= r["n"]:
            break
        steps.append([pos, k, a, src, round(t - t_prev, 4)])
        pos += 1 if r["cfg"] == "base" else a + 1
        t_prev = t
    return {"pieces": pieces(r["out"]), "steps": steps, "dt": r["dt"], "n": r["n"], "identical": r["identical"]}


turns = collections.defaultdict(dict)
for r in recs:
    if r["dataset"] == ds and r.get("trace"):
        turns[r["i"]][r["cfg"]] = r
out = {"turns": [], "summary": {}}
for i, cfgs in sorted(turns.items()):
    w = work[i]
    out["turns"].append({"id": w["id"], "kind": w["kind"], "prompt_tokens": cfgs["base"]["prompt_tokens"], "runs": {c: run(r) for c, r in cfgs.items()}})

agg = {}
for t in out["turns"]:
    for k in (t["kind"], "all"):
        a = agg.setdefault(k, {"turns": 0, "cfg": {}})
        a["turns"] += 1
        for c, r in t["runs"].items():
            b = a["cfg"].setdefault(c, {"tokens": 0, "time": 0.0, "passes": 0, "identical": 0})
            b["tokens"] += r["n"]; b["time"] += r["dt"]; b["passes"] += len(r["steps"]) + (c != "base"); b["identical"] += r["identical"]
out["summary"] = agg


with open("demo/data.js", "w") as fh:
    fh.write("window.VERBATIM = " + json.dumps(out, separators=(",", ":")) + ";\n")
print("wrote demo/data.js", os.path.getsize("demo/data.js") // 1024, "KB")
