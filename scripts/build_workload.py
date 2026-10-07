import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from llama_cpp import Llama

SHARD = Path(sys.argv[1])
MODEL = "claude-3-7-sonnet-20250219"
BUDGET = 7000
QUOTA = {"str_replace": 12, "create": 6, "after_obs": 6, "reasoning": 6}
OMIT = "\n\n[... earlier steps omitted ...]"

tok = Llama("models/qwen2.5-coder-7b-instruct-q4_k_m.gguf", vocab_only=True, verbose=False)


def ntok(s):
    return len(tok.tokenize(s.encode(), add_bos=False, special=True))


def render(msgs):
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n"


def kind(msgs, i):
    c = msgs[i]["content"]
    if "<parameter=command>str_replace</parameter>" in c:
        return "str_replace"
    if "<parameter=command>create</parameter>" in c:
        return "create"
    f = c.find("<function=")
    prose = c if f < 0 else c[:f]
    if len(prose) >= 600 and (f < 0 or len(c) - f < 0.25 * len(c)):
        return "reasoning"
    if len(msgs[i - 1]["content"]) >= 3000:
        return "after_obs"
    return None


def fit(msgs, i):
    head = [msgs[0], {"role": msgs[1]["role"], "content": msgs[1]["content"] + OMIT}]
    prev = msgs[2:i]
    full = render(msgs[:i])
    if ntok(full) <= BUDGET:
        return full
    base = ntok(render(head)) + 8
    kept, used = [], 0
    for m in reversed(prev):
        n = ntok(render([m])) - 3
        if base + used + n > BUDGET:
            break
        kept.insert(0, m)
        used += n
    while kept and kept[0]["role"] != "assistant":
        kept.pop(0)
    if not kept or kept[-1] is not prev[-1]:
        return None
    p = render(head + kept)
    return p if ntok(p) <= BUDGET else None


rows = [r for r in pq.read_table(SHARD, columns=["model", "messages", "instance_id", "traj_id"]).to_pylist() if r["model"] == MODEL]
rows.sort(key=lambda r: r["traj_id"])
random.Random(0).shuffle(rows)

picked, used_traj, counts = [], set(), Counter()
for r in rows:
    if all(counts[k] >= q for k, q in QUOTA.items()):
        break
    msgs = json.loads(r["messages"])
    if len(msgs) < 3 or msgs[0]["role"] != "system" or not all(isinstance(m.get("content"), str) for m in msgs):
        continue
    if r["traj_id"] in used_traj:
        continue
    cands = [(i, kind(msgs, i)) for i in range(2, len(msgs)) if msgs[i]["role"] == "assistant"]
    cands = [(i, k) for i, k in cands if k and counts[k] < QUOTA[k]]
    random.Random(r["traj_id"]).shuffle(cands)
    for i, k in cands:
        prompt = fit(msgs, i)
        if prompt is None:
            continue
        ref = msgs[i]["content"]
        picked.append({
            "id": f"{k}-{counts[k]:02d}",
            "instance_id": r["instance_id"],
            "turn_index": i,
            "kind": k,
            "prompt": prompt,
            "reference": ref,
            "max_tokens": min(1024, int(1.3 * ntok(ref)) + 32),
        })
        counts[k] += 1
        used_traj.add(r["traj_id"])
        break

Path("data").mkdir(exist_ok=True)
with open("data/workload.jsonl", "w") as f:
    for rec in sorted(picked, key=lambda x: x["id"]):
        f.write(json.dumps(rec) + "\n")

lens = [ntok(p["prompt"]) for p in picked]
print("counts", dict(counts), "total", len(picked))
print("prompt tokens min/median/max", min(lens), int(statistics.median(lens)), max(lens))
print("max_tokens median", int(statistics.median(p["max_tokens"] for p in picked)))

Path("data/WORKLOAD.md").write_text(
    "# Workload\n"
    "- Source: SWE-bench/SWE-smith-trajectories (Hugging Face), split `tool`, shard tool-00004-of-00008.parquet.\n"
    "- License: MIT (dataset card `license: mit`).\n"
    f"- Selection: {MODEL} trajectories, one target assistant turn per trajectory, seeded shuffle (seed 0), prompt capped at {BUDGET} Qwen2.5 tokens by dropping oldest middle messages.\n"
    f"- Counts: {', '.join(f'{k}={counts[k]}' for k in QUOTA)} (total {len(picked)}).\n"
    f"- Prompt tokens min/median/max: {min(lens)}/{int(statistics.median(lens))}/{max(lens)}.\n"
)
