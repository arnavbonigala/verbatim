import argparse, json, os, sys

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("shards", nargs="+")
ap.add_argument("--format", default="openhands", choices=["openhands", "swesmith"])
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--trajs", type=int, default=100)
ap.add_argument("--max-ctx", type=int, default=32768)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--out", default="data/regen.jsonl")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verbatim.traj import openhands_messages

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(args.model)
HEAD = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load():
    for shard in args.shards:
        if args.format == "openhands":
            for r in pq.read_table(shard, columns=["trajectory"]).to_pylist():
                yield openhands_messages(r["trajectory"])
        else:
            for r in pq.read_table(shard, columns=["model", "messages"]).to_pylist():
                yield json.loads(r["messages"])


def render(m):
    return f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"


trajs = [t for _, t in zip(range(args.trajs), (t for t in load() if all(isinstance(m.get("content"), str) for m in t)))]
if args.dry_run:
    n_ctx = n_turns = 0
    for t in trajs:
        ctx = ""
        for m in t:
            if m["role"] == "assistant":
                n_turns += 1
                ctx += HEAD + m["content"] + "<|im_end|>\n"
            else:
                ctx += render(m)
        n_ctx += min(len(tok(ctx).input_ids), args.max_ctx)
    print(f"{len(trajs)} trajectories, {n_turns} assistant turns, {n_ctx} context tokens (capped at {args.max_ctx})")
    sys.exit()

from vllm import LLM, SamplingParams

llm = LLM(args.model, enable_prefix_caching=True, max_model_len=args.max_ctx)
sp = SamplingParams(temperature=0, max_tokens=args.max_tokens, stop=["<|im_end|>"])
state = [{"ctx": "", "rest": list(t), "turns": []} for t in trajs]
with open(args.out, "w") as fh:
    while state:
        for s in state:
            while s["rest"] and s["rest"][0]["role"] != "assistant":
                s["ctx"] += render(s["rest"].pop(0))
        live = [s for s in state if s["rest"] and len(tok(s["ctx"] + HEAD).input_ids) + args.max_tokens < args.max_ctx]
        if not live:
            break
        outs = llm.generate([s["ctx"] + HEAD for s in live], sp)
        for s, o in zip(live, outs):
            s["rest"].pop(0)
            text = o.outputs[0].text
            s["turns"].append({"prompt_chars": len(s["ctx"] + HEAD), "response": text})
            s["ctx"] += HEAD + text + "<|im_end|>\n"
        done = [s for s in state if s not in live]
        for s in done:
            fh.write(json.dumps({"text": s["ctx"], "turns": s["turns"]}) + "\n")
        state = live
    for s in state:
        fh.write(json.dumps({"text": s["ctx"], "turns": s["turns"]}) + "\n")
