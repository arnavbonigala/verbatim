import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"none": "No speculation", "pld": "Prompt lookup (built-in)", "verbatim": "Verbatim", "verbatim-noforms": "Verbatim, no emission forms", "pld16": "Prompt lookup, 15 tokens", "verbatim-fixed": "Verbatim, fixed length"}
COLOR = {"none": "#8a8f98", "pld": "#c58b2a", "pld16": "#d9b26b", "verbatim": "#2563eb", "verbatim-noforms": "#7aa2f7", "verbatim-fixed": "#1e3a8a"}


def summarize(d):
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.json")) if not f.endswith("summary.json")]
    empty = [r["id"] for r in recs if not r["runs"]["none"]["out"]]
    if empty:
        print(f"{d}: excluded {len(empty)} turn(s) with no generated tokens: {', '.join(empty)}")
    recs = [r for r in recs if r["runs"]["none"]["out"]]
    cfgs = [c for c in LABEL if c in recs[0]["runs"]]
    groups = defaultdict(list)
    for r in recs:
        groups[r["kind"]].append(r)
        groups["all"].append(r)
    table = {}
    for k, rs in groups.items():
        row = {}
        for c in cfgs:
            toks = sum(len(r["runs"][c]["out"]) for r in rs)
            dt = sum(r["runs"][c]["dt"] for r in rs)
            passes = sum(len(r["runs"][c]["steps"]) for r in rs)
            acc = sum(s["accepted"] for r in rs for s in r["runs"][c]["steps"])
            base = sum(r["runs"]["none"]["dt"] for r in rs)
            row[c] = {
                "tok_s": toks / dt if dt else 0,
                "speedup": base / dt if dt else 0,
                "tokens_per_pass": toks / passes if passes else 0,
                "accepted_per_pass": acc / passes if passes else 0,
                "turn_s": (dt + sum(r["prefill_s"] for r in rs)) / len(rs),
                "identical": sum(r["runs"][c]["identical"] for r in rs),
            }
        table[k] = {"turns": len(rs), "tokens": sum(len(r["runs"]["none"]["out"]) for r in rs), "cfg": row}
    per_turn = [{"id": r["id"], "kind": r["kind"], "n": len(r["runs"]["none"]["out"]), **{c: r["runs"]["none"]["dt"] / r["runs"][c]["dt"] for c in cfgs}} for r in recs]
    return cfgs, table, per_turn, len(recs)


def markdown(name, cfgs, table, n):
    lines = [f"### {name}", "", "| Turn kind | Turns | Output tokens | " + " | ".join(LABEL[c] for c in cfgs) + " |", "|---|---|---|" + "---|" * len(cfgs)]
    for k in ["all"] + sorted(x for x in table if x != "all"):
        t = table[k]
        cells = [f"{t['cfg'][c]['tok_s']:.1f} tok/s, {t['cfg'][c]['speedup']:.2f}x, {t['cfg'][c]['tokens_per_pass']:.2f}/pass" for c in cfgs]
        lines.append(f"| {k} | {t['turns']} | {t['tokens']} | " + " | ".join(cells) + " |")
    lines += ["", "Equivalence (greedy, token IDs vs no speculation): " + ", ".join(f"{LABEL[c]} {table['all']['cfg'][c]['identical']} of {n}" for c in cfgs if c != "none"), ""]
    return "\n".join(lines)


def chart(name, cfgs, table, path):
    kinds = ["all"] + sorted(x for x in table if x != "all")
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    w = 0.8 / len(cfgs)
    for i, c in enumerate(cfgs):
        ys = [table[k]["cfg"][c]["speedup"] for k in kinds]
        bars = ax.bar([x + i * w for x in range(len(kinds))], ys, w, label=LABEL[c], color=COLOR[c])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.axhline(1, color="#444", lw=0.8, ls="--")
    ax.set_xticks([x + w * (len(cfgs) - 1) / 2 for x in range(len(kinds))], kinds)
    ax.set_ylabel("Decode speedup vs no speculation")
    ax.set_title(name, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    md = []
    for d in sys.argv[1:]:
        name = os.path.basename(d.rstrip("/"))
        cfgs, table, per_turn, n = summarize(d)
        json.dump({"table": table, "per_turn": per_turn}, open(f"{d}/summary.json", "w"), indent=1)
        chart(name, cfgs, table, f"{d}/speedup.png")
        md.append(markdown(name, cfgs, table, n))
    print("\n".join(md))
