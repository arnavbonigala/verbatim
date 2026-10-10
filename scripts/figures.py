import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none"})
VB, GREY, MID = "#1d4ed8", "#9aa0a8", "#c9a227"


def tps(f, ds, cfg):
    rs = [r for r in map(json.loads, open(f)) if r["dataset"] == ds and r["cfg"] == cfg]
    return sum(r["n"] for r in rs) / sum(r["dt"] for r in rs)


def bars(ax, labels, series, fmt="{:.2f}x", loc="upper right", ncol=1):
    w = 0.8 / len(series)
    for j, (name, vals, color) in enumerate(series):
        xs = [i + (j - (len(series) - 1) / 2) * w for i in range(len(labels))]
        ax.bar(xs, vals, w * 0.92, label=name, color=color)
        for x, v in zip(xs, vals):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(labels)), labels)
    ax.legend(frameon=False, loc=loc, ncol=ncol)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"docs/{name}.svg")
    plt.close(fig)


ds = [("humaneval", "HumanEval"), ("gsm8k", "GSM8K"), ("mt-bench", "MT-Bench")]
vb = [tps("results/dflash_qwen3_8b_v2.jsonl", d, "sadaptc") / tps("results/dflash_qwen3_8b_v2_stock.jsonl", d, "base") for d, _ in ds]
df = [tps("results/dflash_qwen3_8b.jsonl", d, "dflash5") / tps("results/dflash_qwen3_8b.jsonl", d, "base") for d, _ in ds]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.axhline(1, color=GREY, lw=1, ls="--")
bars(ax, [n for _, n in ds], [("DFlash", df, GREY), ("Verbatim", vb, VB)])
ax.set_ylabel("Speedup over stock MLX")
ax.set_ylim(0, max(vb) * 1.2)
ax.set_title("Qwen3-8B 4-bit, MacBook Air M3", loc="left", fontsize=11, color="#555")
save(fig, "speed_vs_dflash")

sim = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for r in map(json.loads, open("results/sim_frontier.jsonl")):
    for k in (r["kind"], "all"):
        for c, v in r["cfg"].items():
            sim[k][c][0] += r["n"]; sim[k][c][1] += v["passes"]
cfgs = [("pld", "Prompt lookup", "#d7d9dc"), ("suffix", "SuffixDecoding", GREY), ("agspec-forms", "AgSpec", MID), ("verbatim", "Verbatim", VB)]
edit = next(k for k in sim if "str_replace" in k)
fig, ax = plt.subplots(figsize=(7, 3.6))
bars(ax, ["All agent turns", "File edits"], [(n, [sim[k][c][0] / sim[k][c][1] for k in ("all", edit)], col) for c, n, col in cfgs], fmt="{:.2f}", loc="upper left", ncol=2)
ax.set_ylabel("Tokens per verify pass")
ax.set_ylim(0, 6.2)
ax.set_title("Replay of real OpenHands sessions (SWE-rebench)", loc="left", fontsize=11, color="#555")
save(fig, "acceptance")

cost = {k: json.load(open(f"data/verify_costs_{k}.json")) for k in ("stock", "verbatim")}
st, vt = cost["stock"]["False"][1], cost["verbatim"]["True"][1]
n = range(1, min(len(st), len(vt)))
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(n, [st[i] * 1000 for i in n], color=GREY, lw=2.5, marker="o", ms=4, label="Stock MLX")
ax.plot(n, [vt[i] * 1000 for i in n], color=VB, lw=2.5, marker="o", ms=4, label="Verbatim kernels")
ax.set_xlabel("Tokens checked in one forward pass")
ax.set_ylabel("Milliseconds")
ax.set_ylim(0)
ax.legend(frameon=False)
ax.set_title("Cost of one verify pass at 6K context", loc="left", fontsize=11, color="#555")
save(fig, "verify_cost")

base = tps("results/demo_stock.jsonl", "agent", "base")
strict = 100 * (1 - tps("results/demo_verbatim.jsonl", "agent", "sbase") / base)
fig, ax = plt.subplots(figsize=(7, 2.6))
rows = [("Thinking Machines\nbatch-invariant kernels", 61, GREY), ("SGLang\ndeterministic mode", 34, GREY), ("Verbatim strict mode", strict, VB)]
ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
for i, r in enumerate(rows):
    ax.text(r[1] + 1, i, f"{r[1]:.0f}%", va="center")
ax.set_xlabel("Slowdown from bit-exact decoding (lower is better)")
ax.set_xlim(0, 70)
save(fig, "determinism")
print("strict overhead", round(strict, 1), "dflash", [round(x, 2) for x in df], "verbatim", [round(x, 2) for x in vb])
print({k: {c: round(sim[k][c][0] / sim[k][c][1], 2) for c in sim[k]} for k in ("all", edit)})
