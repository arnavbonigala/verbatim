import json, sys
from collections import defaultdict
recs = [json.loads(l) for l in open(sys.argv[1])]
cfgs = list(recs[0]["cfg"])
groups = defaultdict(list)
for r in recs:
    groups[r["kind"]].append(r)
    groups["ALL"].append(r)
print(f"{'kind':12} {'turns':>5} {'tokens':>8} " + " ".join(f"{c:>22}" for c in cfgs))
for k, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    n = sum(r["n"] for r in rs)
    base = sum(r["base_time"] for r in rs)
    cells = []
    for c in cfgs:
        p = sum(r["cfg"][c]["passes"] for r in rs)
        t = sum(r["cfg"][c]["time"] for r in rs)
        cells.append(f"{n/p:5.2f}t/p {base/t:5.2f}x")
    print(f"{k:12} {len(rs):5} {n:8} " + " ".join(f"{x:>22}" for x in cells))
