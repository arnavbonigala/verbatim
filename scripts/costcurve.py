import sys, time, json
sys.path.insert(0, ".")
from verbatim.llama_engine import Engine
model, kw = sys.argv[1], json.loads(sys.argv[2])
e = Engine(model, n_ctx=4096, **kw)
prompt = e.tokenize(b"def f(x):\n    return x\n" * 300)[:2048]
e.prefill(prompt)
res = {}
for b in [1, 2, 4, 8, 9, 12, 16, 24, 32, 33, 48, 64]:
    ts = time.perf_counter()
    for _ in range(4): e._decode(prompt[-b:], len(prompt) - b, True)
    res[b] = round((time.perf_counter() - ts) / 4 * 1000, 1)
print(sys.argv[2], " ".join(f"{b}:{v}({v/res[1]:.1f}x)" for b, v in res.items()))
