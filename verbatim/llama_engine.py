import json
import os
import time

import numpy as np
import llama_cpp
from llama_cpp import Llama


class Engine:
    def __init__(self, model_path, n_ctx=16384, n_batch=512, **kw):
        self.llm = Llama(model_path, n_ctx=n_ctx, n_batch=n_batch, n_gpu_layers=-1, logits_all=False, verbose=False, **kw)
        self.ctx = self.llm._ctx
        self.batch = self.llm._batch
        self.vocab = self.llm._model.vocab
        self.n_vocab = self.llm.n_vocab()
        self.cached = []

    def render(self, messages):
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        meta = self.llm.metadata
        tid = lambda key: self.llm._model.token_get_text(int(meta[key])) if key in meta else ""
        fmt = Jinja2ChatFormatter(meta["tokenizer.chat_template"], tid("tokenizer.ggml.eos_token_id"), tid("tokenizer.ggml.bos_token_id"))
        return fmt(messages=messages).prompt

    def tokenize(self, b: bytes):
        return self.llm.tokenize(b, add_bos=False, special=False)

    def tokenize_prompt(self, text: str):
        return self.llm.tokenize(text.encode(), add_bos=False, special=True)

    def detokenize(self, ids):
        return self.llm.detokenize(ids)

    def _decode(self, toks, n_past, all_logits):
        self.ctx.kv_cache_seq_rm(0, n_past, -1)
        self.batch.set_batch(toks, n_past=n_past, logits_all=all_logits)
        self.ctx.decode(self.batch)

    def cost_profile(self, path="data/cost_profile.json", k_max=64, ctx_len=2048):
        if os.path.exists(path):
            return json.load(open(path))
        prompt = self.tokenize(b"def f(x):\n    return x\n" * (ctx_len // 10))[:ctx_len]
        self.prefill(prompt)
        cost = [0.0]
        for b in range(1, k_max + 2):
            runs = []
            for _ in range(5):
                ts = time.perf_counter()
                self._decode(prompt[-b:], len(prompt) - b, True)
                runs.append(time.perf_counter() - ts)
            cost.append(sorted(runs)[2])
        self.cached = []
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(cost, open(path, "w"))
        return cost

    def _argmax(self, i):
        return int(np.ctypeslib.as_array(self.ctx.get_logits_ith(i), shape=(self.n_vocab,)).argmax())

    def prefill(self, prompt):
        p = 0
        while p < min(len(prompt), len(self.cached)) - 1 and prompt[p] == self.cached[p]:
            p += 1
        n_batch = self.llm.n_batch
        while p < len(prompt):
            chunk = prompt[p : p + n_batch]
            self._decode(chunk, p, False)
            p += len(chunk)
        self.cached = list(prompt)
        return self._argmax(len(chunk) - 1)

    def generate(self, prompt, drafter=None, max_tokens=512):
        t = self.prefill(prompt)
        out, steps = [], []
        t0 = time.perf_counter()
        while True:
            if llama_cpp.llama_vocab_is_eog(self.vocab, t) or len(out) >= max_tokens:
                break
            out.append(t)
            hist = prompt + out
            draft = [] if drafter is None else [int(x) for x in drafter(np.array(hist, dtype=np.intc))]
            draft = draft[: max(0, max_tokens - len(out))]
            ts = time.perf_counter()
            self._decode([t] + draft, len(hist) - 1, True)
            a = 0
            while a < len(draft) and not llama_cpp.llama_vocab_is_eog(self.vocab, draft[a]):
                p = self._argmax(a)
                if p != draft[a]:
                    break
                a += 1
            nxt = self._argmax(a)
            out.extend(draft[:a])
            cost = getattr(drafter, "cost", None)
            if cost is not None and len(draft) + 1 < len(cost):
                cost[len(draft) + 1] = 0.9 * cost[len(draft) + 1] + 0.1 * (time.perf_counter() - ts)
            steps.append({"pos": len(out) - a - 1, "draft": len(draft), "accepted": a, "dt": time.perf_counter() - ts})
            t = nxt
        self.cached = prompt + out
        return out, steps, time.perf_counter() - t0
