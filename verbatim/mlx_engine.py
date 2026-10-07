import json
import os
import time

import mlx.core as mx
import numpy as np
from llama_cpp import Llama
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache


class Engine:
    def __init__(self, model_path, vocab_gguf, draft_path=None, fast=True):
        self.model, hf_tok = load(model_path)
        if fast:
            from verbatim.mlx_fast import accelerate
            accelerate(self.model, f"data/mlx_plan_{os.path.basename(model_path.rstrip('/'))}.json")
        self.vocab = Llama(vocab_gguf, vocab_only=True, verbose=False)
        self.eog = {hf_tok.eos_token_id, *[hf_tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]}
        self.cache = make_prompt_cache(self.model)
        self.cached = []
        self.draft = None
        if draft_path:
            self.draft, _ = load(draft_path)
            self.dcache = make_prompt_cache(self.draft)
            self.dcached = []

    def render(self, messages):
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        meta = self.vocab.metadata
        tid = lambda key: self.vocab._model.token_get_text(int(meta[key])) if key in meta else ""
        fmt = Jinja2ChatFormatter(meta["tokenizer.chat_template"], tid("tokenizer.ggml.eos_token_id"), tid("tokenizer.ggml.bos_token_id"))
        return fmt(messages=messages).prompt

    def tokenize(self, b: bytes):
        return self.vocab.tokenize(b, add_bos=False, special=False)

    def tokenize_prompt(self, text: str):
        return self.vocab.tokenize(text.encode(), add_bos=False, special=True)

    def detokenize(self, ids):
        return self.vocab.detokenize(ids)

    def n_ctx(self):
        return 32768

    def _forward(self, model, cache, toks):
        logits = model(mx.array(toks, dtype=mx.uint32)[None], cache=cache)
        return logits[0].argmax(-1)

    def _sync(self, model, cache, cached, prompt, chunk=512):
        p = 0
        while p < min(len(prompt), len(cached)) - 1 and prompt[p] == cached[p]:
            p += 1
        trim_prompt_cache(cache, len(cached) - p)
        out = None
        while p < len(prompt):
            part = prompt[p : p + chunk]
            out = self._forward(model, cache, part)[-1]
            mx.eval(out)
            p += len(part)
        return int(out.item())

    def prefill(self, prompt):
        t = self._sync(self.model, self.cache, self.cached, prompt)
        self.cached = list(prompt)
        return t

    def cost_profile(self, path, k_max=64, ctx_len=6000):
        if os.path.exists(path):
            return json.load(open(path))
        prompt = self.tokenize(b"def f(x):\n    return x\n" * (ctx_len // 10))[:ctx_len]
        self.prefill(prompt)
        cost = [0.0]
        for b in range(1, k_max + 2):
            runs = []
            for _ in range(5):
                ts = time.perf_counter()
                mx.eval(self._forward(self.model, self.cache, prompt[-b:]))
                runs.append(time.perf_counter() - ts)
                trim_prompt_cache(self.cache, b)
            cost.append(sorted(runs)[2])
        self.cached = list(prompt)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(cost, open(path, "w"))
        return cost

    def _model_draft(self, hist, n):
        self._sync(self.draft, self.dcache, self.dcached, hist[:-1])
        self.dcached = list(hist[:-1])
        toks, t = [], hist[-1]
        for _ in range(n):
            t = int(self._forward(self.draft, self.dcache, [t])[-1].item())
            toks.append(t)
        self.dcached += [hist[-1]] + toks[:-1]
        return toks

    def generate(self, prompt, drafter=None, max_tokens=512, model_draft=0):
        t = self.prefill(prompt)
        out, steps = [], []
        t0 = time.perf_counter()
        while True:
            if t in self.eog or len(out) >= max_tokens:
                break
            out.append(t)
            hist = prompt + out
            draft = [] if drafter is None else [int(x) for x in drafter(np.array(hist, dtype=np.intc))]
            src = "copy" if draft else ""
            if not draft and model_draft and self.draft is not None:
                draft, src = self._model_draft(hist, model_draft), "model"
            draft = draft[: max(0, max_tokens - len(out))]
            ts = time.perf_counter()
            pred = self._forward(self.model, self.cache, [t] + draft).tolist()
            a = 0
            while a < len(draft) and draft[a] not in self.eog and pred[a] == draft[a]:
                a += 1
            trim_prompt_cache(self.cache, len(draft) - a)
            out.extend(draft[:a])
            dt = time.perf_counter() - ts
            cost = getattr(drafter, "cost", None)
            if cost is not None and len(draft) + 1 < len(cost):
                cost[len(draft) + 1] = 0.9 * cost[len(draft) + 1] + 0.1 * dt
            steps.append({"pos": len(out) - a - 1, "draft": len(draft), "accepted": a, "dt": dt, "src": src})
            t = pred[a]
        self.cached = prompt + out
        return out, steps, time.perf_counter() - t0
