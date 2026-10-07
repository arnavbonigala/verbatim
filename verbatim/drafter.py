import json
import re

import numpy as np

LINENO = re.compile(rb"^ *\d+(?:\t|: |:$| \| ?)", re.M)
SEP = b"\x00"


def emission_forms(text: bytes) -> list[bytes]:
    raw = LINENO.sub(b"", text)
    lines = raw.split(b"\n")
    forms = [raw]
    forms += [b"\n".join(p + l for l in lines) for p in (b"+", b"-", b" ")]
    forms.append(json.dumps(raw.decode("utf-8", "replace"))[1:-1].encode())
    forms.append(b"\n".join(b"    " + l for l in lines))
    forms.append(b"\n".join(l[4:] if l.startswith(b"    ") else l for l in lines))
    return forms


class Verbatim:
    def __init__(self, tokenize, detokenize, cost=None, min_match=12, k_max=64, window=16, forms=True):
        self.tokenize, self.detokenize = tokenize, detokenize
        self.cost = cost or [0.0] + [1.0] * (k_max + 1)
        self.min_match, self.k_max, self.window = min_match, min(k_max, len(self.cost) - 2), window
        self.use_forms = forms
        self.seen = np.array([], dtype=np.intc)
        self.stats = []

    def _rebuild(self, ids):
        text = self.detokenize(list(ids))
        forms = emission_forms(text) if self.use_forms else []
        self.base = SEP.join(forms) + SEP if forms else b""
        self.history = text
        self.seen = np.array(ids, dtype=np.intc)
        self.last_draft = []

    def _sync(self, ids):
        n = len(self.seen)
        if n == 0 or len(ids) < n or not np.array_equal(ids[:n], self.seen):
            self._rebuild(ids)
            return
        new = ids[n:]
        if len(new):
            self._adapt([int(t) for t in new])
            self.history += self.detokenize([int(t) for t in new])
            self.seen = np.asarray(ids, dtype=np.intc).copy()

    def _adapt(self, new):
        d = self.last_draft
        if not d:
            self.stats.append(None)
            return
        a = 0
        while a < len(d) and a < len(new) and new[a] == d[a]:
            a += 1
        self.stats.append((len(d), a))
        self.last_draft = []

    def expected_accepts(self, k):
        recent = [x for x in self.stats[-self.window:] if x] + [(0, 1)]
        return sum(k if a >= d else min(a, k) for d, a in recent) / len(recent)

    def draft_len(self, n):
        best, best_k = 1 / self.cost[1], 0
        for k in range(1, min(n, self.k_max) + 1):
            rate = (1 + self.expected_accepts(k)) / self.cost[k + 1]
            if rate > best:
                best, best_k = rate, k
        return best_k

    def match(self):
        hay = self.base + self.history
        end = len(hay) - 1
        lo, hi = self.min_match, min(len(self.history), 4096)
        if lo > hi or hay.rfind(hay[-lo:], 0, end) < 0:
            return None, 0
        pos = hay.rfind(hay[-lo:], 0, end)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            p = hay.rfind(hay[-mid:], 0, end)
            if p >= 0:
                lo, pos = mid, p
            else:
                hi = mid - 1
        start = pos + lo
        stop = hay.find(SEP, start)
        stop = len(hay) if stop < 0 else stop
        return hay[start:min(stop, start + self.k_max * 8)], lo

    def __call__(self, input_ids, /, **kwargs):
        self._sync(np.asarray(input_ids))
        cont, _ = self.match()
        if not cont:
            self.last_draft = []
            return np.array([], dtype=np.intc)
        tail = [int(t) for t in self.seen[-2:]]
        toks = self.tokenize(self.detokenize(tail) + cont)
        draft = toks[len(tail):] if toks[: len(tail)] == tail else self.tokenize(cont)
        draft = draft[:-1]
        draft = draft[: self.draft_len(len(draft))]
        self.last_draft = list(draft)
        return np.array(draft, dtype=np.intc)
