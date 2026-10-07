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
    def __init__(self, tokenize, detokenize, cost=None, min_match=12, k_max=64, window=16, forms=True, branches=1):
        self.tokenize, self.detokenize = tokenize, detokenize
        self.cost = cost or [0.0] + [1.0] * (k_max + 1)
        self.min_match, self.k_max, self.window = min_match, min(k_max, len(self.cost) - 2), window
        self.use_forms, self.branches = forms, branches
        self.seen = np.array([], dtype=np.intc)
        self.stats = []
        self.by_len = {}

    def _rebuild(self, ids):
        self.history = self.detokenize([int(t) for t in ids])
        self.base, self.formed = b"", 0
        self.seen = np.array(ids, dtype=np.intc)
        self.last_draft = []
        self._extend_forms()

    def _extend_forms(self):
        cut = self.history.rfind(b"\n", self.formed) + 1
        if not self.use_forms or cut <= self.formed:
            return
        chunk = self.history[self.formed:cut]
        self.base += b"".join(f + SEP for f in emission_forms(chunk) if f != chunk)
        self.formed = cut

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
            self._extend_forms()

    def _adapt(self, new):
        if not self.last_draft:
            self.stats.append(None)
            return
        d, a = max(((d, accepted_prefix(d, new)) for d in self.last_draft), key=lambda x: x[1])
        d = len(d)
        self.stats.append((d, a))
        self.by_len.setdefault(self.last_bucket, []).append((d, a))
        self.last_draft = []

    def expected_accepts(self, k, bucket):
        recent = self.by_len.get(bucket, [])[-self.window:] + [(0, 1)]
        return sum(k if a >= d else min(a, k) for d, a in recent) / len(recent)

    def draft_len(self, n, bucket):
        best, best_k = 1 / self.cost[1], 0
        for k in range(1, min(n, self.k_max) + 1):
            rate = (1 + self.expected_accepts(k, bucket)) / self.cost[k + 1]
            if rate > best:
                best, best_k = rate, k
        return best_k

    def match(self):
        hay = self.base + self.history
        end = len(hay) - 1
        lo, hi = self.min_match, min(len(self.history), 4096)
        pos = hay.rfind(hay[-lo:], 0, end) if lo <= hi else -1
        if pos < 0:
            return [], 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            p = hay.rfind(hay[-mid:], 0, end)
            if p >= 0:
                lo, pos = mid, p
            else:
                hi = mid - 1
        pat, conts, keys = hay[-lo:], [], set()
        for _ in range(32):
            start = pos + lo
            stop = hay.find(SEP, start)
            cont = hay[start:min(len(hay) if stop < 0 else stop, start + self.k_max * 8)]
            if cont and cont[:16] not in keys:
                keys.add(cont[:16])
                conts.append(cont)
            if len(conts) >= self.branches:
                break
            pos = hay.rfind(pat, 0, pos + lo - 1)
            if pos < 0:
                break
        return conts, lo

    def _tokens(self, cont):
        tail = [int(t) for t in self.seen[-2:]]
        toks = self.tokenize(self.detokenize(tail) + cont)
        draft = toks[len(tail):] if toks[: len(tail)] == tail else self.tokenize(cont)
        return draft[:-1]

    def propose(self, input_ids):
        self._sync(np.asarray(input_ids))
        conts, length = self.match()
        drafts = [d for d in map(self._tokens, conts) if d]
        self.last_draft = []
        if not drafts:
            return []
        self.last_bucket = min(length.bit_length(), 12)
        k = self.draft_len(len(drafts[0]), self.last_bucket)
        if k == 0:
            return []
        drafts = drafts[: max(1, k // 8)]
        self.last_draft = [d[: k // len(drafts)] for d in drafts]
        return self.last_draft

    def __call__(self, input_ids, /, **kwargs):
        drafts = self.propose(input_ids)
        return np.array(drafts[0] if drafts else [], dtype=np.intc)


def accepted_prefix(draft, actual):
    a = 0
    while a < len(draft) and a < len(actual) and draft[a] == actual[a]:
        a += 1
    return a
