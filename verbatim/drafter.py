import json
import re

import numpy as np

LINENO = re.compile(rb"^ *\d+(?:\t|: |:$| \| ?)", re.M)
SEP = b"\x00"


def emission_forms(text: bytes, agspec=False) -> list[bytes]:
    if agspec:
        lines = text.split(b"\n")
        return [text] + [b"\n".join(p + l for l in lines) for p in (b" ", b"-")] + [json.dumps(text.decode("utf-8", "replace"))[1:-1].encode()]
    raw = LINENO.sub(b"", text)
    lines = raw.split(b"\n")
    forms = [raw]
    forms += [b"\n".join(p + l for l in lines) for p in (b"+", b"-", b" ")]
    forms.append(json.dumps(raw.decode("utf-8", "replace"))[1:-1].encode())
    forms.append(b"\n".join(b"    " + l for l in lines))
    forms.append(b"\n".join(l[4:] if l.startswith(b"    ") else l for l in lines))
    return forms


class Verbatim:
    def __init__(self, tokenize, detokenize, cost=None, min_match=4, k_max=64, window=16, forms=True, branches=1, shared=None):
        self.tokenize, self.detokenize = tokenize, detokenize
        self.cost = cost or [0.0] + [1.0] * (k_max + 1)
        self.min_match, self.k_max, self.window = min_match, min(k_max, len(self.cost) - 2), window
        self.use_forms, self.branches = forms, branches
        self.seen = np.array([], dtype=np.intc)
        self.stats = []
        self.by_len = {} if shared is None else shared

    def _rebuild(self, ids):
        self.history = self.detokenize([int(t) for t in ids])
        self.base, self.formed = b"", 0
        self.seen = np.array(ids, dtype=np.intc)
        self.last_draft = []
        self.prev = None
        self._extend_forms()

    def _extend_forms(self):
        cut = self.history.rfind(b"\n", self.formed) + 1
        if not self.use_forms or cut <= self.formed:
            return
        chunk = self.history[self.formed:cut]
        self.base += b"".join(f + SEP for f in emission_forms(chunk, self.use_forms == "agspec") if f != chunk)
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

    def expected_accepts(self, ks, bucket):
        def est(xs):
            d, a = np.array(xs + [(4, 4)]).T
            return np.where(a >= d, np.minimum(ks[:, None], 2 * d), np.minimum(a, ks[:, None])).mean(1)
        recent = [x for x in self.stats[-8:] if x]
        return np.minimum(est(self.by_len.get(bucket, [])[-self.window:]), est(recent))

    def draft_len(self, n, bucket):
        ks = np.arange(1, min(n, self.k_max) + 1)
        if not len(ks):
            return 0
        rate = (1 + self.expected_accepts(ks, bucket)) / np.array(self.cost[2 : len(ks) + 2])
        i = int(rate.argmax())
        return int(ks[i]) if rate[i] > 1 / self.cost[1] else 0

    def match(self):
        hay = self.base + self.history
        end = len(hay) - 1
        cap = min(len(self.history), 4096)
        if self.prev:
            pos, lo, n = self.prev
            grow = len(self.history) - n
            if 0 <= grow and lo + grow <= cap and hay[pos + lo : pos + lo + grow] == self.history[n:] and SEP not in self.history[n:]:
                lo += grow
            else:
                pos = -1
        else:
            pos = -1
        if pos < 0:
            lo = self.min_match
            pos = hay.rfind(hay[-lo:], 0, end) if lo <= cap else -1
            if pos < 0:
                self.prev = None
                return [], 0
            hi = lo
            while hi < cap:
                hi = min(2 * hi, cap)
                p = hay.rfind(hay[-hi:], 0, end)
                if p < 0:
                    hi -= 1
                    break
                lo, pos = hi, p
            while lo < hi:
                mid = (lo + hi + 1) // 2
                p = hay.rfind(hay[-mid:], 0, end)
                if p >= 0:
                    lo, pos = mid, p
                else:
                    hi = mid - 1
        self.prev = (pos, lo, len(self.history))
        pat, conts, keys = hay[-lo:], [], set()
        for _ in range(32):
            start = pos + lo
            stop = hay.find(SEP, start, start + self.k_max * 8)
            cont = hay[start:stop if stop >= 0 else start + self.k_max * 8]
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
        self.last_draft = []
        if not conts:
            return []
        self.last_bucket = (self.prev[0] < len(self.base), min(length.bit_length(), 12))
        k = self.draft_len(self.k_max, self.last_bucket)
        if k == 0:
            return []
        span = 2 * (k + 2) * len(self.history) // max(1, len(self.seen)) + 16
        drafts = [d for d in (self._tokens(c[:span]) for c in conts) if d]
        if not drafts:
            return []
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
