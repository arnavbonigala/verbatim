from collections import Counter

import numpy as np


def _enc(ids):
    return np.asarray(ids, dtype=">u4").tobytes()


class SuffixDecoding:
    def __init__(self, corpus_ids=(), max_depth=64, factor=1.0, offset=0, min_prob=0.1, max_occ=256):
        self.global_hay = _enc(corpus_ids)
        self.max_depth, self.factor, self.offset, self.min_prob, self.max_occ = max_depth, factor, offset, min_prob, max_occ

    def _occurrences(self, hay, pat, limit):
        out, i = [], hay.find(pat)
        while i >= 0 and len(out) < self.max_occ:
            if i % 4 == 0 and i + len(pat) < limit:
                out.append(i + len(pat))
            i = hay.find(pat, i + 1)
        return out

    def _longest(self, hay, ids, limit):
        lo, hi = 0, min(self.max_depth, len(ids))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._occurrences(hay, _enc(ids[-mid:]), limit):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _draft(self, hay, ids, limit):
        p = self._longest(hay, ids, limit)
        if p == 0:
            return [], 0.0
        occ = self._occurrences(hay, _enc(ids[-p:]), limit)
        budget = int(self.factor * p + self.offset)
        draft, prob, score = [], 1.0, 0.0
        while occ and len(draft) < budget:
            nxt = Counter(hay[o : o + 4] for o in occ if o + 4 <= limit)
            if not nxt:
                break
            tok, c = nxt.most_common(1)[0]
            prob *= c / sum(nxt.values())
            if prob < self.min_prob:
                break
            draft.append(int.from_bytes(tok, "big"))
            score += prob
            occ = [o + 4 for o in occ if hay[o : o + 4] == tok]
        return draft, score

    def __call__(self, input_ids, /, **kwargs):
        ids = [int(t) for t in input_ids]
        local = _enc(ids)
        d1, s1 = self._draft(local, ids, len(local))
        d2, s2 = self._draft(self.global_hay, ids, len(self.global_hay)) if self.global_hay else ([], 0.0)
        return np.array(d1 if s1 >= s2 else d2, dtype=np.intc)


if __name__ == "__main__":
    sd = SuffixDecoding()
    seq = [1, 2, 3, 4, 5, 9, 1, 2, 3, 4]
    assert sd(seq).tolist() == [5, 9, 1, 2], sd(seq).tolist()
    sd2 = SuffixDecoding(corpus_ids=[7, 8, 9, 10, 11, 12])
    assert sd2([0, 7, 8, 9]).tolist() == [10, 11, 12], sd2([0, 7, 8, 9]).tolist()
    print("ok")
