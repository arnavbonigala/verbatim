import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verbatim.drafter import Verbatim

tok = lambda b: list(b)
detok = lambda ids: bytes(int(x) for x in ids)
src = b"def area(r):\n    pi = 3.14159\n    return pi * r * r\n"
view = b"".join(b"%6d\t%s\n" % (i + 1, l) for i, l in enumerate(src.split(b"\n")[:-1]))


def drafted(context, generated, **kw):
    d = Verbatim(tok, detok, **kw)
    return detok(d(np.array(tok(context + generated))))


def copies(out, expected):
    return len(out) >= 4 and expected.startswith(out)


def test_copies_line_numbered_view_as_plain_code():
    out = drafted(b"cat -n geo.py\n" + view + b"\nassistant:\n", b"def area(r):\n    pi")
    assert copies(out, b" = 3.14159\n    return pi"), out


def test_diff_prefixed_copy():
    out = drafted(view + b"\nassistant:\n", b"-def area(r):\n-    pi = 3.14159")
    assert copies(out, b"\n-    return pi * r"), out


def test_json_escaped_copy():
    out = drafted(view + b"\nassistant:\n", json.dumps(src.decode())[1:35].encode())
    assert copies(out, json.dumps(src.decode())[35:].encode()), out


def test_no_forms_misses_diff_copy():
    out = drafted(view + b"\nassistant:\n", b"-def area(r):\n-    pi = 3.14159", forms=False)
    assert not copies(out, b"\n-    return pi * r"), out


def test_no_draft_without_match():
    assert drafted(view, b"\nclass Totally(New):") == b""


def test_generated_text_is_indexed():
    d = Verbatim(tok, detok)
    ids = tok(b"x" * 20 + b"\nresult = compute_everything(alpha, beta)\nprint(1)\n")
    d(np.array(ids))
    assert copies(detok(d(np.array(ids + tok(b"result = compute_eve")))), b"rything(alpha, beta)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
