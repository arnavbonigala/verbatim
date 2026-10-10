import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbatim import qmv

def test_split_kernels_batch_invariant():
    mx.random.seed(0)
    for N, K in [(1024, 4096), (4096, 12288)]:
        w, s, b = mx.quantize(mx.random.normal((N, K)).astype(mx.bfloat16), group_size=64, bits=4)
        vb, sb = qmv.repack(w), qmv.pack_sb(s, b)
        X = mx.random.normal((1, 16, K)).astype(mx.bfloat16)
        rows = [qmv.gemv_split(X[:, i : i + 1], vb, sb)[:, 0] for i in range(16)]
        ref = mx.quantized_matmul(X, w, s, b, transpose=True, group_size=64, bits=4).astype(mx.float32)
        for M in (1, 2, 7, 8, 9, 15, 16):
            y = qmv.qmv_split(X[:, :M], vb, sb)
            assert all(mx.array_equal(y[:, i], rows[i]).item() for i in range(M))
        assert (mx.abs(qmv.qmv_split(X, vb, sb).astype(mx.float32) - ref).max() / mx.abs(ref).max()).item() < 0.01

if __name__ == "__main__":
    test_split_kernels_batch_invariant()
    print("ok")
