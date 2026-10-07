import mlx.core as mx

RT = 4
SIMDS = 4

SOURCE = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    int row0 = (threadgroup_position_in_grid.x * SIMDS + sg) * (8 * RT);
    short qid = lane / 4;
    short fm = (qid & 4) + ((lane / 2) % 4);
    short fn = (qid & 2) * 2 + (lane % 2) * 2;
    int words = K / 8;
    int groups = K / 64;
    const device uint2* wv = (const device uint2*)wq;
    simdgroup_matrix<float, 8, 8> C[RT][MT];
    for (int r = 0; r < RT; r++)
        for (int mt = 0; mt < MT; mt++)
            C[r][mt] = simdgroup_matrix<float, 8, 8>(0);
    uint2 wn[RT];
    float sn[RT], bn[RT];
    for (int r = 0; r < RT; r++) {
        int n = row0 + 8 * r + fm;
        wn[r] = wv[(n * words + fn) / 2];
        sn[r] = (float)scales[n * groups];
        bn[r] = (float)biases[n * groups];
    }
    for (int g = 0; g < groups; g++) {
        uint2 wr[RT];
        float s[RT], b[RT];
        for (int r = 0; r < RT; r++) {
            wr[r] = wn[r]; s[r] = sn[r]; b[r] = bn[r];
        }
        if (g + 1 < groups) {
            for (int r = 0; r < RT; r++) {
                int n = row0 + 8 * r + fm;
                wn[r] = wv[(n * words + (g + 1) * 8 + fn) / 2];
                sn[r] = (float)scales[n * groups + g + 1];
                bn[r] = (float)biases[n * groups + g + 1];
            }
        }
        for (int t = 0; t < 8; t++) {
            int k0 = g * 64 + t * 8;
            simdgroup_matrix<half, 8, 8> B[MT];
            for (int mt = 0; mt < MT; mt++)
                simdgroup_load(B[mt], x + mt * 8 * K + k0, K, ulong2(0, 0), true);
            for (int r = 0; r < RT; r++) {
                uint w = (wr[r][t >> 2] >> ((t & 3) * 8)) & 0xFF;
                simdgroup_matrix<half, 8, 8> A;
                A.thread_elements()[0] = (half)(s[r] * (float)(w & 0xF) + b[r]);
                A.thread_elements()[1] = (half)(s[r] * (float)(w >> 4) + b[r]);
                for (int mt = 0; mt < MT; mt++)
                    simdgroup_multiply_accumulate(C[r][mt], A, B[mt], C[r][mt]);
            }
        }
    }
    for (int r = 0; r < RT; r++)
        for (int mt = 0; mt < MT; mt++)
            simdgroup_store(C[r][mt], out + mt * 8 * N + row0 + 8 * r, N, ulong2(0, 0), true);
"""

_kernel = mx.fast.metal_kernel(
    name="verbatim_qmv",
    input_names=["x", "wq", "scales", "biases"],
    output_names=["out"],
    source=SOURCE,
    header="#include <metal_simdgroup_matrix>\nusing namespace metal;\n",
)


def repack(wq):
    import numpy as np
    w = np.array(wq).view(np.uint8)
    nib = np.stack([w & 0xF, w >> 4], -1).reshape(w.shape[0], -1, 64)
    t, j, f = np.meshgrid(np.arange(8), np.arange(2), np.arange(4), indexing="ij")
    old = 8 * t + 2 * f + j
    new = 16 * f + 2 * t + j
    out = np.empty_like(nib)
    out[..., new.ravel()] = nib[..., old.ravel()]
    out = out.reshape(w.shape[0], -1, 2)
    return mx.array((out[..., 0] | (out[..., 1] << 4)).astype(np.uint8).view(np.uint32))


def qmv(x, wq, scales, biases, RT=RT, SIMDS=SIMDS):
    *lead, K = x.shape
    x2 = x.reshape(-1, K).astype(mx.float16)
    M, N = x2.shape[0], wq.shape[0]
    mt = (M + 7) // 8
    if M % 8:
        x2 = mx.concatenate([x2, mx.zeros((mt * 8 - M, K), mx.float16)])
    out = _kernel(
        inputs=[x2, wq, scales, biases],
        template=[("MT", mt), ("K", K), ("N", N), ("RT", RT), ("SIMDS", SIMDS)],
        grid=(N // (8 * RT * SIMDS) * SIMDS * 32, 1, 1),
        threadgroup=(SIMDS * 32, 1, 1),
        output_shapes=[(mt * 8, N)],
        output_dtypes=[mx.float32],
    )[0]
    return out[:M].astype(x.dtype).reshape(*lead, N)


def patch(model, lo=4, hi=16):
    import mlx.nn as nn
    n = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear) and m.bits == 4 and m.group_size == 64 and m.weight.shape[0] % (8 * RT * SIMDS) == 0:
            m._vb = repack(m.weight)
            n += 1
    orig = nn.QuantizedLinear.__call__

    def call(self, x):
        rows = x.size // x.shape[-1]
        if hasattr(self, "_vb") and lo <= rows <= hi:
            y = qmv(x, self._vb, self.scales, self.biases)
            return y + self.bias if "bias" in self else y
        return orig(self, x)

    nn.QuantizedLinear.__call__ = call
    return n


if __name__ == "__main__":
    import time

    def bench(f, n=50):
        for _ in range(5):
            mx.eval(f())
        ts = time.perf_counter()
        for _ in range(n):
            mx.eval(f())
        return (time.perf_counter() - ts) / n

    for N, K in [(4608, 3584), (37888, 3584), (3584, 18944)]:
        w = mx.random.normal((N, K)).astype(mx.bfloat16) * 0.02
        wq0, s, b = mx.quantize(w, group_size=64, bits=4)
        wq = repack(wq0)
        for M in [1, 2, 4, 8, 16]:
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            ref = mx.quantized_matmul(x, wq0, s, b, transpose=True, group_size=64, bits=4).astype(mx.float32)
            got = qmv(x, wq, s, b).astype(mx.float32)
            err = float(mx.abs(got - ref).max() / mx.abs(ref).max())
            solo = mx.concatenate([qmv(x[i : i + 1], wq, s, b) for i in range(M)]).astype(mx.float32)
            inv = bool(mx.array_equal(solo, got))
            t_ref = bench(lambda: mx.quantized_matmul(x, wq0, s, b, transpose=True, group_size=64, bits=4))
            t_new = bench(lambda: qmv(x, wq, s, b))
            gb = N * K / 2 / 1e9
            print(f"N={N} K={K} M={M} err={err:.1e} batch_invariant={inv} mlx={t_ref*1e6:.0f}us new={t_new*1e6:.0f}us ({gb/t_new:.0f} GB/s)")
