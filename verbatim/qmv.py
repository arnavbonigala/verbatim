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
    int groups = K / 64;
    const device uint2* wv = (const device uint2*)wq;
    const device half2* sbv = (const device half2*)sb;
    simdgroup_matrix<float, 8, 8> C[RT][MT];
    for (int r = 0; r < RT; r++)
        for (int mt = 0; mt < MT; mt++)
            C[r][mt] = simdgroup_matrix<float, 8, 8>(0);
    int base[RT];
    uint2 wn[RT];
    half2 sbn[RT];
    for (int r = 0; r < RT; r++) {
        base[r] = ((row0 + 8 * r) / 8) * groups * 8 + fm;
        wn[r] = wv[base[r] * 4 + fn / 2];
        sbn[r] = sbv[base[r]];
    }
    for (int g = 0; g < groups; g++) {
        uint2 wr[RT];
        half2 s[RT], b[RT];
        for (int r = 0; r < RT; r++) {
            wr[r] = wn[r]; s[r] = half2(sbn[r].x); b[r] = half2(sbn[r].y);
        }
        if (g + 1 < groups) {
            for (int r = 0; r < RT; r++) {
                int i = base[r] + (g + 1) * 8;
                wn[r] = wv[i * 4 + fn / 2];
                sbn[r] = sbv[i];
            }
        }
        for (int t = 0; t < 8; t++) {
            int k0 = g * 64 + t * 8;
            simdgroup_matrix<half, 8, 8> B[MT];
            for (int mt = 0; mt < MT; mt++) {
                int m = mt * 8 + fn;
                half v0 = (half)x[min(m, M - 1) * K + k0 + fm];
                half v1 = (half)x[min(m + 1, M - 1) * K + k0 + fm];
                B[mt].thread_elements()[0] = m < M ? v0 : 0.0h;
                B[mt].thread_elements()[1] = m + 1 < M ? v1 : 0.0h;
            }
            for (int r = 0; r < RT; r++) {
                uint pair = ((wr[r][t >> 2] >> ((t & 3) * 4)) & 0x000F000Fu) | 0x64006400u;
                half2 h = fma(as_type<half2>(pair) - half2(1024.0h), s[r], b[r]);
                simdgroup_matrix<half, 8, 8> A;
                A.thread_elements()[0] = h.x;
                A.thread_elements()[1] = h.y;
                for (int mt = 0; mt < MT; mt++)
                    simdgroup_multiply_accumulate(C[r][mt], A, B[mt], C[r][mt]);
            }
        }
    }
    for (int r = 0; r < RT; r++)
        for (int mt = 0; mt < MT; mt++) {
            int m = mt * 8 + fn;
            int n = row0 + 8 * r + fm;
            if (m < M) out[m * N + n] = (OT)C[r][mt].thread_elements()[0];
            if (m + 1 < M) out[(m + 1) * N + n] = (OT)C[r][mt].thread_elements()[1];
        }
"""

_kernel = mx.fast.metal_kernel(
    name="verbatim_qmv",
    input_names=["x", "wq", "sb"],
    output_names=["out"],
    source=SOURCE,
    header="#include <metal_simdgroup_matrix>\nusing namespace metal;\n",
)


GEMV_SOURCE = """
    int n = thread_position_in_grid.x;
    int groups = K / 64;
    int base = (n / 8) * groups * 8 + n % 8;
    const device uint4* w = (const device uint4*)wq;
    const device half2* sbv = (const device half2*)sb;
    threadgroup half xs[SHARED ? K : 1];
    if (SHARED) {
        for (int i = thread_position_in_threadgroup.x; i < K; i += threads_per_threadgroup.x)
            xs[i] = (half)x[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float acc = 0.0f;
    uint4 n0 = w[base * 2], n1 = w[base * 2 + 1];
    half2 sbn = sbv[base];
    for (int g = 0; g < groups; g++) {
        uint ws[8] = {n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w};
        half s = sbn.x, b = sbn.y;
        if (g + 1 < groups) {
            int i = base + (g + 1) * 8;
            n0 = w[i * 2]; n1 = w[i * 2 + 1]; sbn = sbv[i];
        }
        for (int k = 0; k < 64; k++) {
            int t = k / 8, f = (k % 8) / 2, j = k % 2;
            half h = fma((half)((ws[2 * f + t / 4] >> (4 * ((t % 4) + 4 * j))) & 15u), s, b);
            acc = fma((float)h, (float)(SHARED ? xs[g * 64 + k] : (half)x[g * 64 + k]), acc);
        }
    }
    out[n] = (OT)acc;
"""

_gemv = mx.fast.metal_kernel(name="verbatim_gemv", input_names=["x", "wq", "sb"], output_names=["out"], source=GEMV_SOURCE)


def gemv(x, wq, sb, tg=128):
    *lead, K = x.shape
    N = wq.shape[0]
    return _gemv(
        inputs=[x, wq, sb],
        template=[("K", K), ("OT", x.dtype), ("SHARED", K <= 16384)],
        grid=(N, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(*lead, N)],
        output_dtypes=[x.dtype],
    )[0]


DEQ_SOURCE = """
    uint i = thread_position_in_grid.x;
    int groups = K / 64;
    int w = i % 8, fm = (i / 8) % 8, g = (i / 64) % groups, blk = i / (64 * groups);
    int n = blk * 8 + fm - N0;
    if (n < 0 || n >= NR) return;
    half2 sbh = ((const device half2*)sb)[i / 8];
    uint word = wq[i];
    for (int q = 0; q < 8; q++) {
        int k = 32 * (w % 2) + 8 * (q % 4) + 2 * (w / 2) + q / 4;
        out[n * K + g * 64 + k] = (OT)fma((half)((word >> (4 * q)) & 15u), sbh.x, sbh.y);
    }
"""

_deq = mx.fast.metal_kernel(name="verbatim_dequant", input_names=["wq", "sb"], output_names=["out"], source=DEQ_SOURCE)


def dequant(wq, sb, n0, nr, dtype):
    N, W = wq.shape
    K = W * 8
    start, stop = (n0 // 8) * 8 * W, ((n0 + nr + 7) // 8) * 8 * W
    return _deq(
        inputs=[wq.reshape(-1)[start:stop], sb.reshape(-1)[start // 8 : stop // 8]],
        template=[("K", K), ("N0", n0 % 8), ("NR", nr), ("OT", dtype)],
        grid=(stop - start, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(nr, K)],
        output_dtypes=[dtype],
    )[0]


def matmul_large(x, wq, sb, chunk=16384):
    N = wq.shape[0]
    return mx.concatenate([x @ dequant(wq, sb, n0, min(chunk, N - n0), x.dtype).T for n0 in range(0, N, chunk)], axis=-1)


def repack(wq):
    import numpy as np
    w = np.array(wq).view(np.uint8)
    nib = np.stack([w & 0xF, w >> 4], -1).reshape(w.shape[0], -1, 64)
    t, j, f = np.meshgrid(np.arange(8), np.arange(2), np.arange(4), indexing="ij")
    old = 8 * t + 2 * f + j
    new = 16 * f + 8 * (t // 4) + (t % 4) + 4 * j
    out = np.empty_like(nib)
    out[..., new.ravel()] = nib[..., old.ravel()]
    out = out.reshape(w.shape[0], -1, 2)
    words = (out[..., 0] | (out[..., 1] << 4)).astype(np.uint8).view(np.uint32)
    N, G = words.shape[0], words.shape[1] // 8
    return mx.array(words.reshape(N // 8, 8, G, 8).transpose(0, 2, 1, 3).reshape(N, -1))


def pack_sb(scales, biases):
    import numpy as np
    sb = np.stack([np.array(scales.astype(mx.float16)), np.array(biases.astype(mx.float16))], -1)
    N, G = sb.shape[:2]
    return mx.array(sb.reshape(N // 8, 8, G, 2).transpose(0, 2, 1, 3).reshape(N, -1).view(np.uint32))


def qmv(x, wq, sb, RT=RT, SIMDS=SIMDS):
    *lead, K = x.shape
    M, N = x.size // K, wq.shape[0]
    out = _kernel(
        inputs=[x, wq, sb],
        template=[("MT", (M + 7) // 8), ("M", M), ("K", K), ("N", N), ("RT", RT), ("SIMDS", SIMDS), ("OT", x.dtype)],
        grid=(N // (8 * RT * SIMDS) * SIMDS * 32, 1, 1),
        threadgroup=(SIMDS * 32, 1, 1),
        output_shapes=[(*lead, N)],
        output_dtypes=[x.dtype],
    )[0]
    return out


def patch(model, lo=4, hi=16):
    import mlx.nn as nn
    n = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear) and m.bits == 4 and m.group_size == 64 and m.weight.shape[0] % (8 * RT * SIMDS) == 0:
            m._vb = repack(m.weight)
            m._sb = pack_sb(m.scales, m.biases)
            n += 1
    orig = nn.QuantizedLinear.__call__

    def call(self, x):
        rows = x.size // x.shape[-1]
        if hasattr(self, "_vb") and lo <= rows <= hi:
            y = qmv(x, self._vb, self._sb)
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
        sbp = pack_sb(s, b)
        for M in [1, 2, 4, 8, 16]:
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            ref = mx.quantized_matmul(x, wq0, s, b, transpose=True, group_size=64, bits=4).astype(mx.float32)
            got = qmv(x, wq, sbp).astype(mx.float32)
            err = float(mx.abs(got - ref).max() / mx.abs(ref).max())
            solo = mx.concatenate([qmv(x[i : i + 1], wq, sbp) for i in range(M)]).astype(mx.float32)
            inv = bool(mx.array_equal(solo, got))
            t_ref = bench(lambda: mx.quantized_matmul(x, wq0, s, b, transpose=True, group_size=64, bits=4))
            t_new = bench(lambda: qmv(x, wq, sbp))
            gb = N * K / 2 / 1e9
            print(f"N={N} K={K} M={M} err={err:.1e} batch_invariant={inv} mlx={t_ref*1e6:.0f}us new={t_new*1e6:.0f}us ({gb/t_new:.0f} GB/s)")


SPLIT_SOURCE = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    int row0 = threadgroup_position_in_grid.x * 8;
    short qid = lane / 4;
    short fm = (qid & 4) + ((lane / 2) % 4);
    short fn = (qid & 2) * 2 + (lane % 2) * 2;
    int groups = K / 64;
    int per = groups / S;
    const device uint2* wv = (const device uint2*)wq;
    const device half2* sbv = (const device half2*)sb;
    threadgroup float part[S * MT * 64];
    simdgroup_matrix<float, 8, 8> C[MT];
    for (int mt = 0; mt < MT; mt++)
        C[mt] = simdgroup_matrix<float, 8, 8>(0);
    int base = (row0 / 8) * groups * 8 + fm;
    int g0 = sg * per;
    uint2 wn = wv[(base + g0 * 8) * 4 + fn / 2];
    half2 sbn = sbv[base + g0 * 8];
    for (int g = g0; g < g0 + per; g++) {
        uint2 wr = wn;
        half2 s = half2(sbn.x), b = half2(sbn.y);
        if (g + 1 < g0 + per) {
            int i = base + (g + 1) * 8;
            wn = wv[i * 4 + fn / 2];
            sbn = sbv[i];
        }
        for (int t = 0; t < 8; t++) {
            int k0 = g * 64 + t * 8;
            uint pair = ((wr[t >> 2] >> ((t & 3) * 4)) & 0x000F000Fu) | 0x64006400u;
            half2 h = fma(as_type<half2>(pair) - half2(1024.0h), s, b);
            simdgroup_matrix<half, 8, 8> A;
            A.thread_elements()[0] = h.x;
            A.thread_elements()[1] = h.y;
            for (int mt = 0; mt < MT; mt++) {
                int m = mt * 8 + fn;
                half v0 = (half)x[min(m, M - 1) * K + k0 + fm];
                half v1 = (half)x[min(m + 1, M - 1) * K + k0 + fm];
                simdgroup_matrix<half, 8, 8> B;
                B.thread_elements()[0] = m < M ? v0 : 0.0h;
                B.thread_elements()[1] = m + 1 < M ? v1 : 0.0h;
                simdgroup_multiply_accumulate(C[mt], A, B, C[mt]);
            }
        }
    }
    for (int mt = 0; mt < MT; mt++) {
        part[(sg * MT + mt) * 64 + fm * 8 + fn] = C[mt].thread_elements()[0];
        part[(sg * MT + mt) * 64 + fm * 8 + fn + 1] = C[mt].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = sg * 32 + lane; i < MT * 64; i += S * 32) {
        int mt = i / 64, e = i % 64;
        float acc = part[mt * 64 + e];
        for (int j = 1; j < S; j++)
            acc += part[(j * MT + mt) * 64 + e];
        int m = mt * 8 + e % 8, n = row0 + e / 8;
        if (m < M) out[m * N + n] = (OT)acc;
    }
"""

_split = mx.fast.metal_kernel(
    name="verbatim_qmv_split",
    input_names=["x", "wq", "sb"],
    output_names=["out"],
    source=SPLIT_SOURCE,
    header="#include <metal_simdgroup_matrix>\nusing namespace metal;\n",
)

SPLIT_S = 8


def qmv_split(x, wq, sb, S=SPLIT_S):
    *lead, K = x.shape
    M, N = x.size // K, wq.shape[0]
    return _split(
        inputs=[x, wq, sb],
        template=[("MT", (M + 7) // 8), ("M", M), ("K", K), ("N", N), ("S", S), ("OT", x.dtype)],
        grid=(N // 8 * S * 32, 1, 1),
        threadgroup=(S * 32, 1, 1),
        output_shapes=[(*lead, N)],
        output_dtypes=[x.dtype],
    )[0]


GEMV_SPLIT_SOURCE = """
    int t = thread_position_in_threadgroup.x;
    int r = t % R, sl = t / R;
    int n = threadgroup_position_in_grid.x * R + r;
    int groups = K / 64;
    int per = groups / S;
    int base = (n / 8) * groups * 8 + n % 8;
    const device uint4* w = (const device uint4*)wq;
    const device half2* sbv = (const device half2*)sb;
    threadgroup float part[S * R];
    float acc = 0.0f;
    int g0 = sl * per;
    uint4 n0 = w[(base + g0 * 8) * 2], n1 = w[(base + g0 * 8) * 2 + 1];
    half2 sbn = sbv[base + g0 * 8];
    for (int g = g0; g < g0 + per; g++) {
        uint ws[8] = {n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w};
        half s = sbn.x, b = sbn.y;
        if (g + 1 < g0 + per) {
            int i = base + (g + 1) * 8;
            n0 = w[i * 2]; n1 = w[i * 2 + 1]; sbn = sbv[i];
        }
        for (int k = 0; k < 64; k++) {
            int tt = k / 8, f = (k % 8) / 2, j = k % 2;
            half h = fma((half)((ws[2 * f + tt / 4] >> (4 * ((tt % 4) + 4 * j))) & 15u), s, b);
            acc = fma((float)h, (float)(half)x[g * 64 + k], acc);
        }
    }
    part[sl * R + r] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sl == 0) {
        float tot = part[r];
        for (int j = 1; j < S; j++)
            tot += part[j * R + r];
        out[n] = (OT)tot;
    }
"""

_gemv_split = mx.fast.metal_kernel(name="verbatim_gemv_split", input_names=["x", "wq", "sb"], output_names=["out"], source=GEMV_SPLIT_SOURCE)


def gemv_split(x, wq, sb, S=SPLIT_S, R=32):
    *lead, K = x.shape
    N = wq.shape[0]
    return _gemv_split(
        inputs=[x, wq, sb],
        template=[("K", K), ("S", S), ("R", R), ("OT", x.dtype)],
        grid=(N * S, 1, 1),
        threadgroup=(R * S, 1, 1),
        output_shapes=[(*lead, N)],
        output_dtypes=[x.dtype],
    )[0]
