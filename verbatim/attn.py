import mlx.core as mx

SOURCE = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tid = sg * 32 + lane;
    int blk = threadgroup_position_in_grid.x;
    int h = threadgroup_position_in_grid.y;
    int NT = k_shape[2];
    int key0 = blk * BK;
    const device T* K = k + h * k_strides[1];
    const device T* V = v + h * v_strides[1];
    const device T* Q = q + h * R * D;
    threadgroup float S[R * BK];
    threadgroup T P[R * BK];
    for (int i = sg; i < (R / 8) * (BK / 8); i += SG) {
            int rt = i / (BK / 8), ct = i % (BK / 8);
            simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0);
            for (int dt = 0; dt < D / 8; dt++) {
                simdgroup_matrix<T, 8, 8> A, B;
                simdgroup_load(A, Q + rt * 8 * D + dt * 8, D);
                simdgroup_load(B, K + (key0 + ct * 8) * D + dt * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, A, B, acc);
            }
            simdgroup_store(acc, S + rt * 8 * BK + ct * 8, BK);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = tid; r < R; r += SG * 32) {
        int li = r % L;
        int qpos = NT - L + li;
        bool live = r < G * L;
        float m = -INFINITY;
        for (int c = 0; c < BK; c++) {
            int key = key0 + c;
            if (live && key <= qpos)
                m = max(m, S[r * BK + c] * scale);
        }
        float l = 0.0f;
        for (int c = 0; c < BK; c++) {
            int key = key0 + c;
            float p = (live && key <= qpos && m > -INFINITY) ? exp(S[r * BK + c] * scale - m) : 0.0f;
            P[r * BK + c] = (T)p;
            l += p;
        }
        int o = (h * NB + blk) * R + r;
        ms[o] = m > -INFINITY ? m : -1e30f;
        ls[o] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = sg; i < (R / 8) * (D / 8); i += SG) {
            int rt = i / (D / 8), dt = i % (D / 8);
            simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0);
            for (int ct = 0; ct < BK / 8; ct++) {
                simdgroup_matrix<T, 8, 8> A, B;
                simdgroup_load(A, P + rt * 8 * BK + ct * 8, BK);
                simdgroup_load(B, V + (key0 + ct * 8) * D + dt * 8, D);
                simdgroup_multiply_accumulate(acc, A, B, acc);
            }
            simdgroup_store(acc, os + ((h * NB + blk) * R + rt * 8) * D + dt * 8, D);
        }
"""

_kernel = mx.fast.metal_kernel(
    name="verbatim_attn",
    input_names=["q", "k", "v", "scale"],
    output_names=["os", "ms", "ls"],
    source=SOURCE,
    header="#include <metal_simdgroup_matrix>\nusing namespace metal;\n",
    ensure_row_contiguous=False,
)

COMBINE = """
    uint i = thread_position_in_grid.x;
    int d = i % D;
    int r = (i / D) % R;
    int h = i / (D * R);
    float m = -INFINITY;
    for (int b = 0; b < NB; b++)
        m = max(m, ms[(h * NB + b) * R + r]);
    float num = 0.0f, den = 0.0f;
    for (int b = 0; b < NB; b++) {
        int o = (h * NB + b) * R + r;
        float w = exp(ms[o] - m);
        num += w * os[o * D + d];
        den += w * ls[o];
    }
    out[(h * R + r) * D + d] = (T)(num / den);
"""

_combine = mx.fast.metal_kernel(
    name="verbatim_attn_combine",
    input_names=["os", "ms", "ls"],
    output_names=["out"],
    source=COMBINE,
)

def attention(q, k, v, scale, SG=4, BK=None):
    B, H, L, D = q.shape
    kvh, T = k.shape[1], k.shape[2]
    G = H // kvh
    BK = BK or (64 if G <= 4 else 32)
    R = (G * L + 7) // 8 * 8

    NB = (T + BK - 1) // BK
    qg = q.reshape(kvh, G * L, D).astype(k.dtype)
    if R > G * L:
        qg = mx.concatenate([qg, mx.zeros((kvh, R - G * L, D), k.dtype)], axis=1)
    os, ms, ls = _kernel(
        inputs=[qg, k, v, mx.array(scale, mx.float32)],
        template=[("T", k.dtype), ("L", L), ("G", G), ("D", D), ("R", R), ("BK", BK), ("NB", NB), ("SG", SG)],
        grid=(NB * 32 * SG, kvh, 1),
        threadgroup=(32 * SG, 1, 1),
        output_shapes=[(kvh, NB, R, D), (kvh, NB, R), (kvh, NB, R)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = _combine(
        inputs=[os, ms, ls],
        template=[("T", q.dtype), ("D", D), ("R", R), ("NB", NB)],
        grid=(kvh * R * D, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(kvh, R, D)],
        output_dtypes=[q.dtype],
    )[0]
    return out[:, : G * L].reshape(B, H, L, D)

SOURCE2 = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tid = sg * 32 + lane;
    short qid = lane / 4;
    short fm = (qid & 4) + ((lane / 2) % 4);
    int grp = threadgroup_position_in_grid.x;
    int h = threadgroup_position_in_grid.y;
    int NT = k_shape[2];
    const device T* K = k + h * k_strides[1];
    const device T* V = v + h * v_strides[1];
    const device T* Q = q + h * R * D;
    threadgroup float S[R * BK];
    threadgroup T P[R * BK];
    threadgroup float A[R];
    constexpr int TILES = (R / 8) * (D / 8) / SG;
    simdgroup_matrix<float, 8, 8> O[TILES];
    for (int i = 0; i < TILES; i++)
        O[i] = simdgroup_matrix<float, 8, 8>(0);
    float m = -INFINITY, l = 0.0f;
    for (int bi = 0; bi < BPT; bi++) {
        int key0 = (grp * BPT + bi) * BK;
        if (key0 >= NT) break;
        for (int i = sg; i < (R / 8) * (BK / 8); i += SG) {
            int rt = i / (BK / 8), ct = i % (BK / 8);
            simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0);
            for (int dt = 0; dt < D / 8; dt++) {
                simdgroup_matrix<T, 8, 8> Aq, Bk;
                simdgroup_load(Aq, Q + rt * 8 * D + dt * 8, D);
                simdgroup_load(Bk, K + (key0 + ct * 8) * D + dt * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, Aq, Bk, acc);
            }
            simdgroup_store(acc, S + rt * 8 * BK + ct * 8, BK);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < R) {
            int r = tid;
            int qpos = NT - L + r % L;
            bool live = r < G * L;
            float mn = m;
            for (int c = 0; c < BK; c++)
                if (live && key0 + c <= qpos) mn = max(mn, S[r * BK + c] * scale);
            float a = mn == -INFINITY ? 1.0f : exp(m - mn);
            float sum = 0.0f;
            for (int c = 0; c < BK; c++) {
                float p = (live && key0 + c <= qpos && mn > -INFINITY) ? exp(S[r * BK + c] * scale - mn) : 0.0f;
                P[r * BK + c] = (T)p;
                sum += p;
            }
            l = l * a + sum;
            m = mn;
            A[r] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < TILES; j++) {
            int i = sg + j * SG;
            int rt = i / (D / 8), dt = i % (D / 8);
            float a = A[rt * 8 + fm];
            O[j].thread_elements()[0] *= a;
            O[j].thread_elements()[1] *= a;
            for (int ct = 0; ct < BK / 8; ct++) {
                simdgroup_matrix<T, 8, 8> Ap, Bv;
                simdgroup_load(Ap, P + rt * 8 * BK + ct * 8, BK);
                simdgroup_load(Bv, V + (key0 + ct * 8) * D + dt * 8, D);
                simdgroup_multiply_accumulate(O[j], Ap, Bv, O[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int j = 0; j < TILES; j++) {
        int i = sg + j * SG;
        int rt = i / (D / 8), dt = i % (D / 8);
        simdgroup_store(O[j], os + ((h * NB + grp) * R + rt * 8) * D + dt * 8, D);
    }
    if (tid < R) {
        int o = (h * NB + grp) * R + tid;
        ms[o] = m > -INFINITY ? m : -1e30f;
        ls[o] = l;
    }
"""

_kernel2 = mx.fast.metal_kernel(
    name="verbatim_attn2",
    input_names=["q", "k", "v", "scale"],
    output_names=["os", "ms", "ls"],
    source=SOURCE2,
    header="#include <metal_simdgroup_matrix>\nusing namespace metal;\n",
    ensure_row_contiguous=False,
)


def attention2(q, k, v, scale, SG=4, BK=None, BPT=4):
    B, H, L, D = q.shape
    kvh, T = k.shape[1], k.shape[2]
    G = H // kvh
    BK = BK or (64 if G <= 4 else 32)
    R = (G * L + 7) // 8 * 8
    NG = (T + BK * BPT - 1) // (BK * BPT)
    qg = q.reshape(kvh, G * L, D).astype(k.dtype)
    if R > G * L:
        qg = mx.concatenate([qg, mx.zeros((kvh, R - G * L, D), k.dtype)], axis=1)
    os, ms, ls = _kernel2(
        inputs=[qg, k, v, mx.array(scale, mx.float32)],
        template=[("T", k.dtype), ("L", L), ("G", G), ("D", D), ("R", R), ("BK", BK), ("NB", NG), ("SG", SG), ("BPT", BPT)],
        grid=(NG * 32 * SG, kvh, 1),
        threadgroup=(32 * SG, 1, 1),
        output_shapes=[(kvh, NG, R, D), (kvh, NG, R), (kvh, NG, R)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = _combine(
        inputs=[os, ms, ls],
        template=[("T", q.dtype), ("D", D), ("R", R), ("NB", NG)],
        grid=(kvh * R * D, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(kvh, R, D)],
        output_dtypes=[q.dtype],
    )[0]
    return out[:, : G * L].reshape(B, H, L, D)


if __name__ == "__main__":
    import time

    def bench(f, n=20):
        for _ in range(3):
            mx.eval(f())
        ts = time.perf_counter()
        for _ in range(n):
            mx.eval(f())
        return (time.perf_counter() - ts) / n

    T = 6000
    kb = mx.random.normal((1, 4, T + 200, 128)).astype(mx.bfloat16)
    vb = mx.random.normal((1, 4, T + 200, 128)).astype(mx.bfloat16)
    k, v = kb[:, :, :T], vb[:, :, :T]
    for L in (1, 2, 4, 8, 12, 16):
        q = mx.random.normal((1, 28, L, 128)).astype(mx.bfloat16)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.088, mask="causal" if L > 1 else None)
        got = attention(q, k, v, 0.088)
        err = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max())
        t1 = bench(lambda: [mx.fast.scaled_dot_product_attention(q, k, v, scale=0.088, mask="causal") for _ in range(28)])
        t2 = bench(lambda: [attention(q, k, v, 0.088) for _ in range(28)])
        print(f"L={L} err={err:.4f} mlx={t1*1000:.1f}ms fused={t2*1000:.1f}ms")
    q8 = mx.random.normal((1, 28, 8, 128)).astype(mx.bfloat16)
    many = attention(q8, k, v, 0.088)
    for li in range(8):
        n = T - 8 + li + 1
        one = attention(q8[:, :, li : li + 1], kb[:, :, :n], vb[:, :, :n], 0.088)
        print("invariant", li, bool(mx.array_equal(one[:, :, 0], many[:, :, li])))
