"""Bool-masked ``F.scaled_dot_product_attention`` on Apple GPUs as one fused Metal kernel

``masked_attention(q, k, v, mask)`` equals ``F.scaled_dot_product_attention(q, k, v, attn_mask=mask)`` for MPS float32
q, k, v of shape (B, H, L, 32) read through strides (last stride 1) and a bool mask broadcastable to (B, 1, Lq, Lk),
true where a query reads a key, so the transformer's head split (views into one qkv projection) and merge need no
copies. A query with no readable key outputs zeros. fp32 inputs, outputs and accumulation; no score matrix in memory.

Design (after MLX's steel attention, ``mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h``):
- each simdgroup owns TQ 8-row query fragments, so every K and V fragment read from threadgroup memory feeds TQ
  matrix multiplies (a first version with TQ = 1 ran at 2.0-2.3 TFLOPS, this one at 4.4-5.5);
- scores stay in the simdgroup accumulator fragments: the softmax runs on ``thread_elements()``, row maxima by
  ``simd_shuffle_xor`` over lanes 1 and 8, and the probability fragments feed P V directly as left operands, since an
  accumulator fragment has the same lane layout as an input fragment (no threadgroup round trip);
- log2(e) is folded into the query scale, so the exponentials are exp2 (``precise::exp2``: as fast as
  ``fast::exp2`` here and no less accurate in the parity runs);
- no infinities: a masked score is -FLT_MAX and the running maximum starts at -1e38, so a fully masked row or block
  gives exp2(-huge) = 0 and a row sum of 0, which stores zeros; this holds under the default fast-math mode;
- a key block that no row of a simdgroup reads is skipped by that simdgroup;
- per-lane partial row sums, reduced across the row once at the end;
- int32 offsets (64-bit multiplies are emulated on this GPU) and uchar2 mask loads where the mask's key stride is 1.
"""
from __future__ import annotations

import torch

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

constant constexpr int D = 32;             // head dim
constant constexpr int TQ = __TQ__;        // 8-row query fragments per simdgroup
constant constexpr int NS = __NS__;        // simdgroups per threadgroup
constant constexpr int BK = __BK__;        // keys per block
constant constexpr int TK = BK / 8;        // key fragments per block
constant constexpr int QT = 8 * TQ * NS;   // queries per threadgroup
constant constexpr int NT = 32 * NS;       // threads per threadgroup
constant constexpr int LDK = D + 4;        // padded row strides of the K and V tiles
constant constexpr int LDV = D + 4;
constant constexpr float NEG = -FLT_MAX;   // a masked score
constant constexpr float M0 = -1e38f;      // the running max before any readable key: NEG - M0 is still finite
static_assert(TQ * TK * 2 <= 32, "the lane's mask bits fit one uint");

constant constexpr bool MV = __MV__;      // mask key stride 1 with even row and batch strides: uchar2 mask loads

// ints: B, H, Lq, Lk, then element strides q (b, h, n), k (b, h, n), v (b, h, n), out (b, h, n), mask (b, q, k);
// int32: the host checks that every offset fits, and 64-bit multiplies are emulated on this GPU
[[kernel, max_total_threads_per_threadgroup(NT)]]
void masked_attention(device const float* q [[buffer(0)]],
                      device const float* k [[buffer(1)]],
                      device const float* v [[buffer(2)]],
                      device const uchar* mask [[buffer(3)]],
                      device float* out [[buffer(4)]],
                      device const int* ints [[buffer(5)]],
                      constant float& scale [[buffer(6)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const int H = ints[1], Lq = ints[2], Lk = ints[3];
    const int tiles = (Lq + QT - 1) / QT;
    const int bh = int(group) / tiles, q0 = (int(group) % tiles) * QT;
    const int b = bh / H, h = bh % H;
    device const float* qp = q + b * ints[4] + h * ints[5];
    device const float* kp = k + b * ints[7] + h * ints[8];
    device const float* vp = v + b * ints[10] + h * ints[11];
    device float* op = out + b * ints[13] + h * ints[14];
    device const uchar* mp = mask + b * ints[16];
    const int qs = ints[6], ks = ints[9], vs = ints[12], os = ints[15], mq = ints[17], mk = ints[18];

    threadgroup float kt[BK * LDK], vt[BK * LDV];

    // the lane's place in an 8x8 fragment: row fm, columns fn and fn + 1 (MLX BaseMMAFrag::get_coord)
    const short qid = short(lane) / 4;
    const short fm = (qid & 4) + ((short(lane) / 2) % 4);
    const short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;

    // this simdgroup's queries, scaled by scale * log2(e), straight into fragments; rows past Lq are zero
    const int r0 = q0 + int(sg) * 8 * TQ;
    const int rl = r0 + fm;  // the lane's row in query fragment i is rl + 8 i
    simdgroup_float8x8 qm[TQ][4];
    UNROLL for (int i = 0; i < TQ; ++i) {
        const bool in = rl + i * 8 < Lq;
        device const float* src = qp + (in ? rl + i * 8 : 0) * qs + fn;
        UNROLL for (int c = 0; c < 4; ++c) {
            const float2 x = in ? *(device const float2*)(src + c * 8) * scale : float2(0.f);
            qm[i][c].thread_elements()[0] = x[0];
            qm[i][c].thread_elements()[1] = x[1];
        }
    }
    simdgroup_float8x8 om[TQ][4];
    UNROLL for (int i = 0; i < TQ; ++i) UNROLL for (int e = 0; e < 4; ++e) om[i][e] = simdgroup_float8x8(0.f);
    float m[TQ], l[TQ];
    UNROLL for (int i = 0; i < TQ; ++i) { m[i] = M0; l[i] = 0.f; }
    const bool live = r0 < Lq;  // simdgroups wholly past Lq still load tiles and meet the barriers, but skip the math

    for (int k0 = 0; k0 < Lk; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);  // the previous block's tiles are no longer read
        UNROLL for (int n = 0; n < (BK * D / 4 + NT - 1) / NT; ++n) {
            const int e = int(tid) + n * NT;
            if (BK * D / 4 % NT != 0 && e >= BK * D / 4) break;
            const int key = e / (D / 4), d4 = (e % (D / 4)) * 4;
            const bool in = k0 + key < Lk;
            const float4 kx = in ? *(device const float4*)(kp + (k0 + key) * ks + d4) : float4(0.f);
            const float4 vx = in ? *(device const float4*)(vp + (k0 + key) * vs + d4) : float4(0.f);
            *(threadgroup float4*)(kt + key * LDK + d4) = kx;
            *(threadgroup float4*)(vt + key * LDV + d4) = vx;
        }
        // mask bits of this lane's scores, bit (i TK + j) 2 + t; keys past Lk and rows past Lq read nothing
        uint keep = 0;
        if (live) {
            UNROLL for (int i = 0; i < TQ; ++i) {
                const int row = rl + i * 8;
                if (row >= Lq) continue;
                device const uchar* mrow = mp + row * mq;
                UNROLL for (int j = 0; j < TK; ++j) {
                    const int key = k0 + j * 8 + fn;  // even
                    const int bit = (i * TK + j) * 2;
                    if (MV && key + 1 < Lk) {
                        const uchar2 x = *(device const uchar2*)(mrow + key);
                        keep |= (uint(x[0] != 0) | uint(x[1] != 0) << 1) << bit;
                    } else {
                        UNROLL for (int t = 0; t < 2; ++t)
                            keep |= uint(key + t < Lk && mrow[(key + t) * mk] != 0) << (bit + t);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!simd_any(keep != 0)) continue;  // no row of this simdgroup reads this block

        // scores (8 TQ x BK) = q k^T, in log2 units
        simdgroup_float8x8 s[TQ][TK];
        UNROLL for (int i = 0; i < TQ; ++i) UNROLL for (int j = 0; j < TK; ++j) s[i][j] = simdgroup_float8x8(0.f);
        UNROLL for (int j = 0; j < TK; ++j) {
            UNROLL for (int c = 0; c < 4; ++c) {
                simdgroup_float8x8 kk;
                simdgroup_load(kk, kt, LDK, ulong2(c * 8, j * 8), true);
                UNROLL for (int i = 0; i < TQ; ++i) simdgroup_multiply_accumulate(s[i][j], qm[i][c], kk, s[i][j]);
            }
        }
        // online softmax on the fragments
        UNROLL for (int i = 0; i < TQ; ++i) {
            float best = M0;
            UNROLL for (int j = 0; j < TK; ++j) {
                UNROLL for (int t = 0; t < 2; ++t) {
                    const bool in = (keep >> ((i * TK + j) * 2 + t)) & 1u;
                    const float x = in ? s[i][j].thread_elements()[t] : NEG;
                    s[i][j].thread_elements()[t] = x;
                    best = max(best, x);
                }
            }
            best = max(best, simd_shuffle_xor(best, ushort(1)));
            best = max(best, simd_shuffle_xor(best, ushort(8)));
            const float m_new = max(m[i], best);
            const float factor = precise::exp2(m[i] - m_new);
            float sum = 0.f;
            UNROLL for (int j = 0; j < TK; ++j) {
                UNROLL for (int t = 0; t < 2; ++t) {
                    const float p = precise::exp2(s[i][j].thread_elements()[t] - m_new);
                    s[i][j].thread_elements()[t] = p;
                    sum += p;
                }
            }
            l[i] = l[i] * factor + sum;
            m[i] = m_new;
            UNROLL for (int e = 0; e < 4; ++e) {
                om[i][e].thread_elements()[0] *= factor;
                om[i][e].thread_elements()[1] *= factor;
            }
        }
        // out += p v
        UNROLL for (int j = 0; j < TK; ++j) {
            UNROLL for (int e = 0; e < 4; ++e) {
                simdgroup_float8x8 vv;
                simdgroup_load(vv, vt, LDV, ulong2(e * 8, j * 8));
                UNROLL for (int i = 0; i < TQ; ++i) simdgroup_multiply_accumulate(om[i][e], s[i][j], vv, om[i][e]);
            }
        }
    }
    UNROLL for (int i = 0; i < TQ; ++i) {
        float total = l[i];
        total += simd_shuffle_xor(total, ushort(1));
        total += simd_shuffle_xor(total, ushort(8));
        const int row = rl + i * 8;
        if (row >= Lq) continue;
        // a query with no key reads zeros; a true division, one rounding fewer than a reciprocal and a product
        const float denominator = total > 0.f ? total : 1.f;
        device float* dst = op + row * os + fn;
        UNROLL for (int e = 0; e < 4; ++e) {
            *(device float2*)(dst + e * 8) = float2(om[i][e].thread_elements()[0],
                                                    om[i][e].thread_elements()[1]) / denominator;
        }
    }
}
'''

# (TQ, NS, BK): query fragments per simdgroup, simdgroups per threadgroup, keys per block. Sweeps on the M4 Pro
# (B = 16): TQ = 2 is 2.5-5% faster for the 1580-query context calls, TQ = 1 about 5% faster for the 450-query calls,
# whose 64-row tiles (TQ = 2) would leave 62 of 512 rows idle against 30 of 480; BK 16 vs 32 within noise, NS = 2
# and TQ = 4 slower
WIDE, NARROW = (2, 4, 32), (1, 4, 32)


def choose_config(lq):
    """WIDE unless its 64-row tiles idle over 5% more rows than NARROW's 32-row tiles at this query length"""
    return WIDE if -(-lq//64)*64 <= 1.05*(-(-lq//32)*32) else NARROW

_LIBRARIES = {}
_INT32_MAX = 2**31 - 1


def source(tq, ns, bk, mv):
    """The Metal source for TQ query fragments per simdgroup, NS simdgroups per threadgroup, BK-key blocks and
    ``mv`` uchar2 mask loads (mask key stride 1, even batch and query strides and storage offset)"""
    return (_SOURCE.replace('__TQ__', str(tq)).replace('__NS__', str(ns)).replace('__BK__', str(bk))
            .replace('__MV__', 'true' if mv else 'false'))


def library(tq, ns, bk, mv):
    key = (tq, ns, bk, mv)
    if key not in _LIBRARIES:
        _LIBRARIES[key] = torch.mps.compile_shader(source(*key))
    return _LIBRARIES[key]


def fused_attention_suits(q, k, v, mask):
    """Whether :func:`masked_attention` takes this ``F.scaled_dot_product_attention`` call: MPS float32 tensors
    (B, H, L, 32) without autograd, a bool mask broadcastable to (B, 1, Lq, Lk), true where a query reads a key."""
    if mask is None or mask.dtype != torch.bool or q.device.type != 'mps' or torch.is_grad_enabled() and any(
            t.requires_grad for t in (q, k, v)):
        return False
    if any(t.dtype != torch.float32 or t.ndim != 4 or t.shape[-1] != 32 or t.stride(-1) != 1 for t in (q, k, v)):
        return False
    b, h, lq, _ = q.shape
    return (k.shape == v.shape and k.shape[:2] == (b, h) and mask.ndim == 4 and mask.shape[1] == 1
            and mask.shape[0] in (1, b) and mask.shape[2] in (1, lq) and mask.shape[3] == k.shape[2])


def _vector_aligned(t):
    """The kernel reads k and v rows as float4 and q rows as float2: every row must start 16-byte aligned"""
    return t.storage_offset() % 4 == 0 and all(s % 4 == 0 for s in t.stride()[:3])


def _int32_reach(t):
    """Whether every element offset the kernel forms for ``t`` (storage offset included) fits in int32"""
    return t.storage_offset() + sum((n - 1)*abs(s) for n, s in zip(t.shape, t.stride())) <= _INT32_MAX


def masked_attention(q, k, v, mask, config=None):
    """``F.scaled_dot_product_attention(q, k, v, attn_mask=mask)`` for a bool ``mask`` (:func:`fused_attention_suits`)
    by one Metal kernel. The output (B, H, Lq, 32) is a view of a (B, Lq, H, 32) tensor, so merging heads is free; a
    query with no readable key reads zeros. ``config`` overrides the (TQ, NS, BK) tile choice."""
    tq, ns, bk = config or choose_config(q.shape[2])
    q, k, v = (t if _vector_aligned(t) else t.contiguous() for t in (q, k, v))
    b, h, lq, d = q.shape
    lk = k.shape[2]
    mask = mask.expand(b, 1, lq, lk).view(torch.uint8)
    out = torch.empty((b, lq, h, d), dtype=torch.float32, device=q.device).transpose(1, 2)
    if not all(map(_int32_reach, (q, k, v, mask, out))):
        raise ValueError('masked_attention: a tensor spans more than 2**31 elements')
    strides = [*q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *out.stride()[:3], mask.stride(0), mask.stride(2),
               mask.stride(3)]
    mv = mask.stride(3) == 1 and mask.stride(0) % 2 == 0 and mask.stride(2) % 2 == 0 and mask.storage_offset() % 2 == 0
    ints = torch.tensor([b, h, lq, lk, *strides], dtype=torch.int32, device=q.device)
    groups = b*h*(-(-lq//(8*tq*ns)))
    library(tq, ns, bk, mv).masked_attention(q, k, v, mask, out, ints, d**-0.5*1.4426950408889634,
                                             threads=groups*32*ns, group_size=32*ns)
    return out
