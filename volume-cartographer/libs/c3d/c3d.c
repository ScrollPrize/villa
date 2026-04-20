/* c3d — 3D volumetric u8 compression codec.  See LICENSE, PLAN.md, CLAUDE.md.
 *
 * This TU is organised into sections:
 *     §A  scaffolding (panic/assert, bit-io, alignment, Morton-12)
 *     §B  c3d_hash128 (MurmurHash3_x64_128)
 *     §C  rANS engine (scalar + 8-way interleaved)
 *     §D..§L  not yet implemented (see PLAN.md §6)
 */

#include "c3d.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Endianness gate — LE only. */
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#  error "c3d requires a little-endian target"
#endif

#if defined(_OPENMP)
#  include <omp.h>
#endif

/* Architecture-specific SIMD.  Guarded so portable C is always the fallback.
 *
 * Selection priority: C3D_SIMD (from CMake) sets C3D_FORCE_SCALAR /
 * C3D_FORCE_AVX2 / C3D_FORCE_AVX512 / C3D_FORCE_NEON; otherwise we auto-detect
 * from compiler predefines.  Targets that hit neither branch fall through to
 * the portable scalar path, which is always correct (just slower).
 *
 * C3D_HAVE_AVX512 implies C3D_HAVE_AVX2.  C3D_HAVE_NEON is aarch64-only.  The
 * hot kernels are gated with `#if C3D_HAVE_AVX512 / #elif C3D_HAVE_AVX2 /
 * #elif C3D_HAVE_NEON / #else scalar`. */
#if defined(C3D_FORCE_SCALAR)
    /* nothing */
#elif defined(C3D_FORCE_AVX512) || (!defined(C3D_FORCE_AVX2) && !defined(C3D_FORCE_NEON) && defined(__AVX512F__))
#  include <immintrin.h>
#  define C3D_HAVE_AVX2 1
#  define C3D_HAVE_AVX512 1
#elif defined(C3D_FORCE_AVX2) || (!defined(C3D_FORCE_NEON) && defined(__AVX2__))
#  include <immintrin.h>
#  define C3D_HAVE_AVX2 1
#elif defined(C3D_FORCE_NEON) || defined(__ARM_NEON)
#  include <arm_neon.h>
#  define C3D_HAVE_NEON 1
#endif

/* ========================================================================= *
 *  §A  Scaffolding                                                          *
 * ========================================================================= */

static c3d_panic_fn s_panic_hook = NULL;

const char *c3d_version(void)        { return C3D_VERSION_STRING; }
uint32_t    c3d_format_version(void) { return C3D_FORMAT_VERSION; }

void c3d_set_panic_hook(c3d_panic_fn hook) {
    s_panic_hook = hook;
}

_Noreturn void c3d_panic(const char *file, int line, const char *msg) {
    if (s_panic_hook) {
        s_panic_hook(file, line, msg);
        /* Hook must not return; if it does, fall through to abort. */
    }
    fprintf(stderr, "c3d_panic: %s:%d: %s\n", file, line, msg ? msg : "(null)");
    fflush(stderr);
    abort();
}

/* ----- bit-io / integer read-write (memcpy-based for unaligned safety) ---- */

static inline uint16_t c3d_read_u16_le(const uint8_t *p) {
    uint16_t v; memcpy(&v, p, sizeof v); return v;
}
static inline uint32_t c3d_read_u32_le(const uint8_t *p) {
    uint32_t v; memcpy(&v, p, sizeof v); return v;
}
static inline uint64_t c3d_read_u64_le(const uint8_t *p) {
    uint64_t v; memcpy(&v, p, sizeof v); return v;
}
static inline float c3d_read_f32_le(const uint8_t *p) {
    float v; memcpy(&v, p, sizeof v); return v;
}
static inline void c3d_write_u16_le(uint8_t *p, uint16_t v) { memcpy(p, &v, sizeof v); }
static inline void c3d_write_u32_le(uint8_t *p, uint32_t v) { memcpy(p, &v, sizeof v); }
static inline void c3d_write_u64_le(uint8_t *p, uint64_t v) { memcpy(p, &v, sizeof v); }
static inline void c3d_write_f32_le(uint8_t *p, float v)    { memcpy(p, &v, sizeof v); }

/* ----- LEB128 unsigned varints (7-bit groups, high bit = continuation) ----
 * Caller-provided bounds to detect overrun without a separate state struct. */

static size_t c3d_leb128_encode(uint64_t v, uint8_t *out, size_t out_cap) {
    size_t n = 0;
    do {
        c3d_assert(n < out_cap);
        uint8_t b = (uint8_t)(v & 0x7f);
        v >>= 7;
        if (v) b |= 0x80;
        out[n++] = b;
    } while (v);
    return n;
}

/* Decode one LEB128 varint from `in[0..in_len)`; returns consumed bytes,
 * writes value to *out.  Panics on truncation or >10-byte encoding. */
static size_t c3d_leb128_decode(const uint8_t *in, size_t in_len, uint64_t *out) {
    uint64_t v = 0;
    unsigned shift = 0;
    size_t n = 0;
    for (;;) {
        c3d_assert(n < in_len);
        c3d_assert(shift < 64);
        uint8_t b = in[n++];
        v |= (uint64_t)(b & 0x7f) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    *out = v;
    return n;
}

/* ----- alignment ---------------------------------------------------------- */

static inline void c3d_check_voxel_alignment(const void *p) {
    c3d_assert(((uintptr_t)p & (C3D_ALIGN - 1)) == 0);
}

/* ----- 12-bit Morton (shard chunk-index ordering) ------------------------- *
 * Chunk coords are 4 bits each; Morton interleaves as z3 y3 x3 ... z0 y0 x0. */

C3D_CONST
static inline uint32_t c3d_morton12(uint32_t cx, uint32_t cy, uint32_t cz) {
    c3d_assert(cx < 16 && cy < 16 && cz < 16);
    /* Spread 4 bits of each to every third bit, then combine. */
    static const uint32_t spread4[16] = {
        0x000, 0x001, 0x008, 0x009, 0x040, 0x041, 0x048, 0x049,
        0x200, 0x201, 0x208, 0x209, 0x240, 0x241, 0x248, 0x249,
    };
    return spread4[cx] | (spread4[cy] << 1) | (spread4[cz] << 2);
}

static inline void c3d_morton12_decode(uint32_t m,
                                       uint32_t *cx, uint32_t *cy, uint32_t *cz) {
    /* Compact every third bit back to a nibble. */
    uint32_t x = 0, y = 0, z = 0;
    for (unsigned i = 0; i < 4; ++i) {
        x |= ((m >> (3*i + 0)) & 1u) << i;
        y |= ((m >> (3*i + 1)) & 1u) << i;
        z |= ((m >> (3*i + 2)) & 1u) << i;
    }
    *cx = x; *cy = y; *cz = z;
}

/* ========================================================================= *
 *  §B  c3d_hash128  (MurmurHash3_x64_128)                                   *
 * ========================================================================= */

/* MurmurHash3_x64_128 by Austin Appleby (public domain).  Rewritten here for
 * 64-byte-per-block accumulation, adapted for little-endian direct reads.
 * Not cryptographically secure; intended only for content addressing. */

static inline uint64_t c3d_rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t c3d_fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

void c3d_hash128(const void *data, size_t len, uint8_t out[16]) {
    const uint8_t *p = (const uint8_t *)data;
    const size_t   nblocks = len / 16;

    uint64_t h1 = 0;
    uint64_t h2 = 0;

    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;

    for (size_t i = 0; i < nblocks; ++i) {
        uint64_t k1 = c3d_read_u64_le(p + 16*i + 0);
        uint64_t k2 = c3d_read_u64_le(p + 16*i + 8);

        k1 *= c1; k1 = c3d_rotl64(k1, 31); k1 *= c2; h1 ^= k1;
        h1 = c3d_rotl64(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729;

        k2 *= c2; k2 = c3d_rotl64(k2, 33); k2 *= c1; h2 ^= k2;
        h2 = c3d_rotl64(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
    }

    const uint8_t *tail = p + nblocks * 16;
    uint64_t k1 = 0, k2 = 0;
    switch (len & 15) {
    case 15: k2 ^= (uint64_t)tail[14] << 48;  /* fallthrough */
    case 14: k2 ^= (uint64_t)tail[13] << 40;  /* fallthrough */
    case 13: k2 ^= (uint64_t)tail[12] << 32;  /* fallthrough */
    case 12: k2 ^= (uint64_t)tail[11] << 24;  /* fallthrough */
    case 11: k2 ^= (uint64_t)tail[10] << 16;  /* fallthrough */
    case 10: k2 ^= (uint64_t)tail[ 9] <<  8;  /* fallthrough */
    case  9: k2 ^= (uint64_t)tail[ 8] <<  0;
             k2 *= c2; k2 = c3d_rotl64(k2, 33); k2 *= c1; h2 ^= k2;
             /* fallthrough */
    case  8: k1 ^= (uint64_t)tail[ 7] << 56;  /* fallthrough */
    case  7: k1 ^= (uint64_t)tail[ 6] << 48;  /* fallthrough */
    case  6: k1 ^= (uint64_t)tail[ 5] << 40;  /* fallthrough */
    case  5: k1 ^= (uint64_t)tail[ 4] << 32;  /* fallthrough */
    case  4: k1 ^= (uint64_t)tail[ 3] << 24;  /* fallthrough */
    case  3: k1 ^= (uint64_t)tail[ 2] << 16;  /* fallthrough */
    case  2: k1 ^= (uint64_t)tail[ 1] <<  8;  /* fallthrough */
    case  1: k1 ^= (uint64_t)tail[ 0] <<  0;
             k1 *= c1; k1 = c3d_rotl64(k1, 31); k1 *= c2; h1 ^= k1;
             /* fallthrough */
    case  0: break;
    }

    h1 ^= (uint64_t)len;
    h2 ^= (uint64_t)len;
    h1 += h2;
    h2 += h1;
    h1 = c3d_fmix64(h1);
    h2 = c3d_fmix64(h2);
    h1 += h2;
    h2 += h1;

    c3d_write_u64_le(out + 0, h1);
    c3d_write_u64_le(out + 8, h2);
}

/* ========================================================================= *
 *  §C  rANS engine (ryg_rans_byte-style)                                    *
 * ========================================================================= *
 *
 * 32-bit state, byte-at-a-time renormalisation, lower bound RANS_BYTE_L.
 * Encoder writes bytes backward (from buf end toward buf start); decoder
 * reads bytes forward from where the encoder finished.
 *
 * 8-way interleaved variant: 8 independent states encoding symbols dealt
 * round-robin.  Wire format per PLAN §3.4:
 *   rans_header    = 32 B = 8 × u32 final states (little-endian)
 *   rans_renorm    = the forward-read renorm byte stream
 * The decoder loads the 8 states from the header, then consumes renorm
 * bytes on demand as each state's value drops below RANS_BYTE_L.
 */

#define C3D_RANS_BYTE_L    (1u << 23)   /* lower bound of normal state range */
#define C3D_RANS_N_STATES  8u           /* interleaving factor               */

typedef struct {
    uint32_t start;   /* cumulative probability of symbol */
    uint32_t freq;    /* probability of symbol            */
} c3d_rans_sym;

typedef struct {
    uint32_t cum2sym[1u << 14];  /* map cumulative prob → symbol (max M=16384) */
    c3d_rans_sym syms[65];        /* 65 symbols in the c3d alphabet            */
    uint32_t denom_shift;         /* log2(M); M = 1<<denom_shift               */
} c3d_rans_tables;

/* Build fast tables from a (symbol,freq) list that sums to M = 1<<denom_shift.
 * Panics if frequencies don't sum to M. */
static void c3d_rans_build_tables(c3d_rans_tables *t,
                                  uint32_t denom_shift,
                                  const uint32_t *freqs,      /* freqs[0..n_symbols) */
                                  size_t n_symbols)
{
    c3d_assert(denom_shift <= 14);
    const uint32_t M = 1u << denom_shift;
    t->denom_shift = denom_shift;

    uint32_t cum = 0;
    for (size_t s = 0; s < n_symbols; ++s) {
        t->syms[s].start = cum;
        t->syms[s].freq  = freqs[s];
        for (uint32_t i = 0; i < freqs[s]; ++i) {
            t->cum2sym[cum + i] = (uint32_t)s;
        }
        cum += freqs[s];
    }
    c3d_assert(cum == M);
    /* Fill unused alphabet entries (freq=0) with defined start/freq. */
    for (size_t s = n_symbols; s < 65; ++s) {
        t->syms[s].start = cum;
        t->syms[s].freq  = 0;
    }
}

/* ----- scalar rANS ------------------------------------------------------- */

/* Encoder state; bytes are written at *out_p, stepping backward.
 * Initial state = RANS_BYTE_L.  Call rans_flush at end to dump final state. */

static inline void c3d_rans_enc_init(uint32_t *state) {
    *state = C3D_RANS_BYTE_L;
}

static inline void c3d_rans_enc_put(uint32_t *state,
                                    uint8_t **out_p,
                                    const uint8_t *out_begin,
                                    uint32_t start, uint32_t freq,
                                    uint32_t denom_shift)
{
    /* Renormalise: while state * M / freq >= 2^32 — equivalently, while
     * state >= freq * (RANS_BYTE_L >> (denom_shift)) << 8. */
    uint32_t x = *state;
    uint32_t x_max = ((C3D_RANS_BYTE_L >> denom_shift) << 8) * freq;
    while (x >= x_max) {
        c3d_assert(*out_p > out_begin);
        *--(*out_p) = (uint8_t)(x & 0xff);
        x >>= 8;
    }
    /* Mix in the symbol. */
    *state = ((x / freq) << denom_shift) + (x % freq) + start;
}

static inline void c3d_rans_enc_flush(uint32_t state,
                                      uint8_t **out_p, const uint8_t *out_begin)
{
    c3d_assert(*out_p - out_begin >= 4);
    *--(*out_p) = (uint8_t)((state >> 24) & 0xff);
    *--(*out_p) = (uint8_t)((state >> 16) & 0xff);
    *--(*out_p) = (uint8_t)((state >>  8) & 0xff);
    *--(*out_p) = (uint8_t)((state >>  0) & 0xff);
}

static inline void c3d_rans_dec_init(uint32_t *state,
                                     const uint8_t **in_p, const uint8_t *in_end)
{
    c3d_assert(in_end - *in_p >= 4);
    uint32_t x = (*in_p)[0]
              | ((uint32_t)(*in_p)[1] <<  8)
              | ((uint32_t)(*in_p)[2] << 16)
              | ((uint32_t)(*in_p)[3] << 24);
    *in_p += 4;
    *state = x;
}

/* Decode one symbol using the cum2sym lookup. */
static inline uint32_t c3d_rans_dec_get(uint32_t *state,
                                        const c3d_rans_tables *t)
{
    const uint32_t M_mask = (1u << t->denom_shift) - 1u;
    uint32_t slot = *state & M_mask;
    uint32_t sym  = t->cum2sym[slot];
    /* advance state */
    *state = t->syms[sym].freq * (*state >> t->denom_shift)
           + slot - t->syms[sym].start;
    return sym;
}

/* After decoding a symbol, the decoder may need to consume renorm bytes. */
static inline void c3d_rans_dec_renorm(uint32_t *state,
                                       const uint8_t **in_p, const uint8_t *in_end)
{
    while (*state < C3D_RANS_BYTE_L) {
        c3d_assert(*in_p < in_end);
        *state = (*state << 8) | **in_p;
        ++*in_p;
    }
}

/* ----- 8-way interleaved rANS -------------------------------------------- *
 *
 * Encoder:  8 independent states, symbols dispatched round-robin by index.
 * Output layout in the per-subband bitstream:
 *     [ 8 × u32 final states (32 B) ][ renorm bytes, forward-read ]
 *
 * During encoding we don't know where the renorm byte stream will end up
 * relative to the 32 B state header until the encode completes, because bytes
 * are written backward.  We allocate a working buffer, run the encoder
 * writing backward from the end, then:
 *     1. Copy the resulting renorm bytes into the final output in FORWARD
 *        order (because the decoder reads forward).
 *     2. Write the 8 final states as 32 B little-endian u32s at the
 *        start.
 *
 * So the encoder needs O(max_output_size) scratch memory per subband. */

typedef struct {
    /* Scratch layout used during encode: bytes are written at scratch+scratch_end
     * growing downward to scratch+scratch_head. */
    uint8_t *scratch;
    size_t   scratch_size;
} c3d_rans_enc_scratch;

/* Encode `symbols[0..n_symbols)` interleaved over 8 states.
 * `symbol_of` is a function that maps an alphabet index to (start, freq).
 * Output format: out[0..32) = 8 u32 states; out[32..out_len) = renorm bytes.
 * Returns the total number of bytes written to out; panics on out_cap exceed. */
static size_t c3d_rans_enc_x8(const uint8_t *symbols,     /* each < 65 */
                              size_t n_symbols,
                              const c3d_rans_tables *t,
                              uint8_t *scratch, size_t scratch_size,
                              uint8_t *out, size_t out_cap)
{
    /* We need scratch_size large enough to hold all renorm bytes.  Worst case:
     * each symbol triggers up to 2 renorm bytes (rare at denom_shift ≤ 14 with
     * start state = RANS_BYTE_L).  We demand scratch_size ≥ n_symbols * 2 + 32. */
    c3d_assert(scratch_size >= n_symbols * 2u + 32u);
    (void)scratch_size;

    uint8_t       *buf_end = scratch + scratch_size;
    uint8_t       *buf_ptr = buf_end;
    const uint8_t *buf_beg = scratch;

    uint32_t states[C3D_RANS_N_STATES];
    for (unsigned i = 0; i < C3D_RANS_N_STATES; ++i) {
        c3d_rans_enc_init(&states[i]);
    }

    /* Encode from back to front so that decode-forward reads renorm bytes in
     * the right order.  Each symbol i is assigned to stream (i % 8).
     * We reverse that: iterate i from n_symbols-1 down to 0.
     *
     * Fast path for symbol 0: hoist its (freq, start=0, x_max) so the common
     * case avoids the t->syms[s] dependent load that feeds the renorm check.
     *
     * Subband sizes are always multiples of 8 (8³ .. 128³), so we unroll
     * 8 iterations per loop step — one touch of each lane's state.  Exposes
     * inter-lane independence to the scheduler, mirroring the decoder's
     * 8-lane unroll. */
    c3d_assert((n_symbols & 7u) == 0);
    const uint32_t ds       = t->denom_shift;
    const uint32_t freq0    = t->syms[0].freq;
    c3d_assert(t->syms[0].start == 0);
    const uint32_t x_max_0  = freq0 ? ((C3D_RANS_BYTE_L >> ds) << 8) * freq0
                                    : UINT32_MAX;

    /* Per-lane enc: fast path for sym 0, else call into c3d_rans_enc_put. */
    #define ENC_LANE(LANE, SYM) do {                                              \
        uint8_t s_ = (SYM);                                                        \
        if (__builtin_expect(s_ == 0, 1)) {                                        \
            uint32_t x = states[LANE];                                             \
            while (x >= x_max_0) {                                                 \
                c3d_assert(buf_ptr > buf_beg);                                     \
                *--buf_ptr = (uint8_t)(x & 0xff);                                  \
                x >>= 8;                                                           \
            }                                                                      \
            states[LANE] = ((x / freq0) << ds) + (x % freq0);                      \
        } else {                                                                   \
            const c3d_rans_sym *sym_ = &t->syms[s_];                               \
            c3d_assert(sym_->freq > 0);                                            \
            c3d_rans_enc_put(&states[LANE], &buf_ptr, buf_beg,                     \
                             sym_->start, sym_->freq, ds);                         \
        }                                                                          \
    } while (0)

    /* Process 8 symbols per step, in backward lane order (7,6,5,4,3,2,1,0)
     * which matches the backward symbol-index walk idx = i-1 … i-8. */
    for (size_t i = n_symbols; i >= 8; i -= 8) {
        ENC_LANE(7, symbols[i - 1]);
        ENC_LANE(6, symbols[i - 2]);
        ENC_LANE(5, symbols[i - 3]);
        ENC_LANE(4, symbols[i - 4]);
        ENC_LANE(3, symbols[i - 5]);
        ENC_LANE(2, symbols[i - 6]);
        ENC_LANE(1, symbols[i - 7]);
        ENC_LANE(0, symbols[i - 8]);
    }
    #undef ENC_LANE

    size_t renorm_bytes = (size_t)(buf_end - buf_ptr);
    size_t total = 32u + renorm_bytes;
    c3d_assert(total <= out_cap);

    /* Write the 8 final states (u32 LE each) to out[0..32). */
    for (unsigned i = 0; i < C3D_RANS_N_STATES; ++i) {
        c3d_write_u32_le(out + 4u * i, states[i]);
    }
    /* Copy renorm bytes; in our scratch they run [buf_ptr .. buf_end), and the
     * decoder reads them forward starting at out+32. */
    memcpy(out + 32, buf_ptr, renorm_bytes);
    return total;
}

/* Context-aware variant of c3d_rans_enc_x8: each symbol selects between
 * tbl_z (lane's previous symbol was 0) and tbl_nz (non-zero).  Context is
 * derived from sub_symbols[idx-8] (same lane, previous batch), which is
 * already written during the forward quant pass. */
static size_t c3d_rans_enc_x8_ctx(const uint8_t *symbols, size_t n_symbols,
                                   const c3d_rans_tables *tbl_z,
                                   const c3d_rans_tables *tbl_nz,
                                   uint8_t *scratch, size_t scratch_size,
                                   uint8_t *out, size_t out_cap)
{
    c3d_assert((n_symbols & 7u) == 0);
    c3d_assert(scratch_size >= n_symbols * 2u + 32u);
    (void)scratch_size;

    uint8_t       *buf_end = scratch + scratch_size;
    uint8_t       *buf_ptr = buf_end;
    const uint8_t *buf_beg = scratch;

    uint32_t states[C3D_RANS_N_STATES];
    for (unsigned i = 0; i < C3D_RANS_N_STATES; ++i)
        c3d_rans_enc_init(&states[i]);

    const uint32_t ds = tbl_z->denom_shift;  /* both tables share denom_shift */

    for (size_t i = n_symbols; i >= 8; i -= 8) {
        /* Process 8 symbols per step (backward). */
        for (unsigned lane = 8; lane-- > 0; ) {
            size_t idx = i - 8 + lane;
            bool ctx = (idx >= 8) ? (symbols[idx - 8] != 0) : false;
            const c3d_rans_tables *t = ctx ? tbl_nz : tbl_z;
            uint8_t s = symbols[idx];
            const c3d_rans_sym *sym = &t->syms[s];
            c3d_assert(sym->freq > 0);
            c3d_rans_enc_put(&states[lane], &buf_ptr, buf_beg,
                             sym->start, sym->freq, ds);
        }
    }

    size_t renorm_bytes = (size_t)(buf_end - buf_ptr);
    size_t total = 32u + renorm_bytes;
    c3d_assert(total <= out_cap);
    for (unsigned i = 0; i < C3D_RANS_N_STATES; ++i)
        c3d_write_u32_le(out + 4u * i, states[i]);
    memcpy(out + 32, buf_ptr, renorm_bytes);
    return total;
}

/* Decode n_symbols from the packed rans block at `in[0..in_len)`; writes
 * symbols[0..n_symbols).  Panics on truncation.
 *
 * 8 lanes are unrolled into independent local state variables so the compiler
 * can schedule their dec_get / renorm operations as parallel ILP.  This is
 * pure C — no intrinsics — and gets a meaningful speedup on out-of-order
 * cores by exposing inter-lane independence to the scheduler. */
static void c3d_rans_dec_x8(const uint8_t *in, size_t in_len,
                            const c3d_rans_tables *t,
                            uint8_t *symbols, size_t n_symbols)
{
    c3d_assert(in_len >= 32);
    uint32_t s0 = c3d_read_u32_le(in +  0);
    uint32_t s1 = c3d_read_u32_le(in +  4);
    uint32_t s2 = c3d_read_u32_le(in +  8);
    uint32_t s3 = c3d_read_u32_le(in + 12);
    uint32_t s4 = c3d_read_u32_le(in + 16);
    uint32_t s5 = c3d_read_u32_le(in + 20);
    uint32_t s6 = c3d_read_u32_le(in + 24);
    uint32_t s7 = c3d_read_u32_le(in + 28);
    const uint8_t *r   = in + 32;
    const uint8_t *r_e = in + in_len;
    const uint32_t M_mask  = (1u << t->denom_shift) - 1u;
    const uint32_t ds      = t->denom_shift;
    const uint32_t *cum2sym = t->cum2sym;
    const c3d_rans_sym *syms = t->syms;
    /* Symbol 0 occupies cumulative slots [0, freq0) — typically 70-90 % of M
     * in quantized wavelet subbands.  Hoist it so the hot branch avoids the
     * cum2sym and syms[] dependent loads. */
    const uint32_t freq0 = syms[0].freq;

    /* dec_get inlined: sym = cum2sym[state & M], state = freq*(state>>ds) + (state&M) - start.
     * Fast path: if slot < freq0 then sym=0, start=0, freq=freq0 (all loads avoidable). */
    #define DEC_LANE(SREF, OUT) do {                                          \
        uint32_t st = (SREF);                                                 \
        uint32_t slot = st & M_mask;                                          \
        if (__builtin_expect(slot < freq0, 1)) {                              \
            (OUT) = 0;                                                        \
            (SREF) = freq0 * (st >> ds) + slot;                               \
        } else {                                                              \
            uint32_t sym = cum2sym[slot];                                     \
            c3d_assert(sym < 65);                                             \
            (OUT) = (uint8_t)sym;                                             \
            (SREF) = syms[sym].freq * (st >> ds) + slot - syms[sym].start;    \
        }                                                                     \
    } while (0)
    #define RENORM(SREF) do {                                                 \
        while ((SREF) < C3D_RANS_BYTE_L) {                                    \
            c3d_assert(r < r_e);                                              \
            __builtin_prefetch(r + 64, 0, 0);                                 \
            (SREF) = ((SREF) << 8) | *r++;                                    \
        }                                                                     \
    } while (0)

    /* Process 8 lanes per iteration: lane k handles symbols i where i % 8 == k. */
    size_t full = n_symbols & ~(size_t)7u;
    for (size_t i = 0; i < full; i += 8) {
        DEC_LANE(s0, symbols[i + 0]);  RENORM(s0);
        DEC_LANE(s1, symbols[i + 1]);  RENORM(s1);
        DEC_LANE(s2, symbols[i + 2]);  RENORM(s2);
        DEC_LANE(s3, symbols[i + 3]);  RENORM(s3);
        DEC_LANE(s4, symbols[i + 4]);  RENORM(s4);
        DEC_LANE(s5, symbols[i + 5]);  RENORM(s5);
        DEC_LANE(s6, symbols[i + 6]);  RENORM(s6);
        DEC_LANE(s7, symbols[i + 7]);  RENORM(s7);
    }
    /* Tail: at most 7 symbols left.  Walk lane 0..min(7, remaining-1). */
    size_t left = n_symbols - full;
    uint32_t *tail_states[8] = {&s0,&s1,&s2,&s3,&s4,&s5,&s6,&s7};
    for (size_t k = 0; k < left; ++k) {
        DEC_LANE(*tail_states[k], symbols[full + k]);
        RENORM(*tail_states[k]);
    }
    #undef DEC_LANE
    #undef RENORM

    c3d_assert(r == r_e);
}

/* Context-aware variant of c3d_rans_dec_x8: uses 2 tables selected by
 * lane-local context (was previous symbol on this lane zero?). */
static void c3d_rans_dec_x8_ctx(const uint8_t *in, size_t in_len,
                                const c3d_rans_tables *tbl_z,
                                const c3d_rans_tables *tbl_nz,
                                uint8_t *symbols, size_t n_symbols)
{
    c3d_assert(in_len >= 32 && (n_symbols & 7u) == 0);
    uint32_t s0 = c3d_read_u32_le(in +  0);
    uint32_t s1 = c3d_read_u32_le(in +  4);
    uint32_t s2 = c3d_read_u32_le(in +  8);
    uint32_t s3 = c3d_read_u32_le(in + 12);
    uint32_t s4 = c3d_read_u32_le(in + 16);
    uint32_t s5 = c3d_read_u32_le(in + 20);
    uint32_t s6 = c3d_read_u32_le(in + 24);
    uint32_t s7 = c3d_read_u32_le(in + 28);
    const uint8_t *rd   = in + 32;
    const uint8_t *r_e = in + in_len;

    bool ctx0=0,ctx1=0,ctx2=0,ctx3=0,ctx4=0,ctx5=0,ctx6=0,ctx7=0;

    #define DEC_CTX(SREF, OUT, CTX) do {                                       \
        const c3d_rans_tables *t_ = (CTX) ? tbl_nz : tbl_z;                   \
        const uint32_t M_ = (1u << t_->denom_shift) - 1u;                     \
        const uint32_t ds_ = t_->denom_shift;                                 \
        uint32_t st_ = (SREF);                                                \
        uint32_t slot_ = st_ & M_;                                            \
        uint32_t sym_ = t_->cum2sym[slot_];                                   \
        c3d_assert(sym_ < 65);                                                \
        (OUT) = (uint8_t)sym_;                                                 \
        (SREF) = t_->syms[sym_].freq * (st_ >> ds_) + slot_                   \
               - t_->syms[sym_].start;                                        \
        while ((SREF) < C3D_RANS_BYTE_L) {                                    \
            c3d_assert(rd < r_e);                                              \
            (SREF) = ((SREF) << 8) | *rd++;                                   \
        }                                                                      \
        (CTX) = (sym_ != 0);                                                   \
    } while (0)

    for (size_t i = 0; i < n_symbols; i += 8) {
        DEC_CTX(s0, symbols[i+0], ctx0);
        DEC_CTX(s1, symbols[i+1], ctx1);
        DEC_CTX(s2, symbols[i+2], ctx2);
        DEC_CTX(s3, symbols[i+3], ctx3);
        DEC_CTX(s4, symbols[i+4], ctx4);
        DEC_CTX(s5, symbols[i+5], ctx5);
        DEC_CTX(s6, symbols[i+6], ctx6);
        DEC_CTX(s7, symbols[i+7], ctx7);
    }
    #undef DEC_CTX
    c3d_assert(rd == r_e);
}

/* ========================================================================= *
 *  §D  Per-subband frequency tables (build + serialise + parse)             *
 * ========================================================================= *
 *
 * Wire format (PLAN §3.4):
 *     u8  denom_shift        log2(M); M is the cumulative-frequency denominator
 *     u8  n_nonzero          1..65
 *     n_nonzero × { u8 symbol_index, LEB128 freq }
 * Invariant: Σ freq[i] == M exactly.  Parser verifies and panics on mismatch.
 *
 * For encoding, we build a histogram from the symbol buffer, normalise so the
 * sum is exactly M, then serialise.  Normalisation preserves the "present →
 * freq ≥ 1" invariant required by rANS (a zero-prob symbol can't be encoded).
 */

#ifdef C3D_BUILD_REF
/* Count each symbol value 0..64 into hist[65].  Only used by c3d_test for
 * round-trip checks of the freq-table build path. */
static void c3d_histogram65(const uint8_t *symbols, size_t n, uint32_t hist[65]) {
    memset(hist, 0, 65 * sizeof(uint32_t));
    for (size_t i = 0; i < n; ++i) {
        c3d_assert(symbols[i] < 65u);
        hist[symbols[i]]++;
    }
}
#endif

/* Normalise a 65-entry histogram so the nonzero entries sum to M = 1<<denom_shift.
 * Every originally-nonzero entry ends ≥ 1.  Writes freqs[65].
 *
 * Algorithm (after ryg_rans):
 *   1. alloc[i] = (hist[i] * M) / T       (where T = Σ hist)
 *   2. if hist[i] > 0 and alloc[i] == 0: bump alloc[i] to 1
 *   3. adjust total up or down by shaving from / adding to the largest entry
 *      until sum == M. */
static void c3d_normalise_freqs(const uint32_t hist[65], uint32_t denom_shift,
                                uint32_t freqs[65])
{
    c3d_assert(denom_shift >= 1 && denom_shift <= 14);
    const uint32_t M = 1u << denom_shift;

    uint64_t T = 0;
    for (unsigned i = 0; i < 65; ++i) T += hist[i];
    c3d_assert(T > 0);

    /* Initial floor-scale allocation. */
    uint64_t used = 0;
    for (unsigned i = 0; i < 65; ++i) {
        if (hist[i] == 0) { freqs[i] = 0; continue; }
        uint64_t f = ((uint64_t)hist[i] * M) / T;
        if (f == 0) f = 1;
        freqs[i] = (uint32_t)f;
        used += f;
    }

    /* Adjust to hit M exactly.  Iteratively trim/give at the largest entry;
     * a single pass usually suffices, but we loop to be safe. */
    while (used != M) {
        unsigned best = 0;
        uint32_t best_f = freqs[0];
        for (unsigned i = 1; i < 65; ++i) {
            if (freqs[i] > best_f) { best_f = freqs[i]; best = i; }
        }
        if (used > M) {
            uint64_t over = used - M;
            /* Trim from `best`, but never drop to 0 if it was nonzero. */
            uint32_t keep_min = (hist[best] > 0) ? 1u : 0u;
            uint32_t max_trim = freqs[best] - keep_min;
            uint32_t trim = (over < max_trim) ? (uint32_t)over : max_trim;
            freqs[best] -= trim;
            used -= trim;
            c3d_assert(trim > 0);   /* forward progress */
        } else {
            uint64_t under = M - used;
            freqs[best] += (uint32_t)under;
            used += under;
        }
    }

    /* Sanity: every originally-nonzero symbol still has freq ≥ 1; sum == M. */
    uint64_t check = 0;
    for (unsigned i = 0; i < 65; ++i) {
        if (hist[i] > 0) c3d_assert(freqs[i] >= 1u);
        check += freqs[i];
    }
    c3d_assert(check == M);
}

/* Serialise: writes denom_shift, n_nonzero, then per-symbol (sym, LEB128 freq).
 * Returns bytes written.  out_cap must be large enough (worst case: 2 + 65*(1+10) = 717 B). */
static size_t c3d_freqs_serialise(uint32_t denom_shift, const uint32_t freqs[65],
                                  uint8_t *out, size_t out_cap)
{
    c3d_assert(out_cap >= 2);
    unsigned n_nonzero = 0;
    for (unsigned i = 0; i < 65; ++i) if (freqs[i] > 0) n_nonzero++;
    c3d_assert(n_nonzero >= 1 && n_nonzero <= 65);

    size_t w = 0;
    out[w++] = (uint8_t)denom_shift;
    out[w++] = (uint8_t)n_nonzero;
    for (unsigned i = 0; i < 65; ++i) {
        if (freqs[i] == 0) continue;
        c3d_assert(w < out_cap);
        out[w++] = (uint8_t)i;
        w += c3d_leb128_encode(freqs[i], out + w, out_cap - w);
    }
    return w;
}

/* Parse the reverse.  Writes denom_shift, freqs[65] (zero-filled first).
 * Returns bytes consumed.  Panics on sum != M or malformed input. */
static size_t c3d_freqs_parse(const uint8_t *in, size_t in_len,
                              uint32_t *denom_shift, uint32_t freqs[65])
{
    c3d_assert(in_len >= 2);
    uint32_t ds = in[0];
    unsigned n_nonzero = in[1];
    c3d_assert(ds >= 1 && ds <= 14);
    c3d_assert(n_nonzero >= 1 && n_nonzero <= 65);

    size_t r = 2;
    memset(freqs, 0, 65 * sizeof(uint32_t));

    uint64_t sum = 0;
    int last_sym = -1;
    for (unsigned k = 0; k < n_nonzero; ++k) {
        c3d_assert(r < in_len);
        uint8_t sym = in[r++];
        c3d_assert(sym < 65);
        c3d_assert((int)sym > last_sym);   /* symbols must be ascending */
        last_sym = (int)sym;

        uint64_t f = 0;
        r += c3d_leb128_decode(in + r, in_len - r, &f);
        c3d_assert(f >= 1 && f <= (1u << ds));
        freqs[sym] = (uint32_t)f;
        sum += f;
    }
    c3d_assert(sum == (uint64_t)(1u << ds));
    *denom_shift = ds;
    return r;
}

/* ========================================================================= *
 *  §E  CDF 9/7 lifting DWT (1D and separable 3D)                            *
 * ========================================================================= *
 *
 * Lifting cascade per JPEG 2000 Part 1, Annex H (informative):
 *     1. d[i] += α (s[i-1] + s[i+1])    (predict 1)
 *     2. s[i] += β (d[i-1] + d[i])      (update  1)
 *     3. d[i] += γ (s[i-1] + s[i+1])    (predict 2)
 *     4. s[i] += δ (d[i-1] + d[i])      (update  2)
 *     5. s *= 1/K,  d *= K              (scaling)
 *
 * Whole-sample symmetric boundary extension at both ends:
 *     x[-1]       = x[1]
 *     x[N]        = x[N-2]
 *
 * After the 1D lift, we deinterleave: s in x[0..N/2), d in x[N/2..N).
 * Inverse 1D: reinterleave, then run the cascade in reverse.
 *
 * 3D: apply 1D along X, Y, Z (in that order), then recurse on the LLL octant.
 * 5 levels on 256³ → LLL_5 at [0:8, 0:8, 0:8].
 */

#define C3D_CDF97_ALPHA (-1.586134342059924f)
#define C3D_CDF97_BETA  (-0.052980118572961f)
#define C3D_CDF97_GAMMA ( 0.882911075530934f)
#define C3D_CDF97_DELTA ( 0.443506852043971f)
#define C3D_CDF97_K     ( 1.230174104914001f)
#define C3D_CDF97_INV_K ( 0.812893066115961f)

/* In-place 1D lift on interleaved samples x[0..N), N even. */
static void c3d_cdf97_lift_fwd(float *x, size_t N) {
    c3d_assert(N >= 4 && (N & 1) == 0);

    /* Predict 1: odd += α (even_L + even_R).  WSS at right: x[N] = x[N-2]. */
    for (size_t i = 1; i + 1 < N; i += 2)
        x[i] += C3D_CDF97_ALPHA * (x[i-1] + x[i+1]);
    x[N-1] += 2.0f * C3D_CDF97_ALPHA * x[N-2];

    /* Update 1: even += β (odd_L + odd_R).  WSS at left: x[-1] = x[1]. */
    x[0] += 2.0f * C3D_CDF97_BETA * x[1];
    for (size_t i = 2; i < N; i += 2)
        x[i] += C3D_CDF97_BETA * (x[i-1] + x[i+1]);

    /* Predict 2: γ. */
    for (size_t i = 1; i + 1 < N; i += 2)
        x[i] += C3D_CDF97_GAMMA * (x[i-1] + x[i+1]);
    x[N-1] += 2.0f * C3D_CDF97_GAMMA * x[N-2];

    /* Update 2: δ. */
    x[0] += 2.0f * C3D_CDF97_DELTA * x[1];
    for (size_t i = 2; i < N; i += 2)
        x[i] += C3D_CDF97_DELTA * (x[i-1] + x[i+1]);

    /* Scale: even (s) *= 1/K, odd (d) *= K. */
    for (size_t i = 0; i < N; i += 2) x[i]   *= C3D_CDF97_INV_K;
    for (size_t i = 1; i < N; i += 2) x[i]   *= C3D_CDF97_K;
}

static void c3d_cdf97_lift_inv(float *x, size_t N) {
    c3d_assert(N >= 4 && (N & 1) == 0);

    /* Undo scaling. */
    for (size_t i = 0; i < N; i += 2) x[i] *= C3D_CDF97_K;
    for (size_t i = 1; i < N; i += 2) x[i] *= C3D_CDF97_INV_K;

    /* Undo update 2: even -= δ (odd_L + odd_R). */
    x[0] -= 2.0f * C3D_CDF97_DELTA * x[1];
    for (size_t i = 2; i < N; i += 2)
        x[i] -= C3D_CDF97_DELTA * (x[i-1] + x[i+1]);

    /* Undo predict 2. */
    for (size_t i = 1; i + 1 < N; i += 2)
        x[i] -= C3D_CDF97_GAMMA * (x[i-1] + x[i+1]);
    x[N-1] -= 2.0f * C3D_CDF97_GAMMA * x[N-2];

    /* Undo update 1. */
    x[0] -= 2.0f * C3D_CDF97_BETA * x[1];
    for (size_t i = 2; i < N; i += 2)
        x[i] -= C3D_CDF97_BETA * (x[i-1] + x[i+1]);

    /* Undo predict 1. */
    for (size_t i = 1; i + 1 < N; i += 2)
        x[i] -= C3D_CDF97_ALPHA * (x[i-1] + x[i+1]);
    x[N-1] -= 2.0f * C3D_CDF97_ALPHA * x[N-2];
}

/* Deinterleave x[0..N) → [evens | odds] using aux[0..N) as scratch.
 * NEON: vld2q_f32 is a hardware stride-2 deinterleaving load; two 4-wide
 * stores emit the split halves.  ~2-3× speed vs the scalar two-pass loop. */
static void c3d_deinterleave(float *x, size_t N, float *aux) {
    size_t half = N / 2;
#ifdef C3D_HAVE_NEON
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        float32x4x2_t p = vld2q_f32(x + i);
        vst1q_f32(aux + i / 2,        p.val[0]);
        vst1q_f32(aux + half + i / 2, p.val[1]);
    }
    for (; i < N; i += 2) {
        aux[i / 2]        = x[i];
        aux[half + i / 2] = x[i + 1];
    }
#elif defined(C3D_HAVE_AVX2)
    /* AVX2 has no hardware stride-2 deinterleaving load (unlike NEON vld2q).
     * Shuffle + permute2f128 chains are possible but the scalar pair-copy
     * below vectorises cleanly under -O3 via gathers / simple lane swaps, so
     * the extra code is not worth it for a rarely-called helper.  Left here
     * as a hook if profiling shows this path is hot on amd64. */
    for (size_t i = 0; i < half; ++i) aux[i]        = x[2 * i];
    for (size_t i = 0; i < half; ++i) aux[half + i] = x[2 * i + 1];
#else
    for (size_t i = 0; i < half; ++i) aux[i]        = x[2 * i];
    for (size_t i = 0; i < half; ++i) aux[half + i] = x[2 * i + 1];
#endif
    memcpy(x, aux, N * sizeof(float));
}
/* Interleave [evens | odds] back into x[0..N).  Mirror of the above. */
static void c3d_interleave(float *x, size_t N, float *aux) {
    size_t half = N / 2;
#ifdef C3D_HAVE_NEON
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        float32x4x2_t p;
        p.val[0] = vld1q_f32(x + i / 2);
        p.val[1] = vld1q_f32(x + half + i / 2);
        vst2q_f32(aux + i, p);
    }
    for (; i < N; i += 2) {
        aux[i]     = x[i / 2];
        aux[i + 1] = x[half + i / 2];
    }
#else
    for (size_t i = 0; i < half; ++i) aux[2 * i]     = x[i];
    for (size_t i = 0; i < half; ++i) aux[2 * i + 1] = x[half + i];
#endif
    memcpy(x, aux, N * sizeof(float));
}

/* 1D DWT: lift + deinterleave.  aux must be N floats of scratch. */
static void c3d_dwt_1d_fwd(float *x, size_t N, float *aux) {
    c3d_cdf97_lift_fwd(x, N);
    c3d_deinterleave(x, N, aux);
}
static void c3d_dwt_1d_inv(float *x, size_t N, float *aux) {
    c3d_interleave(x, N, aux);
    c3d_cdf97_lift_inv(x, N);
}

/* 4-column-parallel 1D lift + deinterleave, used for Y/Z axis passes.
 *
 * `x` holds four columns interleaved: x[i*4 + c] is column c at index i.
 * Each `#pragma GCC ivdep` / unrolled `for (c)` loop is 4 parallel FMAs that
 * the compiler trivially vectorises to one 128-bit NEON op.  No intrinsics. */
/* Tile width for the Y/Z passes.  8 columns at a time fits two 128-bit NEON
 * FMAs per step without reloading the lane-pair; the compiler autovectorises
 * both halves cleanly.  Keep this a compile-time constant so loops can unroll. */
#define C3D_TILE_X 8

/* Copy 8 contiguous floats (32 bytes = TILE_X × f32).  Used for every Y/Z
 * tile gather/scatter inside the 3D DWT; replacing the glibc memcpy call
 * with a direct 2×NEON-load+store saves the __memcpy_chk trampoline and
 * makes the hot inner memory ops inlineable.  Compiler trivially fuses
 * the pair when the data is aligned. */
static inline void c3d_copy8(float *restrict dst, const float *restrict src) {
#ifdef C3D_HAVE_NEON
    float32x4_t a = vld1q_f32(src);
    float32x4_t b = vld1q_f32(src + 4);
    vst1q_f32(dst,     a);
    vst1q_f32(dst + 4, b);
#else
    memcpy(dst, src, 8 * sizeof(float));
#endif
}

static void c3d_cdf97_lift_fwd_x4(float *restrict x, size_t N) {
    c3d_assert(N >= 4 && (N & 1) == 0);
    /* Predict 1. */
    for (size_t i = 1; i + 1 < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] += C3D_CDF97_ALPHA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[(N-1)*C3D_TILE_X + c] += 2.0f * C3D_CDF97_ALPHA * x[(N-2)*C3D_TILE_X + c];
    /* Update 1. */
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[0 + c] += 2.0f * C3D_CDF97_BETA * x[1*C3D_TILE_X + c];
    for (size_t i = 2; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] += C3D_CDF97_BETA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    /* Predict 2. */
    for (size_t i = 1; i + 1 < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] += C3D_CDF97_GAMMA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[(N-1)*C3D_TILE_X + c] += 2.0f * C3D_CDF97_GAMMA * x[(N-2)*C3D_TILE_X + c];
    /* Update 2. */
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[0 + c] += 2.0f * C3D_CDF97_DELTA * x[1*C3D_TILE_X + c];
    for (size_t i = 2; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] += C3D_CDF97_DELTA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    /* Scale. */
    for (size_t i = 0; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) x[i*C3D_TILE_X + c] *= C3D_CDF97_INV_K;
    for (size_t i = 1; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) x[i*C3D_TILE_X + c] *= C3D_CDF97_K;
}

static void c3d_cdf97_lift_inv_x4(float *restrict x, size_t N) {
    c3d_assert(N >= 4 && (N & 1) == 0);
    /* Undo scale. */
    for (size_t i = 0; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) x[i*C3D_TILE_X + c] *= C3D_CDF97_K;
    for (size_t i = 1; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) x[i*C3D_TILE_X + c] *= C3D_CDF97_INV_K;
    /* Undo update 2. */
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[0 + c] -= 2.0f * C3D_CDF97_DELTA * x[1*C3D_TILE_X + c];
    for (size_t i = 2; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] -= C3D_CDF97_DELTA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    /* Undo predict 2. */
    for (size_t i = 1; i + 1 < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] -= C3D_CDF97_GAMMA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[(N-1)*C3D_TILE_X + c] -= 2.0f * C3D_CDF97_GAMMA * x[(N-2)*C3D_TILE_X + c];
    /* Undo update 1. */
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[0 + c] -= 2.0f * C3D_CDF97_BETA * x[1*C3D_TILE_X + c];
    for (size_t i = 2; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] -= C3D_CDF97_BETA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    /* Undo predict 1. */
    for (size_t i = 1; i + 1 < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] -= C3D_CDF97_ALPHA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[(N-1)*C3D_TILE_X + c] -= 2.0f * C3D_CDF97_ALPHA * x[(N-2)*C3D_TILE_X + c];
}

/* Deinterleave TILE_X interleaved columns in-place: per column, evens go to
 * first half, odds to second half.  aux must be N*TILE_X floats of scratch. */
static void c3d_deinterleave_x4(float *restrict x, size_t N, float *restrict aux) {
    size_t half = N / 2;
    for (size_t i = 0; i < half; ++i)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) aux[i*C3D_TILE_X + c]        = x[(2*i)*C3D_TILE_X + c];
    for (size_t i = 0; i < half; ++i)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) aux[(half+i)*C3D_TILE_X + c] = x[(2*i+1)*C3D_TILE_X + c];
    memcpy(x, aux, N * C3D_TILE_X * sizeof(float));
}
static void c3d_interleave_x4(float *restrict x, size_t N, float *restrict aux) {
    size_t half = N / 2;
    for (size_t i = 0; i < half; ++i)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) aux[(2*i)*C3D_TILE_X + c]   = x[i*C3D_TILE_X + c];
    for (size_t i = 0; i < half; ++i)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) aux[(2*i+1)*C3D_TILE_X + c] = x[(half+i)*C3D_TILE_X + c];
    memcpy(x, aux, N * C3D_TILE_X * sizeof(float));
}

static void c3d_dwt_1d_fwd_x4(float *restrict x, size_t N, float *restrict aux) {
    c3d_cdf97_lift_fwd_x4(x, N);
    c3d_deinterleave_x4(x, N, aux);
}
static void c3d_dwt_1d_inv_x4(float *restrict x, size_t N, float *restrict aux) {
    c3d_interleave_x4(x, N, aux);
    c3d_cdf97_lift_inv_x4(x, N);
}

/* 3D single-level forward on the [0:side, 0:side, 0:side] sub-cube of a 256³
 * volume.  scratch must be at least 8 * C3D_CHUNK_SIDE floats. */
#define C3D_STRIDE_Y ((size_t)C3D_CHUNK_SIDE)
#define C3D_STRIDE_Z ((size_t)C3D_CHUNK_SIDE * C3D_CHUNK_SIDE)

/* For Y/Z tiled passes we use TILE_X contiguous X columns at a time.  Scratch
 * layout: tile[N*TILE_X] + aux[N*TILE_X] = 2 * TILE_X * side floats. */
#define C3D_Y_TILE  C3D_TILE_X
#define C3D_Z_TILE  C3D_TILE_X

static void c3d_dwt3_fwd_level(float *restrict buf, size_t side, float *scratch) {
    (void)scratch;  /* per-thread buffers supersede the shared scratch. */
    buf = __builtin_assume_aligned(buf, C3D_ALIGN);
    /* side ∈ {256,128,64,32,16}: power-of-2 ≥ 16, ≤ 256. */
    c3d_invariant(side >= 16u && side <= 256u);
    c3d_invariant((side & (side - 1u)) == 0u);

    /* X pass — row stride 1, contiguous.  Outer z loop is parallelised;
     * each thread gets its own aux (512 floats = 2 KB on stack). */
    #pragma omp parallel
    {
        float aux[C3D_CHUNK_SIDE];
        #pragma omp for schedule(static)
        for (size_t z = 0; z < side; ++z) {
            for (size_t y = 0; y < side; ++y) {
                float *row = &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y];
                c3d_dwt_1d_fwd(row, side, aux);
            }
        }
    }
    /* Y pass — 4 adjacent X-columns at a time (cache-line-sized load/store). */
    c3d_assert((side & 3u) == 0);
    #pragma omp parallel
    {
        float tile[C3D_TILE_X * C3D_CHUNK_SIDE];
        float aux [C3D_TILE_X * C3D_CHUNK_SIDE];
        #pragma omp for schedule(static)
        for (size_t z = 0; z < side; ++z) {
            for (size_t xb = 0; xb < side; xb += C3D_Y_TILE) {
                for (size_t y = 0; y < side; ++y)
                    c3d_copy8(&tile[y * C3D_TILE_X],
                              &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb]);
                c3d_dwt_1d_fwd_x4(tile, side, aux);
                for (size_t y = 0; y < side; ++y)
                    c3d_copy8(&buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb],
                              &tile[y * C3D_TILE_X]);
            }
        }
    }
    /* Z pass — same tiling, parallelise over outer y. */
    #pragma omp parallel
    {
        float tile[C3D_TILE_X * C3D_CHUNK_SIDE];
        float aux [C3D_TILE_X * C3D_CHUNK_SIDE];
        #pragma omp for schedule(static)
        for (size_t y = 0; y < side; ++y) {
            for (size_t xb = 0; xb < side; xb += C3D_Z_TILE) {
                for (size_t z = 0; z < side; ++z)
                    c3d_copy8(&tile[z * C3D_TILE_X],
                              &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb]);
                c3d_dwt_1d_fwd_x4(tile, side, aux);
                for (size_t z = 0; z < side; ++z)
                    c3d_copy8(&buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb],
                              &tile[z * C3D_TILE_X]);
            }
        }
    }
}

static void c3d_dwt3_inv_level(float *restrict buf, size_t side, float *scratch) {
    (void)scratch;  /* per-thread buffers supersede the shared scratch. */
    buf = __builtin_assume_aligned(buf, C3D_ALIGN);
    c3d_invariant(side >= 16u && side <= 256u);
    c3d_invariant((side & (side - 1u)) == 0u);

    c3d_assert((side & 3u) == 0);
    /* Inverse order: Z, Y, X — each pass parallelised over its outer loop. */
    #pragma omp parallel
    {
        float tile[C3D_TILE_X * C3D_CHUNK_SIDE];
        float aux [C3D_TILE_X * C3D_CHUNK_SIDE];
        #pragma omp for schedule(static)
        for (size_t y = 0; y < side; ++y) {
            for (size_t xb = 0; xb < side; xb += C3D_Z_TILE) {
                for (size_t z = 0; z < side; ++z)
                    c3d_copy8(&tile[z * C3D_TILE_X],
                              &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb]);
                c3d_dwt_1d_inv_x4(tile, side, aux);
                for (size_t z = 0; z < side; ++z)
                    c3d_copy8(&buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb],
                              &tile[z * C3D_TILE_X]);
            }
        }
    }
    #pragma omp parallel
    {
        float tile[C3D_TILE_X * C3D_CHUNK_SIDE];
        float aux [C3D_TILE_X * C3D_CHUNK_SIDE];
        #pragma omp for schedule(static)
        for (size_t z = 0; z < side; ++z) {
            for (size_t xb = 0; xb < side; xb += C3D_Y_TILE) {
                for (size_t y = 0; y < side; ++y)
                    c3d_copy8(&tile[y * C3D_TILE_X],
                              &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb]);
                c3d_dwt_1d_inv_x4(tile, side, aux);
                for (size_t y = 0; y < side; ++y)
                    c3d_copy8(&buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb],
                              &tile[y * C3D_TILE_X]);
            }
        }
    }
    #pragma omp parallel
    {
        float aux[C3D_CHUNK_SIDE];
        #pragma omp for schedule(static)
        for (size_t z = 0; z < side; ++z) {
            for (size_t y = 0; y < side; ++y) {
                float *row = &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y];
                c3d_dwt_1d_inv(row, side, aux);
            }
        }
    }
}

/* Full 5-level forward DWT on a 256³ f32 buffer.
 * scratch must be ≥ 8 * C3D_CHUNK_SIDE floats (2 KiB). */
static void c3d_dwt3_fwd(float *restrict buf, float *scratch) {
    size_t side = C3D_CHUNK_SIDE;
    for (unsigned lvl = 0; lvl < C3D_N_DWT_LEVELS; ++lvl) {
        c3d_dwt3_fwd_level(buf, side, scratch);
        side /= 2;
    }
}

/* Inverse `n_synth_levels` levels, 0 ≤ n ≤ 5.
 *   n=0 → no inverse (output is LLL_5 at [0:8, 0:8, 0:8]).
 *   n=k → synthesise levels 5, 4, ..., 6-k; output is LLL_{5-k} at [0:(8<<k), ...].
 *   n=5 → full inverse, output at [0:256, 0:256, 0:256]. */
static void c3d_dwt3_inv_levels(float *restrict buf, unsigned n_synth_levels, float *scratch) {
    c3d_assert(n_synth_levels <= C3D_N_DWT_LEVELS);
    for (unsigned i = 0; i < n_synth_levels; ++i) {
        size_t active_side = (size_t)16u << i;   /* 16, 32, 64, 128, 256 */
        c3d_dwt3_inv_level(buf, active_side, scratch);
    }
}

/* ========================================================================= *
 *  §F  Quantizer, symbol mapping (zigzag + escape), and subband info        *
 * ========================================================================= */

/* Forward decl — defined in §F.  Used by the dz lookup below. */
static unsigned c3d_kind_h_count(unsigned kind);

/* Dead-zone widening, per subband kind.  Wavelet coefficients have a
 * "spike at 0 + Laplacian tail" shape that a wider dead zone absorbs
 * efficiently.  HF subbands have heavier tails and are dominated by
 * ringing/noise → benefit from more aggressive zeroing; LF carries the
 * structural signal → standard 0.50 dead zone.  Per-kind lookup picked
 * via per-ratio × per-h_count sweep on scroll CT.  No format change —
 * decoder computes dz_half from the same subband index. */
static inline float c3d_dz_ratio_for_kind(unsigned h_count) {
    /* h_count = number of HF axes in the subband (0..3).
     * Calibrated by per-kind sweep: LLL_5 carries the structural DC residual
     * (preserve everything), details all benefit equally from 0.55. */
    /* Per-kind tuning was tested (LLL=0.50 + per-h table, aggressive HF
     * variants); gains are <0.02 dB avg vs global 0.55.  The R-D allocator
     * already redistributes bits via per-subband step coarsening, so per-
     * kind dz is a near-redundant degree of freedom on this data.  Kept
     * the per-kind plumbing for future format-changing work; current
     * table just sets every kind to the global 0.55. */
    static const float ratios[4] = { 0.55f, 0.55f, 0.55f, 0.55f };
    return ratios[h_count <= 3 ? h_count : 3];
}

/* Dead-zone uniform quantizer.  dz_half is computed by the caller from
 * the subband's kind via c3d_dz_ratio_for_kind() × step.
 *   |c| < dz_half     → 0
 *   |c| ≥ dz_half     → sign(c) * (floor((|c| - dz_half) / step) + 1) */
C3D_CONST
static inline int32_t c3d_quant(float c, float step, float dz_half) {
    float ac = (c < 0.0f) ? -c : c;
    if (ac < dz_half) return 0;
    int32_t q = (int32_t)((ac - dz_half) / step) + 1;
    return (c < 0.0f) ? -q : q;
}

/* Mid-tread dequantizer.  Bin k (k≥1) spans [dz_half + (k-1)·step,
 * dz_half + k·step]; reconstruction = dz_half + (k - 1 + α)·step
 * where α∈[0.25,0.50] picks the Laplacian-optimal position in the bin. */
C3D_CONST
static inline float c3d_dequant(int32_t q, float step, float dz_half, float alpha) {
    if (q == 0) return 0.0f;
    float aq  = (float)((q < 0) ? -q : q);
    float mag = dz_half + (aq - 1.0f + alpha) * step;
    return (q < 0) ? -mag : mag;
}

/* Look up dz_half for a subband from its kind. */
C3D_CONST
static inline float c3d_dz_half_for_kind(unsigned kind, float step) {
    return c3d_dz_ratio_for_kind(c3d_kind_h_count(kind)) * step;
}

/* Standard 32-bit zigzag: signed ↔ unsigned bijection.
 *   0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, ...  */
C3D_CONST
static inline uint32_t c3d_zigzag32(int32_t v) {
    return ((uint32_t)v << 1) ^ (uint32_t)(v >> 31);
}
C3D_CONST
static inline int32_t c3d_unzigzag32(uint32_t z) {
    return (int32_t)((z >> 1) ^ (uint32_t)-(int32_t)(z & 1u));
}

/* --- Sign-predictive symbol mapping ---
 *
 * Encodes (magnitude, sign_prediction_error) instead of zigzag:
 *   sym 0:     |qv| = 0  (zero coeff, no sign)
 *   sym 2k-1:  |qv| = k, sign prediction CORRECT  (k = 1..31)
 *   sym 2k:    |qv| = k, sign prediction WRONG     (k = 1..31)
 *   sym 63:    escape (|qv| ≥ 32), sign CORRECT
 *   sym 64:    escape (|qv| ≥ 32), sign WRONG
 *
 * Prediction = sign of previous non-zero coeff in raster order (default +).
 * At ~65-75 % accuracy on scroll CT, "correct" symbols get higher probability
 * → lower rANS bits → ~5-15 % tighter at r ≥ 25.  Same 65-symbol alphabet
 * and rANS infrastructure; only the mapping changes.
 *
 * Escape payload: LEB128 of |qv| (unsigned magnitude). */
#define C3D_SYM_ESCAPE_LO 63u  /* escape + sign correct */
#define C3D_SYM_ESCAPE_HI 64u  /* escape + sign wrong   */
#define C3D_N_SYMBOLS      65u

#define C3D_SYM_IS_ESCAPE(s)  ((s) >= C3D_SYM_ESCAPE_LO)

static inline uint8_t c3d_quant_to_symbol(int32_t q, uint32_t *escape_mag_out,
                                          bool *sign_pred)
{
    if (q == 0) { *escape_mag_out = 0; return 0; }
    uint32_t mag = (uint32_t)(q < 0 ? -q : q);
    bool actual_pos = (q > 0);
    bool correct = (actual_pos == *sign_pred);
    *sign_pred = actual_pos;
    if (mag < 32u) {
        *escape_mag_out = 0;
        return (uint8_t)(1u + (mag - 1u) * 2u + (correct ? 0u : 1u));
    }
    *escape_mag_out = mag;
    return correct ? (uint8_t)C3D_SYM_ESCAPE_LO : (uint8_t)C3D_SYM_ESCAPE_HI;
}

static inline int32_t c3d_symbol_to_quant(uint8_t sym, uint32_t escape_mag,
                                          bool *sign_pred)
{
    if (sym == 0) return 0;
    bool correct; int32_t mag;
    if (C3D_SYM_IS_ESCAPE(sym)) {
        correct = (sym == C3D_SYM_ESCAPE_LO);
        mag = (int32_t)escape_mag;
    } else {
        uint32_t s1 = (uint32_t)(sym - 1u);
        correct = ((s1 & 1u) == 0u);
        mag = (int32_t)(s1 / 2u + 1u);
    }
    bool actual_pos = correct ? *sign_pred : !*sign_pred;
    *sign_pred = actual_pos;
    return actual_pos ? mag : -mag;
}

/* Helper: extract |qv| from a sign-predictive symbol (for α fit / wsum).
 * Does NOT update sign_pred — read-only convenience. */
C3D_CONST
static inline uint32_t c3d_sym_magnitude(uint8_t sym) {
    if (sym == 0) return 0;
    if (C3D_SYM_IS_ESCAPE(sym)) return 32u; /* approximate; real mag in escape stream */
    return ((uint32_t)(sym - 1u)) / 2u + 1u;
}

/* --- Subband descriptor --------------------------------------------------- *
 *
 * Canonical indexing (PLAN §2.3):
 *   index 0            = LLL_5   at (0,0,0), side 8
 *   index 1..7         = level-5 details, side 8, in 7 non-LLL octants of [0..16]³
 *   index 8..14        = level-4 details, side 16, in [0..32]³
 *   index 15..21       = level-3 details, side 32, in [0..64]³
 *   index 22..28       = level-2 details, side 64, in [0..128]³
 *   index 29..35       = level-1 details, side 128, in [0..256]³
 *
 * Within each level the 7-detail ordering is
 *     HHH, HHL, HLH, LHH, HLL, LHL, LLH
 * i.e. kind indices 1..7 respectively.  The letter order is ZYX (first = Z).
 * Octant offset: each H adds +side on its axis.
 *
 *   kind 1 HHH → (+s, +s, +s)
 *   kind 2 HHL → (+s, +s,  0 )
 *   kind 3 HLH → (+s,  0 , +s)
 *   kind 4 LHH → ( 0 , +s, +s)
 *   kind 5 HLL → (+s,  0 ,  0 )
 *   kind 6 LHL → ( 0 , +s,  0 )
 *   kind 7 LLH → ( 0 ,  0 , +s)
 */

typedef struct {
    unsigned level;    /* 1..5                                               */
    unsigned kind;     /* 0 = LLL (only for LLL_5); else 1..7 per table above */
    uint32_t side;     /* coefficient count per axis                         */
    uint32_t z0, y0, x0; /* origin inside the 256³ coefficient buffer        */
} c3d_subband_info;

static void c3d_subband_info_of(unsigned idx, c3d_subband_info *info) {
    c3d_assert(idx < C3D_N_SUBBANDS);
    if (idx == 0) {
        info->level = 5;
        info->kind  = 0;
        info->side  = 8;
        info->z0 = info->y0 = info->x0 = 0;
        return;
    }
    unsigned i = idx - 1;           /* 0..34 over detail subbands      */
    unsigned level_from_deep = i / 7;  /* 0 = level 5, 4 = level 1        */
    unsigned kind_minus_1    = i % 7;
    info->level = 5u - level_from_deep;
    info->kind  = kind_minus_1 + 1u;
    info->side  = 8u << level_from_deep;   /* 8, 16, 32, 64, 128          */

    uint32_t s = info->side;
    uint32_t z_hi = 0, y_hi = 0, x_hi = 0;
    switch (info->kind) {
    case 1: z_hi = 1; y_hi = 1; x_hi = 1; break; /* HHH */
    case 2: z_hi = 1; y_hi = 1;            break; /* HHL */
    case 3: z_hi = 1;            x_hi = 1; break; /* HLH */
    case 4:            y_hi = 1; x_hi = 1; break; /* LHH */
    case 5: z_hi = 1;                      break; /* HLL */
    case 6:            y_hi = 1;           break; /* LHL */
    case 7:                       x_hi = 1; break; /* LLH */
    default: c3d_panic(__FILE__, __LINE__, "bad subband kind");
    }
    info->z0 = z_hi * s;
    info->y0 = y_hi * s;
    info->x0 = x_hi * s;
}

#ifdef C3D_BUILD_REF
/* Extract / scatter a subband region into a packed flat array.  Only used
 * by c3d_test for round-trip verification; the live codec path operates on
 * the 3D buffer in place via subband_info coordinates. */
static size_t c3d_subband_extract(const float *buf,
                                  const c3d_subband_info *sb,
                                  float *out_flat)
{
    size_t count = 0;
    for (uint32_t z = sb->z0; z < sb->z0 + sb->side; ++z)
    for (uint32_t y = sb->y0; y < sb->y0 + sb->side; ++y)
    for (uint32_t x = sb->x0; x < sb->x0 + sb->side; ++x) {
        out_flat[count++] = buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x];
    }
    return count;
}
static void c3d_subband_scatter(float *buf,
                                const c3d_subband_info *sb,
                                const float *in_flat)
{
    size_t count = 0;
    for (uint32_t z = sb->z0; z < sb->z0 + sb->side; ++z)
    for (uint32_t y = sb->y0; y < sb->y0 + sb->side; ++y)
    for (uint32_t x = sb->x0; x < sb->x0 + sb->side; ++x) {
        buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x] = in_flat[count++];
    }
}
#endif

/* ========================================================================= *
 *  §G/§H/§I  Chunk encoder + decoder (SELF mode only for now)               *
 * ========================================================================= *
 *
 * Chunk layout (PLAN §2.2):
 *    0     chunk header (40 B)
 *   40     qmul[36]            (144 B)
 *  184     subband_offset[36]  (144 B)
 *  328     lod_offset[6]       (24 B)
 *  352     alpha_per_subband[36] (36 B)  — per-chunk Laplacian dequant α
 *  388     entropy payload (variable, resolution-first)
 *
 * Per-subband bitstream (PLAN §3.4):
 *    0   u16 freq_table_size
 *    2   freq_table (denom_shift u8, n_nonzero u8, n_nonzero × {u8 sym, LEB128 freq})
 *    ?   u32 n_symbols
 *    ?   u32 rans_block_size
 *    ?   rans_header (32 B) + rans_renorm (variable)
 *    ?   escape_stream (variable)
 */

#define C3D_CHUNK_FIXED_SIZE 388u
#define C3D_CHUNK_ALPHA_OFFSET 352u
#define C3D_CHUNK_FLAGS_OFFSET 16u

/* uint8 <-> α float in [0.40, 0.50] — clamp + linear map.  Q1 writes one
 * byte per subband; decoder reads it back for the Laplacian-optimal
 * dequant offset (c3d_default_alpha is the fallback when no per-subband
 * α is available, e.g. for the empty-subband sentinel or ctx overrides).
 *
 * Range was [0.25, 0.50] — narrowed to [0.40, 0.50] after empirical
 * sweep showed scroll CT data is NOT pure Laplacian.  The Laplacian fit
 * formula α* = 1/u - 1/(exp(u)-1) drives toward 0 for HF subbands at
 * high ratios, biasing reconstruction toward dz_half (bin start).  But
 * the actual NON-ZERO coefficients in HF subbands cluster nearer the
 * bin midpoint (real edges, not noise) — so the fit underestimates α.
 * Floor at 0.40 forces reconstructions closer to bin midpoint; gives
 * +0.02 dB at r=25-100 with no regression at low ratios.  Format
 * change — old-chunk α bytes decode to a different physical value. */
static inline uint8_t c3d_alpha_to_u8(float a) {
    if (a < 0.40f) a = 0.40f;
    if (a > 0.50f) a = 0.50f;
    float v = (a - 0.40f) * (255.0f / 0.10f) + 0.5f;
    return (uint8_t)v;
}
static inline float c3d_alpha_from_u8(uint8_t v) {
    return 0.40f + (float)v * (0.10f / 255.0f);
}
#define C3D_Q_MIN            (1.0f / 4096.0f) /* 2^-12 (was 2^-6 — wider range
                                                  needed to reach big target
                                                  budgets under perceptual
                                                  weighting which compresses
                                                  HF subbands aggressively). */
#define C3D_Q_MAX            4096.0f          /* 2^12 */

/* --- Perceptual per-subband quantizer weights --------------------------- *
 *
 * CDF 9/7 synthesis gains squared per axis (||synthesis basis||² from JPEG
 * 2000 Part 1 Annex F.2).  An error ε in a subband with L_count low-pass
 * axes and H_count high-pass axes contributes to reconstruction MSE by
 *    ε² · (G_L²)^L_count · (G_H²)^H_count
 * For R-D-optimal bit allocation, subband step ∝ 1/sqrt(weight), so deep
 * low-frequency bands get fine quantization and high-frequency bands get
 * coarse.  Normalised so geomean across 36 subbands == 1.0, preserving the
 * chunk_scalar q range semantics.  Computed lazily on first use. */

#define C3D_CDF97_GAIN_L_SQ 2.08f
#define C3D_CDF97_GAIN_H_SQ 0.48f

/* Bins in the fine-histogram cache used by the R-D allocator (§I1 / §Q3).
 * 1024 bins × 36 subbands × 4 B = 144 KiB per encoder.  Enough resolution
 * that trial-step dead-zone boundaries land on stable bin indices across
 * bisection rounds. */
#define C3D_FINE_BINS 1024u

static float c3d_subband_baseline_table[C3D_N_SUBBANDS];
static bool  c3d_subband_baseline_init = false;

C3D_CONST
static unsigned c3d_kind_h_count(unsigned kind) {
    switch (kind) {
    case 0: return 0;                       /* LLL_5               */
    case 1: return 3;                       /* HHH                 */
    case 2: case 3: case 4: return 2;       /* HHL, HLH, LHH       */
    case 5: case 6: case 7: return 1;       /* HLL, LHL, LLH       */
    default: return 0;
    }
}

/* Fill baselines[36] using 1/w^softness weighting.  w = product of axis
 * CDF97 synthesis gains² per subband; strict R-D-optimal is softness=0.5 but
 * that collapses step dynamic range so much the rate-control loop saturates
 * at q_min.  Default 0.25 keeps control responsive; adaptive path varies
 * softness mildly with target_ratio (§G).  Normalised to geomean 1.0 so the
 * chunk_scalar q semantics are preserved across softness values. */
static void c3d_fill_subband_baselines(float softness, float baselines[C3D_N_SUBBANDS]) {
    float b[C3D_N_SUBBANDS];
    double log_sum = 0.0;
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_subband_info sb; c3d_subband_info_of(i, &sb);
        unsigned h = c3d_kind_h_count(sb.kind);
        float log_w = (float)(3u * (sb.level - 1u) + (3u - h)) * logf(C3D_CDF97_GAIN_L_SQ)
                    + (float)h * logf(C3D_CDF97_GAIN_H_SQ);
        b[i] = expf(-softness * log_w);
        log_sum += -(double)softness * (double)log_w;
    }
    float scale = expf(-(float)(log_sum / (double)C3D_N_SUBBANDS));
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i)
        baselines[i] = b[i] * scale;
}

static void c3d_compute_subband_baselines(void) {
    c3d_fill_subband_baselines(0.25f, c3d_subband_baseline_table);
    c3d_subband_baseline_init = true;
}

static inline float c3d_subband_baseline(unsigned i) {
    if (!c3d_subband_baseline_init) c3d_compute_subband_baselines();
    return c3d_subband_baseline_table[i];
}

/* Default rANS denom_shift per subband.  LLL_5 needs M=16384 (only 512 coeffs
 * need a tight histogram); every other subband uses M=4096.  Tried pushing
 * finest HF bands to M=1024: LUT shrinks from 4 KiB to 1 KiB but each symbol
 * consumes fewer fractional bits of state → renorm reads ~12 % more bytes,
 * net decode regressed 15-20 % at fine q.  The LUT was already L1-resident. */
C3D_CONST
static inline uint32_t c3d_default_denom_shift(unsigned i) {
    return (i == 0) ? 14u : 12u;
}

/* Per-subband default Laplacian-optimal dead-zone offset (dequant α).  The
 * reconstruction is (|q| - 0.5 + α) * step; smaller α biases toward zero
 * (matches heavier-tailed distributions in pure-HF subbands), larger α
 * toward bin midpoint (better for the near-uniform LLL_5 DC residual).
 * 0.375 was the previous global default — retained for mid-frequency bands.
 * Ctx overrides still win; this only changes the fall-through default. */
static inline float c3d_default_alpha(unsigned s) {
    if (s == 0) return 0.45f;  /* LLL_5: broad distribution, closer to midpoint */
    c3d_subband_info sb; c3d_subband_info_of(s, &sb);
    unsigned h = c3d_kind_h_count(sb.kind);
    /* h=1 mixed-LF: 0.40;  h=2 mixed: 0.375;  h=3 pure HF (HHH): 0.33 */
    switch (h) {
    case 1: return 0.40f;
    case 2: return 0.375f;
    case 3: return 0.33f;
    default: return 0.375f;
    }
}

/* Per-subband softness (1 / w^softness perceptual weighting).
 *
 * Original calibration found s=0.50 optimal (R-D theory exponent for
 * Gaussian sources).  After the R-D allocator + dz=0.55 + dead-zone
 * widening landed, a re-sweep (s ∈ {0.45..0.80} × r ∈ {5..200}) shows
 * the optimum has shifted to s≈0.60 across every ratio: +0.02 to +0.04
 * dB vs s=0.50, with ~0.5-2% ratio overshoot the bisection absorbs.
 * Beyond 0.60 PSNR plateaus then regresses (rate-control saturates).
 * Env override kept for future sweeps. */
static float c3d_adaptive_softness(float target_ratio) {
    (void)target_ratio;
    const char *env = getenv("C3D_SOFTNESS");
    if (env) {
        float v = (float)atof(env);
        if (v >= 0.05f && v <= 0.9f) return v;
    }
    return 0.60f;
}

/* Number of subbands required to decode each LOD (prefix of canonical order). */
static const unsigned c3d_n_subbands_for_lod[C3D_N_LODS] = {
    C3D_N_SUBBANDS,  /* LOD 0: all 36                          */
    29,              /* LOD 1: excludes level-1 details (29..35) */
    22,              /* LOD 2: excludes levels 1 and 2           */
    15,              /* LOD 3                                    */
    8,               /* LOD 4: LLL_5 + level-5 details           */
    1,               /* LOD 5: LLL_5 only                        */
};

/* -- Reusable encoder / decoder scratch contexts -------------------------- *
 *
 * c3d_encoder owns ~115 MiB:  coeff_buf 64M + sub_symbols 2M + sub_escapes 2M
 *                              + rans_scratch 8M + small DWT scratch.
 * c3d_decoder owns ~80 MiB:   coeff_buf 64M + sub_symbols 2M + small.
 *
 * Both are exposed to callers as opaque handles via c3d_{encoder,decoder}_new.
 * The stateless c3d_chunk_encode/decode functions are now thin wrappers that
 * allocate a temporary context per call. */

/* Max OpenMP threads we reserve per-thread scratch for.  Anything above
 * this falls back to thread-0 scratch (still correct, just not parallel). */
#define C3D_OMP_MAX_THREADS 32

struct c3d_encoder {
    float   *coeff_buf;
    uint8_t *sub_symbols;
    uint8_t *sub_escapes;
    uint8_t *rans_scratch;
    /* §S11 — per-thread scratch pools for the parallel subband-encode.
     * Each pointer aliases the above at index 0; [1..N-1] lazy-allocated
     * on first parallel encode.  thread_out_scratch holds one subband's
     * output bytes before they're concatenated into the chunk buffer. */
    uint8_t *thread_sub_symbols[C3D_OMP_MAX_THREADS];
    uint8_t *thread_sub_escapes[C3D_OMP_MAX_THREADS];
    uint8_t *thread_rans_scratch[C3D_OMP_MAX_THREADS];
    uint8_t *thread_out_scratch [C3D_OMP_MAX_THREADS];
    float    dwt_scratch[2 * C3D_TILE_X * C3D_CHUNK_SIDE];
    /* Dynamic per-subband baselines (adaptive perceptual softness).  Populated
     * by c3d_encoder_chunk_encode from target_ratio; unused by encode_at_q
     * (which falls back to the cached default-softness table). */
    float    dyn_baselines[C3D_N_SUBBANDS];
    bool     has_dyn_baselines;
    /* Per-subband max |coeff| after prepare_chunk.  Lets the estimator skip
     * the full quant scan on subbands that are definitely all-zero at the
     * trial step — matches the empty-subband fast path in the real emit. */
    float    max_abs_per_subband[C3D_N_SUBBANDS];
    bool     has_max_abs;
    /* Raw post-DWT max|coeff| (= coeff_scale).  Absorbed into per-subband
     * step at emit/estimate time so the normalise-to-[-1,1] scan can be
     * skipped — see c3d_prepare_chunk. */
    float    coeff_scale;
    /* Warm start for rate-control bisection.  Populated from the previous
     * call at the same target_ratio; successive chunks in a shard usually
     * converge to a very similar q, so this cuts bisection from ~8 iters
     * to ~3-4. */
    float    last_q;
    float    last_target_ratio;
    /* Per-subband fine histogram + running prefix sum over |c|.  Built once
     * post-DWT in c3d_prepare_chunk; lets the R-D allocator evaluate rate at
     * any trial step with an O(65) range sum instead of an O(N) quant scan.
     * fine_prefix[s][i] = count of coeffs in subband s with |c|/bin_width < i,
     * where bin_width = max_abs_per_subband[s] / C3D_FINE_BINS.
     * The +1 in [C3D_FINE_BINS+1] is the standard "one past last" slot so
     * range_sum(lo, hi) = fine_prefix[hi] - fine_prefix[lo] never indexes
     * out-of-bounds. */
    uint32_t fine_prefix[C3D_N_SUBBANDS][C3D_FINE_BINS + 1];
    bool     has_fine_hist;
    /* Per-subband step chosen by the R-D allocator (§Q3).  When
     * has_allocator_steps is set, c3d_emit_entropy_at_q uses these values
     * directly instead of step = q*baseline*coeff_scale. */
    float    allocator_steps[C3D_N_SUBBANDS];
    bool     has_allocator_steps;

    /* §T14 — learned R-D slope for rate-control shortcut.  Tracks
     * d(log bytes)/d(log q) from consecutive estimator samples (EMA,
     * alpha=0.3).  Used to pick the next q via Newton-in-log-space
     * instead of geometric bisection; collapses typical bisection
     * from 3-8 iters to 1-3 since the rate curve is close to a
     * straight line in log-log.  Seeded to -1.5 on first encode. */
    float    log_rd_slope;
    bool     has_log_rd_slope;

    /* Lazy-allocated 16 MiB u8 scratch used by the _masked encode variants
     * as the filled-input buffer (0-voxels replaced by global-min non-zero
     * before the regular encode path). */
    uint8_t *in_scratch;
};

struct c3d_decoder {
    float   *coeff_buf;
    uint8_t *sub_symbols;
    float    dwt_scratch[2 * C3D_TILE_X * C3D_CHUNK_SIDE];
    /* §S11 — per-thread sub_symbols scratch for the parallel subband
     * decode.  Index 0 aliases `sub_symbols` (lazy-allocated above);
     * indices 1..N-1 are allocated on first parallel decode and reused
     * across chunks.  Arena-style to avoid malloc/free on the hot path. */
    uint8_t *thread_sub_symbols[C3D_OMP_MAX_THREADS];
    /* rANS table cache for stable EXTERNAL ctx.  cached_tables is lazily
     * allocated (2.4 MiB total, 36 × ~65 KiB) on first EXTERNAL-ctx decode;
     * reused across chunks as long as the ctx id matches. */
    c3d_rans_tables *cached_tables;
    uint8_t          cached_ctx_id[16];
    bool             cached_valid;

};

c3d_encoder *c3d_encoder_new(void) {
    c3d_encoder *e = malloc(sizeof *e);
    c3d_assert(e);
    e->coeff_buf    = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK * sizeof(float));
    e->sub_symbols  = malloc((size_t)128 * 128 * 128);
    e->sub_escapes  = malloc((size_t)128 * 128 * 128 / 4 + 1024);
    e->rans_scratch = malloc((size_t)128 * 128 * 128 * 2 + 1024);
    c3d_assert(e->coeff_buf && e->sub_symbols && e->sub_escapes && e->rans_scratch);
    memset(e->thread_sub_symbols, 0, sizeof e->thread_sub_symbols);
    memset(e->thread_sub_escapes, 0, sizeof e->thread_sub_escapes);
    memset(e->thread_rans_scratch,0, sizeof e->thread_rans_scratch);
    memset(e->thread_out_scratch, 0, sizeof e->thread_out_scratch);
    e->thread_sub_symbols [0] = e->sub_symbols;
    e->thread_sub_escapes [0] = e->sub_escapes;
    e->thread_rans_scratch[0] = e->rans_scratch;
    /* thread_out_scratch[0] stays NULL; thread 0 writes directly to `out` when
     * it's the only thread (single-thread path bypasses the scratch copy). */
    e->has_dyn_baselines = false;
    e->has_max_abs = false;
    e->last_q = 0.0f;
    e->last_target_ratio = 0.0f;
    e->has_fine_hist = false;
    e->has_allocator_steps = false;
    e->log_rd_slope = 0.0f;
    e->has_log_rd_slope = false;
    e->in_scratch = NULL;   /* lazy-allocated on first _masked call */
    return e;
}
void c3d_encoder_free(c3d_encoder *e) {
    if (!e) return;
    free(e->coeff_buf);   free(e->sub_symbols);
    free(e->sub_escapes); free(e->rans_scratch);
    /* thread_* slot 0 aliased the three above; slots 1..N-1 are independently
     * allocated.  thread_out_scratch[0] was NULL. */
    for (unsigned i = 1; i < C3D_OMP_MAX_THREADS; ++i) {
        free(e->thread_sub_symbols [i]);
        free(e->thread_sub_escapes [i]);
        free(e->thread_rans_scratch[i]);
    }
    for (unsigned i = 0; i < C3D_OMP_MAX_THREADS; ++i)
        free(e->thread_out_scratch[i]);
    free(e->in_scratch);
    free(e);
}

c3d_decoder *c3d_decoder_new(void) {
    c3d_decoder *d = malloc(sizeof *d);
    c3d_assert(d);
    d->coeff_buf   = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK * sizeof(float));
    d->sub_symbols = malloc((size_t)128 * 128 * 128);
    c3d_assert(d->coeff_buf && d->sub_symbols);
    memset(d->thread_sub_symbols, 0, sizeof d->thread_sub_symbols);
    d->thread_sub_symbols[0] = d->sub_symbols;
    d->cached_tables = NULL;
    memset(d->cached_ctx_id, 0, 16);
    d->cached_valid = false;
    return d;
}
void c3d_decoder_free(c3d_decoder *d) {
    if (!d) return;
    free(d->coeff_buf); free(d->sub_symbols); free(d->cached_tables);
    /* Index 0 aliases sub_symbols (already freed).  Free the rest. */
    for (unsigned i = 1; i < C3D_OMP_MAX_THREADS; ++i)
        free(d->thread_sub_symbols[i]);
    free(d);
}

/* -- Stage 1: ingest + DWT + compute coeff_scale, normalise. -------------- *
 * Writes the 40 B chunk header and zero-fills table regions to byte 352.
 * Returns true if the chunk is nonempty (needs entropy payload); false if
 * uniform-after-centering (just emit the 352 B header with all-zero tables). */
static bool c3d_prepare_chunk(const uint8_t *in, uint8_t *out,
                              c3d_encoder *s,
                              float *out_dc_offset, float *out_coeff_scale)
{
    /* Header skeleton. */
    memcpy(out + 0, "C3DC", 4);
    c3d_write_u16_le(out + 4, 1u);
    out[6] = 0; out[7] = 0;  /* reserved, reserved */
    /* dc_offset, coeff_scale filled later. */
    memset(out + 16, 0, 8 + 16);  /* reserved2 + reserved3 */
    /* Zero-fill tables (will be overwritten). */
    memset(out + 40, 0, C3D_CHUNK_FIXED_SIZE - 40);

    /* Ingest: u8 → f32 − 128 − dc_offset.  Two passes over `in` (integer
     * accumulator on the first, no coeff_buf write) + one pass that writes
     * coeff_buf exactly once.  Saves one 64 MiB coeff_buf round-trip vs the
     * naive (write f32-128 then subtract dc) ordering.
     *
     * Uniform-chunk fast path: scan tracks min/max alongside sum.  If
     * min == max, every voxel is the same and the DWT of the centred
     * buffer is exactly zero — skip DWT (~80 ms saved per uniform chunk).
     * Critical for masked scroll shards where 75-85 % of chunks are
     * either all-air or all-material. */
    uint64_t u8_sum = 0;
    uint8_t  u8_min = 255, u8_max = 0;
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) {
        uint8_t v = in[i];
        u8_sum += v;
        if (v < u8_min) u8_min = v;
        if (v > u8_max) u8_max = v;
    }
    float dc_offset;
    if (u8_min == u8_max) {
        dc_offset = (float)u8_min - 128.0f;
        c3d_write_f32_le(out + 8, dc_offset);
        c3d_write_f32_le(out + 12, 1.0f);
        *out_dc_offset = dc_offset;
        *out_coeff_scale = 1.0f;
        return false;   /* empty entropy: 388 B header reconstructs uniform u8 */
    }
    dc_offset = (float)u8_sum / (float)C3D_VOXELS_PER_CHUNK - 128.0f;
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) {
        s->coeff_buf[i] = (float)in[i] - 128.0f - dc_offset;
    }

    /* Forward 3D DWT in place. */
    c3d_dwt3_fwd(s->coeff_buf, s->dwt_scratch);

    /* Single fused pass: per-subband max |coeff| + overall max.  Covers all
     * 36 subbands (= every coefficient in the 256³ buffer exactly once) so
     * the global max is just the max over the per-subband table — no extra
     * 64 MiB scan.  Saves ~12 ms/chunk vs separate loops. */
    float max_abs = 0.0f;
    /* 36 independent per-subband scans.  Dynamic scheduling balances the
     * wide range of subband sizes (8³=512 vox to 128³=2 M vox). */
    #pragma omp parallel for reduction(max:max_abs) schedule(dynamic,1)
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_subband_info sb; c3d_subband_info_of(i, &sb);
        float mx = 0.0f;
        for (uint32_t z = sb.z0; z < sb.z0 + sb.side; ++z)
        for (uint32_t y = sb.y0; y < sb.y0 + sb.side; ++y)
        for (uint32_t x = sb.x0; x < sb.x0 + sb.side; ++x) {
            float a = fabsf(s->coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x]);
            if (a > mx) mx = a;
        }
        s->max_abs_per_subband[i] = mx;
        if (mx > max_abs) max_abs = mx;
    }

    c3d_write_f32_le(out + 8, dc_offset);
    *out_dc_offset = dc_offset;

    if (max_abs == 0.0f) {
        c3d_write_f32_le(out + 12, 1.0f);
        *out_coeff_scale = 1.0f;
        return false;   /* empty path: return 352 B chunk with all-zero tables */
    }

    /* Raw-coefficient format (post-DC, post-DWT, un-normalised).  Per-subband
     * step values absorb coeff_scale at emit time (step = q*baseline*coeff_scale)
     * so quant sees matching units; the decoder dequantizes directly into the
     * raw range and skips the old post-IDWT *coeff_scale multiply.  Skipping
     * the normalise-to-[-1,1] scan saves ~12 ms/chunk.  coeff_scale is still
     * written to the header for inspection but is no longer used on decode —
     * preserved so c3d_inspect / downstream tools see the pre-encode magnitude. */
    float coeff_scale = max_abs;
    c3d_write_f32_le(out + 12, coeff_scale);
    *out_coeff_scale = coeff_scale;

    s->coeff_scale = coeff_scale;
    s->has_max_abs = true;
    s->has_fine_hist = false;   /* lazy-built on demand by the R-D allocator */
    return true;
}

/* Lazily build per-subband fine histograms + prefix sums over |c|.  Called
 * from the R-D allocator (§Q3) on encoders where rate control needs O(1)
 * trial-step rate estimation.  Idempotent once has_fine_hist is set.
 * Each bin i counts coefficients with |c| in [i·w, (i+1)·w) where
 * w = max_abs_per_subband / C3D_FINE_BINS.  Prefix[i] = total count of
 * bins 0..i-1 so range_sum is one subtract. */
static void c3d_build_fine_hist(c3d_encoder *s) {
    if (s->has_fine_hist) return;
    c3d_assert(s->has_max_abs);
    /* 36 independent per-subband histogram builds.  Each writes a
     * disjoint fine_prefix[sidx] row.  Work is uneven so dynamic-1. */
    #pragma omp parallel for schedule(dynamic,1)
    for (unsigned sidx = 0; sidx < C3D_N_SUBBANDS; ++sidx) {
        uint32_t *pref = s->fine_prefix[sidx];
        memset(pref, 0, (C3D_FINE_BINS + 1) * sizeof(uint32_t));
        float mx = s->max_abs_per_subband[sidx];
        if (mx <= 0.0f) continue;
        float inv_w = (float)C3D_FINE_BINS / mx;
        c3d_subband_info sb; c3d_subband_info_of(sidx, &sb);
        /* Two-phase per row: NEON computes 4-wide fabs + bin index;
         * scalar pass increments the histogram (scatter dependency). */
        uint32_t bin_row[128];
        for (uint32_t z = sb.z0; z < sb.z0 + sb.side; ++z)
        for (uint32_t y = sb.y0; y < sb.y0 + sb.side; ++y) {
            const float *row = &s->coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + sb.x0];
#ifdef C3D_HAVE_NEON
            float32x4_t vinv = vdupq_n_f32(inv_w);
            uint32x4_t vclip = vdupq_n_u32(C3D_FINE_BINS - 1u);
            uint32_t xx = 0;
            for (; xx + 4 <= sb.side; xx += 4) {
                float32x4_t a = vabsq_f32(vld1q_f32(row + xx));
                uint32x4_t b = vcvtq_u32_f32(vmulq_f32(a, vinv));
                b = vminq_u32(b, vclip);
                vst1q_u32(bin_row + xx, b);
            }
            for (; xx < sb.side; ++xx) {
                float ac = fabsf(row[xx]);
                uint32_t b = (uint32_t)(ac * inv_w);
                if (b >= C3D_FINE_BINS) b = C3D_FINE_BINS - 1u;
                bin_row[xx] = b;
            }
#else
            for (uint32_t xx = 0; xx < sb.side; ++xx) {
                float ac = fabsf(row[xx]);
                uint32_t b = (uint32_t)(ac * inv_w);
                if (b >= C3D_FINE_BINS) b = C3D_FINE_BINS - 1u;
                bin_row[xx] = b;
            }
#endif
            for (uint32_t xh = 0; xh < sb.side; ++xh) pref[bin_row[xh] + 1]++;
        }
        for (unsigned i = 1; i <= C3D_FINE_BINS; ++i) pref[i] += pref[i - 1];
    }
    s->has_fine_hist = true;
}

/* -- Stage 2: per-subband encode (quantize → symbols + escapes → freq table
 *             → rANS → pack).  Writes bytes to `out`.  Returns bytes written.
 * If `external_freqs` is non-NULL, use those frequencies (EXTERNAL mode):
 * all chunk symbols must have freq ≥ 1 in the provided table, otherwise rANS
 * encoding panics.  freq_table_size is written as 0 in this mode.
 * §T13: dz_ratio is looked up by caller (ctx override or kind default). */
static size_t c3d_encode_one_subband(
    const float *restrict coeff_buf, const c3d_subband_info *sb,
    float step, float dz_ratio, uint32_t denom_shift,
    const uint32_t *external_freqs,
    uint8_t *restrict sub_symbols, uint8_t *restrict sub_escapes,
    uint8_t *restrict rans_scratch, size_t rans_scratch_size,
    uint8_t *restrict out, size_t out_cap,
    float max_abs, float *out_alpha)
{
    /* Subband sides are always one of {8,16,32,64,128}, power-of-2, and
     * ≥ 8 (LLL_5).  Telling the compiler lets it narrow loop bounds, pick
     * aligned loads, and avoid the (rare) "side == 0" corner at codegen. */
    c3d_invariant(sb->side >= 8u && sb->side <= 128u);
    c3d_invariant((sb->side & (sb->side - 1u)) == 0u);
    size_t n = (size_t)sb->side * sb->side * sb->side;

    float dz_half = dz_ratio * step;

    /* All-zero fast path (before the quant scan): if max|c| < dz_half,
     * every coefficient will quantize to 0 and we can emit the 2-byte
     * sentinel without scanning the subband at all. */
    if (max_abs < dz_half) {
        c3d_assert(out_cap >= 2);
        c3d_write_u16_le(out, 0xFFFFu);
        (void)denom_shift; (void)external_freqs;
        (void)sub_symbols; (void)sub_escapes;
        (void)rans_scratch; (void)rans_scratch_size;
        return 2;
    }

    /* Pass 1: quantize + symbol + escape + histogram.
     * §T1c spatial sign prediction: prev_sign_zy[(y-y0)*side + (x-x0)]
     * tracks the last non-zero sign seen at this (y, x) column as we
     * iterate z.  Each new (z, y, x) predicts from (z-1, y, x) — wavelet
     * coefficients in CT data correlate strongly across slice (z) so
     * sign prediction accuracy improves vs the old raster-prev scheme. */
    uint32_t hist[65] = {0};
    uint32_t hist_ctx[2][65] = {{0}, {0}};  /* [0]=after-zero, [1]=after-nonzero */
    size_t escape_pos = 0;
    size_t idx = 0;
    /* §T12: only memset the portion we'll actually use.  Side ranges from 8
     * to 128, so this zeroes 64 B to 16 KiB instead of always 16 KiB — a 4-256×
     * reduction in zero-fill work on smaller subbands.  Must NOT use `= {0}`
     * on the VLA-sized subarray because that zeros the whole 128*128 block. */
    bool prev_sign_zy[128 * 128];
    memset(prev_sign_zy, 0, (size_t)sb->side * sb->side);
    bool lane_ctx[8] = {false,false,false,false,false,false,false,false};
    /* §T12: hoist sb-> struct reads to locals so the compiler doesn't
     * reload them every inner iteration (sb is not restrict-qualified). */
    const uint32_t sb_z0 = sb->z0, sb_y0 = sb->y0, sb_x0 = sb->x0;
    const uint32_t sb_side = sb->side;
    /* §S11 two-phase quant/symbol.  Phase 1 (SIMD): vectorised float→int
     * quant over a full x-row into qv_row[].  Phase 2 (scalar): stateful
     * sign-prediction + sym mapping + histogram + escape emit.  The scalar
     * state (sp, lane_ctx, hist) is unchanged; only the float arithmetic
     * moves to 4-lane NEON fma/fabs/vcvt. */
    int32_t qv_row[128];
#ifdef C3D_HAVE_NEON
    const float inv_step = 1.0f / step;
#endif
    for (uint32_t z = sb_z0; z < sb_z0 + sb_side; ++z)
    for (uint32_t y = sb_y0; y < sb_y0 + sb_side; ++y) {
        const float *crow = &coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + sb_x0];
        /* Phase 1. */
#ifdef C3D_HAVE_NEON
        float32x4_t vdz    = vdupq_n_f32(dz_half);
        float32x4_t vinv   = vdupq_n_f32(inv_step);
        float32x4_t vzero  = vdupq_n_f32(0.0f);
        uint32_t x = 0;
        for (; x + 4 <= sb_side; x += 4) {
            float32x4_t c  = vld1q_f32(crow + x);
            float32x4_t ac = vabsq_f32(c);
            uint32x4_t below = vcltq_f32(ac, vdz);             /* zero mask */
            float32x4_t s  = vmulq_f32(vsubq_f32(ac, vdz), vinv);
            int32x4_t  qi  = vcvtq_s32_f32(s);
            qi = vaddq_s32(qi, vdupq_n_s32(1));
            uint32x4_t neg = vcltq_f32(c, vzero);
            int32x4_t  qn  = vnegq_s32(qi);
            int32x4_t  q   = vbslq_s32(neg, qn, qi);
            q = vbslq_s32(below, vdupq_n_s32(0), q);
            vst1q_s32(qv_row + x, q);
        }
        for (; x < sb_side; ++x)
            qv_row[x] = c3d_quant(crow[x], step, dz_half);
#else
        for (uint32_t x = 0; x < sb_side; ++x)
            qv_row[x] = c3d_quant(crow[x], step, dz_half);
#endif
        /* Phase 2 — scalar, keeps sp/lane_ctx/hist/escape correct. */
        for (uint32_t xp = 0; xp < sb_side; ++xp) {
            uint32_t escape_mag;
            bool *sp = &prev_sign_zy[(y - sb_y0) * sb_side + xp];
            uint8_t sym = c3d_quant_to_symbol(qv_row[xp], &escape_mag, sp);
            sub_symbols[idx] = sym;
            hist[sym]++;
            unsigned lane = idx & 7u;
            hist_ctx[lane_ctx[lane] ? 1 : 0][sym]++;
            lane_ctx[lane] = (sym != 0);
            idx++;
            if (c3d_unlikely(C3D_SYM_IS_ESCAPE(sym))) {
                escape_pos += c3d_leb128_encode(escape_mag,
                                                sub_escapes + escape_pos, 5);
            }
        }
    }
    c3d_assert(idx == n);

    /* Laplacian-α fit (Q1).  Closed form:
     *   β_hat  = step * Σ(|q|·hist[q]) / Σ(hist[q])
     *   u      = step / β_hat
     *   α_opt  = 1/u - 1/(exp(u) - 1)         (independent of bin k)
     * Clamped to [0.25, 0.50].  When escapes are rare, β_hat from the
     * direct-symbol histogram is a tight estimate (bin-0 bias is small).
     * For every-bin-0 chunks we never reach this path (all-zero sentinel). */
    {
        double wsum = 0.0;
        for (unsigned s = 1; s < C3D_SYM_ESCAPE_LO; ++s)
            wsum += (double)c3d_sym_magnitude((uint8_t)s) * hist[s];
        wsum += 32.0 * (double)(hist[C3D_SYM_ESCAPE_LO] + hist[C3D_SYM_ESCAPE_HI]);
        double beta = (double)step * wsum / (double)n;
        if (beta <= 1e-12) {
            *out_alpha = 0.25f;
        } else {
            double u = (double)step / beta;
            double denom = exp(u) - 1.0;
            double a = 1.0 / u - (denom > 1e-12 ? 1.0 / denom : 0.0);
            if (a < 0.40) a = 0.40;
            if (a > 0.50) a = 0.50;
            *out_alpha = (float)a;
        }
    }

    /* All-zero fast path: every coefficient quantizes to symbol 0.  Emit a
     * 2-byte sentinel (freq_table_size = 0xFFFF) and stop — decoder zero-fills
     * the subband.  Saves ~40 bytes/subband on sparse chunks (rANS state
     * header alone is 32 B).  Common case for HF subbands at moderate ratios. */
    if (hist[0] == n) {
        c3d_assert(out_cap >= 2);
        c3d_write_u16_le(out, 0xFFFFu);
        return 2;
    }

    /* Estimate 2-table (lane-local context) rate vs 1-table rate.
     * 2-table encodes each symbol conditioned on whether the same lane's
     * previous symbol was zero; the two histograms are more concentrated
     * than the joint, giving lower entropy — but cost 2 freq tables.
     * Only use 2-table when the entropy savings outweigh the extra table. */
    bool use_2table = false;
    uint32_t ctx_freqs[2][65];
    {
        uint32_t n_z = 0, n_nz = 0;
        for (unsigned k = 0; k < 65; ++k) { n_z += hist_ctx[0][k]; n_nz += hist_ctx[1][k]; }
        double bits_1t = 0.0;
        { double inv = 1.0 / (double)n;
          for (unsigned k = 0; k < 65; ++k) {
              if (!hist[k]) continue;
              double p = (double)hist[k] * inv;
              bits_1t += -(double)hist[k] * log2(p);
          }
        }
        double bits_2t = 0.0;
        if (n_z > 0) {
            double inv_z = 1.0 / (double)n_z;
            for (unsigned k = 0; k < 65; ++k) {
                if (!hist_ctx[0][k]) continue;
                double p = (double)hist_ctx[0][k] * inv_z;
                bits_2t += -(double)hist_ctx[0][k] * log2(p);
            }
        }
        if (n_nz > 0) {
            double inv_nz = 1.0 / (double)n_nz;
            for (unsigned k = 0; k < 65; ++k) {
                if (!hist_ctx[1][k]) continue;
                double p = (double)hist_ctx[1][k] * inv_nz;
                bits_2t += -(double)hist_ctx[1][k] * log2(p);
            }
        }
        /* Overhead of 2nd freq table ≈ 2 + 3*nnz bytes.  Plus 1 ctx_mode byte. */
        unsigned nnz_z = 0, nnz_nz = 0;
        for (unsigned k = 0; k < 65; ++k) {
            if (hist_ctx[0][k]) nnz_z++;
            if (hist_ctx[1][k]) nnz_nz++;
        }
        double overhead_2t = 1.0 + (2.0 + 3.0*(double)nnz_z) + (2.0 + 3.0*(double)nnz_nz);
        unsigned nnz_1t = 0;
        for (unsigned k = 0; k < 65; ++k) if (hist[k]) nnz_1t++;
        double overhead_1t = 1.0 + (2.0 + 3.0*(double)nnz_1t);
        double cost_1t = bits_1t / 8.0 + overhead_1t;
        double cost_2t = bits_2t / 8.0 + overhead_2t;
        use_2table = (cost_2t < cost_1t - 8.0) && (n_z > 0) && (n_nz > 0);
    }
    if (use_2table) {
        c3d_normalise_freqs(hist_ctx[0], denom_shift, ctx_freqs[0]);
        c3d_normalise_freqs(hist_ctx[1], denom_shift, ctx_freqs[1]);
    }

    /* Pick the frequency source per-subband.  When external freqs are
     * available, compare actual byte cost against the SELF path.
     * 2-table mode overrides both (external ctx doesn't support it yet). */
    uint32_t local_freqs[65];
    const uint32_t *freqs_to_use;
    bool use_external = false;
    c3d_normalise_freqs(hist, denom_shift, local_freqs);
    if (external_freqs && !use_2table) {
        bool ext_covers = true;
        for (unsigned k = 0; k < 65; ++k) {
            if (hist[k] > 0 && external_freqs[k] == 0) { ext_covers = false; break; }
        }
        if (ext_covers) {
            /* Both paths are decodable; compare actual rate. */
            double log2_M = (double)denom_shift;
            double ext_bits = 0.0, self_bits = 0.0;
            double inv_n = 1.0 / (double)n;
            for (unsigned k = 0; k < 65; ++k) {
                if (!hist[k]) continue;
                ext_bits  += (double)hist[k] * (log2_M - log2((double)external_freqs[k]));
                double p = (double)hist[k] * inv_n;
                self_bits += -(double)hist[k] * log2(p);
            }
            /* SELF pays ftable_serialised_size extra bytes; ~2 + 3*nnz. */
            unsigned nnz = 0;
            for (unsigned k = 0; k < 65; ++k) if (hist[k]) nnz++;
            double self_overhead_bytes = 2.0 + 3.0 * (double)nnz;
            double ext_bytes  = ext_bits / 8.0;
            double self_bytes = self_bits / 8.0 + self_overhead_bytes;
            use_external = ext_bytes <= self_bytes;
        }
    }
    if (use_external) {
        freqs_to_use = external_freqs;
    } else {
        freqs_to_use = local_freqs;
    }
    c3d_rans_tables tbl;
    c3d_rans_build_tables(&tbl, denom_shift, freqs_to_use, 65);

    /* Emit per-subband bitstream layout. */
    size_t w = 0;

    /* [freq_table_size u16] — 0 for EXTERNAL (table lives in ctx). */
    c3d_assert(w + 2 <= out_cap);
    size_t ftable_size_pos = w;
    w += 2;

    /* [freq_table region] layout:
     *   [ctx_mode u8] — 0 = 1 table, 1 = 2-table lane-local ctx
     *   [freq_table_0 ...]
     *   [freq_table_1 ...] (only if ctx_mode == 1)
     * External mode: ftable_bytes = 0 (no ctx_mode byte either). */
    size_t ftable_bytes;
    if (use_external) {
        ftable_bytes = 0;
    } else {
        out[w] = use_2table ? 1u : 0u;
        size_t ft_w = 1;
        if (use_2table) {
            ft_w += c3d_freqs_serialise(denom_shift, ctx_freqs[0],
                                        out + w + ft_w, out_cap - w - ft_w);
            ft_w += c3d_freqs_serialise(denom_shift, ctx_freqs[1],
                                        out + w + ft_w, out_cap - w - ft_w);
        } else {
            ft_w += c3d_freqs_serialise(denom_shift, local_freqs,
                                        out + w + ft_w, out_cap - w - ft_w);
        }
        ftable_bytes = ft_w;
    }
    c3d_assert(ftable_bytes <= 65535);
    c3d_write_u16_le(out + ftable_size_pos, (uint16_t)ftable_bytes);
    w += ftable_bytes;

    /* [n_symbols u32][rans_block_size u32 placeholder] */
    c3d_assert(w + 8 <= out_cap);
    c3d_write_u32_le(out + w, (uint32_t)n); w += 4;
    size_t rans_size_pos = w;
    w += 4;

    /* [rans_header 32 B][rans_renorm variable] */
    size_t rans_bytes;
    if (use_2table) {
        /* Build 2 rANS tables and encode with per-lane context. */
        c3d_rans_tables tbl_z, tbl_nz;
        c3d_rans_build_tables(&tbl_z,  denom_shift, ctx_freqs[0], 65);
        c3d_rans_build_tables(&tbl_nz, denom_shift, ctx_freqs[1], 65);
        /* c3d_rans_enc_x8 with lane-local context selection. */
        rans_bytes = c3d_rans_enc_x8_ctx(
            sub_symbols, n, &tbl_z, &tbl_nz,
            rans_scratch, rans_scratch_size,
            out + w, out_cap - w);
    } else {
        rans_bytes = c3d_rans_enc_x8(
            sub_symbols, n, &tbl,
            rans_scratch, rans_scratch_size,
            out + w, out_cap - w);
    }
    c3d_write_u32_le(out + rans_size_pos, (uint32_t)rans_bytes);
    w += rans_bytes;

    /* [escape_stream variable] */
    c3d_assert(w + escape_pos <= out_cap);
    memcpy(out + w, sub_escapes, escape_pos);
    w += escape_pos;

    return w;
}

/* Cheap entropy estimator: quantize + histogram + Shannon (or cross-entropy
 * vs external freqs when applicable) + escape LEB128 size, no rANS encode,
 * no freq-table normalisation, no serialise.  Used by the rate-control
 * bisection so each iteration costs ~1 quantize pass instead of a full emit.
 * Returns estimated subband byte size (double so errors aggregate cleanly).
 * Matches c3d_encode_one_subband's "use external iff external covers every
 * symbol present in this chunk" decision so the estimate tracks reality. */
static double c3d_estimate_one_subband_bytes(
    const float *restrict coeff_buf, const c3d_subband_info *sb,
    float step, float dz_ratio, uint32_t denom_shift,
    const uint32_t *external_freqs,
    float max_abs)
{
    c3d_invariant(sb->side >= 8u && sb->side <= 128u);
    c3d_invariant((sb->side & (sb->side - 1u)) == 0u);
    float dz_half = dz_ratio * step;
    /* Fast reject: if max |c| in this subband quantizes to 0, the whole band
     * is empty → matches c3d_encode_one_subband's 2-byte sentinel, and we
     * skip the O(N) quant loop entirely.  On sparse chunks (typical at
     * r≥50) this hits for 10-20 of 36 subbands per estimator iteration. */
    if (max_abs < dz_half) return 2.0;

    size_t n = (size_t)sb->side * sb->side * sb->side;
    uint32_t hist[65] = {0};
    uint32_t hist_ctx[2][65] = {{0},{0}};
    size_t escape_bytes = 0;
    bool prev_sign_zy[128 * 128];
    memset(prev_sign_zy, 0, (size_t)sb->side * sb->side);   /* §T12 */
    bool lane_ctx_est[8] = {false,false,false,false,false,false,false,false};
    size_t est_idx = 0;
    /* §S11 two-phase quant (same as c3d_encode_one_subband). */
    int32_t qv_row[128];
#ifdef C3D_HAVE_NEON
    const float inv_step = 1.0f / step;
#endif
    const uint32_t sb_side = sb->side;
    for (uint32_t z = sb->z0; z < sb->z0 + sb_side; ++z)
    for (uint32_t y = sb->y0; y < sb->y0 + sb_side; ++y) {
        const float *crow = &coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + sb->x0];
#ifdef C3D_HAVE_NEON
        float32x4_t vdz    = vdupq_n_f32(dz_half);
        float32x4_t vinv   = vdupq_n_f32(inv_step);
        float32x4_t vzero  = vdupq_n_f32(0.0f);
        uint32_t x = 0;
        for (; x + 4 <= sb_side; x += 4) {
            float32x4_t c  = vld1q_f32(crow + x);
            float32x4_t ac = vabsq_f32(c);
            uint32x4_t below = vcltq_f32(ac, vdz);
            float32x4_t s  = vmulq_f32(vsubq_f32(ac, vdz), vinv);
            int32x4_t  qi  = vcvtq_s32_f32(s);
            qi = vaddq_s32(qi, vdupq_n_s32(1));
            uint32x4_t neg = vcltq_f32(c, vzero);
            int32x4_t  qn  = vnegq_s32(qi);
            int32x4_t  q   = vbslq_s32(neg, qn, qi);
            q = vbslq_s32(below, vdupq_n_s32(0), q);
            vst1q_s32(qv_row + x, q);
        }
        for (; x < sb_side; ++x)
            qv_row[x] = c3d_quant(crow[x], step, dz_half);
#else
        for (uint32_t x = 0; x < sb_side; ++x)
            qv_row[x] = c3d_quant(crow[x], step, dz_half);
#endif
        for (uint32_t xp = 0; xp < sb_side; ++xp) {
            uint32_t escape_mag;
            bool *sp = &prev_sign_zy[(y - sb->y0) * sb_side + xp];
            uint8_t sym = c3d_quant_to_symbol(qv_row[xp], &escape_mag, sp);
            hist[sym]++;
            unsigned lane = est_idx & 7u;
            hist_ctx[lane_ctx_est[lane] ? 1 : 0][sym]++;
            lane_ctx_est[lane] = (sym != 0);
            est_idx++;
            if (c3d_unlikely(C3D_SYM_IS_ESCAPE(sym))) {
                uint32_t v = escape_mag;
                do { escape_bytes++; v >>= 7; } while (v);
            }
        }
    }

    /* All-zero subband fast path matches c3d_encode_one_subband. */
    if (hist[0] == n) return 2.0;

    /* Compute 1-table rate. */
    double inv_n = 1.0 / (double)n;
    double self_bits = 0.0;
    for (unsigned k = 0; k < 65; ++k) {
        if (!hist[k]) continue;
        double p = (double)hist[k] * inv_n;
        self_bits += -(double)hist[k] * log2(p);
    }
    unsigned nnz = 0;
    for (unsigned k = 0; k < 65; ++k) if (hist[k]) nnz++;
    double self_ftable = 1.0 + 2.0 + 3.0 * (double)nnz;  /* ctx_mode + ftable */
    double self_total = 2.0 + self_ftable + 8.0 + self_bits / 8.0 + 32.0 + (double)escape_bytes;

    /* Compute 2-table (lane-local context) rate; pick min. */
    {
        uint32_t n_z = 0, n_nz = 0;
        for (unsigned k = 0; k < 65; ++k) { n_z += hist_ctx[0][k]; n_nz += hist_ctx[1][k]; }
        if (n_z > 0 && n_nz > 0) {
            double bits_2t = 0.0;
            double inv_z = 1.0/(double)n_z, inv_nz = 1.0/(double)n_nz;
            for (unsigned k = 0; k < 65; ++k) {
                if (hist_ctx[0][k]) { double p=(double)hist_ctx[0][k]*inv_z; bits_2t += -(double)hist_ctx[0][k]*log2(p); }
                if (hist_ctx[1][k]) { double p=(double)hist_ctx[1][k]*inv_nz; bits_2t += -(double)hist_ctx[1][k]*log2(p); }
            }
            unsigned nnz_z=0, nnz_nz=0;
            for (unsigned k=0;k<65;++k) { if(hist_ctx[0][k])nnz_z++; if(hist_ctx[1][k])nnz_nz++; }
            double ft2 = 1.0 + (2.0+3.0*(double)nnz_z) + (2.0+3.0*(double)nnz_nz);
            double total_2t = 2.0 + ft2 + 8.0 + bits_2t/8.0 + 32.0 + (double)escape_bytes;
            if (total_2t < self_total) self_total = total_2t;
        }
    }

    if (external_freqs) {
        bool ext_covers = true;
        for (unsigned k = 0; k < 65; ++k) {
            if (hist[k] > 0 && external_freqs[k] == 0) { ext_covers = false; break; }
        }
        if (ext_covers) {
            double log2_M = (double)denom_shift;
            double ext_bits = 0.0;
            for (unsigned k = 0; k < 65; ++k) {
                if (!hist[k]) continue;
                ext_bits += (double)hist[k] * (log2_M - log2((double)external_freqs[k]));
            }
            double ext_total = 2.0 + 0.0 + 8.0 + ext_bits / 8.0 + 32.0 + (double)escape_bytes;
            if (ext_total < self_total) return ext_total;
        }
    }
    return self_total;
}

/* Rate-distortion estimate for a single subband at a trial step, using the
 * pre-built fine histogram (§I1).  Rate returned in bytes (Shannon + a
 * constant per-subband framing overhead).  Distortion estimated by treating
 * each fine bin's count as concentrated at the bin midpoint, then summing
 * (c_mid - dequant(quant(c_mid, step), step, α))².  Used by the R-D
 * allocator (§Q3) to pick per-subband optimal steps. */
static void c3d_rd_estimate_subband(const c3d_encoder *s, unsigned sidx,
                                    float step, float alpha,
                                    double *out_rate_bytes, double *out_dist)
{
    c3d_subband_info sb; c3d_subband_info_of(sidx, &sb);
    size_t n = (size_t)sb.side * sb.side * sb.side;
    float mx = s->max_abs_per_subband[sidx];

    /* All-zero fast path: every coef quantizes to 0 → 2 B sentinel. */
    if (mx < 0.5f * step) {
        if (out_rate_bytes) *out_rate_bytes = 2.0;
        if (out_dist) {
            /* Quantization error = c² for all voxels ≤ 0.5*step.  Use an
             * upper bound = n * (0.5*step)² / 3 (uniform on [0, 0.5*step]). */
            double d_per = (double)(0.5f * step) * (double)(0.5f * step) / 3.0;
            *out_dist = d_per * (double)n;
        }
        return;
    }

    const uint32_t *pref = s->fine_prefix[sidx];
    float w = mx / (float)C3D_FINE_BINS;
    float inv_w = (float)C3D_FINE_BINS / mx;

    /* Build trial histogram via range sums over the fine prefix.
     * NOTE: this estimator uses the *theoretical* dz=0.5*step model
     * rather than the actual c3d_dz_half_for_kind value.  Empirically,
     * matching the actual dz here regressed the allocator by ~0.05 dB
     * — the theoretical model interacts more cleanly with the
     * single-point rate calibration at mult=1. */
    uint32_t trial_hist[65];
    uint32_t idx_last;
    {
        /* Bin 0: |c| < 0.5*step (dead zone, dequants to 0). */
        float c0 = 0.5f * step;
        uint32_t i0 = (c0 <= 0.0f) ? 0u : (uint32_t)(c0 * inv_w);
        if (i0 > C3D_FINE_BINS) i0 = C3D_FINE_BINS;
        trial_hist[0] = pref[i0];
        idx_last = i0;
    }
    for (unsigned k = 1; k < 64; ++k) {
        float chi = ((float)k + 0.5f) * step;
        uint32_t ihi = (chi <= 0.0f) ? 0u : (uint32_t)(chi * inv_w);
        if (ihi > C3D_FINE_BINS) ihi = C3D_FINE_BINS;
        trial_hist[k] = pref[ihi] - pref[idx_last];
        idx_last = ihi;
    }
    /* Escape (|q| ≥ 64): everything past 63.5*step. */
    trial_hist[64] = pref[C3D_FINE_BINS] - pref[idx_last];

    /* Shannon entropy, in bits. */
    double rate_bits = 0.0;
    double inv_n = 1.0 / (double)n;
    unsigned nnz = 0;
    for (unsigned k = 0; k < 65; ++k) {
        if (!trial_hist[k]) continue;
        nnz++;
        double p = (double)trial_hist[k] * inv_n;
        rate_bits += -(double)trial_hist[k] * log2(p);
    }
    /* Per-subband framing (matches c3d_estimate_one_subband_bytes):
     *   2  freq_table_size u16
     *   ~2 + 3*nnz  ftable_bytes (SELF mode; encoder chooses this for us)
     *   8  n_symbols + rans_block_size u32s
     *   32 rANS initial state header */
    double ftable_bytes = 2.0 + 3.0 * (double)nnz;
    /* Accurate escape LEB128 cost: sum per escape fine bin, with zigzag
     * estimated from bin midpoint.  zigzag ≈ 2·|q| for positive; LEB size
     * = ceil((log2(zigzag+1))/7), floored to 1. */
    double esc_leb_bytes = 0.0;
    for (unsigned i = 0; i < C3D_FINE_BINS; ++i) {
        uint32_t h = pref[i + 1] - pref[i];
        if (!h) continue;
        float c_mid = ((float)i + 0.5f) * w;
        if (c_mid < 63.5f * step) continue;
        uint32_t k = (uint32_t)((c_mid - 0.5f * step) / step) + 1u;
        if (k < 64u) continue;
        uint32_t z = 2u * k;            /* zigzag for positive |q| */
        unsigned leb = 1;
        uint32_t v = z;
        while ((v >>= 7) != 0) leb++;
        esc_leb_bytes += (double)h * (double)leb;
    }
    double rate_bytes = rate_bits / 8.0
                      + 2.0 + ftable_bytes + 8.0 + 32.0
                      + esc_leb_bytes;

    /* Distortion: sum over fine bins of count × (c_mid - reconstruction)².
     * For bin i with count h_i at midpoint c_mid = (i+0.5)*w, reconstruction
     * depends on which trial quantizer bin c_mid lands in. */
    double dist = 0.0;
    for (unsigned i = 0; i < C3D_FINE_BINS; ++i) {
        uint32_t h = pref[i + 1] - pref[i];
        if (!h) continue;
        float c_mid = ((float)i + 0.5f) * w;
        float recon;
        if (c_mid < 0.5f * step) {
            recon = 0.0f;
        } else {
            uint32_t k = (uint32_t)((c_mid - 0.5f * step) / step) + 1u;
            if (k >= 64u) {
                /* Escape: assume exact reconstruction up to step resolution. */
                recon = ((float)k - 0.5f + 0.5f) * step;   /* alpha≈0.5 for escape */
            } else {
                recon = ((float)k - 0.5f + alpha) * step;
            }
        }
        float e = c_mid - recon;
        dist += (double)h * (double)e * (double)e;
    }

    if (out_rate_bytes) *out_rate_bytes = rate_bytes;
    if (out_dist) *out_dist = dist;
}

/* Lookup the effective "baseline step" for a subband — mirrors the emit-time
 * selection between ctx override, dynamic softness table, and the cached
 * default.  Used by the R-D allocator to centre its per-subband step grid. */
static inline float c3d_emit_baseline(const c3d_encoder *s, unsigned i)
{
    if (s->has_dyn_baselines)               return s->dyn_baselines[i];
    return c3d_subband_baseline(i);
}

/* Hybrid R-D allocator (§Q3 v2): after global-q bisection determines the
 * byte budget, re-distribute bits across subbands to minimise total
 * distortion while keeping the total byte count close.  Uses the accurate
 * quant-scan estimator (c3d_estimate_one_subband_bytes) so rate and distortion
 * are internally consistent — sidesteps the calibration gap that sank the
 * fine-histogram-based first cut. */
#define C3D_RD_NCAND 9
static void c3d_rd_allocate_hybrid(c3d_encoder *s,
                                   double target_bytes,
                                   float q_center,
                                   const double *actual_bytes_per_sb)
{
    c3d_build_fine_hist(s);

    static const float mults[C3D_RD_NCAND] = {
        0.70f, 0.78f, 0.86f, 0.93f, 1.0f, 1.07f, 1.16f, 1.27f, 1.43f
    };

    double rate[C3D_N_SUBBANDS][C3D_RD_NCAND];
    double dist[C3D_N_SUBBANDS][C3D_RD_NCAND];
    float  step_grid[C3D_N_SUBBANDS][C3D_RD_NCAND];
    /* Subbands vary 500×+ in voxel count (LL_5 = 512 vs LL_1 = 2 M).  Dynamic
     * scheduling lets finished threads pick up the tiny LL_5 tail while a
     * few threads grind through the 2-level subbands.  Dependencies: the
     * fine_prefix cache (c3d_build_fine_hist) must already be populated —
     * we called c3d_build_fine_hist() above, so the reads here are race-
     * free.  `ctx`, `s` fields other than allocator_steps are read-only.
     * rate/dist/step_grid writes are disjoint per `i`. */
    #pragma omp parallel for schedule(dynamic,1)
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_subband_info sb; c3d_subband_info_of(i, &sb);
        float base_step = q_center * c3d_emit_baseline(s, i) * s->coeff_scale;
        unsigned h = c3d_kind_h_count(sb.kind);
        double log_w = (double)(3u * (sb.level - 1u) + (3u - h)) * log((double)C3D_CDF97_GAIN_L_SQ)
                     + (double)h * log((double)C3D_CDF97_GAIN_H_SQ);
        double w_px = exp(log_w);

        /* Per-subband rate calibration: ratio of actual rANS bytes (from
         * the first emit) to the fine-histogram Shannon+overhead prediction.
         * Corrects the per-subband Shannon→rANS gap so the allocator's
         * bit trades reflect real byte costs. */
        double cal = 1.0;
        if (actual_bytes_per_sb) {
            double r_center, d_center;
            c3d_rd_estimate_subband(s, i, base_step, 0.375f,
                                    &r_center, &d_center);
            if (r_center > 2.0)
                cal = actual_bytes_per_sb[i] / r_center;
            if (cal < 0.5) cal = 0.5;
            if (cal > 2.0) cal = 2.0;
        }

        for (unsigned j = 0; j < C3D_RD_NCAND; ++j) {
            float step = base_step * mults[j];
            if (step <= 0.0f) step = 1e-9f;
            step_grid[i][j] = step;
            double r_j, d_j;
            c3d_rd_estimate_subband(s, i, step, 0.375f, &r_j, &d_j);
            rate[i][j] = r_j * cal;
            dist[i][j] = d_j * w_px;
        }
    }

    double lam_lo = 1e-8, lam_hi = 1e8;
    for (int it = 0; it < 16; ++it) {
        double lam = sqrt(lam_lo * lam_hi);
        double total_rate = 0.0;
        for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
            unsigned bj = 0;
            double bc = dist[i][0] + lam * rate[i][0];
            for (unsigned j = 1; j < C3D_RD_NCAND; ++j) {
                double c = dist[i][j] + lam * rate[i][j];
                if (c < bc) { bc = c; bj = j; }
            }
            total_rate += rate[i][bj];
        }
        if (fabs(total_rate - target_bytes) < 0.005 * target_bytes) break;
        if (total_rate > target_bytes) lam_lo = lam;
        else                           lam_hi = lam;
    }

    double lam = sqrt(lam_lo * lam_hi);
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        unsigned bj = 0;
        double bc = dist[i][0] + lam * rate[i][0];
        for (unsigned j = 1; j < C3D_RD_NCAND; ++j) {
            double c = dist[i][j] + lam * rate[i][j];
            if (c < bc) { bc = c; bj = j; }
        }
        s->allocator_steps[i] = step_grid[i][bj];
    }
    s->has_allocator_steps = true;
}

/* Original fine-histogram R-D allocator — kept for reference / future
 * calibration work.  Gated behind C3D_RD_ALLOCATOR env; see PLAN.md.
 *
 * Picks per-subband step minimising
 *     Σ d_s(step_s) + λ · Σ r_s(step_s)
 * under a fixed byte target, using the fine-histogram cache (§I1). */
#define C3D_RD_NCAND_LEGACY 10
static void c3d_rd_allocate(c3d_encoder *s,
                            double target_bytes)
{
    c3d_build_fine_hist(s);

    /* Per-subband trial step grid, centred on q_seed * baseline_s *
     * coeff_scale, log-spaced from /8 to ×8 so the allocator can reallocate
     * bits freely.  q_seed from warm-start if the caller is re-using the
     * encoder at the same target ratio; else the classic sqrt(ratio)/64
     * heuristic, which centres the grid near the true optimum. */
    float q_seed;
    if (s->last_q > 0.0f && target_bytes > 0.0) {
        q_seed = s->last_q;
    } else {
        /* ratio ≈ C3D_VOXELS / (target + fixed_header); solve for q heuristic. */
        double target_ratio = (double)C3D_VOXELS_PER_CHUNK
                            / (target_bytes + (double)C3D_CHUNK_FIXED_SIZE);
        if (target_ratio < 1.0) target_ratio = 1.0;
        q_seed = (float)(sqrt(target_ratio) / 64.0);
    }
    if (q_seed < C3D_Q_MIN) q_seed = C3D_Q_MIN;
    if (q_seed > C3D_Q_MAX) q_seed = C3D_Q_MAX;

    float grid[C3D_N_SUBBANDS][C3D_RD_NCAND_LEGACY];
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        float base = c3d_emit_baseline(s, i) * s->coeff_scale;
        float centre = q_seed * base;
        /* Factors: 1/8, 1/4, 1/2.5, 1/1.5, 1/1.1, 1.1, 1.5, 2.5, 4, 8. */
        static const float mults[C3D_RD_NCAND_LEGACY] = {
            1.0f/8.0f, 1.0f/4.0f, 1.0f/2.5f, 1.0f/1.5f, 1.0f/1.1f,
            1.1f, 1.5f, 2.5f, 4.0f, 8.0f
        };
        for (unsigned j = 0; j < C3D_RD_NCAND_LEGACY; ++j) {
            float t = centre * mults[j];
            if (t <= 0.0f) t = 1e-9f;
            grid[i][j] = t;
        }
    }

    /* Pre-compute r_ij = rate(step_ij), d_ij = dist(step_ij).  Cached so
     * λ bisection is O(subbands × ncand × 1) lookups instead of rebuilds. */
    double rate[C3D_N_SUBBANDS][C3D_RD_NCAND_LEGACY];
    double dist[C3D_N_SUBBANDS][C3D_RD_NCAND_LEGACY];
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        for (unsigned j = 0; j < C3D_RD_NCAND_LEGACY; ++j) {
            c3d_rd_estimate_subband(s, i, grid[i][j], /*alpha=*/0.375f,
                                    &rate[i][j], &dist[i][j]);
        }
    }

    /* Bisect λ in log space.  For each λ, pick per-subband j that minimises
     * d_ij + λ·r_ij; sum over subbands and compare vs target. */
    double lam_lo = 1e-8, lam_hi = 1e8;
    /* Safe upper and lower brackets: pick coarsest and finest steps. */
    double rate_coarse = 0.0, rate_fine = 0.0;
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        rate_coarse += rate[i][C3D_RD_NCAND_LEGACY - 1];   /* coarsest = last grid slot */
        rate_fine   += rate[i][0];                   /* finest = first slot */
    }
    if (rate_fine <= target_bytes) {
        /* Target is easy — even finest steps fit.  Pick finest everywhere. */
        for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i)
            s->allocator_steps[i] = grid[i][0];
        s->has_allocator_steps = true;
        if (getenv("C3D_RD_DEBUG"))
            fprintf(stderr, "RD: shortcut FINE  target=%.0f rate_fine=%.0f rate_coarse=%.0f\n",
                    target_bytes, rate_fine, rate_coarse);
        return;
    }
    if (rate_coarse >= target_bytes) {
        /* Target is hopelessly tight even at coarsest; pick coarsest. */
        for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i)
            s->allocator_steps[i] = grid[i][C3D_RD_NCAND_LEGACY - 1];
        s->has_allocator_steps = true;
        if (getenv("C3D_RD_DEBUG"))
            fprintf(stderr, "RD: shortcut COARSE target=%.0f rate_fine=%.0f rate_coarse=%.0f\n",
                    target_bytes, rate_fine, rate_coarse);
        return;
    }

    float best_j_all[C3D_N_SUBBANDS];
    for (int it = 0; it < 14; ++it) {
        double lam = sqrt(lam_lo * lam_hi);
        double total_rate = 0.0;
        for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
            unsigned best_j = 0;
            double   best_cost = dist[i][0] + lam * rate[i][0];
            for (unsigned j = 1; j < C3D_RD_NCAND_LEGACY; ++j) {
                double c = dist[i][j] + lam * rate[i][j];
                if (c < best_cost) { best_cost = c; best_j = j; }
            }
            best_j_all[i] = (float)best_j;   /* stash as float to reuse the array */
            total_rate += rate[i][best_j];
        }
        if (fabs(total_rate - target_bytes) < 0.01 * target_bytes) break;
        if (total_rate > target_bytes) lam_lo = lam;   /* need more penalty on rate */
        else                           lam_hi = lam;
    }

    /* Final picks at the last λ. */
    double lam = sqrt(lam_lo * lam_hi);
    double total_est_rate = 0.0;
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        unsigned best_j = 0;
        double   best_cost = dist[i][0] + lam * rate[i][0];
        for (unsigned j = 1; j < C3D_RD_NCAND_LEGACY; ++j) {
            double c = dist[i][j] + lam * rate[i][j];
            if (c < best_cost) { best_cost = c; best_j = j; }
        }
        s->allocator_steps[i] = grid[i][best_j];
        total_est_rate += rate[i][best_j];
    }
    s->has_allocator_steps = true;
    (void)best_j_all; (void)total_est_rate; (void)rate_fine; (void)rate_coarse;
    (void)lam;
}

/* Cheap whole-chunk estimate: sum of per-subband estimates under the same
 * baseline / denom_shift / ctx-override logic as c3d_emit_entropy_at_q. */
static double c3d_estimate_entropy_at_q(float q, const c3d_encoder *s)
{
    double total = 0.0;
    /* 36 subbands, each an independent full quant-scan over its
     * coefficient cube.  Work is uneven (LL_5=512 vox vs LL_1=2 M) so
     * dynamic scheduling with a chunk size of 1 avoids straggler threads. */
    #pragma omp parallel for reduction(+:total) schedule(dynamic,1)
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_subband_info sb;
        c3d_subband_info_of(i, &sb);
        float baseline = c3d_emit_baseline(s, i);
        float step = q * baseline * s->coeff_scale;
        uint32_t denom_shift = c3d_default_denom_shift(i);
        float dz_ratio = c3d_dz_ratio_for_kind(c3d_kind_h_count(sb.kind));
        float max_abs = s->has_max_abs ? s->max_abs_per_subband[i] : s->coeff_scale;
        total += c3d_estimate_one_subband_bytes(
            s->coeff_buf, &sb, step, dz_ratio, denom_shift,
            NULL,
            max_abs);
    }
    return total;
}

/* -- Stage 3: emit all subbands given normalised coeff_buf and chunk_scalar q.
 * If `ctx` is non-NULL, applies its overrides:
 *   - has_quantizer_baseline: step = q * baseline[s]
 *   - has_freq_tables: uses ctx's freqs, emits freq_table_size = 0 per subband
 * Writes entropy payload into out[352..], fills qmul/subband_offset/lod_offset
 * tables.  Returns total chunk size (352 + entropy bytes). */
static size_t c3d_emit_entropy_at_q(float q, c3d_encoder *s,
                                    uint8_t *out, size_t out_cap)
{
    uint8_t *qmul_ptr   = out + 40;
    uint8_t *suboff_ptr = out + 40 + 144;
    uint8_t *lodoff_ptr = out + 40 + 144 + 144;
    uint8_t *alpha_ptr  = out + C3D_CHUNK_ALPHA_OFFSET;

    const size_t entropy_cap = out_cap - C3D_CHUNK_FIXED_SIZE;
    const size_t rans_scratch_size = (size_t)128 * 128 * 128 * 2 + 1024;

    /* Fast path: single-thread.  Skip the parallel scaffolding (thread
     * scratch allocation, per-subband malloc + memcpy) and write the
     * subband bytes straight into `out` as before. */
    int n_threads_now = 1;
#ifdef _OPENMP
    n_threads_now = omp_get_max_threads();
#endif
    if (n_threads_now <= 1) {
        size_t entropy_pos = 0;
        for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
            c3d_subband_info sb;
            c3d_subband_info_of(i, &sb);
            float step;
            if (s->has_allocator_steps) {
                step = s->allocator_steps[i];
            } else {
                float baseline = c3d_emit_baseline(s, i);
                step = q * baseline * s->coeff_scale;
            }
            uint32_t denom_shift = c3d_default_denom_shift(i);
            float dz_ratio = c3d_dz_ratio_for_kind(c3d_kind_h_count(sb.kind));
            c3d_write_f32_le(qmul_ptr + 4 * i, step);
            c3d_write_u32_le(suboff_ptr + 4 * i, (uint32_t)entropy_pos);
            float max_abs = s->has_max_abs ? s->max_abs_per_subband[i] : s->coeff_scale;
            float fitted_alpha = c3d_default_alpha(i);
            size_t bytes = c3d_encode_one_subband(
                s->coeff_buf, &sb, step, dz_ratio, denom_shift,
                NULL,
                s->sub_symbols, s->sub_escapes,
                s->rans_scratch, rans_scratch_size,
                out + C3D_CHUNK_FIXED_SIZE + entropy_pos,
                entropy_cap - entropy_pos,
                max_abs, &fitted_alpha);
            alpha_ptr[i] = c3d_alpha_to_u8(fitted_alpha);
            entropy_pos += bytes;
        }
        c3d_write_u32_le(lodoff_ptr + 4 * 5, c3d_read_u32_le(suboff_ptr + 4 * 1));
        c3d_write_u32_le(lodoff_ptr + 4 * 4, c3d_read_u32_le(suboff_ptr + 4 * 8));
        c3d_write_u32_le(lodoff_ptr + 4 * 3, c3d_read_u32_le(suboff_ptr + 4 * 15));
        c3d_write_u32_le(lodoff_ptr + 4 * 2, c3d_read_u32_le(suboff_ptr + 4 * 22));
        c3d_write_u32_le(lodoff_ptr + 4 * 1, c3d_read_u32_le(suboff_ptr + 4 * 29));
        c3d_write_u32_le(lodoff_ptr + 4 * 0, (uint32_t)entropy_pos);
        return C3D_CHUNK_FIXED_SIZE + entropy_pos;
    }

    /* Per-subband output sizes + fitted α + step, collected by the parallel
     * region and consumed serially below. */
    size_t  sub_bytes[C3D_N_SUBBANDS];
    float   sub_alpha[C3D_N_SUBBANDS];
    float   sub_step [C3D_N_SUBBANDS];

    /* Precomputed upper-bound byte offsets per subband.  Each subband gets a
     * dedicated slot in the encoder's subband_scratch region so threads can
     * write directly to the final spot without a malloc/memcpy handoff.
     * Upper bound per subband: 2× its raw voxel count + 1 KiB of framing.
     * Sum across all 36 subbands = ~33 MiB — allocated once, reused. */
    size_t sub_max_offset[C3D_N_SUBBANDS + 1];
    sub_max_offset[0] = 0;
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_subband_info sb;
        c3d_subband_info_of(i, &sb);
        size_t v = (size_t)sb.side * sb.side * sb.side;
        sub_max_offset[i + 1] = sub_max_offset[i] + 2 * v + 1024;
    }
    const size_t subband_scratch_size = sub_max_offset[C3D_N_SUBBANDS];
    /* Lazy-allocate the subband scratch in thread_out_scratch[0] — we repurpose
     * this one existing pointer since the per-thread out_scratch is gone now. */
    if (!s->thread_out_scratch[0]) {
        s->thread_out_scratch[0] = malloc(subband_scratch_size);
        c3d_assert(s->thread_out_scratch[0]);
    }
    uint8_t *subband_scratch = s->thread_out_scratch[0];

    #pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        if (tid >= C3D_OMP_MAX_THREADS) tid = 0;

        /* Lazy-alloc per-thread scratch pools.  Thread 0 aliases the encoder's
         * original sub_symbols / sub_escapes / rans_scratch — no extra alloc.
         * thread_out_scratch[0] is repurposed as the subband_scratch arena
         * (allocated above), not a per-thread buffer. */
        #pragma omp critical(c3d_enc_alloc)
        {
            if (!s->thread_sub_symbols [tid]) s->thread_sub_symbols [tid] = malloc((size_t)128*128*128);
            if (!s->thread_sub_escapes [tid]) s->thread_sub_escapes [tid] = malloc((size_t)128*128*128/4 + 1024);
            if (!s->thread_rans_scratch[tid]) s->thread_rans_scratch[tid] = malloc(rans_scratch_size);
            c3d_assert(s->thread_sub_symbols [tid]);
            c3d_assert(s->thread_sub_escapes [tid]);
            c3d_assert(s->thread_rans_scratch[tid]);
        }

        uint8_t *t_syms = s->thread_sub_symbols [tid];
        uint8_t *t_esc  = s->thread_sub_escapes [tid];
        uint8_t *t_rans = s->thread_rans_scratch[tid];

        /* 36 subbands with wide size range → dynamic,1.  Each worker writes
         * its subband output directly into subband_scratch[sub_max_offset[i]..]
         * — no intermediate t_out buffer, no per-subband malloc. */
        #pragma omp for schedule(dynamic,1)
        for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
            c3d_subband_info sb;
            c3d_subband_info_of(i, &sb);
            float step;
            if (s->has_allocator_steps) {
                step = s->allocator_steps[i];
            } else {
                float baseline = c3d_emit_baseline(s, i);
                step = q * baseline * s->coeff_scale;
            }
            uint32_t denom_shift = c3d_default_denom_shift(i);
            float dz_ratio = c3d_dz_ratio_for_kind(c3d_kind_h_count(sb.kind));
            float max_abs = s->has_max_abs ? s->max_abs_per_subband[i] : s->coeff_scale;

            float fitted_alpha = c3d_default_alpha(i);
            uint8_t *slot = subband_scratch + sub_max_offset[i];
            size_t slot_cap = sub_max_offset[i + 1] - sub_max_offset[i];
            size_t bytes = c3d_encode_one_subband(
                s->coeff_buf, &sb, step, dz_ratio, denom_shift,
                NULL,
                t_syms, t_esc, t_rans, rans_scratch_size,
                slot, slot_cap,
                max_abs, &fitted_alpha);

            sub_bytes[i] = bytes;
            sub_alpha[i] = fitted_alpha;
            sub_step [i] = step;
        }
    }

    /* Compute final entropy offsets serially (data dependency), write
     * qmul/suboff/alpha metadata, then parallel-memcpy the subband bytes
     * from scratch to `out` at the tight offsets. */
    size_t entropy_pos = 0;
    size_t sub_final_off[C3D_N_SUBBANDS];
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_write_f32_le(qmul_ptr + 4 * i, sub_step[i]);
        c3d_write_u32_le(suboff_ptr + 4 * i, (uint32_t)entropy_pos);
        alpha_ptr[i] = c3d_alpha_to_u8(sub_alpha[i]);
        sub_final_off[i] = entropy_pos;
        c3d_assert(entropy_pos + sub_bytes[i] <= entropy_cap);
        entropy_pos += sub_bytes[i];
    }
    #pragma omp parallel for schedule(dynamic,1)
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        memcpy(out + C3D_CHUNK_FIXED_SIZE + sub_final_off[i],
               subband_scratch + sub_max_offset[i], sub_bytes[i]);
    }

    /* LOD offsets: the cumulative sizes at resolution boundaries.  Subband
     * indices 1, 8, 15, 22, 29 are the first detail subband of levels 5..1. */
    c3d_write_u32_le(lodoff_ptr + 4 * 5, c3d_read_u32_le(suboff_ptr + 4 * 1));
    c3d_write_u32_le(lodoff_ptr + 4 * 4, c3d_read_u32_le(suboff_ptr + 4 * 8));
    c3d_write_u32_le(lodoff_ptr + 4 * 3, c3d_read_u32_le(suboff_ptr + 4 * 15));
    c3d_write_u32_le(lodoff_ptr + 4 * 2, c3d_read_u32_le(suboff_ptr + 4 * 22));
    c3d_write_u32_le(lodoff_ptr + 4 * 1, c3d_read_u32_le(suboff_ptr + 4 * 29));
    c3d_write_u32_le(lodoff_ptr + 4 * 0, (uint32_t)entropy_pos);

    return C3D_CHUNK_FIXED_SIZE + entropy_pos;
}

/* Public: encode with an explicit q.  One pass, no rate control. */
size_t c3d_chunk_encode_at_q(const uint8_t *in, float q,
                             uint8_t *out, size_t out_cap)
{
    c3d_assert(in && out);
    c3d_check_voxel_alignment(in);
    c3d_assert(out_cap >= C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(q >= C3D_Q_MIN && q <= C3D_Q_MAX);

    c3d_encoder *e = c3d_encoder_new();
    size_t r = c3d_encoder_chunk_encode_at_q(e, in, q, out, out_cap);
    c3d_encoder_free(e);
    return r;
}

size_t c3d_encoder_chunk_encode_at_q(c3d_encoder *e, const uint8_t *in, float q,
                                     uint8_t *out, size_t out_cap)
{
    c3d_assert(e && in && out);
    c3d_check_voxel_alignment(in);
    c3d_assert(out_cap >= C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(q >= C3D_Q_MIN && q <= C3D_Q_MAX);

    /* Fixed-q path skips adaptive softness — falls back to the cached default. */
    e->has_dyn_baselines = false;

    float dc, cs;
    bool has_entropy = c3d_prepare_chunk(in, out, e, &dc, &cs);
    if (!has_entropy) return C3D_CHUNK_FIXED_SIZE;
    return c3d_emit_entropy_at_q(q, e, out, out_cap);
}

/* Forward declaration — defined in §Q2 below but called by the encoder
 * to compute the in-loop denoise alpha stored in header byte 7. */
static float c3d_denoise_strength(size_t in_len);

/* Public: rate-controlled encode targeting `target_ratio`.
 * Uses log-space bisection on q, capped at 8 iterations.  Last attempt's
 * output is committed (may not be best if didn't converge). */
size_t c3d_chunk_encode(const uint8_t *in, float target_ratio,
                        uint8_t *out, size_t out_cap)
{
    c3d_encoder *e = c3d_encoder_new();
    size_t r = c3d_encoder_chunk_encode(e, in, target_ratio, out, out_cap);
    c3d_encoder_free(e);
    return r;
}

size_t c3d_encoder_chunk_encode(c3d_encoder *e, const uint8_t *in,
                                float target_ratio,
                                uint8_t *out, size_t out_cap)
{
    c3d_assert(e && in && out);
    c3d_check_voxel_alignment(in);
    c3d_assert(out_cap >= C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(target_ratio > 1.0f);

    /* Adaptive perceptual softness: fill the encoder's per-subband dynamic
     * baselines so emit / estimate use target-ratio-aware weighting.  ctx
     * overrides take precedence at the use site, so this is harmless when
     * the caller supplies a full quantizer_baseline. */
    float softness = c3d_adaptive_softness(target_ratio);
    c3d_fill_subband_baselines(softness, e->dyn_baselines);
    e->has_dyn_baselines = true;

    float dc, cs;
    bool has_entropy = c3d_prepare_chunk(in, out, e, &dc, &cs);
    size_t total;
    if (!has_entropy) {
        return C3D_CHUNK_FIXED_SIZE;
    }

    double target_bytes_d = (double)C3D_VOXELS_PER_CHUNK / (double)target_ratio
                          - (double)C3D_CHUNK_FIXED_SIZE;
    if (target_bytes_d < 64.0) target_bytes_d = 64.0;

    /* Legacy fine-histogram R-D allocator (§Q3 v1) stays gated under
     * C3D_RD_ALLOCATOR env — its rate estimate misses by ~20 % on real
     * data.  The hybrid allocator below fires by default. */
    e->has_allocator_steps = false;
    if (getenv("C3D_RD_ALLOCATOR")) {
        c3d_rd_allocate(e, target_bytes_d);
    }

    /* Warm-start q from the previous chunk when target_ratio hasn't changed.
     * Bracket narrows to [q/4, q*4] — still wide enough to converge even if
     * chunk content varies sharply, but cuts typical iteration count in half.
     * Only used as a fallback here (R-D path above is primary). */
    float q, q_lo, q_hi;
    if (e->last_q > 0.0f && e->last_target_ratio == target_ratio) {
        q = e->last_q;
        q_lo = q * 0.25f;
        q_hi = q * 4.0f;
        if (q_lo < C3D_Q_MIN) q_lo = C3D_Q_MIN;
        if (q_hi > C3D_Q_MAX) q_hi = C3D_Q_MAX;
    } else {
        q = sqrtf(target_ratio) / 64.0f;
        if (q < C3D_Q_MIN) q = C3D_Q_MIN;
        if (q > C3D_Q_MAX) q = C3D_Q_MAX;
        q_lo = C3D_Q_MIN;
        q_hi = C3D_Q_MAX;
    }

    /* Rate-control on the cheap estimator (quantize + Shannon, no rANS) to
     * pick q, then run the true emit exactly once.  ~3-4× encode speedup vs
     * the per-iteration full-emit loop; final output is always the real
     * encode.
     *
     * §S7: when warm-started, accept the warm q if it lands within ±5% of
     * target (R-D allocator can correct the rest).  Saves 2-4 iterations on
     * sequential chunks in a shard.
     *
     * §T14: the rate curve log(bytes) vs log(q) is nearly linear for
     * DWT+rANS on CT data, with slope ~-1.5.  We track that slope as an EMA
     * over successive estimator samples and jump directly to the predicted q
     * (Newton-in-log-space) instead of geometric bisection.  The bracket
     * q_lo/q_hi still advances every iter as a safety net: when the Newton
     * step lands at the bracket edge we fall back to the geometric midpoint.
     * Typical iter counts: cold 2-3 (vs 5-8), warm 1-2 (vs 2-4). */
    bool warm = (e->last_q > 0.0f && e->last_target_ratio == target_ratio);
    double prev_log_q = 0.0, prev_log_b = 0.0;
    bool have_prev = false;
    for (int iter = 0; iter < 10 && !e->has_allocator_steps; ++iter) {
        double est_bytes = c3d_estimate_entropy_at_q(q, e);
        double err = est_bytes - target_bytes_d;
        double rel = (err < 0 ? -err : err) / target_bytes_d;
        if (rel < 0.01) break;
        if (iter == 0 && warm && rel < 0.05) break;   /* §S7 early-exit */

        /* Advance bracket from the new sample. */
        if (est_bytes > target_bytes_d) q_lo = q;
        else                            q_hi = q;

        /* Online slope update (EMA) from consecutive samples. */
        double log_q = log((double)q);
        double log_b = log(est_bytes);
        if (have_prev) {
            double dq = log_q - prev_log_q;
            if (dq > 1e-4 || dq < -1e-4) {
                double sample = (log_b - prev_log_b) / dq;
                if (sample < -4.0)  sample = -4.0;
                if (sample > -0.25) sample = -0.25;
                e->log_rd_slope = e->has_log_rd_slope
                    ? (0.7f * e->log_rd_slope + 0.3f * (float)sample)
                    : (float)sample;
                e->has_log_rd_slope = true;
            }
        }
        prev_log_q = log_q; prev_log_b = log_b; have_prev = true;

        /* Newton step in log-space using the learned slope.  Seed -1.5 when
         * no slope has been observed yet — typical value for DWT+rANS. */
        float slope = e->has_log_rd_slope ? e->log_rd_slope : -1.5f;
        double d_log_q = log(target_bytes_d / est_bytes) / (double)slope;
        /* Cap step to 4× / quarter per iter so a bad slope can't slingshot
         * us outside the valid range. */
        if (d_log_q > 1.386)  d_log_q = 1.386;   /* log 4 */
        if (d_log_q < -1.386) d_log_q = -1.386;
        float new_q = (float)((double)q * exp(d_log_q));

        /* Keep the jump strictly inside the current bracket; if it lands on
         * the edge, fall back to the geometric midpoint (old behaviour). */
        if (new_q <= q_lo * 1.001f || new_q >= q_hi * 0.999f) {
            new_q = sqrtf(q_lo * q_hi);
            if (new_q <= q_lo * 1.001f) break;
            if (new_q >= q_hi * 0.999f) break;
        }
        if (new_q < C3D_Q_MIN) new_q = C3D_Q_MIN;
        if (new_q > C3D_Q_MAX) new_q = C3D_Q_MAX;
        q = new_q;
    }
    /* R-D allocator (§Q3 v3): two-pass calibrated Lagrangian.
     * Pass 1: emit at global q → measure actual per-subband bytes.
     * Pass 2: Lagrangian with calibrated fine-histogram rate estimates
     * and synthesis-gain-weighted distortion → re-emit at per-subband
     * optimal steps.  +0.10 dB avg across all ratios, ~9% encode cost.
     * Disable via C3D_NO_RD=1 for speed-critical paths. */
    if (!e->has_allocator_steps && !getenv("C3D_NO_RD")) {
        /* Pass 1: emit at global q. */
        total = c3d_emit_entropy_at_q(q, e, out, out_cap);
        double target_entropy = (double)(total - C3D_CHUNK_FIXED_SIZE);

        /* Read actual per-subband bytes from the emitted header. */
        const uint8_t *suboff = out + 40 + 144;
        const uint8_t *lodoff = out + 40 + 144 + 144;
        uint32_t lod0 = c3d_read_u32_le(lodoff);
        double actual_sb[C3D_N_SUBBANDS];
        for (unsigned s = 0; s < C3D_N_SUBBANDS; ++s) {
            uint32_t start = c3d_read_u32_le(suboff + 4 * s);
            uint32_t end = (s + 1 < C3D_N_SUBBANDS)
                         ? c3d_read_u32_le(suboff + 4 * (s + 1))
                         : lod0;
            actual_sb[s] = (double)(end - start);
        }

        /* Pass 2: calibrated R-D optimization + re-emit. */
        c3d_rd_allocate_hybrid(e, target_entropy, q, actual_sb);

        /* §T3b: skip the second emit when the allocator's steps are within
         * ±1% of the global step everywhere — pass 1's output already
         * matches what pass 2 would produce.  Common on uniform chunks
         * where the global-q allocation is already R-D optimal. */
        bool degenerate = true;
        for (unsigned s = 0; s < C3D_N_SUBBANDS; ++s) {
            float baseline = c3d_emit_baseline(e, s);
            float global_step = q * baseline * e->coeff_scale;
            float ratio = (global_step > 0.0f)
                        ? e->allocator_steps[s] / global_step : 1.0f;
            if (ratio < 0.99f || ratio > 1.01f) { degenerate = false; break; }
        }
        if (degenerate) {
            e->has_allocator_steps = false;
            /* total + out are already from pass 1 with global steps. */
        } else {
            total = c3d_emit_entropy_at_q(q, e, out, out_cap);
        }
    } else {
        total = c3d_emit_entropy_at_q(q, e, out, out_cap);
    }

    /* In-loop denoiser (§Q2 v2): encoder picks the post-decode denoise alpha
     * and writes it to the chunk header (byte 7).  The decoder reads it
     * instead of computing from in_len.  For now the encoder uses the same
     * formula as the old decoder-side auto-alpha; the value is just stored
     * explicitly so the decoder doesn't have to guess, and future versions
     * can do per-chunk alpha optimization (self-decode sweep). */
    {
        float dn = c3d_denoise_strength(total);
        const char *dn_env = getenv("C3D_DENOISE_ALPHA");
        if (dn_env) dn = (float)atof(dn_env);
        int dv = (int)(dn * 400.0f + 0.5f);
        if (dv > 255) dv = 255;
        out[7] = (uint8_t)dv;
    }

    e->last_q = q;
    e->last_target_ratio = target_ratio;

    return total;
}

size_t c3d_chunk_encode_max_size(void) { return C3D_CHUNK_ENCODE_MAX_SIZE; }

/* Fill `out` from `in`, replacing every voxel with value 0 by the minimum
 * non-zero value found in `in`.  For scroll data preprocessed to zero all
 * voxels below threshold, the minimum non-zero value is the threshold itself,
 * so this reproduces floor-clamp semantics automatically — air and material
 * sit at the same floor, so the DWT sees no boundary step and the detail
 * bands stay concentrated near zero.  If the input contains no non-zero
 * voxels, the output is a verbatim copy (the regular encoder's all-zero
 * fast path handles that case cheaply). */
static void c3d_fill_mask_ignore(const uint8_t *restrict in,
                                 uint8_t *restrict out)
{
    uint8_t m_min = 0;
    bool any_nonzero = false;
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) {
        uint8_t v = in[i];
        if (v != 0) {
            if (!any_nonzero || v < m_min) m_min = v;
            any_nonzero = true;
        }
    }
    if (!any_nonzero) {
        memcpy(out, in, C3D_VOXELS_PER_CHUNK);
        return;
    }
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) {
        uint8_t v = in[i];
        out[i] = v == 0 ? m_min : v;
    }
}

/* Ensure e->in_scratch is allocated.  Called from the _masked entry points. */
static void c3d_encoder_ensure_in_scratch(c3d_encoder *e) {
    if (!e->in_scratch) {
        e->in_scratch = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
        c3d_assert(e->in_scratch);
    }
}

/* Public: rate-controlled encode treating 0-valued voxels as don't-care.
 *
 * The encoder replaces every 0-voxel with the minimum non-zero value from
 * the input before running the standard encode pipeline.  The output
 * bitstream is a regular v1 chunk that any c3d decoder can read; no format
 * changes, no version bump.
 *
 * Caller contract:
 *   - Mark don't-care voxels by setting them to 0 in the input.
 *   - At display / use time, re-apply your mask to re-zero those regions.
 *     Small non-zero values (typically 1–3) may appear in previously-zero
 *     regions due to wavelet ringing; either threshold them away or use
 *     your own mask to gate decoded output.
 *
 * When is the win worth it?  When a meaningful fraction of voxels carry no
 * information the caller cares about (e.g. air / void in CT scans).  On
 * Vesuvius scroll data with ~40% air, c3d_encoder_chunk_encode_masked gives
 * roughly +1 dB full-cube PSNR at matched target ratio vs compressing the
 * raw noisy air region, and slightly better material-only PSNR.  The gain
 * grows with air fraction. */
size_t c3d_encoder_chunk_encode_masked(c3d_encoder *e, const uint8_t *in,
                                       float target_ratio,
                                       uint8_t *out, size_t out_cap)
{
    c3d_assert(e && in && out);
    c3d_check_voxel_alignment(in);
    c3d_encoder_ensure_in_scratch(e);
    c3d_fill_mask_ignore(in, e->in_scratch);
    return c3d_encoder_chunk_encode(e, e->in_scratch, target_ratio,
                                    out, out_cap);
}

/* Public: _at_q variant of the masked encode — bypasses rate control, uses
 * the given q directly.  Useful for R-D sweeps and deterministic tests. */
size_t c3d_encoder_chunk_encode_masked_at_q(c3d_encoder *e, const uint8_t *in,
                                            float q,
                                            uint8_t *out, size_t out_cap)
{
    c3d_assert(e && in && out);
    c3d_check_voxel_alignment(in);
    c3d_encoder_ensure_in_scratch(e);
    c3d_fill_mask_ignore(in, e->in_scratch);
    return c3d_encoder_chunk_encode_at_q(e, e->in_scratch, q, out, out_cap);
}

/* Stateless wrappers — allocate a fresh encoder per call. */
size_t c3d_chunk_encode_masked(const uint8_t *in, float target_ratio,
                               uint8_t *out, size_t out_cap)
{
    c3d_encoder *e = c3d_encoder_new();
    size_t r = c3d_encoder_chunk_encode_masked(e, in, target_ratio, out, out_cap);
    c3d_encoder_free(e);
    return r;
}

size_t c3d_chunk_encode_masked_at_q(const uint8_t *in, float q,
                                    uint8_t *out, size_t out_cap)
{
    c3d_encoder *e = c3d_encoder_new();
    size_t r = c3d_encoder_chunk_encode_masked_at_q(e, in, q, out, out_cap);
    c3d_encoder_free(e);
    return r;
}

/* §I3.  Batched multi-chunk encode — loop the single-chunk API. */
void c3d_encoder_chunks_encode(c3d_encoder *e,
                               const uint8_t *const *inputs,
                               size_t n_chunks,
                               float target_ratio,
                               uint8_t *const *outs,
                               size_t *out_sizes)
{
    c3d_assert(e && inputs && outs && out_sizes);
    for (size_t i = 0; i < n_chunks; ++i) {
        out_sizes[i] = c3d_encoder_chunk_encode(
            e, inputs[i], target_ratio, outs[i], C3D_CHUNK_ENCODE_MAX_SIZE);
    }
}

/* ------------------------------------------------------------------------- *
 *  §Q2  Post-decode denoiser                                                *
 * ------------------------------------------------------------------------- *
 *
 * Separable 3×3×3 box blur blended with the identity at strength α:
 *   out = (1−α)·in + α · mean3( ... ) applied independently on each axis.
 * Three single-pass in-place 1D passes, no extra scratch buffer.  Cumulative
 * smoothing is roughly proportional to α.  Called from the decoder at LOD 0
 * after IDWT, before the u8 denormalise — strength chosen from encoded/input
 * byte ratio so the filter is a no-op at near-lossless and a mild blur at
 * high ratios where quantisation ringing dominates. */

/* §T10: 3-tap 3D denoiser.  Previous implementation maintained l/c/r state
 * across the inner loop iterations, which created a first-order recurrence
 * the auto-vectorizer can't break.  This rewrite uses a single scratch row
 * (X pass) and a scratch plane (Y/Z passes) so each inner loop is a pure
 * unit-stride SIMD-friendly filter with no iteration-carried state.  Clang
 * and GCC both vectorize the inner float loops to 4/8-wide NEON/AVX2. */

static inline void c3d_denoise_blur_x_row(float *restrict row, size_t N,
                                          float alpha, float *restrict scratch)
{
    if (N < 3) return;
    float one_minus_a = 1.0f - alpha;
    float third = alpha * (1.0f / 3.0f);
    memcpy(scratch, row, N * sizeof(float));
    /* Boundary: replicate endpoints (scratch[0] used as left of row[0],
     * scratch[N-1] as right of row[N-1]). */
    row[0] = one_minus_a * scratch[0]
           + third * (scratch[0] + scratch[0] + scratch[1]);
    for (size_t i = 1; i + 1 < N; ++i) {
        row[i] = one_minus_a * scratch[i]
               + third * (scratch[i - 1] + scratch[i] + scratch[i + 1]);
    }
    row[N - 1] = one_minus_a * scratch[N - 1]
               + third * (scratch[N - 2] + scratch[N - 1] + scratch[N - 1]);
}

/* Generic "blur one axis" pass.  Treats `buf` as a 3D array of side³ with
 * strides (SZ, SY, 1) in (z, y, x) order.  Blurs along `axis` in {0=Z, 1=Y, 2=X}
 * using a 3-tap [1, 1, 1] / 3 kernel weighted by alpha: out = (1-α)·c + (α/3)·(l+c+r),
 * with edge replication at boundaries.
 *
 * Implementation: for each (slice, row) of the two non-axis dimensions, read
 * three adjacent rows/columns, write one filtered output row/column into
 * scratch_plane, then copy scratch_plane back.  Inner loop is always unit-stride
 * on the innermost axis, so vectorization works even on Y/Z passes. */
static void c3d_denoise_axis(float *restrict buf, size_t side,
                             unsigned axis, float alpha,
                             float *restrict scratch_plane,
                             float *restrict scratch_row)
{
    (void)scratch_plane; (void)scratch_row;   /* per-thread now */
    const size_t SY = C3D_STRIDE_Y;
    const size_t SZ = C3D_STRIDE_Z;
    const float one_minus_a = 1.0f - alpha;
    const float third = alpha * (1.0f / 3.0f);

    if (axis == 2u) {
        /* X pass: each (z,y) row is independent.  One per-thread row scratch. */
        #pragma omp parallel
        {
            float row_scratch[256];
            #pragma omp for schedule(static)
            for (size_t z = 0; z < side; ++z)
            for (size_t y = 0; y < side; ++y)
                c3d_denoise_blur_x_row(&buf[z * SZ + y * SY], side, alpha,
                                       row_scratch);
        }
        return;
    }

    /* Y pass: each z-slice is independent.  Per-thread plane scratch. */
    if (axis == 1u) {
        #pragma omp parallel
        {
            float plane_scratch[256 * 256];
            #pragma omp for schedule(static)
            for (size_t z = 0; z < side; ++z) {
                float *plane = buf + z * SZ;
                for (size_t y = 0; y < side; ++y) {
                    size_t y_l = (y > 0) ? y - 1 : y;
                    size_t y_r = (y + 1 < side) ? y + 1 : y;
                    const float *rl = plane + y_l * SY;
                    const float *rc = plane + y   * SY;
                    const float *rr = plane + y_r * SY;
                    float *out = plane_scratch + y * side;
                    for (size_t x = 0; x < side; ++x) {
                        out[x] = one_minus_a * rc[x]
                               + third * (rl[x] + rc[x] + rr[x]);
                    }
                }
                for (size_t y = 0; y < side; ++y) {
                    memcpy(plane + y * SY, plane_scratch + y * side,
                           side * sizeof(float));
                }
            }
        }
        return;
    }

    /* Z pass: each y-slab is independent.  Per-thread scratch. */
    if (axis == 0u) {
        #pragma omp parallel
        {
            float plane_scratch[256 * 256];
            #pragma omp for schedule(static)
            for (size_t y = 0; y < side; ++y) {
                for (size_t z = 0; z < side; ++z) {
                    size_t z_l = (z > 0) ? z - 1 : z;
                    size_t z_r = (z + 1 < side) ? z + 1 : z;
                    const float *rl = buf + z_l * SZ + y * SY;
                    const float *rc = buf + z   * SZ + y * SY;
                    const float *rr = buf + z_r * SZ + y * SY;
                    float *out = plane_scratch + z * side;
                    for (size_t x = 0; x < side; ++x) {
                        out[x] = one_minus_a * rc[x]
                               + third * (rl[x] + rc[x] + rr[x]);
                    }
                }
                for (size_t z = 0; z < side; ++z) {
                    memcpy(buf + z * SZ + y * SY, plane_scratch + z * side,
                           side * sizeof(float));
                }
            }
        }
        return;
    }
}

static void c3d_denoise_3d(float *buf, size_t side, float alpha)
{
    if (alpha <= 0.0f || side < 3) return;
    /* Scratch: one plane (side²) for Y/Z passes, one row (side) for X pass.
     * At side=256 that's 256 KiB + 1 KiB on the stack.  Fits within a
     * typical 8 MiB pthread stack comfortably; for smaller LODs it's <<1 KiB. */
    float scratch_plane[256 * 256];
    float scratch_row[256];
    c3d_assert(side <= 256);
    c3d_denoise_axis(buf, side, 2, alpha, scratch_plane, scratch_row); /* X */
    c3d_denoise_axis(buf, side, 1, alpha, scratch_plane, scratch_row); /* Y */
    c3d_denoise_axis(buf, side, 0, alpha, scratch_plane, scratch_row); /* Z */
}

/* Denoiser strength as a piecewise function of achieved ratio.
 * Re-calibrated after R-D allocator + dz=0.55 + softness=0.60 + T1c
 * spatial sign + T2a α clamp landed.  The sweet spot grows much
 * faster with ratio than the old curve assumed:
 *   r=5:    α=0 (denoise hurts near-lossless)
 *   r=10:   α=0.04 (+0.07 dB)
 *   r=25:   α=0.05 (+0.02 dB)
 *   r=50:   α=0.07 (+0.03 dB)
 *   r=100:  α=0.22 (+0.08 dB)
 *   r=200:  α=0.35 (+0.12 dB)
 * Sharp bump above r≈70 because at high ratios the wavelet noise
 * floor (sentinel-zeroed coefficients reconstructing to dz_half×sign
 * artifacts in the spatial domain) becomes the dominant error term —
 * a wide denoise blur smooths it out cheaply. */
static float c3d_denoise_strength(size_t in_len)
{
    double ratio = (double)C3D_VOXELS_PER_CHUNK / (double)in_len;
    if (ratio <=   8.0) return 0.0f;
    if (ratio <=  30.0) return 0.05f;
    if (ratio <=  70.0) return 0.08f;
    if (ratio <= 150.0) return 0.22f;
    return 0.35f;
}

/* ------------------------------------------------------------------------- *
 *  §I  Chunk decoder (SELF mode)                                            *
 * ------------------------------------------------------------------------- */

/* Decodes one subband's bitstream (its full byte range), dequantizes, and
 * scatters reconstructed float coefficients into coeff_buf at the subband's
 * spatial position.
 * If `external_freqs` is non-NULL and freq_table_size == 0, uses those
 * frequencies with the provided `external_denom_shift`. */
static void c3d_decode_one_subband(
    const uint8_t *restrict in, size_t in_size,
    float step, float dz_ratio, float alpha,
    const uint32_t *external_freqs, uint32_t external_denom_shift,
    float *restrict coeff_buf, const c3d_subband_info *sb,
    uint8_t *restrict sub_symbols, c3d_rans_tables *tbl_scratch,
    const c3d_rans_tables *cached_external_tbl)
{
    c3d_invariant(sb->side >= 8u && sb->side <= 128u);
    c3d_invariant((sb->side & (sb->side - 1u)) == 0u);
    size_t n = (size_t)sb->side * sb->side * sb->side;
    size_t r = 0;

    /* freq_table_size + freq_table */
    c3d_assert(in_size >= 2);
    uint16_t ftable_bytes = c3d_read_u16_le(in + r);
    r += 2;

    /* All-zero subband sentinel (encoder's fast path): zero-fill and return.
     * §T12: row-level memset is several times faster than the per-coef
     * triple-nested loop for large subbands, and this branch fires for
     * ~10-20 of 36 subbands at r≥50 per c3d_estimate_one_subband_bytes. */
    if (ftable_bytes == 0xFFFFu) {
        c3d_assert(in_size == 2);
        const size_t row_bytes = (size_t)sb->side * sizeof(float);
        for (uint32_t z = sb->z0; z < sb->z0 + sb->side; ++z)
        for (uint32_t y = sb->y0; y < sb->y0 + sb->side; ++y) {
            memset(&coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + sb->x0],
                   0, row_bytes);
        }
        (void)step; (void)alpha;
        (void)sub_symbols; (void)tbl_scratch;
        return;
    }

    uint32_t denom_shift;
    uint32_t local_freqs[65];
    bool ctx_2table = false;
    uint32_t ctx_freqs_z[65], ctx_freqs_nz[65];

    if (ftable_bytes == 0) {
        /* EXTERNAL: require ctx to have provided tables. */
        c3d_assert(external_freqs != NULL);
        denom_shift = external_denom_shift;
    } else {
        c3d_assert(r + ftable_bytes <= in_size);
        /* First byte = ctx_mode: 0 = single table, 1 = 2-table lane-local. */
        uint8_t ctx_mode = in[r];
        r++;
        if (ctx_mode == 1) {
            ctx_2table = true;
            size_t c1 = c3d_freqs_parse(in + r, ftable_bytes - 1, &denom_shift, ctx_freqs_z);
            r += c1;
            uint32_t ds2;
            size_t c2 = c3d_freqs_parse(in + r, ftable_bytes - 1 - c1, &ds2, ctx_freqs_nz);
            c3d_assert(ds2 == denom_shift);
            r += c2;
        } else {
            size_t consumed = c3d_freqs_parse(in + r, ftable_bytes - 1, &denom_shift, local_freqs);
            r += consumed;
        }
    }

    /* n_symbols + rans_block_size */
    c3d_assert(r + 8 <= in_size);
    uint32_t n_symbols = c3d_read_u32_le(in + r); r += 4;
    uint32_t rans_block_size = c3d_read_u32_le(in + r); r += 4;
    c3d_assert(n_symbols == n);
    c3d_assert(r + rans_block_size <= in_size);

    if (ctx_2table) {
        c3d_rans_tables tbl_z, tbl_nz;
        c3d_rans_build_tables(&tbl_z,  denom_shift, ctx_freqs_z, 65);
        c3d_rans_build_tables(&tbl_nz, denom_shift, ctx_freqs_nz, 65);
        c3d_rans_dec_x8_ctx(in + r, rans_block_size, &tbl_z, &tbl_nz, sub_symbols, n);
    } else if (cached_external_tbl && ftable_bytes == 0) {
        c3d_rans_dec_x8(in + r, rans_block_size, cached_external_tbl, sub_symbols, n);
    } else {
        const uint32_t *ftu = (ftable_bytes == 0) ? external_freqs : local_freqs;
        c3d_rans_build_tables(tbl_scratch, denom_shift, ftu, 65);
        c3d_rans_dec_x8(in + r, rans_block_size, tbl_scratch, sub_symbols, n);
    }
    r += rans_block_size;

    /* escape_stream spans [r..in_size). */
    const uint8_t *esc_ptr = in + r;
    size_t esc_remaining = in_size - r;

    /* Dequantize + scatter into coeff_buf (sign-predictive symbols).
     * §T12: hoist sb-> fields to locals to help the compiler eliminate
     * repeated struct-field reloads inside the hot inner loop.
     * §T13: dz_half comes from caller's dz_ratio (ctx override or default). */
    float dz_half = dz_ratio * step;
    bool prev_sign_zy[128 * 128];
    memset(prev_sign_zy, 0, (size_t)sb->side * sb->side);
    const uint32_t sb_z0 = sb->z0, sb_y0 = sb->y0, sb_x0 = sb->x0;
    const uint32_t sb_side = sb->side;
    size_t idx = 0;
    /* Row-wise two-phase dequant.  Phase 1 (scalar): read sub_symbols, do
     * the stateful sign-prediction + escape LEB128 decode, emit qv into a
     * stack-local row buffer.  Phase 2 (NEON): turn qv → float via the
     * dequant formula, 4 lanes at a time, write contiguous x-run.
     * The scalar phase keeps sp/escape flow trivially correct; NEON only
     * touches the pure float math. */
    int32_t qv_row[128];   /* max sb_side = 128 */
    const float inv_step_unused = 0.0f; (void)inv_step_unused;
    for (uint32_t z = sb_z0; z < sb_z0 + sb_side; ++z)
    for (uint32_t y = sb_y0; y < sb_y0 + sb_side; ++y) {
        /* Phase 1 — scalar, keeps sp + escape state correct. */
        for (uint32_t x = 0; x < sb_side; ++x) {
            uint8_t sym = sub_symbols[idx++];
            uint32_t escape_mag = 0;
            if (c3d_unlikely(C3D_SYM_IS_ESCAPE(sym))) {
                uint64_t zv = 0;
                size_t c = c3d_leb128_decode(esc_ptr, esc_remaining, &zv);
                c3d_assert(zv <= 0xffffffffull);
                escape_mag = (uint32_t)zv;
                esc_ptr += c;
                esc_remaining -= c;
            }
            bool *sp = &prev_sign_zy[(y - sb_y0) * sb_side + x];
            qv_row[x] = c3d_symbol_to_quant(sym, escape_mag, sp);
        }
        /* Phase 2 — NEON dequant over the row. */
        float *out_row = &coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + sb_x0];
#ifdef C3D_HAVE_NEON
        float32x4_t vdz    = vdupq_n_f32(dz_half);
        float32x4_t vstep  = vdupq_n_f32(step);
        float32x4_t vbias  = vdupq_n_f32(alpha - 1.0f);
        float32x4_t vzero  = vdupq_n_f32(0.0f);
        uint32_t x = 0;
        for (; x + 4 <= sb_side; x += 4) {
            int32x4_t q   = vld1q_s32(qv_row + x);
            uint32x4_t isz = vceqq_s32(q, vdupq_n_s32(0));
            int32x4_t aq   = vabsq_s32(q);
            float32x4_t af = vcvtq_f32_s32(aq);
            float32x4_t mag = vfmaq_f32(vdz, vstep, vaddq_f32(af, vbias));
            uint32x4_t neg = vcltq_s32(q, vdupq_n_s32(0));
            float32x4_t res = vbslq_f32(neg, vnegq_f32(mag), mag);
            res = vbslq_f32(isz, vzero, res);
            vst1q_f32(out_row + x, res);
        }
        for (; x < sb_side; ++x)
            out_row[x] = c3d_dequant(qv_row[x], step, dz_half, alpha);
#else
        for (uint32_t x = 0; x < sb_side; ++x)
            out_row[x] = c3d_dequant(qv_row[x], step, dz_half, alpha);
#endif
    }
    c3d_assert(esc_remaining == 0);
}

void c3d_decoder_chunk_decode_lod(c3d_decoder *d,
                                  const uint8_t *in, size_t in_len, uint8_t lod,
                                  uint8_t *out)
{
    c3d_assert(d && in && out);
    c3d_check_voxel_alignment(out);
    c3d_assert(in_len >= C3D_CHUNK_FIXED_SIZE);
    c3d_assert(lod < C3D_N_LODS);
    c3d_assert(memcmp(in, "C3DC", 4) == 0);
    uint16_t version = c3d_read_u16_le(in + 4);
    c3d_assert(version == 1);

    float dc_offset   = c3d_read_f32_le(in + 8);
    float coeff_scale = c3d_read_f32_le(in + 12);

    const uint8_t *qmul_ptr   = in + 40;
    const uint8_t *suboff_ptr = in + 40 + 144;
    const uint8_t *lodoff_ptr = in + 40 + 144 + 144;
    const uint8_t *alpha_ptr  = in + C3D_CHUNK_ALPHA_OFFSET;
    const uint8_t *entropy    = in + C3D_CHUNK_FIXED_SIZE;

    uint32_t lod_end = c3d_read_u32_le(lodoff_ptr + 4 * lod);
    size_t out_side = (size_t)C3D_CHUNK_SIDE >> lod;
    size_t out_vox  = out_side * out_side * out_side;

    if (lod_end == 0) {
        float v = dc_offset + 128.0f;
        int iv = (int)(v + 0.5f);
        if (iv < 0) iv = 0; else if (iv > 255) iv = 255;
        memset(out, (uint8_t)iv, out_vox);
        return;
    }

    unsigned n_sb = c3d_n_subbands_for_lod[lod];

    const c3d_rans_tables *cached_ext = NULL;

    /* §T9 — quality-scalable truncation.  If `in_len` is shorter than the
     * emitted chunk (caller truncated for streaming / bandwidth-adaptive
     * decode), any subband whose entropy range extends past the supplied
     * bytes is zero-filled instead of panicking.  Subsequent subbands get
     * the same treatment.  Output quality degrades gracefully from finest
     * (LOD 0 full) to coarsest (LL_5 only) as bytes drop, and is monotonic
     * — appending bytes can only improve quality. */
    size_t entropy_avail = (in_len > C3D_CHUNK_FIXED_SIZE)
                         ? (in_len - C3D_CHUNK_FIXED_SIZE) : 0;

    /* Pre-scan the per-subband offset table to find the first truncated
     * subband.  Subbands before it decode in parallel; subbands at or
     * after it get zero-filled serially (cheap).  This hoists the
     * data-dependent early-break out of the parallel region. */
    unsigned first_trunc = n_sb;
    for (unsigned s = 0; s < n_sb; ++s) {
        uint32_t sub_start = c3d_read_u32_le(suboff_ptr + 4 * s);
        uint32_t sub_end   = (s + 1 < n_sb)
                           ? c3d_read_u32_le(suboff_ptr + 4 * (s + 1))
                           : lod_end;
        c3d_assert(sub_end >= sub_start);
        if (sub_end > entropy_avail) { first_trunc = s; break; }
    }

    /* Parallel decode of [0, first_trunc).  Each thread uses its own
     * per-thread sub_symbols scratch from d->thread_sub_symbols (arena:
     * allocated once, reused across chunks).  tbl is per-thread on stack
     * (~100 KB — fine).  Dynamic scheduling with a chunk size of 1
     * balances the wide range of subband sizes (8³=512 vox to 128³=2 M). */
    #pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        uint8_t *tls_syms;
        if (tid < C3D_OMP_MAX_THREADS) {
            if (!d->thread_sub_symbols[tid]) {
                d->thread_sub_symbols[tid] = malloc((size_t)128 * 128 * 128);
                c3d_assert(d->thread_sub_symbols[tid]);
            }
            tls_syms = d->thread_sub_symbols[tid];
        } else {
            /* Over-subscribed: fall back to thread-0 scratch.  Rare but
             * safe — only one such thread runs any given subband at a time
             * under dynamic scheduling. */
            tls_syms = d->thread_sub_symbols[0];
        }
        c3d_rans_tables tls_tbl;
        #pragma omp for schedule(dynamic,1)
        for (unsigned s = 0; s < first_trunc; ++s) {
            c3d_subband_info sb;
            c3d_subband_info_of(s, &sb);
            float step  = c3d_read_f32_le(qmul_ptr + 4 * s);
            float alpha;
            {
                uint8_t av = alpha_ptr[s];
                alpha = av ? c3d_alpha_from_u8(av) : c3d_default_alpha(s);
            }
            const uint32_t *ext_freqs = NULL;
            uint32_t ext_ds           = 0u;
            float dz_ratio = c3d_dz_ratio_for_kind(c3d_kind_h_count(sb.kind));
            uint32_t sub_start = c3d_read_u32_le(suboff_ptr + 4 * s);
            uint32_t sub_end   = (s + 1 < n_sb)
                               ? c3d_read_u32_le(suboff_ptr + 4 * (s + 1))
                               : lod_end;
            c3d_decode_one_subband(entropy + sub_start, sub_end - sub_start,
                                   step, dz_ratio, alpha,
                                   ext_freqs, ext_ds,
                                   d->coeff_buf, &sb, tls_syms, &tls_tbl,
                                   cached_ext ? &cached_ext[s] : NULL);
        }
        /* tls_syms is arena-owned by the decoder; do not free here. */
    }

    /* Serial zero-fill for truncated tail (§T9). */
    for (unsigned sz = first_trunc; sz < n_sb; ++sz) {
        c3d_subband_info sbz;
        c3d_subband_info_of(sz, &sbz);
        for (uint32_t z = sbz.z0; z < sbz.z0 + sbz.side; ++z)
        for (uint32_t y = sbz.y0; y < sbz.y0 + sbz.side; ++y)
        for (uint32_t x = sbz.x0; x < sbz.x0 + sbz.side; ++x)
            d->coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x] = 0.0f;
    }

    unsigned n_synth = C3D_N_DWT_LEVELS - lod;
    c3d_dwt3_inv_levels(d->coeff_buf, n_synth, d->dwt_scratch);

    /* Q2 post-decode denoiser at LOD 0 only.  In-loop: read the encoder-
     * chosen alpha from header byte 7.  If zero (old-format or encode_at_q
     * which doesn't set it), fall back to the ratio-based heuristic.
     * Opt-out via C3D_DENOISE=0 for bench comparisons. */
    if (lod == 0) {
        uint8_t hdr_dn = in[7];
        float denoise_alpha = hdr_dn ? ((float)hdr_dn / 400.0f)
                                     : c3d_denoise_strength(in_len);
        const char *env = getenv("C3D_DENOISE");
        if (env && env[0] == '0') denoise_alpha = 0.0f;
        c3d_denoise_3d(d->coeff_buf, C3D_CHUNK_SIDE, denoise_alpha);
    }

    /* Encoder v2: coeff_scale is already absorbed into per-subband step, so
     * dequant produces raw-magnitude coefficients.  coeff_scale in the header
     * is informational only (preserved for c3d_inspect and downstream tools). */
    /* §T10: branch-free u8 output cast.  Previous version had `? 0.5 : -0.5`
     * and a clamping if-ladder, both of which blocked auto-vectorization.
     * This form is pure float arithmetic with fminf/fmaxf and single cast —
     * clang and gcc both auto-vectorize the innermost x-loop.  Inputs to the
     * cast live in [0, 255] after the fmaxf/fminf so (uint8_t)(v_c + 0.5f)
     * rounds correctly (no negative-rounding case needed). */
    (void)coeff_scale;
    for (size_t z = 0; z < out_side; ++z) {
        for (size_t y = 0; y < out_side; ++y) {
            const float *restrict row = d->coeff_buf + z * C3D_STRIDE_Z
                                      + y * C3D_STRIDE_Y;
            uint8_t *restrict orow = out + z * out_side * out_side
                                   + y * out_side;
            for (size_t x = 0; x < out_side; ++x) {
                float v = row[x] + dc_offset + 128.0f;
                float v_c = fminf(fmaxf(v, 0.0f), 255.0f);
                orow[x] = (uint8_t)(v_c + 0.5f);
            }
        }
    }
}

void c3d_decoder_chunk_decode(c3d_decoder *d, const uint8_t *in, size_t in_len,
                              uint8_t *out)
{
    c3d_decoder_chunk_decode_lod(d, in, in_len, 0, out);
}

/* §I3.  Batched multi-chunk decode — loop the single-chunk API.  When all
 * chunks share the same EXTERNAL ctx, the decoder's per-ctx rANS table
 * cache (S2) stays warm across the batch. */
void c3d_decoder_chunks_decode(c3d_decoder *d,
                               const uint8_t *const *ins,
                               const size_t *in_sizes,
                               size_t n_chunks,
                               uint8_t *const *outs)
{
    c3d_assert(d && ins && in_sizes && outs);
    for (size_t i = 0; i < n_chunks; ++i) {
        c3d_decoder_chunk_decode_lod(d, ins[i], in_sizes[i], 0, outs[i]);
    }
}

void c3d_chunk_decode_lod(const uint8_t *in, size_t in_len, uint8_t lod,
                          uint8_t *out)
{
    c3d_decoder *d = c3d_decoder_new();
    c3d_decoder_chunk_decode_lod(d, in, in_len, lod, out);
    c3d_decoder_free(d);
}

void c3d_chunk_decode(const uint8_t *in, size_t in_len, uint8_t *out) {
    c3d_chunk_decode_lod(in, in_len, 0, out);
}

/* Cheap metadata peek — no entropy decode. */
void c3d_chunk_inspect(const uint8_t *in, size_t in_len, c3d_chunk_info *info) {
    c3d_assert(in && info);
    c3d_assert(in_len >= C3D_CHUNK_FIXED_SIZE);
    c3d_assert(memcmp(in, "C3DC", 4) == 0);
    uint16_t version = c3d_read_u16_le(in + 4);
    c3d_assert(version == 1);
    info->dc_offset   = c3d_read_f32_le(in + 8);
    info->coeff_scale = c3d_read_f32_le(in + 12);
    const uint8_t *lodoff = in + 40 + 144 + 144;
    for (unsigned k = 0; k < C3D_N_LODS; ++k)
        info->lod_offsets[k] = c3d_read_u32_le(lodoff + 4 * k);
}

/* Non-panicking structural check — does NOT run entropy decode. */
bool c3d_chunk_validate(const uint8_t *in, size_t in_len) {
    if (!in || in_len < C3D_CHUNK_FIXED_SIZE)             return false;
    if (memcmp(in, "C3DC", 4) != 0)                       return false;
    if (c3d_read_u16_le(in + 4) != 1)                     return false;

    const uint8_t *suboff = in + 40 + 144;
    const uint8_t *lodoff = in + 40 + 144 + 144;
    uint32_t lod0 = c3d_read_u32_le(lodoff + 0);

    /* entropy region length = in_len - 352 must ≥ lod0 */
    if (in_len < (size_t)C3D_CHUNK_FIXED_SIZE + lod0)     return false;

    /* Empty chunk: all lod_offsets zero, lod0 == 0. */
    if (lod0 == 0) {
        for (unsigned k = 0; k < C3D_N_LODS; ++k)
            if (c3d_read_u32_le(lodoff + 4 * k) != 0)     return false;
        return true;
    }

    /* Check monotonic lod_offsets (lod_offset[5] ≤ [4] ≤ ... ≤ [0]). */
    uint32_t prev = 0;
    for (unsigned k = C3D_N_LODS; k-- > 0; ) {
        uint32_t v = c3d_read_u32_le(lodoff + 4 * k);
        if (v < prev) return false;
        prev = v;
    }
    if (prev != lod0) return false;

    /* Check monotonic subband_offsets. */
    prev = 0;
    for (unsigned s = 0; s < C3D_N_SUBBANDS; ++s) {
        uint32_t v = c3d_read_u32_le(suboff + 4 * s);
        if (v < prev) return false;
        if (v > lod0) return false;
        prev = v;
    }

    return true;
}


/* ========================================================================= *
 *  §13  c3d_downsample_chunk_2x (box 2^3 average)                           *
 * ========================================================================= */

void c3d_downsample_chunk_2x(const uint8_t *in, uint32_t side, uint8_t *out) {
    c3d_assert(side == 256 || side == 128 || side == 64 || side == 32 || side == 16);
    uint32_t half = side / 2;
    for (uint32_t z = 0; z < half; ++z)
    for (uint32_t y = 0; y < half; ++y)
    for (uint32_t x = 0; x < half; ++x) {
        uint32_t sum = 0;
        for (uint32_t dz = 0; dz < 2; ++dz)
        for (uint32_t dy = 0; dy < 2; ++dy)
        for (uint32_t dx = 0; dx < 2; ++dx) {
            sum += in[(2*z + dz) * side * side + (2*y + dy) * side + (2*x + dx)];
        }
        /* Round to nearest, ties to even (banker's rounding). */
        uint32_t rounded = (sum + 4) >> 3;
        /* Banker's rounding on ties: if sum is odd-half (sum+4 is odd), stays same.
         * For perfectly even ties (sum mod 8 == 4), bias to even. */
        if ((sum & 7u) == 4u && (rounded & 1u)) --rounded;
        out[z * half * half + y * half + x] = (uint8_t)rounded;
    }
}
