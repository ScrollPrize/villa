/* c3d — a 3D volumetric u8 compression codec for larger-than-RAM X-ray data.
 * See LICENSE.  See PLAN.md for the full design spec.  This header is the
 * canonical public API; c3d.c is the canonical implementation.
 *
 * Library-wide rules (short form — full version in PLAN.md §0):
 *   - C23, single TU (c3d.c), libc only.
 *   - In-memory API.  Library never touches disk, network, or fds.
 *   - Fatal on error: every invalid input, OOM, or parser inconsistency calls
 *     c3d_panic() which aborts.  No status codes.  Happy path is the only path.
 *   - Little-endian only (build-time static-assert).
 *   - Same-binary encode is byte-deterministic; cross-binary is not.
 */

#ifndef C3D_H
#define C3D_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── library + format version ───────────────────────────────────────────── */

#define C3D_FORMAT_VERSION 1u

#define C3D_VERSION_MAJOR 1
#define C3D_VERSION_MINOR 0
#define C3D_VERSION_PATCH 0
#define C3D_VERSION_STRING "1.0.0"

const char *c3d_version(void);           /* "1.0.0" */
uint32_t    c3d_format_version(void);    /* C3D_FORMAT_VERSION */

/* ─── fixed hierarchy constants ──────────────────────────────────────────── */

#define C3D_BLOCK_SIDE     16u     /* caller-side RAM cache granularity          */
#define C3D_CHUNK_SIDE     256u    /* codec atom: one encode/decode call         */
#define C3D_VOXELS_PER_CHUNK ((size_t)C3D_CHUNK_SIDE * C3D_CHUNK_SIDE * C3D_CHUNK_SIDE)

#define C3D_N_LODS         6u      /* LOD 0 (256^3) .. LOD 5 (8^3)               */
#define C3D_N_DWT_LEVELS   5u
#define C3D_N_SUBBANDS     36u     /* 1 LLL_5 + 5*7 details = 36                 */

#define C3D_ALIGN          32u     /* required alignment for raw voxel buffers   */

/* ─── magic identifiers ──────────────────────────────────────────────────── */

#define C3D_CHUNK_MAGIC  "C3DC"

/* Side at the given LOD.  LOD 0 = 256³ (full), LOD 5 = 8³ (coarsest). */
static inline uint32_t c3d_side_per_lod(uint8_t lod) {
    return (lod <= 5u) ? (C3D_CHUNK_SIDE >> lod) : 0u;
}
static inline size_t c3d_voxels_per_lod(uint8_t lod) {
    size_t s = (size_t)c3d_side_per_lod(lod);
    return s * s * s;
}

/* Cheap magic sniff.  True iff `in` has at least 4 bytes and the first 4
 * are "C3DC". */
static inline bool c3d_is_chunk(const uint8_t *in, size_t n) {
    return n >= 4u
        && in[0] == (uint8_t)'C' && in[1] == (uint8_t)'3'
        && in[2] == (uint8_t)'D' && in[3] == (uint8_t)'C';
}

/* Upper bound on a c3d_chunk_encode output. */
#define C3D_CHUNK_ENCODE_MAX_SIZE \
    (((size_t)16 * 1024 * 1024 + 388 + 4096 + (size_t)(C3D_ALIGN - 1)) \
     & ~(size_t)(C3D_ALIGN - 1))

/* ─── panic / assert ─────────────────────────────────────────────────────── */

typedef void (*c3d_panic_fn)(const char *file, int line, const char *msg);
void c3d_set_panic_hook(c3d_panic_fn hook);

#ifdef __cplusplus
[[noreturn]]
#else
_Noreturn
#endif
void c3d_panic(const char *file, int line, const char *msg);

#define c3d_assert(cond)                                                      \
    do {                                                                      \
        if (!(cond)) c3d_panic(__FILE__, __LINE__, #cond);                    \
    } while (0)

#ifdef NDEBUG
#  if defined(__clang__)
#    define c3d_invariant(cond) __builtin_assume(cond)
#  elif defined(__GNUC__)
#    define c3d_invariant(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#  else
#    define c3d_invariant(cond) ((void)0)
#  endif
#else
#  define c3d_invariant(cond) c3d_assert(cond)
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define c3d_likely(cond)   __builtin_expect(!!(cond), 1)
#  define c3d_unlikely(cond) __builtin_expect(!!(cond), 0)
#else
#  define c3d_likely(cond)   (cond)
#  define c3d_unlikely(cond) (cond)
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define C3D_CONST __attribute__((const))
#  define C3D_PURE  __attribute__((pure))
#else
#  define C3D_CONST
#  define C3D_PURE
#endif

/* ─── u64 voxel key ──────────────────────────────────────────────────────── */
/* Layout: [ lod:4 ][ z:20 ][ y:20 ][ x:20 ].  Planar (not Morton). */

static inline uint64_t c3d_key(uint32_t x, uint32_t y, uint32_t z, uint8_t lod) {
    return ((uint64_t)(lod & 0xfu) << 60)
         | ((uint64_t)(z & 0xfffffu) << 40)
         | ((uint64_t)(y & 0xfffffu) << 20)
         | ((uint64_t)(x & 0xfffffu));
}
static inline void c3d_unkey(uint64_t k, uint32_t *x, uint32_t *y, uint32_t *z, uint8_t *lod) {
    *x   = (uint32_t)( k         & 0xfffffu);
    *y   = (uint32_t)((k >> 20)  & 0xfffffu);
    *z   = (uint32_t)((k >> 40)  & 0xfffffu);
    *lod = (uint8_t )((k >> 60)  & 0xfu);
}

/* ─── 128-bit content hash (MurmurHash3_x64_128) ─────────────────────────── */

void c3d_hash128(const void *data, size_t len, uint8_t out[16]);

/* ─── chunk inspection (metadata-only, fast) ─────────────────────────────── */

typedef struct {
    uint32_t lod_offsets[C3D_N_LODS];
    float    dc_offset;
    float    coeff_scale;
} c3d_chunk_info;

void c3d_chunk_inspect(const uint8_t *in, size_t in_len, c3d_chunk_info *info);

bool c3d_chunk_validate(const uint8_t *in, size_t in_len);

/* ─── stateless chunk codec ──────────────────────────────────────────────── */

/* Reusable encoder/decoder scratch.  Create once per thread, reuse across
 * many chunks to avoid alloc/free churn (50-100 ms/chunk saved).
 *
 * Thread-safety: a c3d_encoder / c3d_decoder instance is NOT thread-safe.
 * For multi-threaded encode/decode, allocate one per worker thread. */
typedef struct c3d_encoder c3d_encoder;
typedef struct c3d_decoder c3d_decoder;

c3d_encoder *c3d_encoder_new(void);
void         c3d_encoder_free(c3d_encoder *);
c3d_decoder *c3d_decoder_new(void);
void         c3d_decoder_free(c3d_decoder *);

/* Reusable-context variants — same semantics as the stateless calls below
 * but allocate-once-reuse-many.  Recommended for any caller doing >1 chunk. */
size_t c3d_encoder_chunk_encode(c3d_encoder *, const uint8_t *in,
                                float target_ratio,
                                uint8_t *out, size_t out_cap);
size_t c3d_encoder_chunk_encode_at_q(c3d_encoder *, const uint8_t *in,
                                     float q,
                                     uint8_t *out, size_t out_cap);
void   c3d_decoder_chunk_decode(c3d_decoder *, const uint8_t *in, size_t in_len,
                                uint8_t *out);
void   c3d_decoder_chunk_decode_lod(c3d_decoder *, const uint8_t *in, size_t in_len,
                                    uint8_t lod, uint8_t *out);

/* Multi-chunk batched encode. */
void c3d_encoder_chunks_encode(c3d_encoder *e,
                               const uint8_t *const *inputs,
                               size_t n_chunks,
                               float target_ratio,
                               uint8_t *const *outs,
                               size_t *out_sizes);

/* Multi-chunk batched decode — analogous to the encode batch. */
void c3d_decoder_chunks_decode(c3d_decoder *d,
                               const uint8_t *const *ins,
                               const size_t *in_sizes,
                               size_t n_chunks,
                               uint8_t *const *outs);

size_t c3d_chunk_encode_max_size(void);   /* returns C3D_CHUNK_ENCODE_MAX_SIZE */

/* target_ratio must be > 1.0.  Returns bytes written. */
size_t c3d_chunk_encode(const uint8_t *in,
                        float target_ratio,
                        uint8_t *out, size_t out_cap);

/* Bypass rate control; use the given quantizer scalar q ∈ [2^-6, 2^12]. */
size_t c3d_chunk_encode_at_q(const uint8_t *in,
                             float q,
                             uint8_t *out, size_t out_cap);

/* "Zero means ignore" encode variants.  Voxels with value 0 in the input are
 * treated as don't-care: the encoder replaces them with the minimum non-zero
 * value found in the chunk, so the DWT sees no step between air and material
 * and the bit budget concentrates on the material voxels. */
size_t c3d_chunk_encode_masked(const uint8_t *in, float target_ratio,
                               uint8_t *out, size_t out_cap);
size_t c3d_chunk_encode_masked_at_q(const uint8_t *in, float q,
                                    uint8_t *out, size_t out_cap);
size_t c3d_encoder_chunk_encode_masked(c3d_encoder *, const uint8_t *in,
                                       float target_ratio,
                                       uint8_t *out, size_t out_cap);
size_t c3d_encoder_chunk_encode_masked_at_q(c3d_encoder *, const uint8_t *in,
                                            float q,
                                            uint8_t *out, size_t out_cap);

/* LOD 0 decode — writes 256^3 u8 into out. */
void c3d_chunk_decode(const uint8_t *in, size_t in_len,
                      uint8_t *out);

/* LOD decode, lod ∈ 0..5.  out must be sized (256>>lod)^3 bytes. */
void c3d_chunk_decode_lod(const uint8_t *in, size_t in_len, uint8_t lod,
                          uint8_t *out);

/* Post-decode 2× box-average downsample for caller-side pyramids.
 * side ∈ {256, 128, 64, 32, 16}.  Writes (side/2)^3 to out. */
void c3d_downsample_chunk_2x(const uint8_t *in, uint32_t side, uint8_t *out);

#ifdef __cplusplus
}
#endif

#endif /* C3D_H */
