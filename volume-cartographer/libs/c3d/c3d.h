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

/* Format version written into every c3d chunk / shard / .c3dx blob.  Frozen
 * at 1 for the lifetime of the codec (per PLAN §0: re-encoding from raw u8
 * is the upgrade path during development, not on-the-wire compat).  Useful
 * for downstream code that wants to record the codec identity in an
 * enclosing metadata blob (e.g. zarr `.zarray` codec config). */
#define C3D_FORMAT_VERSION 1u

/* Library version string, semver.  Matches `project(c3d VERSION ...)` in
 * CMakeLists.txt; written out at install time via the generated .pc file. */
#define C3D_VERSION_MAJOR 1
#define C3D_VERSION_MINOR 0
#define C3D_VERSION_PATCH 0
#define C3D_VERSION_STRING "1.0.0"

/* Runtime accessors.  Useful when the caller links against the shared lib
 * (libc3d.so) and wants to check the header-vs-library version match. */
const char *c3d_version(void);           /* "1.0.0" */
uint32_t    c3d_format_version(void);    /* C3D_FORMAT_VERSION */

/* ─── fixed hierarchy constants ──────────────────────────────────────────── */

#define C3D_BLOCK_SIDE     16u     /* caller-side RAM cache granularity          */
#define C3D_CHUNK_SIDE     256u    /* codec atom: one encode/decode call         */
#define C3D_SHARD_SIDE     4096u   /* 16^3 = 4096 chunks per shard               */
#define C3D_VOXELS_PER_CHUNK ((size_t)C3D_CHUNK_SIDE * C3D_CHUNK_SIDE * C3D_CHUNK_SIDE)

#define C3D_N_LODS         6u      /* LOD 0 (256^3) .. LOD 5 (8^3)               */
#define C3D_N_DWT_LEVELS   5u
#define C3D_N_SUBBANDS     36u     /* 1 LLL_5 + 5*7 details = 36                 */

#define C3D_ALIGN          32u     /* required alignment for raw voxel buffers   */

/* ─── magic identifiers ──────────────────────────────────────────────────── */

/* Every encoded chunk starts with the 4 bytes "C3DC" at offset 0.
 * Every serialised shard starts with "C3DS".
 * Every serialised .c3dx context block starts with "C3DX".
 *
 * Downstream zarr / archive tooling can sniff the chunk magic cheaply to
 * dispatch between codecs (e.g. c3d vs legacy blosc/zstd) without parsing
 * the full header. */
#define C3D_CHUNK_MAGIC  "C3DC"
#define C3D_SHARD_MAGIC  "C3DS"
#define C3D_CTX_MAGIC    "C3DX"

/* Side at the given LOD.  LOD 0 = 256³ (full), LOD 5 = 8³ (coarsest).
 * Valid for lod ∈ [0, 5]; higher values clamp to 0.  The buffer passed to
 * c3d_chunk_decode_lod() must be at least c3d_voxels_per_lod(lod) bytes. */
static inline uint32_t c3d_side_per_lod(uint8_t lod) {
    return (lod <= 5u) ? (C3D_CHUNK_SIDE >> lod) : 0u;
}
static inline size_t c3d_voxels_per_lod(uint8_t lod) {
    size_t s = (size_t)c3d_side_per_lod(lod);
    return s * s * s;
}

/* Cheap magic sniff.  True iff `in` has at least 4 bytes and the first 4
 * are "C3DC".  Does NOT validate version, sizes, or entropy payload — use
 * c3d_chunk_validate() for structural integrity.  The intended use is a
 * codec-dispatch decision inside a multi-codec archive reader: "is this a
 * c3d chunk?" → yes → call the c3d path; no → try the other codec. */
static inline bool c3d_is_chunk(const uint8_t *in, size_t n) {
    return n >= 4u
        && in[0] == (uint8_t)'C' && in[1] == (uint8_t)'3'
        && in[2] == (uint8_t)'D' && in[3] == (uint8_t)'C';
}
static inline bool c3d_is_shard(const uint8_t *in, size_t n) {
    return n >= 4u
        && in[0] == (uint8_t)'C' && in[1] == (uint8_t)'3'
        && in[2] == (uint8_t)'D' && in[3] == (uint8_t)'S';
}
static inline bool c3d_is_ctx(const uint8_t *in, size_t n) {
    return n >= 4u
        && in[0] == (uint8_t)'C' && in[1] == (uint8_t)'3'
        && in[2] == (uint8_t)'D' && in[3] == (uint8_t)'X';
}

/* Upper bound on a c3d_chunk_encode output: raw u8 + fixed header + tables
 * + a small range-coder safety margin.  Small enough to stack-allocate.
 * (Fixed header = 388 B: 40 B base + 144 B qmul + 144 B subband_offset
 *  + 24 B lod_offset + 36 B per-subband Laplacian α.) */
/* 16 MiB + header + slack, rounded up to a multiple of C3D_ALIGN so the
 * constant is a valid `size` argument to aligned_alloc (standard C11
 * requires `size` divide evenly by `align`; ASan enforces it strictly). */
#define C3D_CHUNK_ENCODE_MAX_SIZE \
    (((size_t)16 * 1024 * 1024 + 388 + 4096 + (size_t)(C3D_ALIGN - 1)) \
     & ~(size_t)(C3D_ALIGN - 1))

/* ─── panic / assert ─────────────────────────────────────────────────────── */

/* Default: fputs a short message to stderr, then abort().
 * Install your own to capture in tests (but hooks must not return). */
typedef void (*c3d_panic_fn)(const char *file, int line, const char *msg);
void c3d_set_panic_hook(c3d_panic_fn hook);

/* Called by every internal failure.  Does not return.
 * Hooks that attempt to longjmp or otherwise resume control are undefined
 * behaviour; library state is unrecoverable after a panic. */
_Noreturn void c3d_panic(const char *file, int line, const char *msg);

#define c3d_assert(cond)                                                      \
    do {                                                                      \
        if (!(cond)) c3d_panic(__FILE__, __LINE__, #cond);                    \
    } while (0)

/* c3d_invariant: hot-path invariant hint.
 *
 * Debug builds: same as c3d_assert — runtime check, panic on failure.
 * Release builds: compiler hint (`[[assume]]` / `__builtin_unreachable`)
 *                 — no runtime code, but the optimiser may narrow value
 *                 ranges, drop bounds checks, pick aligned loads, etc.
 *
 * Use ONLY for invariants that are genuinely true by construction on every
 * internal call path.  If one is false at runtime in release, the compiler
 * assumes it's true and may emit UB-producing code: silent corruption, not
 * a clean panic.  That's why safety-critical checks (buffer bounds, version
 * match, external inputs) stay on c3d_assert. */
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

/* Portable branch hints. */
#if defined(__GNUC__) || defined(__clang__)
#  define c3d_likely(cond)   __builtin_expect(!!(cond), 1)
#  define c3d_unlikely(cond) __builtin_expect(!!(cond), 0)
#else
#  define c3d_likely(cond)   (cond)
#  define c3d_unlikely(cond) (cond)
#endif

/* Mark pure/const helpers for CSE.  pure = reads memory, no side effects.
 * const = no memory reads (only args), no side effects — stronger. */
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

/* ─── chunk state enum ───────────────────────────────────────────────────── */

typedef enum {
    C3D_CHUNK_ABSENT  = 0,  /* shard index: (UINT64_MAX, 0) — never written    */
    C3D_CHUNK_ZERO    = 1,  /* shard index: (0, 0) — definitionally empty      */
    C3D_CHUNK_PRESENT = 2,  /* real payload exists                              */
} c3d_chunk_state;

/* ─── chunk inspection (metadata-only, fast) ─────────────────────────────── */

typedef struct {
    uint8_t  context_mode;       /* 0 = SELF, 1 = EXTERNAL                      */
    uint8_t  context_id[16];     /* zero if SELF                                */
    uint32_t lod_offsets[C3D_N_LODS];
    float    dc_offset;
    float    coeff_scale;
} c3d_chunk_info;

void c3d_chunk_inspect(const uint8_t *in, size_t in_len, c3d_chunk_info *info);

/* Non-panicking structural check: magic, version, header sizes, table offsets,
 * per-subband frame sizes, TLV bounds.  Does NOT run entropy decode — a
 * structurally-valid chunk with a bad rANS state inside will still panic on
 * actual decode.
 *
 * Validate-then-decode pattern for untrusted input (e.g. reading zarr chunks
 * from an archive that may contain corrupted or foreign bytes):
 *
 *     if (!c3d_is_chunk(in, in_len))           return CODEC_NOT_C3D;
 *     if (!c3d_chunk_validate(in, in_len))     return CODEC_CORRUPT;
 *     c3d_decoder_chunk_decode(dec, in, in_len, ctx, out);
 *
 * The library is fatal-on-error by design (c3d_panic → abort).  A caller
 * who wants exception-like semantics (continue after a bad chunk) should
 * (a) gate the decode on c3d_chunk_validate(), and (b) install a panic
 * hook that `longjmp`s back to a known setjmp target BEFORE calling the
 * decode path.  The longjmp-after-panic pattern is documented as UB
 * against the library's own internal assertions, but it does work for
 * the specific case of a deliberately-corrupt encoded byte stream, as
 * long as the panic hook never returns. */
bool c3d_chunk_validate(const uint8_t *in, size_t in_len);

/* ─── stateless chunk codec ──────────────────────────────────────────────── */

typedef struct c3d_ctx c3d_ctx;

/* Reusable encoder/decoder scratch.  Owns ~115 MiB (encoder) / ~80 MiB
 * (decoder) of buffers; create once per thread, reuse across many chunks
 * to avoid alloc/free churn (50-100 ms/chunk saved).
 *
 * Thread-safety: a c3d_encoder / c3d_decoder instance is NOT thread-safe.
 * The internal scratch arenas are mutated on every call.  For multi-threaded
 * encode/decode, allocate one encoder / decoder per worker thread.  The
 * stateless functions (c3d_chunk_encode etc.) are safe to call concurrently
 * from any thread — they malloc scratch per call, which is what you pay to
 * avoid the per-thread-context dance. */
typedef struct c3d_encoder c3d_encoder;
typedef struct c3d_decoder c3d_decoder;

c3d_encoder *c3d_encoder_new(void);
void         c3d_encoder_free(c3d_encoder *);
c3d_decoder *c3d_decoder_new(void);
void         c3d_decoder_free(c3d_decoder *);

/* Reset inter-chunk prediction state (Morton-neighbour LL_5 cache +
 * rate-control warm-start).  Call at the start of a fresh shard or
 * when switching to non-sequential decode. */
void c3d_encoder_reset_inter_chunk(c3d_encoder *);
void c3d_decoder_reset_inter_chunk(c3d_decoder *);

/* Enable / disable inter-chunk Morton-neighbour LL_5 prediction on an
 * encoder.  Disabled by default — the stateless chunk API is
 * byte-deterministic across reused-vs-fresh encoders.  Enable inside
 * a shard-level sequential encode loop; the encoder will use its
 * running prev_ll5 as a prediction reference for each subsequent
 * chunk.  c3d_shard_encode_all enables this automatically. */
void c3d_encoder_enable_inter_chunk(c3d_encoder *, bool enabled);

/* Reusable-context variants — same semantics as the stateless calls below
 * but allocate-once-reuse-many.  Recommended for any caller doing >1 chunk. */
size_t c3d_encoder_chunk_encode(c3d_encoder *, const uint8_t *in,
                                float target_ratio, const c3d_ctx *ctx,
                                uint8_t *out, size_t out_cap);
size_t c3d_encoder_chunk_encode_at_q(c3d_encoder *, const uint8_t *in,
                                     float q, const c3d_ctx *ctx,
                                     uint8_t *out, size_t out_cap);
void   c3d_decoder_chunk_decode(c3d_decoder *, const uint8_t *in, size_t in_len,
                                const c3d_ctx *ctx, uint8_t *out);
void   c3d_decoder_chunk_decode_lod(c3d_decoder *, const uint8_t *in, size_t in_len,
                                    uint8_t lod, const c3d_ctx *ctx, uint8_t *out);

/* Multi-chunk batched encode — thin wrapper around c3d_encoder_chunk_encode
 * that runs n_chunks encodes back-to-back on the same encoder and writes
 * the per-chunk output sizes to out_sizes[].  Same output as calling the
 * single-chunk API n_chunks times (warm-start kicks in after the first
 * chunk).  Exists so callers can express shard-style work in one call and
 * so future versions can overlap DWT / rANS across chunks without changing
 * the API shape.  All `inputs` must be C3D_ALIGN-aligned, all `outs` must
 * each point to a buffer of at least C3D_CHUNK_ENCODE_MAX_SIZE bytes. */
void c3d_encoder_chunks_encode(c3d_encoder *e,
                               const uint8_t *const *inputs,
                               size_t n_chunks,
                               float target_ratio, const c3d_ctx *ctx,
                               uint8_t *const *outs,
                               size_t *out_sizes);

/* Multi-chunk batched decode — analogous to the encode batch.  Each chunk's
 * encoded size is taken from in_sizes[i]; each out must be 256³ bytes. */
void c3d_decoder_chunks_decode(c3d_decoder *d,
                               const uint8_t *const *ins,
                               const size_t *in_sizes,
                               size_t n_chunks,
                               const c3d_ctx *ctx,
                               uint8_t *const *outs);

size_t c3d_chunk_encode_max_size(void);   /* returns C3D_CHUNK_ENCODE_MAX_SIZE */

/* target_ratio must be > 1.0; ctx may be NULL (→ SELF chunk), else EXTERNAL.
 * Returns bytes written.  Aborts if out_cap < C3D_CHUNK_ENCODE_MAX_SIZE. */
size_t c3d_chunk_encode(const uint8_t *in,
                        float target_ratio,
                        const c3d_ctx *ctx,
                        uint8_t *out, size_t out_cap);

/* Bypass rate control; use the given quantizer scalar q ∈ [2^-6, 2^12].
 * Useful for R-D sweeps and deterministic per-test encodes. */
size_t c3d_chunk_encode_at_q(const uint8_t *in,
                             float q,
                             const c3d_ctx *ctx,
                             uint8_t *out, size_t out_cap);

/* LOD 0 decode — writes 256^3 u8 into out. */
void c3d_chunk_decode(const uint8_t *in, size_t in_len,
                      const c3d_ctx *ctx,
                      uint8_t *out);

/* LOD decode, lod ∈ 0..5.  out must be sized (256>>lod)^3 bytes. */
void c3d_chunk_decode_lod(const uint8_t *in, size_t in_len, uint8_t lod,
                          const c3d_ctx *ctx,
                          uint8_t *out);

/* Post-decode 2× box-average downsample for caller-side pyramids.
 * side ∈ {256, 128, 64, 32, 16}.  Writes (side/2)^3 to out.
 * Intentionally does not match the codec's internal wavelet-synthesis LODs. */
void c3d_downsample_chunk_2x(const uint8_t *in, uint32_t side, uint8_t *out);

/* ─── context block (.c3dx) ──────────────────────────────────────────────── */

size_t   c3d_ctx_max_size(void);              /* upper bound: 65535           */
size_t   c3d_ctx_serialized_size(const c3d_ctx *);  /* exact bytes for this ctx */

/* Always deep-copies.  Caller may free `in` immediately. */
c3d_ctx *c3d_ctx_parse(const uint8_t *in, size_t in_len);

size_t   c3d_ctx_serialize(const c3d_ctx *, uint8_t *out, size_t out_cap);
void     c3d_ctx_id(const c3d_ctx *, uint8_t out[16]);
void     c3d_ctx_free(c3d_ctx *);

/* Builder (single-threaded). */
typedef struct c3d_ctx_builder c3d_ctx_builder;

c3d_ctx_builder *c3d_ctx_builder_new(void);
void             c3d_ctx_builder_observe_chunk(c3d_ctx_builder *,
                                               const uint8_t *in);
/* §T13 — supply a per-subband dead-zone ratio for emission in the produced
 * ctx.  Must be called before c3d_ctx_builder_finish.  Typically used by
 * training tools that run a grid search.  Array length is C3D_N_SUBBANDS=36.
 * Without this call, builder emits no DZ_RATIO TLV and encoders fall back
 * to kind-based c3d_dz_ratio_for_kind(). */
void             c3d_ctx_builder_set_dz_ratio(c3d_ctx_builder *,
                                              const float dz_ratio[36]);
/* §T15 — supply a per-subband Laplacian α for dequantization.  Same
 * conventions as set_dz_ratio: call before finish(), 36-long array.
 * Without this call, decoder falls back to the per-chunk alpha byte or
 * the kind-based default c3d_default_alpha(). */
void             c3d_ctx_builder_set_laplacian_alpha(c3d_ctx_builder *,
                                                     const float alpha[36]);
c3d_ctx         *c3d_ctx_builder_finish(c3d_ctx_builder *, bool include_freq_tables);
void             c3d_ctx_builder_free(c3d_ctx_builder *);

/* ─── shard (in-memory parsed form) ──────────────────────────────────────── */

typedef struct c3d_shard c3d_shard;

c3d_shard *c3d_shard_new(const uint32_t origin[3], uint8_t shard_lod);

/* Non-copy: shard holds pointers into `in`; `in` must outlive the shard.
 *   - Mutating ops (put_chunk, mark_zero, set_ctx) allocate their own memory.
 *   - Returned pointers from c3d_shard_chunk_bytes for non-mutated chunks
 *     point into the original `in`; for mutated chunks, into shard memory. */
c3d_shard *c3d_shard_parse     (const uint8_t *in, size_t in_len);
c3d_shard *c3d_shard_parse_copy(const uint8_t *in, size_t in_len);

size_t     c3d_shard_max_serialized_size(const c3d_shard *);
size_t     c3d_shard_serialize(const c3d_shard *, uint8_t *out, size_t out_cap);
void       c3d_shard_free(c3d_shard *);

c3d_chunk_state c3d_shard_chunk_state(const c3d_shard *,
                                      uint32_t cx, uint32_t cy, uint32_t cz);
uint32_t        c3d_shard_chunk_count(const c3d_shard *, c3d_chunk_state);

/* Returns pointer into shard memory and size.  Panics on ABSENT or ZERO. */
const uint8_t  *c3d_shard_chunk_bytes(const c3d_shard *,
                                      uint32_t cx, uint32_t cy, uint32_t cz,
                                      size_t *out_size);

void c3d_shard_put_chunk  (c3d_shard *,
                           uint32_t cx, uint32_t cy, uint32_t cz,
                           const uint8_t *in, size_t in_len);
void c3d_shard_mark_zero  (c3d_shard *,
                           uint32_t cx, uint32_t cy, uint32_t cz);

/* Embedded context block (0 or 1 per shard).  set_ctx deep-copies. */
void             c3d_shard_set_ctx(c3d_shard *, const c3d_ctx *);
const c3d_ctx   *c3d_shard_ctx    (const c3d_shard *);

/* Convenience: use the shard's embedded ctx (if any) for encode/decode. */
void c3d_shard_encode_chunk     (c3d_shard *,
                                 uint32_t cx, uint32_t cy, uint32_t cz,
                                 const uint8_t *in, float target_ratio);
void c3d_shard_decode_chunk     (const c3d_shard *,
                                 uint32_t cx, uint32_t cy, uint32_t cz,
                                 uint8_t *out);
void c3d_shard_decode_chunk_lod (const c3d_shard *,
                                 uint32_t cx, uint32_t cy, uint32_t cz,
                                 uint8_t lod, uint8_t *out);

/* ─── shard-level batch helpers ──────────────────────────────────────────── *
 *
 * These wrap c3d_shard_set_ctx + per-chunk encode/decode in a single call so
 * callers who have the full shard up-front get the inter-chunk wins (per-
 * shard freq tables, LL_5 reference) without having to wire ctx training
 * themselves.  The per-chunk APIs above remain fully supported — these
 * helpers are pure additions for the shard-level workflow.
 *
 * c3d_shard_auto_train_ctx: build a ctx from a sample of training chunks
 *   and embed it in the shard.  Subsequent c3d_shard_encode_chunk / decode
 *   calls automatically use the embedded ctx.  When the training chunks
 *   are spatially close (same shard), the corpus-average LL_5 reference
 *   inside the ctx tightens the LL_5 entropy by 5-15 %.
 */
void c3d_shard_auto_train_ctx(c3d_shard *,
                              const uint8_t *const *training_chunks,
                              size_t n_training);

/* c3d_shard_encode_all: auto-train the ctx from the input chunks (or use
 * the shard's existing ctx), then encode every chunk.  `coords[i]` holds
 * (cx, cy, cz) for chunk i.  All chunks must be valid 256³ u8 buffers
 * (32-byte aligned).  After return the shard contains all chunks; serialise
 * with c3d_shard_serialize. */
void c3d_shard_encode_all(c3d_shard *,
                          const uint8_t *const *chunks,
                          const uint32_t (*coords)[3],
                          size_t n_chunks,
                          float target_ratio);

#ifdef __cplusplus
}
#endif

#endif /* C3D_H */
