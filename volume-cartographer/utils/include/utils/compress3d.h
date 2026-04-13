#ifndef COMPRESS3D_H
#define COMPRESS3D_H

#include <stdint.h>
#include <stddef.h>

#define C3D_BLOCK_SIZE 32
#define C3D_BLOCK_VOXELS (C3D_BLOCK_SIZE * C3D_BLOCK_SIZE * C3D_BLOCK_SIZE)

/* Compressed data returned by c3d_compress. Caller must free `data`. */
typedef struct {
    uint8_t *data;
    size_t   size;
} c3d_compressed_t;

/*
 * Compress a 32^3 grayscale 8-bit volume.
 *   input:   32768 bytes, row-major (x fastest, then y, then z)
 *   quality: 1 (smallest/worst) to 100 (largest/best), 101 = lossless
 *            Clamped to [1, 101]. NULL input returns {NULL, 0}.
 * Returns compressed blob. Caller must free result.data.
 * On error, returns {NULL, 0}. Thread-safe (no shared state).
 */
c3d_compressed_t c3d_compress(const uint8_t *input, int quality);

/*
 * Decompress back to NxNxN grayscale 8-bit volume.
 * The size is read from the compressed header.
 *   compressed: blob from c3d_compress
 *   output:     must point to at least size^3 bytes
 * Returns 0 on success, -1 on error. NULL params return -1.
 * Thread-safe (no shared state).
 */
int c3d_decompress(const uint8_t *compressed, size_t compressed_size, uint8_t *output);

/*
 * Read the cube size from a compressed blob without decompressing.
 * Returns the size (8..256) on success, -1 on error. NULL-safe.
 */
int c3d_get_size(const uint8_t *compressed, size_t compressed_size);

/*
 * Returns the maximum possible compressed size for a 32^3 volume.
 * Caller can pre-allocate a buffer of this size.
 */
size_t c3d_compress_bound(void);

/*
 * Compress into a caller-provided buffer (always uses size=32).
 *   output must be at least c3d_compress_bound() bytes.
 *   quality: clamped to [1, 101]. NULL input/output returns 0.
 * Returns actual compressed size, or 0 on error. Thread-safe.
 */
size_t c3d_compress_to(const uint8_t *input, int quality, uint8_t *output, size_t output_cap);

/*
 * Decompress from compressed data into caller-provided output buffer.
 * Returns 0 on success, -1 on error. NULL params return -1.
 */
int c3d_decompress_to(const uint8_t *compressed, size_t compressed_size, uint8_t *output);

/* Compress a shard of multiple 32^3 chunks with inter-chunk DC delta coding.
 * chunks: array of pointers to 32768-byte volumes, in raster order (x fastest)
 * nx, ny, nz: number of chunks along each axis (e.g., 4,4,4 for 128^3)
 * quality: 1-101 (clamped). NULL chunks returns {NULL, 0}.
 * Returns compressed blob. Caller must free result.data.
 */
c3d_compressed_t c3d_compress_shard(const uint8_t **chunks, int nx, int ny, int nz, int quality, int num_threads);

/* Decompress a shard back to individual chunks.
 * chunks: array of pointers to pre-allocated 32768-byte buffers
 * Returns 0 on success, -1 on error. NULL params return -1.
 */
int c3d_decompress_shard(const uint8_t *compressed, size_t compressed_size,
                          uint8_t **chunks, int nx, int ny, int nz, int num_threads);

/* Compress multiple 32^3 chunks in parallel using a thread pool.
 * inputs: array of pointers to 32768-byte volumes
 * count: number of chunks
 * quality: 1-101 (clamped). NULL inputs/outputs returns -1.
 * num_threads: number of worker threads (0 = auto-detect CPU count)
 * outputs: array of c3d_compressed_t results (caller must free each .data)
 * Returns 0 on success, -1 on error.
 */
int c3d_compress_batch(const uint8_t **inputs, int count, int quality,
                       int num_threads, c3d_compressed_t *outputs);

/* Decompress multiple chunks in parallel.
 * compressed: array of {data, size} pairs
 * count: number of chunks
 * num_threads: number of worker threads (0 = auto-detect)
 * outputs: array of pointers to pre-allocated 32768-byte buffers
 * Returns 0 on success, -1 on error. NULL params return -1.
 */
int c3d_decompress_batch(const c3d_compressed_t *compressed, int count,
                         int num_threads, uint8_t **outputs);

/* Compute 3D SSIM between two 32^3 volumes. Returns value in [0, 1].
 * NULL params return 0.0. Thread-safe. */
double c3d_ssim(const uint8_t *original, const uint8_t *reconstructed);

/* Opaque workspace (~264KB). Amortizes allocation across compress/decompress calls.
 * Thread safety: each thread needs its own workspace. Not safe to share. */
typedef struct c3d_workspace c3d_workspace_t;

/* Create a reusable workspace. Returns NULL on allocation failure. */
c3d_workspace_t *c3d_workspace_create(void);

/* Free a workspace. NULL-safe. */
void c3d_workspace_free(c3d_workspace_t *ws);

/* Compress using pre-allocated workspace (avoids internal malloc).
 * quality: clamped to [1, 101]. NULL params return 0. Not thread-safe for same ws. */
size_t c3d_compress_ws(const uint8_t *input, int quality, uint8_t *output, size_t output_cap, c3d_workspace_t *ws);

/* Decompress using pre-allocated workspace.
 * NULL params return -1. Not thread-safe for same ws. */
int c3d_decompress_ws(const uint8_t *compressed, size_t compressed_size, uint8_t *output, c3d_workspace_t *ws);

/* Error codes */
#define C3D_ERR_OK        0
#define C3D_ERR_FORMAT   (-1)
#define C3D_ERR_CHECKSUM (-2)
#define C3D_ERR_ALLOC    (-3)  /* Memory allocation failed */
#define C3D_ERR_PARAM    (-4)  /* Invalid parameter (NULL pointer, bad dimensions) */
#define C3D_ERR_SIZE     (-5)  /* Buffer too small */

/* Header flags (stored in header byte 7) */
#define C3D_FLAG_INTERLEAVED 0x02
#define C3D_FLAG_HAS_CRC  0x04
#define C3D_FLAG_HAS_META 0x08
#define C3D_FLAG_SPARSE   0x10
#define C3D_FLAG_COEFF_PRED 0x20

/* Metadata for medical/scientific volumes */
typedef struct {
    float voxel_size[3];     /* Voxel dimensions in mm (0 = unspecified) */
    float origin[3];         /* Volume origin in world coordinates */
    uint16_t modality;       /* 0=unknown, 1=CT, 2=MRI, 3=XRay */
    uint16_t bits_per_voxel; /* Original bit depth (8 for our case) */
} c3d_metadata_t;

/*
 * Compress with metadata and CRC32 integrity check.
 * meta may be NULL for no metadata (CRC is still added).
 * quality: clamped to [1, 101]. NULL input returns {NULL, 0}.
 */
c3d_compressed_t c3d_compress_meta(const uint8_t *input, int quality, const c3d_metadata_t *meta);

/*
 * Read metadata from a compressed blob without decompressing.
 * Returns 0 on success, -1 if no metadata present or format error.
 */
int c3d_get_metadata(const uint8_t *compressed, size_t size, c3d_metadata_t *meta);

/* Transform type flags (stored in header byte 5) */
#define C3D_TRANSFORM_DCT     0
#define C3D_TRANSFORM_WAVELET 1

/* Compression mode flags (stored in header byte 6) */
#define C3D_MODE_LOSSY    0
#define C3D_MODE_LOSSLESS 1

/*
 * Compress using CDF 5/3 (Le Gall) wavelet transform instead of DCT.
 * Better for sharp edges (less ringing). Integer-based, faster than DCT.
 * Same interface as c3d_compress. Decompress auto-detects transform type.
 * quality: clamped to [1, 101]. NULL input returns {NULL, 0}. Thread-safe.
 */
c3d_compressed_t c3d_compress_wavelet(const uint8_t *input, int quality);

/*
 * Wavelet version of c3d_compress_to.
 * quality: clamped to [1, 101]. NULL input/output returns 0.
 */
size_t c3d_compress_wavelet_to(const uint8_t *input, int quality, uint8_t *output, size_t output_cap);

/*
 * Compress with progressive bitstream. Can be truncated at any byte
 * boundary for progressive decode (lower quality but valid output).
 * Coefficients are encoded in zigzag order using raw VLQ+RLE (no rANS).
 * quality: clamped to [1, 101]. NULL input returns {NULL, 0}. Thread-safe.
 */
c3d_compressed_t c3d_compress_progressive(const uint8_t *input, int quality);

/*
 * Decompress progressive bitstream. Handles truncated data gracefully —
 * missing coefficients are treated as zero. Returns 0 on success, -1 on error.
 */
int c3d_decompress_progressive(const uint8_t *compressed, size_t compressed_size, uint8_t *output);

/* Compress with automatic transform selection (DCT or wavelet).
 * Analyzes edge strength to pick the best transform per chunk.
 * High edge strength -> wavelet (less ringing), low -> DCT (better compaction).
 * quality: clamped to [1, 101]. NULL input returns {NULL, 0}. Thread-safe. */
c3d_compressed_t c3d_compress_auto(const uint8_t *input, int quality);

/* Compress to a target PSNR (dB). Binary searches quality 1-100.
 * NULL input returns {NULL, 0}. Thread-safe. */
c3d_compressed_t c3d_compress_target_psnr(const uint8_t *input, double target_psnr);

/* Compress to a target file size (bytes). Binary searches quality 1-100.
 * NULL input returns {NULL, 0}. Thread-safe. */
c3d_compressed_t c3d_compress_target_size(const uint8_t *input, size_t target_size);

/* Compress to a target SSIM (0.0-1.0). Binary searches quality 1-100.
 * NULL input returns {NULL, 0}. Thread-safe. */
c3d_compressed_t c3d_compress_target_ssim(const uint8_t *input, double target_ssim);

/* Decompress a single chunk from a shard by chunk index (0-based, raster order).
 * output: must point to at least 32768 bytes.
 * Returns 0 on success, -1 on error.
 */
int c3d_decompress_shard_chunk(const uint8_t *compressed, size_t compressed_size,
                                int chunk_index, uint8_t *output);

/* Get the number of chunks in a shard. Returns count or -1 on error. */
int c3d_shard_chunk_count(const uint8_t *compressed, size_t compressed_size);

/* ── Memory-mapped shard access (POSIX only) ── */
#ifndef _WIN32

/* Opaque handle for a memory-mapped shard file. */
typedef struct c3d_shard_map c3d_shard_map_t;

/* Open a shard file with mmap for random-access chunk reading.
 * Returns handle on success, NULL on error. */
c3d_shard_map_t *c3d_shard_mmap_open(const char *path);

/* Get chunk count from mapped shard. */
int c3d_shard_mmap_chunk_count(const c3d_shard_map_t *map);

/* Decompress a single chunk from the mapped shard.
 * output must point to at least 32768 bytes. Returns 0 on success. */
int c3d_shard_mmap_read_chunk(const c3d_shard_map_t *map, int chunk_index, uint8_t *output);

/* Close and unmap. */
void c3d_shard_mmap_close(c3d_shard_map_t *map);

#endif /* !_WIN32 */

/* ── Streaming shard writer ── */

typedef struct c3d_shard_writer c3d_shard_writer_t;

/* Begin writing a shard. Writes header placeholder. */
c3d_shard_writer_t *c3d_shard_writer_open(const char *path, int nx, int ny, int nz, int quality);

/* Add the next chunk (must be added in raster order). Returns 0 on success. */
int c3d_shard_writer_add_chunk(c3d_shard_writer_t *w, const uint8_t *chunk);

/* Finalize: write the offset table and close. Returns 0 on success. */
int c3d_shard_writer_finish(c3d_shard_writer_t *w);

/* ── Streaming API ── */

/* Callback for receiving compressed chunk data. */
typedef void (*c3d_write_cb)(const uint8_t *data, size_t size, int chunk_index, void *userdata);

/* Callback for receiving decompressed chunk data. */
typedef void (*c3d_read_cb)(uint8_t *output, int chunk_index, void *userdata);

typedef struct c3d_stream c3d_stream_t;

/* Create a streaming compressor.
 * Compresses chunks as they arrive via c3d_stream_push.
 * Calls write_cb for each compressed chunk.
 * quality: clamped to [1, 101]. NULL write_cb returns NULL.
 * num_threads: worker thread count (0 = auto)
 */
c3d_stream_t *c3d_stream_compress_create(int quality, int num_threads,
                                          c3d_write_cb write_cb, void *userdata);

/* Push a 32^3 chunk into the stream. Non-blocking if worker threads available.
 * Returns 0 on success, -1 on error. */
int c3d_stream_push(c3d_stream_t *stream, const uint8_t *chunk);

/* Flush remaining chunks and wait for completion. */
int c3d_stream_flush(c3d_stream_t *stream);

/* Destroy stream. */
void c3d_stream_free(c3d_stream_t *stream);

/* ── ROI (Region of Interest) coding ── */

typedef struct {
    int x0, y0, z0;  /* ROI start (inclusive) */
    int x1, y1, z1;  /* ROI end (exclusive) */
    int roi_quality;  /* Quality for the ROI region (1-100) */
} c3d_roi_t;

/* Compress with ROI: background at `quality`, ROI region at `roi.roi_quality`.
 * quality: clamped to [1, 101]. NULL input/roi returns {NULL, 0}.
 * Returns compressed blob. Caller must free result.data.
 */
c3d_compressed_t c3d_compress_roi(const uint8_t *input, int quality, const c3d_roi_t *roi);

/* Apply 3D deblocking filter along chunk boundaries.
 * volume: pointer to the full assembled volume (all chunks combined)
 * vx, vy, vz: full volume dimensions in voxels (must be multiples of 32)
 * strength: 0.0 (no filtering) to 1.0 (maximum smoothing)
 * Only smooths across 32-voxel chunk boundaries, not within chunks.
 */
void c3d_deblock(uint8_t *volume, int vx, int vy, int vz, float strength);

#endif
