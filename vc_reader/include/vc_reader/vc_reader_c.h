/* vc_reader C API: a render engine for volume viewers in any language.
 *
 * Opens a volume's pyramid (local, HTTP, S3, SFTP; disk and memory caches), samples
 * it along planes and tifxyz/vertex-grid surfaces, draws frames progressively from
 * the caches while missing chunks are read in the background, reduces layers to
 * composites and turns samples into display pixels. Link libvc_reader_c.
 *
 * Conventions
 *  - Every function that can fail returns vcr_status; vcr_last_error() then holds a
 *    message for the calling thread until its next failing call.
 *  - Handles (vcr_volume, vcr_surface, vcr_pending) are released by their _release
 *    function. A pending read set keeps the volumes and surfaces it reads alive, so
 *    handles may be released while reads are outstanding.
 *  - Volumes and surfaces are immutable after creation and safe to use from several
 *    threads at once. A vcr_pending is used by one thread at a time.
 *  - Outputs go to caller-provided buffers; nothing allocated here is freed by callers
 *    except through the _release functions.
 *  - Structs with a struct_size field are initialised by their _init function; later
 *    versions only append fields.
 *  - Points and geometry are XYZ in the volume's level-0 voxels; array shapes are ZYX.
 */
#ifndef VC_READER_C_H
#define VC_READER_C_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(VCR_C_BUILDING)
#    define VCR_API __declspec(dllexport)
#  else
#    define VCR_API __declspec(dllimport)
#  endif
#else
#  define VCR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VCR_C_ABI_VERSION 1

typedef enum vcr_status {
    VCR_OK = 0,
    VCR_ERROR_INVALID_ARGUMENT = 1,
    VCR_ERROR_RUNTIME = 2,           /* I/O, decoding, a remote that failed */
    VCR_ERROR_OUT_OF_MEMORY = 3,
    VCR_ERROR_CANCELLED = 4,
    VCR_ERROR_BUFFER_TOO_SMALL = 5
} vcr_status;

/* VCR_C_ABI_VERSION of the loaded library. */
VCR_API int vcr_abi_version(void);
/* Nonzero when the library was built with CUDA and a device is available. */
VCR_API int vcr_has_cuda(void);
/* The calling thread's last error message; never NULL. */
VCR_API const char* vcr_last_error(void);

typedef struct vcr_volume vcr_volume;
typedef struct vcr_surface vcr_surface;
typedef struct vcr_pending vcr_pending;

/* ---- volumes ------------------------------------------------------------------- */

typedef struct vcr_open_options {
    size_t struct_size;
    /* Directories holding (and receiving) disk caches; "vc3d" names VC3D's cache root. */
    const char* const* cache_dirs;
    size_t cache_dir_count;
    int cache_writes;            /* nonzero (default): fetched chunks are written to the cache */
    int offline;                 /* remote volumes: metadata and chunks only from the caches */
    int64_t memory_cache_bytes;  /* decoded chunks kept in memory, shared by the levels; default 2 GiB */
    int64_t gpu_cache_bytes;     /* >0: uint8 levels also sample on the GPU, from one brick cache of
                                    this many bytes that the volume's levels share (least recently
                                    used bricks of any level make room) */
    size_t threads;              /* reader threads per level; 0: the reader's default */
} vcr_open_options;

VCR_API void vcr_open_options_init(vcr_open_options* options);
/* Opens every pyramid level of a Zarr / OME-Zarr volume (or a single array as level 0).
 * Levels that fail to open are absent. options may be NULL for the defaults. */
VCR_API vcr_status vcr_volume_open(const char* location, const vcr_open_options* options, vcr_volume** out);
VCR_API void vcr_volume_release(vcr_volume* volume);
/* Bytes of the volume's GPU brick cache in use, all levels together (0 without one). */
VCR_API int64_t vcr_volume_gpu_resident_bytes(const vcr_volume* volume);

typedef struct vcr_level_info {
    size_t struct_size;
    int present;
    int64_t shape[3];         /* ZYX */
    int64_t chunks[3];        /* ZYX */
    char dtype[16];           /* numpy typestr, e.g. "|u1" */
    double voxel_extent[3];   /* level-0 voxels one index spans, ZYX */
    int gpu;                  /* nonzero while this level samples on the GPU */
} vcr_level_info;

VCR_API void vcr_level_info_init(vcr_level_info* info);
/* Number of pyramid levels, absent ones included (level numbers are their indices). */
VCR_API int vcr_volume_level_count(const vcr_volume* volume);
VCR_API vcr_status vcr_volume_level_info(const vcr_volume* volume, int level, vcr_level_info* out);

/* The level a view at pixels_per_voxel (view pixels per level-0 voxel) samples:
 * the coarsest present level whose voxels are at most 1/(pixels_per_voxel*bias). */
VCR_API int vcr_level_for_view(const vcr_volume* volume, float pixels_per_voxel, float quality_bias);
/* How many coarser levels may fill a width x height view's holes (at most maximum). */
VCR_API int vcr_fallback_levels(const vcr_volume* volume, int start_level, int width, int height,
                                float pixels_per_voxel, int maximum);

/* Progress of a prefill; return nonzero to cancel. */
typedef int (*vcr_progress_fn)(void* user, size_t resolved, size_t total);
/* Reads every chunk of a level through its caches (warms the disk cache).
 * VCR_ERROR_CANCELLED when the callback asked to stop; completed chunks stay cached. */
VCR_API vcr_status vcr_volume_prefill(vcr_volume* volume, int level, vcr_progress_fn progress, void* user);

/* ---- surfaces ------------------------------------------------------------------ */

typedef enum vcr_interpolation { VCR_INTERPOLATION_BICUBIC = 0, VCR_INTERPOLATION_LINEAR = 1 } vcr_interpolation;

typedef struct vcr_surface_options {
    size_t struct_size;
    int interpolation;    /* vcr_interpolation; default bicubic */
    float spacing[3];     /* physical size of a voxel along X, Y, Z (normals follow it); default 1 */
    int flip_normals;
} vcr_surface_options;

VCR_API void vcr_surface_options_init(vcr_surface_options* options);
/* A surface from rows x cols XYZ vertices (row_stride floats between rows; 0 for
 * cols*3). valid may be NULL: then nonfinite vertices are invalid. The data is copied. */
VCR_API vcr_status vcr_surface_create(const float* xyz, int rows, int cols, size_t row_stride,
                                      const uint8_t* valid, const vcr_surface_options* options,
                                      vcr_surface** out);
/* A tifxyz directory (x/y/z.tif, meta.json, mask.tif when present). scale (rows,
 * cols per full-resolution pixel, from meta.json) is written when not NULL. */
VCR_API vcr_status vcr_surface_open_tifxyz(const char* location, const vcr_surface_options* options,
                                           vcr_surface** out, double scale[2]);
VCR_API void vcr_surface_shape(const vcr_surface* surface, int* rows, int* cols);
VCR_API void vcr_surface_release(vcr_surface* surface);

/* A raster over a surface's grid: pixel (i, j) at column x0 + i*sx + j*kx,
 * row y0 + i*ky + j*sy. */
typedef struct vcr_raster {
    double x0, y0, sx, sy;
    int width, height;
    double kx, ky;
} vcr_raster;

/* Positions, normals and tangents at n (column, row) grid positions; NaN where the
 * surface has none. Any output may be NULL. */
VCR_API vcr_status vcr_surface_points_at(const vcr_surface* surface, const float* xy, size_t n, float* xyz,
                                         float* normals, float* column_tangents, float* row_tangents);
/* Positions and normals at every raster pixel, row-major, 3 floats each. */
VCR_API vcr_status vcr_surface_resample(const vcr_surface* surface, const vcr_raster* raster, float* xyz,
                                        float* normals);
/* View pixels per voxel of a raster over the surface (its most magnified direction);
 * 0 when the surface has no point under the raster. */
VCR_API float vcr_surface_pixels_per_voxel(const vcr_surface* surface, const vcr_raster* raster);

/* ---- frames and requests ------------------------------------------------------- */

/* A plane raster: pixel (i, j) at origin + i*dx + j*dy; layers move along normal. */
typedef struct vcr_plane {
    float origin[3], dx[3], dy[3], normal[3];
    int width, height;
} vcr_plane;

/* What a frame samples: plane, or surface with raster (set exactly one of plane, surface). */
typedef struct vcr_frame {
    const vcr_plane* plane;
    const vcr_surface* surface;
    vcr_raster raster;
} vcr_frame;

typedef enum vcr_composite_method {
    VCR_COMPOSITE_MEAN = 0,
    VCR_COMPOSITE_MAX = 1,
    VCR_COMPOSITE_MIN = 2,
    VCR_COMPOSITE_MEDIAN = 3,
    VCR_COMPOSITE_MIN_ABS = 4,
    VCR_COMPOSITE_ALPHA = 5,
    VCR_COMPOSITE_BEER_LAMBERT = 6
} vcr_composite_method;

typedef struct vcr_composite {
    int method;  /* vcr_composite_method */
    float iso_cutoff;
    float alpha_min, alpha_max, alpha_opacity, alpha_cutoff;
    float extinction, emission, ambient;
} vcr_composite;

VCR_API void vcr_composite_init(vcr_composite* composite, int method);

typedef struct vcr_request {
    size_t struct_size;
    const vcr_volume* volume;
    int start_level;       /* the view's level */
    int fallback_levels;   /* coarser levels that fill its holes */
    int order;             /* 0 nearest, 1 linear (default) */
    const float* offsets;  /* layers along the normal; NULL: one layer at 0 */
    size_t offset_count;
    const uint8_t* pixels; /* optional per-pixel mask: only nonzero pixels are sampled */
    const vcr_composite* composite;  /* optional: reduce each pixel's layers to one value */
    int cached_only;       /* vcr_sample only: read the caches alone */
} vcr_request;

VCR_API void vcr_request_init(vcr_request* request);

#define VCR_LEVEL_NONE (-2)     /* no layer of the pixel had a point */
#define VCR_LEVEL_MISSING (-1)  /* a layer's chunks were in no level's caches */

typedef struct vcr_result {
    size_t struct_size;
    /* Caller buffers. values: width*height*layers floats, layer-major (NaN: no value);
     * levels: width*height pyramid level numbers that supplied each pixel (the coarsest
     * of its layers), or VCR_LEVEL_NONE / VCR_LEVEL_MISSING. levels may be NULL. */
    float* values;
    size_t values_capacity;
    int8_t* levels;
    size_t levels_capacity;
    /* Written: */
    int width, height, layers;
    int gpu_failed;  /* a GPU pass failed and the frame was sampled on the CPU; the
                        volume samples on the CPU from now on (vcr_last_error says why) */
} vcr_result;

VCR_API void vcr_result_init(vcr_result* result);
/* Floats a request's values need (width*height*layers). */
VCR_API size_t vcr_request_value_count(const vcr_frame* frame, const vcr_request* request);

/* Samples a frame, reading missing chunks (blocking) unless request->cached_only. */
VCR_API vcr_status vcr_sample(const vcr_frame* frame, const vcr_request* request, vcr_result* result);

/* Progressive frames: sample from the caches alone, now; pending (may be NULL)
 * collects the reads of what the frame lacked: a preview from the coarsest fallback
 * level, then the start level. Run them (e.g. on a worker thread) with
 * vcr_pending_load, then draw the frame again. */
VCR_API vcr_status vcr_sample_cached(const vcr_frame* frame, const vcr_request* request, vcr_pending* pending,
                                     vcr_result* result);
VCR_API vcr_status vcr_pending_create(vcr_pending** out);
VCR_API int vcr_pending_empty(const vcr_pending* pending);
VCR_API void vcr_pending_clear(vcr_pending* pending);
/* Called after the preview and after the start level were read; return 0 to stop. */
typedef int (*vcr_part_fn)(void* user);
VCR_API vcr_status vcr_pending_load(const vcr_pending* pending, vcr_part_fn after_part, void* user);
VCR_API void vcr_pending_release(vcr_pending* pending);

/* ---- display ------------------------------------------------------------------- */

/* Floats to bytes (truncating, clamped; NaN to 0); coverage (may be NULL) is 1
 * where the input is finite, else 0. */
VCR_API void vcr_to_uint8(const float* input, size_t n, uint8_t* output, uint8_t* coverage);
/* 256-entry ARGB lookup tables. */
VCR_API void vcr_window_level_lut(uint32_t lut[256], float low, float high);
VCR_API void vcr_palette_lut(uint32_t lut[256], const uint32_t palette[256], float low, float high, int black_zero);
VCR_API void vcr_overlay_lut(uint32_t lut[256], const uint32_t color[256], float low, float high,
                             int intensity_as_alpha, int premultiplied);
VCR_API void vcr_apply_lut(const uint8_t* input, size_t rows, size_t cols, size_t input_stride,
                           const uint32_t lut[256], uint32_t* output, size_t output_stride);
VCR_API void vcr_blend(const uint32_t* base, const uint32_t* overlay, size_t n, float opacity, uint32_t* output);
/* Reduce `layers` rows of n values each (layer-major) to one value per column. */
VCR_API vcr_status vcr_composite_reduce(const float* values, size_t layers, size_t n, const vcr_composite* composite,
                                       float* output);

/* Volumetric slab compositing: front-to-back over uint8 layers through colour and
 * opacity tables, orthographic or perspective (see vc_reader/slab.hpp). */
typedef struct vcr_slab_layer {
    const uint8_t* values;
    const uint8_t* coverage;
    size_t value_stride, coverage_stride;
} vcr_slab_layer;

typedef struct vcr_slab_projection {
    int perspective;
    float m00, m01, m10, m11;
    const float* layer_offsets;  /* (u, v) pairs, one per layer */
    float position[3], ray_base[3], e1_over_f[3], e2_over_f[3];
    float center_u, center_v;
    const float* layer_num;
} vcr_slab_projection;

typedef struct vcr_slab {
    const vcr_slab_layer* layers;
    int count, rows, cols;
    int left, top, width, height;
    vcr_slab_projection projection;
    const uint32_t* colors;   /* 256 ARGB */
    const float* opacity;     /* 256 */
    float lighting;
} vcr_slab;

VCR_API void vcr_opacity_lut(float lut[256], float low, float high, float opacity, float gamma, uint8_t cutoff,
                             float segment_length);
/* rgb: width*height*3 bytes; coverage: width*height bytes. */
VCR_API vcr_status vcr_composite_slab(const vcr_slab* slab, uint8_t* rgb, uint8_t* coverage);

#ifdef __cplusplus
}
#endif

#endif /* VC_READER_C_H */
