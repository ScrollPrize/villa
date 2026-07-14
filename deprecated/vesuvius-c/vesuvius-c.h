#ifndef VESUVIUS_H
#define VESUVIUS_H

// Vesuvius-c notes:
// - in order to use Vesuvius-c, define VESUVIUS_IMPL in one .c file and then #include "vesuvius-c.h"
// - when passing pointers to a _new function in order to fill out fields in the struct (e.g. vs_mesh_new)
//   the struct will take ownership of the pointer and the pointer shall be cleaned up in the _free function.
//   The caller loses ownership of the pointer. This does NOT include char* for strings like paths or URLs
//     - e.g. vs_vol_new(char* cache, char* url) does _not_ subsume either pointer. If they are not literals
//       then the caller is responsible for cleaning up the char*
// - index order is in Z Y X order
// - a 0 return code indicates success for functions that do NOT return a pointer
// - a non zero return code _can_ indicate failure
//    - this is often on a case by case basis, but is quite often the case for functions that take out parameters
//      or are otherwise side-effect-ful
// - a NULL pointer indicates failure for functions that return a pointer
// - It is the caller's responsibility to clean up pointers returned by Vesuvius APIs
//   - some structures, such as chunk and volume, have custom _free functions which should be called
//     which will free any pointers contained within the structure that have been allocated, f.ex. in _new
//     AND will also free the pointer itself
//   - this applies to both function return values and out parameters passed as pointer pointers
//   - for pointers to primitive types the caller should just call free() on the pointer

#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <json-c/json.h>
#include <blosc2.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <float.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>

#include <curl/curl.h>
#include <blosc2.h>

#if defined(__linux__) || defined(__GLIBC__)
#include <execinfo.h>
#endif

#ifdef NDEBUG
#define ASSERT(expr, msg, ...) ((void)0)
#else
#define ASSERT(expr, msg, ...) do{if(!(expr)){fprintf(stderr,msg __VA_OPT__(,)#__VA_ARGS__); vs__assert_fail_with_backtrace(#expr, __FILE__, __LINE__, __func__);}}while(0)
#endif

#ifdef MAX
#undef MAX
#endif
#define MAX(a,b) (a > b ? a : b)


#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) (a < b ? a : b)

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
typedef float f32;
typedef double f64;


typedef struct histogram {
    s32 num_bins;
    f32 min_value;
    f32 max_value;
    f32 bin_width;
    u32 *bins;
} histogram;

typedef struct hist_stats {
    f32 mean;
    f32 median;
    f32 mode;
    u32 mode_count;
    f32 std_dev;
} hist_stats;

typedef struct {
    char* buffer;
    size_t size;
} DownloadBuffer;

typedef struct {
    CURLM* multi_handle;
    CURL* easy_handle;
    DownloadBuffer chunk;
    bool complete;
    long http_code;
} MultiDownloadState;

typedef struct chunk {
    int dims[3];
    float data[];
} chunk __attribute__((aligned(16)));

typedef struct slice {
    int dims[2];
    float data[];
} slice __attribute__((aligned(16)));

// meshes are triangle only. every 3 entries in vertices corresponds to a new vertex
// normals are 3 component
typedef struct {
    f32 *vertices; // cannot be null
    s32 *indices; // cannot be null
    f32 *normals; // can be null if no normals
    u8* colors; // can be null if no colors. RGB format, 3 u8 per vertex
    s32 vertex_count;
    s32 index_count;
} mesh;

#define MAX_LINE_LENGTH 1024
#define MAX_HEADER_LINES 100

typedef struct {
    char type[32];
    s32 dimension;
    char space[32];
    s32 sizes[16];
    f32 space_directions[16][3];
    char endian[16];
    char encoding[32];
    f32 space_origin[3];

    size_t data_size;
    void* data;

    bool is_valid;
} nrrd;


// PPM format types
typedef enum ppm_type {
    P3,  // ASCII format
    P6   // Binary format
} ppm_type;

typedef struct ppm {
    u32 width;
    u32 height;
    u8 max_val;
    u8* data;  // RGB data in row-major order
} ppm;

// tiff
#define TIFFTAG_SUBFILETYPE 254
#define TIFFTAG_IMAGEWIDTH 256
#define TIFFTAG_IMAGELENGTH 257
#define TIFFTAG_BITSPERSAMPLE 258
#define TIFFTAG_COMPRESSION 259
#define TIFFTAG_PHOTOMETRIC 262
#define TIFFTAG_IMAGEDESCRIPTION 270
#define TIFFTAG_SOFTWARE 305
#define TIFFTAG_DATETIME 306
#define TIFFTAG_SAMPLESPERPIXEL 277
#define TIFFTAG_ROWSPERSTRIP 278
#define TIFFTAG_PLANARCONFIG 284
#define TIFFTAG_RESOLUTIONUNIT 296
#define TIFFTAG_XRESOLUTION 282
#define TIFFTAG_YRESOLUTION 283
#define TIFFTAG_SAMPLEFORMAT 339
#define TIFFTAG_STRIPOFFSETS 273
#define TIFFTAG_STRIPBYTECOUNTS 279

#define TIFF_BYTE 1
#define TIFF_ASCII 2
#define TIFF_SHORT 3
#define TIFF_LONG 4
#define TIFF_RATIONAL 5

typedef struct {
    uint32_t offset;
    uint32_t byteCount;
} StripInfo;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint16_t bitsPerSample;
    uint16_t compression;
    uint16_t photometric;
    uint16_t samplesPerPixel;
    uint32_t rowsPerStrip;
    uint16_t planarConfig;
    uint16_t sampleFormat;
    StripInfo stripInfo;
    char imageDescription[256];
    char software[256];
    char dateTime[20];
    float xResolution;
    float yResolution;
    uint16_t resolutionUnit;
    uint32_t subfileType;
} DirectoryInfo;

typedef struct {
    DirectoryInfo* directories;
    uint16_t depth;
    size_t dataSize;
    void* data;
    bool isValid;
    char errorMsg[256];
} TiffImage;

//zarr
typedef struct zarr_compressor_settings {
    int32_t blocksize;
    int32_t clevel;
    char cname[32];
    char id[32];
    int32_t shuffle;
} zarr_compressor_settings;

typedef struct zarr_metadata {
    int32_t shape[3];
    int32_t chunks[3];
    zarr_compressor_settings compressor;
    char dtype[8];
    int32_t fill_value;
    char order; // Single character 'C' or 'F'
    int32_t zarr_format;
    char dimension_separator;
} zarr_metadata;

typedef struct rgb {
    u8 r, g, b;
} rgb __attribute__((packed));

// vol
// A volume is an entire scroll at a given pixel density
//     - for Scroll 1 it is all 14376 x 7888 x 8096 voxels
//         - for the 2x scaled down Scroll 1 you would need a separate volume
//     - the dtype is uint8
//     - wraps a zarr array
//     - a volume takes a url and a local directory
//         - the url and path should both contain the .zarray
//             - "/path/to/my/zarr" would contain "/path/to/my/zarr/.zarray"
//             - "https://example.com/path/to/my/zarr" would contain "https://example.com/path/to/my/zarr/.zarray"
//         - blocks are read from the cache if they exist, otherwise downloaded and written to disk


typedef struct volume {
    char cache_dir [1024];
    char url [1024];
    zarr_metadata metadata;
} volume;

typedef struct {
    volume* vol;
    s32 vol_start[3];
    s32 chunk_dims[3];
    chunk* ret;
    int z, y, x;
    int zstart, ystart, xstart;
    int zend, yend, xend;
    MultiDownloadState* download;
    bool downloading;
} ChunkLoadState;


typedef enum {
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    LOG_FATAL
} vs__log_level_e;

#define LOG_INFO(...) vs__log_msg(LOG_INFO, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...) vs__log_msg(LOG_WARN, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) vs__log_msg(LOG_ERROR, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define LOG_FATAL(...) vs__log_msg(LOG_FATAL, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)


// Public APIs
// - These are exported and meant to be used by users of vesuvius-c.h

// chamfer
f32 vs_chamfer_distance(const f32* set1, s32 size1, const f32* set2, s32 size2);

// curl
long vs_download(const char* url, void** out_buffer);
MultiDownloadState* vs_download_start(const char* url);
bool vs_download_poll(MultiDownloadState* state, void** out_buffer, long* out_size);

// histogram
histogram *vs_histogram_new(s32 num_bins, f32 min_value, f32 max_value);
void vs_histogram_free(histogram *hist);
histogram* vs_slice_histogram(const f32* data, s32 dimy, s32 dimx, s32 num_bins);
histogram* vs_chunk_histogram(const f32* data, s32 dimz, s32 dimy, s32 dimx, s32 num_bins);
s32 vs_write_histogram_to_csv(const histogram *hist, const char *filename);
hist_stats vs_calculate_histogram_stats(const histogram *hist);

// math
chunk *vs_chunk_new(int dims[static 3]);
void vs_chunk_free(chunk *chunk);
slice *vs_slice_new(int dims[static 2]);
void vs_slice_free(slice *slice);
f32 vs_slice_get(slice *slice, s32 y, s32 x);
void vs_slice_set(slice *slice, s32 y, s32 x, f32 data);
f32 vs_chunk_get(chunk *chunk, s32 z, s32 y, s32 x);
void vs_chunk_set(chunk *chunk, s32 z, s32 y, s32 x, f32 data);
chunk* vs_maxpool(chunk* inchunk, s32 kernel, s32 stride);
chunk *vs_avgpool(chunk *inchunk, s32 kernel, s32 stride);
chunk *vs_sumpool(chunk *inchunk, s32 kernel, s32 stride);
chunk* vs_unsharp_mask_3d(chunk* input, float amount, s32 kernel_size);
chunk* vs_normalize_chunk(chunk* input);
chunk* vs_transpose(chunk* input, const char* input_layout, const char* output_layout);
chunk* vs_dilate(chunk* inchunk, s32 kernel);
chunk* vs_erode(chunk* inchunk, s32 kernel);
f32 vs_chunk_min(chunk *chunk);
f32 vs_chunk_max(chunk *chunk);
s32 vs_flood_fill(chunk* c, s32 z, s32 y, s32 x, chunk* visited, s32 max_size);
chunk* vs_remove_small_components(chunk* c, s32 min_size);
chunk* vs_threshold(chunk* inchunk, f32 threshold, f32 lo, f32 hi);
chunk* vs_histogram_equalize(chunk* inchunk, s32 num_bins);
chunk* vs_mask(chunk* inchunk, chunk* mask);
s32 vs_count_labels(chunk* labeled_chunk, s32** counts);
chunk* vs_connected_components_3d(chunk* in_chunk);

// mesh
mesh* vs_mesh_new(f32 *vertices, f32 *normals, s32 *indices, u8* colors, s32 vertex_count, s32 index_count);
void vs_mesh_free(mesh *mesh);
void vs_mesh_get_bounds(const mesh *m,
                    f32 *origin_z, f32 *origin_y, f32 *origin_x,
                    f32 *length_z, f32 *length_y, f32 *length_x);
void vs_mesh_translate(mesh *m, f32 z, f32 y, f32 x);
void vs_mesh_scale(mesh *m, f32 scale_z, f32 scale_y, f32 scale_x);
s32 vs_march_cubes(const f32* values,
                s32 dimz, s32 dimy, s32 dimx,
                f32 isovalue,
                f32** out_vertices,      //  [z,y,x,z,y,x,...]
                f32** out_colors,        //  [value, value, value, ...]
                s32** out_indices,
                s32* out_vertex_count,
                s32* out_index_count);
rgb vs_colormap_viridis(u8 val);
int vs_colorize(const f32* grayscale, rgb* colors, s32 vertex_count, f32 min, f32 max);

// nrrd
nrrd* vs_nrrd_read(const char* filename);
void vs_nrrd_free(nrrd* nrrd);

// obj
s32 vs_read_obj(const char* filename,
            f32** vertices, s32** indices,
            s32* vertex_count, s32* index_count);
s32 vs_write_obj(const char* filename,
             const f32* vertices, const s32* indices,
             s32 vertex_count, s32 index_count);

// ply
s32 vs_ply_write(const char *filename,
                    const f32 *vertices,
                    const f32 *normals,
                    const rgb *colors,
                    const s32 *indices,
                    s32 vertex_count,
                    s32 index_count);
s32 vs_ply_read(const char *filename,
                          f32 **out_vertices,
                          f32 **out_normals,
                          s32 **out_indices,
                          s32 *out_vertex_count,
                          s32 *out_normal_count,
                          s32 *out_index_count);

//ppm
ppm* vs_ppm_new(u32 width, u32 height);
inline void vs_ppm_free(ppm* img);
ppm* vs_ppm_read(const char* filename);
int vs_ppm_write(const char* filename, const ppm* img, ppm_type type);
void vs_ppm_set_pixel(ppm* img, u32 x, u32 y, u8 r, u8 g, u8 b);
void vs_ppm_get_pixel(const ppm* img, u32 x, u32 y, u8* r, u8* g, u8* b);
void vs_write_ppm_frame(FILE* fp, const chunk* r_chunk, const chunk* g_chunk,
                    const chunk* b_chunk, int frame_idx);
void vs_chunks_to_video(const chunk* r_chunk, const chunk* g_chunk, const chunk* b_chunk,
                    const char* output_filename, int fps);

//tiff
TiffImage* vs_tiff_read(const char* filename);
void vs_tiff_free(TiffImage* img);
const char* vs_tiff_compression_name(uint16_t compression);
const char* vs_tiff_photometric_name(uint16_t photometric);
const char* vs_tiff_planar_config_name(uint16_t config);
const char* vs_tiff_sample_format_name(uint16_t format);
const char* vs_tiff_resolution_unit_name(uint16_t unit);
void vs_tiff_print_tags(const TiffImage* img, int directory);
void vs_tiff_print_all_tags(const TiffImage* img);
size_t vs_tiff_directory_size(const TiffImage* img, int directory);
void* vs_tiff_read_directory_data(const TiffImage* img, int directory);
uint16_t vs_tiff_pixel16(const uint16_t* buffer, int y, int x, int width);
uint8_t vs_tiff_pixel8(const uint8_t* buffer, int y, int x, int width);
int vs_tiff_write(const char* filename, const TiffImage* img, bool littleEndian);
TiffImage* vs_tiff_create(uint32_t width, uint32_t height, uint16_t depth, uint16_t bitsPerSample);

// vcps
int vs_vcps_read(const char* filename,
              size_t* width, size_t* height, size_t* dim,
              void* data, const char* dst_type);
int vs_vcps_write(const char* filename,
               size_t width, size_t height, size_t dim,
               const void* data, const char* src_type, const char* dst_type);

// volume
volume* vs_vol_new(char* cache_dir, char* url);
void vs_vol_free(volume* vol);
chunk* vs_vol_get_chunk(volume* vol, s32 chunk_pos[static 3], s32 chunk_dims[static 3]);

// zarr
zarr_metadata vs_zarr_parse_zarray(char *path);
chunk* vs_zarr_read_chunk(char* path, zarr_metadata metadata);
int vs_zarr_compress_chunk(chunk* c, zarr_metadata metadata, void** compressed_data);
chunk* vs_zarr_decompress_chunk(long size, void* compressed_data, zarr_metadata metadata);
int vs_zarr_parse_metadata(const char *json_string, zarr_metadata *metadata);
chunk* vs_zarr_fetch_block(char* url, zarr_metadata metadata);
int vs_zarr_write_chunk(char *path, zarr_metadata metadata, chunk* c);

// vesuvius specific
chunk *vs_tiff_to_chunk(const char *tiffpath);
slice *vs_tiff_to_slice(const char *tiffpath, int index);
int vs_slice_fill(slice *slice, volume *vol, int start[static 2], int axis);
int vs_chunk_fill(chunk *chunk, volume *vol, int start[static 3]);

#ifdef VESUVIUS_IMPL

// Private APIs
// - These are not exported and are only meant to be used within vesuvius-c.h itself

// utils
static void vs__trim(char* str);
static void vs__skip_line(FILE *fp);
static bool vs__str_starts_with(const char* str, const char* prefix);
static int vs__mkdir_p(const char* path);
static bool vs__path_exists(const char *path);
static void vs__print_backtrace(void);
static void vs__print_assert_details(const char* expr, const char* file, int line, const char* func);
static void vs__assert_fail_with_backtrace(const char* expr, const char* file, int line, const char* func);

//log
static void vs__log_msg(vs__log_level_e level, const char* file, const char* func, int line, const char* fmt, ...);

//chamfer
static f32 vs__squared_distance(const f32* p1, const f32* p2);
static f32 vs__min_distance_to_set(const f32* point, const f32* point_set, s32 set_size);

//curl
static size_t vs__write_callback(void *contents, size_t size, size_t nmemb, void *userp);

//histogram
static s32 vs__get_bin_index(const histogram* hist, f32 value);
static f32 vs__get_slice_value(const f32* data, s32 y, s32 x, s32 dimx);
static f32 vs__get_chunk_value(const f32* data, s32 z, s32 y, s32 x, s32 dimy, s32 dimx);

// math
static float vs__maxfloat(float a, float b);
static float vs__minfloat(float a, float b);
static float vs__avgfloat(float *data, int len);
static chunk *vs__create_box_kernel(s32 size);
static chunk* vs__convolve3d(chunk* input, chunk* kernel);

// mesh
static void vs__interpolate_vertex(f32 isovalue,
                                    f32 v1, f32 v2,
                                    f32 x1, f32 y1, f32 z1,
                                    f32 x2, f32 y2, f32 z2,
                                    f32* out_x, f32* out_y, f32* out_z);
static void vs__process_cube(const f32* values,
                        s32 x, s32 y, s32 z,
                        s32 dimx, s32 dimy, s32 dimz,
                        f32 isovalue,
                        f32* vertices,
                        f32* colors,
                        s32* indices,
                        s32* vertex_count,
                        s32* index_count);
static f32 vs__get_value(const f32* values, s32 x, s32 y, s32 z, s32 dimx, s32 dimy, s32 dimz);

//nrrd
static int vs__nrrd_parse_sizes(char* value, nrrd* nrrd);
static int vs__nrrd_parse_space_directions(char* value, nrrd* nrrd);
static int vs__nrrd_parse_space_origin(char* value, nrrd* nrrd);
static size_t vs__nrrd_get_type_size(const char* type);
static int vs__nrrd_read_raw_data(FILE* fp, nrrd* nrrd);
static int vs__nrrd_read_gzip_data(FILE* fp, nrrd* nrrd);

//ppm
static void vs__skip_whitespace_and_comments(FILE* fp);
static bool vs__ppm_read_header(FILE* fp, ppm_type* type, u32* width, u32* height, u8* max_val);

//tiff
static uint32_t vs__tiff_read_bytes(FILE* fp, int count, int littleEndian);
static void vs__tiff_read_string(FILE* fp, char* str, uint32_t offset, uint32_t count, long currentPos);
static float vs__tiff_read_rational(FILE* fp, uint32_t offset, int littleEndian, long currentPos);
static void vs__tiff_read_ifd_entry(FILE* fp, DirectoryInfo* dir, int littleEndian, long ifdStart);
static bool vs__tiff_validate_directory(DirectoryInfo* dir, TiffImage* img);
static void vs__tiff_write_bytes(FILE* fp, uint32_t value, int count, int littleEndian);
static void vs__tiff_write_string(FILE* fp, const char* str, uint32_t offset);
static void vs__tiff_write_rational(FILE* fp, float value, uint32_t offset, int littleEndian);
static void vs__tiff_current_date_time(char* dateTime);
static uint32_t vs__tiff_write_ifd_entry(FILE* fp, uint16_t tag, uint16_t type, uint32_t count, uint32_t value, int littleEndian);

//vcps
static int vs__vcps_read_binary_data(FILE* fp, void* out_data, const char* src_type, const char* dst_type, size_t count);
static int vs__vcps_write_binary_data(FILE* fp, const void* data, const char* src_type, const char* dst_type, size_t count);

//zarr
static void vs__json_parse_int32_array(json_object *array_obj, int32_t output[3]);
static void vs__log_msg(vs__log_level_e level, const char* file, const char* func, int line, const char* fmt, ...) {

    static const char* level_strings[] = {
        "INFO",
        "WARN",
        "ERROR",
        "FATAL"
    };

    time_t now;
    time(&now);
    char* date = ctime(&now);
    date[strlen(date) - 1] = '\0'; // Remove newline

    fprintf(stderr, "%s [%s] %s:%s:%d: ", date, level_strings[level], file, func, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
    fflush(stderr);
}


static void vs__trim(char* str) {
  char* end;
  while(isspace(*str)) str++;
  if(*str == 0) return;
  end = str + strlen(str) - 1;
  while(end > str && isspace(*end)) end--;
  end[1] = '\0';
}

static bool vs__str_starts_with(const char* str, const char* prefix) {
  return strncmp(str, prefix, strlen(prefix)) == 0;
}

static int vs__mkdir_p(const char* path) {
    char tmp[1024];
    char* p = NULL;
    size_t len;
    int status = 0;

    // Check for NULL path
    if (path == NULL) {
        return 1;
    }

    // Copy path to temporary buffer
    if (snprintf(tmp, sizeof(tmp), "%s", path) >= sizeof(tmp)) {
        return 1;  // Path too long
    }

    len = strlen(tmp);
    if (len == 0) {
        return 1;  // Empty path
    }

    // Remove trailing slash if present
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = 0;
    }

    // Handle absolute path
    p = (tmp[0] == '/') ? tmp + 1 : tmp;

    // Create parent directories
    for (; *p; p++) {
        if (*p == '/') {
            *p = 0;  // Temporarily terminate string at this position

#ifdef _WIN32
            status = mkdir(tmp);
#else
            status = mkdir(tmp, 0755);
#endif
            // Ignore "Already exists" error, fail on other errors
            if (status != 0 && errno != EEXIST) {
                return 1;
            }

            *p = '/';  // Restore the slash
        }
    }

    // Create the final directory
#ifdef _WIN32
    status = mkdir(tmp);
#else
    status = mkdir(tmp, 0755);
#endif

    // Return 0 if directory was created or already exists
    return (status == 0 || errno == EEXIST) ? 0 : 1;
}

static bool vs__path_exists(const char *path) {
    return access(path, F_OK) == 0 ? true : false;
}

static char* vs__basename(const char* path) {
    if (path == NULL) {
        return NULL;
    }

    // Handle empty string
    if (path[0] == '\0') {
        char* result = malloc(2);
        if (result) {
            strcpy(result, ".");
        }
        return result;
    }

    // Create a copy of the path that we can modify
    char* path_copy = strdup(path);
    if (path_copy == NULL) {
        return NULL;
    }

    // Remove trailing slashes
    size_t len = strlen(path_copy);
    while (len > 1 && path_copy[len - 1] == '/') {
        path_copy[--len] = '\0';
    }

    // Find the last separator
    char* last_slash = strrchr(path_copy, '/');

    // Handle different cases
    char* result;
    if (last_slash == NULL) {
        // No slash found - return "."
        result = malloc(2);
        if (result) {
            strcpy(result, ".");
        }
    } else if (last_slash == path_copy) {
        // Slash is at the beginning - return "/"
        result = malloc(2);
        if (result) {
            strcpy(result, "/");
        }
    } else {
        // Normal case - return everything up to the last slash
        *last_slash = '\0';
        result = strdup(path_copy);
    }

    free(path_copy);
    return result;
}

static char* vs__filename(const char* path) {
    if (path == NULL) {
        return NULL;
    }

    // Find the last separator
    const char* last_slash = strrchr(path, '/');

    if (last_slash == NULL) {
        // No slash found - return copy of entire string
        return strdup(path);
    }

    // Move past the slash to get the last component
    last_slash++;

    // Return empty string if the path ends in a slash
    if (*last_slash == '\0') {
        char* result = malloc(1);
        if (result) {
            result[0] = '\0';
        }
        return result;
    }

    // Return copy of everything after the last slash
    return strdup(last_slash);
}


static void vs__print_backtrace(void) {
#if defined(__linux__) || defined(__GLIBC__)

    void *stack_frames[64];
    int frame_count;
    char **frame_strings;

    // Get the stack frames
    frame_count = backtrace(stack_frames, 64);

    // Convert addresses to strings
    frame_strings = backtrace_symbols(stack_frames, frame_count);
    if (frame_strings == NULL) {
        perror("backtrace_symbols");
        exit(EXIT_FAILURE);
    }

    // Print the backtrace
    fprintf(stderr, "\nBacktrace:\n");
    for (int i = 0; i < frame_count; i++) {
        fprintf(stderr, "  [%d] %s\n", i, frame_strings[i]);
    }

    free(frame_strings);
#else
    printf("cannot print a backtrace on non linux systems\n");
#endif
}

static void vs__print_assert_details(const char* expr, const char* file, int line, const char* func) {
    fprintf(stderr, "\nAssertion failed!\n");
    fprintf(stderr, "Expression: %s\n", expr);
    fprintf(stderr, "Location  : %s:%d\n", file, line);
    fprintf(stderr, "Function  : %s\n", func);
}

static void vs__assert_fail_with_backtrace(const char* expr, const char* file, int line, const char* func) {
    vs__print_assert_details(expr, file, line, func);
    vs__print_backtrace();
    abort();
}

// chamfer

static f32 vs__squared_distance(const f32* p1, const f32* p2) {
  f32 dz = p1[0] - p2[0];
  f32 dy = p1[1] - p2[1];
  f32 dx = p1[2] - p2[2];
  return dx*dx + dy*dy + dz*dz;
}


static f32 vs__min_distance_to_set(const f32* point, const f32* point_set, s32 set_size) {
  f32 min_dist = FLT_MAX;

  for (s32 i = 0; i < set_size; i++) {
    f32 dist = vs__squared_distance(point, &point_set[i * 3]);
    if (dist < min_dist) {
      min_dist = dist;
    }
  }
  return min_dist;
}

f32 vs_chamfer_distance(const f32* set1, s32 size1, const f32* set2, s32 size2) {
  f32 sum1 = 0.0f;
  f32 sum2 = 0.0f;

  for (s32 i = 0; i < size1; i++) {
    sum1 += vs__min_distance_to_set(&set1[i * 3], set2, size2);
  }

  for (s32 i = 0; i < size2; i++) {
    sum2 += vs__min_distance_to_set(&set2[i * 3], set1, size1);
  }

  return sqrtf((sum1 / size1 + sum2 / size2) / 2.0f);
}

//curl
static size_t vs__write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    DownloadBuffer *mem = userp;

    char *new_buffer = realloc(mem->buffer, mem->size + realsize + 1);  // +1 for null terminator
    if (!new_buffer) {
        return 0; // Signal error to curl
    }

    memcpy(new_buffer + mem->size, contents, realsize);
    mem->buffer = new_buffer;
    mem->size += realsize;
    mem->buffer[mem->size] = 0; // Null terminate

    return realsize;
}

long vs_download(const char* url, void** out_buffer) {
    CURL* curl;
    CURLcode res;
    long http_code = 0;

    DownloadBuffer chunk = {
        .buffer = malloc(1),
        .size = 0
    };

    if (!chunk.buffer) {
        return -1;
    }
    chunk.buffer[0] = 0;  // Ensure null terminated

    curl = curl_easy_init();
    if (!curl) {
        free(chunk.buffer);
        return -1;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, vs__write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");

    //TODO: can we remove these? they were necessary for bearssl but unsure on openssl / gnutls / others
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

    res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        LOG_ERROR("curl_easy_perform() failed: %s", curl_easy_strerror(res));
        free(chunk.buffer);
        curl_easy_cleanup(curl);
        return -1;
    }

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (http_code != 200) {
        free(chunk.buffer);
        return -1;
    }

    *out_buffer = chunk.buffer;
    return chunk.size;
}

MultiDownloadState* vs_download_start(const char* url) {
    MultiDownloadState* state = malloc(sizeof(MultiDownloadState));
    if (!state) return NULL;

    state->chunk.buffer = malloc(1);
    state->chunk.size = 0;
    state->complete = false;
    state->http_code = 0;

    if (!state->chunk.buffer) {
        free(state);
        return NULL;
    }

    state->chunk.buffer[0] = 0;

    state->multi_handle = curl_multi_init();
    state->easy_handle = curl_easy_init();

    curl_easy_setopt(state->easy_handle, CURLOPT_URL, url);
    curl_easy_setopt(state->easy_handle, CURLOPT_WRITEFUNCTION, vs__write_callback);
    curl_easy_setopt(state->easy_handle, CURLOPT_WRITEDATA, (void *)&state->chunk);
    curl_easy_setopt(state->easy_handle, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(state->easy_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");
    curl_easy_setopt(state->easy_handle, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(state->easy_handle, CURLOPT_SSL_VERIFYHOST, 0L);

    curl_multi_add_handle(state->multi_handle, state->easy_handle);
    return state;
}

bool vs_download_poll(MultiDownloadState* state, void** out_buffer, long* out_size) {
    if (state->complete) return true;

    int still_running;
    curl_multi_perform(state->multi_handle, &still_running);

    if (!still_running) {
        curl_easy_getinfo(state->easy_handle, CURLINFO_RESPONSE_CODE, &state->http_code);
        curl_multi_remove_handle(state->multi_handle, state->easy_handle);
        curl_easy_cleanup(state->easy_handle);
        curl_multi_cleanup(state->multi_handle);

        state->complete = true;

        if (state->http_code == 200 && out_buffer && out_size) {
            *out_buffer = state->chunk.buffer;
            *out_size = state->chunk.size;
            return true;
        }

        free(state->chunk.buffer);
        free(state);
        return true;
    }

    return false;
}

// histogram
histogram *vs_histogram_new(s32 num_bins, f32 min_value, f32 max_value) {
  histogram *hist = malloc(sizeof(histogram));
  if (!hist) {
    return NULL;
  }

  hist->bins = calloc(num_bins, sizeof(u32));
  if (!hist->bins) {
    free(hist);
    return NULL;
  }

  hist->num_bins = num_bins;
  hist->min_value = min_value;
  hist->max_value = max_value;
  hist->bin_width = (max_value - min_value) / num_bins;

  return hist;
}

void vs_histogram_free(histogram *hist) {
  if (hist) {
    free(hist->bins);
    free(hist);
  }
}


static s32 vs__get_bin_index(const histogram* hist, f32 value) {
    if (value <= hist->min_value) return 0;
    if (value >= hist->max_value) return hist->num_bins - 1;

    s32 bin = (s32)((value - hist->min_value) / hist->bin_width);
    if (bin >= hist->num_bins) bin = hist->num_bins - 1;
    return bin;
}

histogram* vs_slice_histogram(const f32* data,
                                      s32 dimy, s32 dimx,
                                      s32 num_bins) {
    if (!data || num_bins <= 0) {
        return NULL;
    }

    f32 min_val = FLT_MAX;
    f32 max_val = -FLT_MAX;

    s32 total_pixels = dimy * dimx;
    for (s32 i = 0; i < total_pixels; i++) {
        f32 val = data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    histogram* hist = vs_histogram_new(num_bins, min_val, max_val);
    if (!hist) {
        return NULL;
    }

    for (s32 i = 0; i < total_pixels; i++) {
        s32 bin = vs__get_bin_index(hist, data[i]);
        hist->bins[bin]++;
    }

    return hist;
}

histogram* vs_chunk_histogram(const f32* data,
                                      s32 dimz, s32 dimy, s32 dimx,
                                      s32 num_bins) {
    if (!data || num_bins <= 0) {
        return NULL;
    }

    f32 min_val = FLT_MAX;
    f32 max_val = -FLT_MAX;

    s32 total_voxels = dimz * dimy * dimx;
    for (s32 i = 0; i < total_voxels; i++) {
        f32 val = data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    histogram* hist = vs_histogram_new(num_bins, min_val, max_val);
    if (!hist) {
        return NULL;
    }

    for (s32 i = 0; i < total_voxels; i++) {
        s32 bin = vs__get_bin_index(hist, data[i]);
        hist->bins[bin]++;
    }

    return hist;
}

static f32 vs__get_slice_value(const f32* data, s32 y, s32 x, s32 dimx) {
    return data[y * dimx + x];
}

static f32 vs__get_chunk_value(const f32* data, s32 z, s32 y, s32 x,
                                  s32 dimy, s32 dimx) {
    return data[z * (dimy * dimx) + y * dimx + x];
}
s32 vs_write_histogram_to_csv(const histogram *hist, const char *filename) {
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    return 1;
  }

  fprintf(fp, "bin_start,bin_end,count\n");

  for (s32 i = 0; i < hist->num_bins; i++) {
    f32 bin_start = hist->min_value + i * hist->bin_width;
    f32 bin_end = bin_start + hist->bin_width;
    fprintf(fp, "%.6f,%.6f,%u\n", bin_start, bin_end, hist->bins[i]);
  }

  fclose(fp);
  return 0;
}

hist_stats vs_calculate_histogram_stats(const histogram *hist) {
  hist_stats stats = {0};

  unsigned long long total_count = 0;
  f64 weighted_sum = 0.0;
  u32 max_count = 0;

  for (s32 i = 0; i < hist->num_bins; i++) {
    f32 bin_center = hist->min_value + (i + 0.5f) * hist->bin_width;
    weighted_sum += bin_center * hist->bins[i];
    total_count += hist->bins[i];

    if (hist->bins[i] > max_count) {
      max_count = hist->bins[i];
      stats.mode = bin_center;
      stats.mode_count = hist->bins[i];
    }
  }

  stats.mean = (f32) (weighted_sum / total_count);

  f64 variance_sum = 0;
  for (s32 i = 0; i < hist->num_bins; i++) {
    f32 bin_center = hist->min_value + (i + 0.5f) * hist->bin_width;
    f32 diff = bin_center - stats.mean;
    variance_sum += diff * diff * hist->bins[i];
  }
  stats.std_dev = (f32) sqrt(variance_sum / total_count);

  u64 median_count = total_count / 2;
  u64 running_count = 0;
  for (s32 i = 0; i < hist->num_bins; i++) {
    running_count += hist->bins[i];
    if (running_count >= median_count) {
      stats.median = hist->min_value + (i + 0.5f) * hist->bin_width;
      break;
    }
  }

  return stats;
}

// math


// A chunk is a 3d cross section of data
//   - this could be a 512x512x512 section starting at 2000x2000x2000 and ending at 2512 x 2512 x 2512
//   - the dtype is float32
//   - increasing Z means increasing through the slice. e.g. 1000.tif -> 1001.tif
//   - increasing Y means looking farther down in a slice
//   - increasing X means looking farther right in a slice
// A slice is a 2d cross section of data
//   - increasing Y means looking farther down in a slice
//   - increasing X means looking farther right in a slice


static float vs__maxfloat(float a, float b) { return a > b ? a : b; }
static float vs__minfloat(float a, float b) { return a < b ? a : b; }
static float vs__avgfloat(float *data, int len) {
  double sum = 0.0;
  for (int i = 0; i < len; i++) sum += data[i];
  return sum / len;
}

chunk *vs_chunk_new(int dims[static 3]) {
  chunk *ret = malloc(sizeof(chunk) + dims[0] * dims[1] * dims[2] * sizeof(float));

  if (ret == NULL) {
    return NULL;
  }

  for (int i = 0; i < 3; i++) {
    ret->dims[i] = dims[i];
  }
  return ret;
}

void vs_chunk_free(chunk *chunk) {
  free(chunk);
}

slice *vs_slice_new(int dims[static 2]) {
  slice *ret = malloc(sizeof(slice) + dims[0] * dims[1] * sizeof(float));

  if (ret == NULL) {
    return NULL;
  }

  for (int i = 0; i < 2; i++) {
    ret->dims[i] = dims[i];
  }
  return ret;
}

void vs_slice_free(slice *slice) {
  free(slice);
}

f32 vs_slice_get(slice *slice, s32 y, s32 x) {
  return slice->data[y * slice->dims[1] + x];
}

void vs_slice_set(slice *slice, s32 y, s32 x, f32 data) {
  slice->data[y * slice->dims[1] + x] = data;
}

f32 vs_chunk_get(chunk *chunk, s32 z, s32 y, s32 x) {
  return chunk->data[z * chunk->dims[1] * chunk->dims[2] + y * chunk->dims[2] + x];
}

void vs_chunk_set(chunk *chunk, s32 z, s32 y, s32 x, f32 data) {
  chunk->data[z * chunk->dims[1] * chunk->dims[2] + y * chunk->dims[2] + x] = data;
}

int vs_chunk_graft(chunk* dest, chunk* src, s32 src_start[static 3], s32 dest_start[static 3], s32 dims[static 3]) {
  if (!dest || !src || !src_start || !dest_start || !dims) {
      LOG_ERROR("a param is NULL");
    return -1;
  }

  for (int i = 0; i < 3; i++) {
    if (dims[i] <= 0) {
        LOG_ERROR("a dimension is <= 0");
      return -1;
    }
  }

  for (int i = 0; i < 3; i++) {
    if (src_start[i] < 0 ||
        src_start[i] + dims[i] > src->dims[i]) {
        LOG_ERROR("out of bounds src dimension");
      return -1;
        }
  }

  for (int i = 0; i < 3; i++) {
    if (dest_start[i] < 0 ||
        dest_start[i] + dims[i] > dest->dims[i]) {
        LOG_ERROR("out of bounds dest dimension");
      return -1;
        }
  }

  for (int z = 0; z < dims[0]; z++) {
    for (int y = 0; y < dims[1]; y++) {
      for (int x = 0; x < dims[2]; x++) {
        f32 value = vs_chunk_get(src, src_start[0] + z, src_start[1] + y, src_start[2] + x);
        vs_chunk_set(dest, dest_start[0] + z, dest_start[1] + y, dest_start[2] + x, value);
      }
    }
  }
  return 0;
}

chunk* vs_remove_small_components(chunk* c, s32 min_size) {
    s32 dims[3] = {c->dims[0], c->dims[1], c->dims[2]};
    chunk* visited = vs_chunk_new(dims);
    chunk* ret = vs_chunk_new(dims);
    memcpy(ret->data, c->data, dims[0] * dims[1] * dims[2] * sizeof(float));

    // Pre-calculate offsets for 6 directions (+-z, +-y, +-x)
    s32 offsets[6] = {
        dims[1] * dims[2],      // +z
        -dims[1] * dims[2],     // -z
        dims[2],                // +y
        -dims[2],               // -y
        1,                      // +x
        -1                      // -x
    };

    // Stack for DFS - pre-allocate once
    s32* stack = malloc(dims[0] * dims[1] * dims[2] * sizeof(s32));

    for (s32 i = 0; i < dims[0] * dims[1] * dims[2]; i++) {
        if (ret->data[i] > 0.0f && visited->data[i] == 0.0f) {
            s32 stack_top = 0;
            s32 size = 0;
            stack[stack_top++] = i;

            // First pass - count size and mark visited
            while (stack_top > 0) {
                s32 curr = stack[--stack_top];
                if (visited->data[curr] > 0.0f) continue;

                visited->data[curr] = 1.0f;
                size++;

                if (size >= min_size) break; // Early exit if size requirement met

                // Get z,y,x from linear index
                s32 z = curr / (dims[1] * dims[2]);
                s32 y = (curr % (dims[1] * dims[2])) / dims[2];
                s32 x = curr % dims[2];

                // Check all 6 directions
                for (s32 dir = 0; dir < 6; dir++) {
                    // Check if move is valid
                    if ((dir == 0 && z >= dims[0]-1) ||
                        (dir == 1 && z <= 0) ||
                        (dir == 2 && y >= dims[1]-1) ||
                        (dir == 3 && y <= 0) ||
                        (dir == 4 && x >= dims[2]-1) ||
                        (dir == 5 && x <= 0)) continue;

                    s32 next = curr + offsets[dir];
                    if (ret->data[next] > 0.0f && visited->data[next] == 0.0f) {
                        stack[stack_top++] = next;
                    }
                }
            }

            // If component is too small, remove it
            if (size < min_size) {
                for (s32 j = 0; j < dims[0] * dims[1] * dims[2]; j++) {
                    if (visited->data[j] == 1.0f) {
                        ret->data[j] = 0.0f;
                    }
                }
            }

            // Reset visited markers for found component
            for (s32 j = 0; j < dims[0] * dims[1] * dims[2]; j++) {
                if (visited->data[j] == 1.0f) {
                    visited->data[j] = 2.0f;  // Mark as processed
                }
            }
        }
    }

    free(stack);
    vs_chunk_free(visited);
    return ret;
}

s32 vs_flood_fill(chunk* c, s32 start_z, s32 start_y, s32 start_x, chunk* visited, s32 max_size) {
    s32 dims[3] = {c->dims[0], c->dims[1], c->dims[2]};
    s32 start_idx = start_z * dims[1] * dims[2] + start_y * dims[2] + start_x;

    // Early exit checks
    if (start_z < 0 || start_z >= dims[0] ||
        start_y < 0 || start_y >= dims[1] ||
        start_x < 0 || start_x >= dims[2] ||
        c->data[start_idx] == 0.0f ||
        visited->data[start_idx] > 0.0f) {
        return 0;
    }

    s32 offsets[6] = {
        dims[1] * dims[2],  // +z
        -dims[1] * dims[2], // -z
        dims[2],            // +y
        -dims[2],          // -y
        1,                 // +x
        -1                // -x
    };

    s32* stack = malloc(dims[0] * dims[1] * dims[2] * sizeof(s32));
    s32 stack_top = 0;
    s32 count = 0;

    stack[stack_top++] = start_idx;

    while (stack_top > 0) {
        s32 curr = stack[--stack_top];
        if (visited->data[curr] > 0.0f) continue;

        visited->data[curr] = 1.0f;
        count++;

        if (max_size > 0 && count >= max_size) {
            free(stack);
            return max_size;
        }

        s32 z = curr / (dims[1] * dims[2]);
        s32 y = (curr % (dims[1] * dims[2])) / dims[2];
        s32 x = curr % dims[2];

        for (s32 dir = 0; dir < 6; dir++) {
            if ((dir == 0 && z >= dims[0]-1) ||
                (dir == 1 && z <= 0) ||
                (dir == 2 && y >= dims[1]-1) ||
                (dir == 3 && y <= 0) ||
                (dir == 4 && x >= dims[2]-1) ||
                (dir == 5 && x <= 0)) continue;

            s32 next = curr + offsets[dir];
            if (c->data[next] > 0.0f && visited->data[next] == 0.0f) {
                stack[stack_top++] = next;
            }
        }
    }

    free(stack);
    return count;
}

chunk* vs_erode(chunk* inchunk, s32 kernel) {
  s32 dims[3] = {inchunk->dims[0], inchunk->dims[1], inchunk->dims[2]};
  chunk* ret = vs_chunk_new(dims);

  s32 offset = kernel / 2;
  for (s32 z = 0; z < dims[0]; z++)
    for (s32 y = 0; y < dims[1]; y++)
      for (s32 x = 0; x < dims[2]; x++) {
        f32 min_val = vs_chunk_get(inchunk, z, y, x);
        for (s32 zi = -offset; zi <= offset; zi++)
          for (s32 yi = -offset; yi <= offset; yi++)
            for (s32 xi = -offset; xi <= offset; xi++) {
              s32 nz = z + zi;
              s32 ny = y + yi;
              s32 nx = x + xi;

              if (nz < 0 || nz >= dims[0] ||
                  ny < 0 || ny >= dims[1] ||
                  nx < 0 || nx >= dims[2]) {
                continue;
              }

              f32 val = vs_chunk_get(inchunk, nz, ny, nx);
              if (val < min_val) {
                min_val = val;
              }
            }
        vs_chunk_set(ret, z, y, x, min_val);
      }
  return ret;
}

chunk* vs_dilate(chunk* inchunk, s32 kernel) {
  s32 dims[3] = {inchunk->dims[0], inchunk->dims[1], inchunk->dims[2]};
  chunk* ret = vs_chunk_new(dims);

  s32 offset = kernel / 2;
  for (s32 z = 0; z < dims[0]; z++)
    for (s32 y = 0; y < dims[1]; y++)
      for (s32 x = 0; x < dims[2]; x++) {
        f32 max_val = 0.0f;
        for (s32 zi = -offset; zi <= offset; zi++)
          for (s32 yi = -offset; yi <= offset; yi++)
            for (s32 xi = -offset; xi <= offset; xi++) {
              s32 nz = z + zi;
              s32 ny = y + yi;
              s32 nx = x + xi;

              if (nz < 0 || nz >= dims[0] ||
                  ny < 0 || ny >= dims[1] ||
                  nx < 0 || nx >= dims[2]) {
                continue;
              }

              f32 val = vs_chunk_get(inchunk, nz, ny, nx);
              if (val > max_val) {
                max_val = val;
              }
            }
        vs_chunk_set(ret, z, y, x, max_val);
      }
  return ret;
}

f32 vs_chunk_max(chunk *chunk) {
    f32 max_val = chunk->data[0];
    s32 total = chunk->dims[0] * chunk->dims[1] * chunk->dims[2];

    for (s32 i = 1; i < total; i++) {
        if (chunk->data[i] > max_val) {
            max_val = chunk->data[i];
        }
    }
    return max_val;
}

f32 vs_chunk_min(chunk *chunk) {
    f32 min_val = chunk->data[0];
    s32 total = chunk->dims[0] * chunk->dims[1] * chunk->dims[2];

    for (s32 i = 1; i < total; i++) {
        if (chunk->data[i] < min_val) {
            min_val = chunk->data[i];
        }
    }
    return min_val;
}

chunk* vs_threshold(chunk* inchunk, f32 threshold, f32 lo, f32 hi) {
    chunk* ret = vs_chunk_new(inchunk->dims);

    for (s32 z = 0; z < ret->dims[0]; z++) {
        for (s32 y = 0; y < ret->dims[1]; y++) {
            for (s32 x = 0; x < ret->dims[2]; x++) {
                f32 current_value = vs_chunk_get(inchunk, z, y, x);
                f32 output_value = (current_value < threshold) ? lo : hi;
                vs_chunk_set(ret, z, y, x, output_value);
            }
        }
    }

    return ret;
}

chunk* vs_maxpool(chunk* inchunk, s32 kernel, s32 stride) {
  s32 dims[3] = {
    (inchunk->dims[0] + stride - 1) / stride, (inchunk->dims[1] + stride - 1) / stride,
    (inchunk->dims[2] + stride - 1) / stride
  };
  chunk *ret = vs_chunk_new(dims);
  for (s32 z = 0; z < ret->dims[0]; z++)
    for (s32 y = 0; y < ret->dims[1]; y++)
      for (s32 x = 0; x < ret->dims[2]; x++) {
        f32 max32 = -INFINITY;
        f32 val32;
        for (s32 zi = 0; zi < kernel; zi++)
          for (s32 yi = 0; yi < kernel; yi++)
            for (s32 xi = 0; xi < kernel; xi++) {
              if (z + zi > inchunk->dims[0] || y + yi > inchunk->dims[1] || x + xi > inchunk->dims[2]) { continue; }

              if ((val32 = vs_chunk_get(inchunk, z * stride + zi, y * stride + yi,
                                                                       x * stride + xi)) > max32) { max32 = val32; }
            }
        vs_chunk_set(ret, z, y, x, max32);
      }
  return ret;
}

chunk *vs_avgpool(chunk *inchunk, s32 kernel, s32 stride) {
  s32 dims[3] = {
    (inchunk->dims[0] + stride - 1) / stride, (inchunk->dims[1] + stride - 1) / stride,
    (inchunk->dims[2] + stride - 1) / stride
  };
  chunk *ret = vs_chunk_new(dims);
  s32 len = kernel * kernel * kernel;
  s32 i = 0;
  f32 *data = malloc(len * sizeof(f32));
  for (s32 z = 0; z < ret->dims[0]; z++)
    for (s32 y = 0; y < ret->dims[1]; y++)
      for (s32 x = 0; x < ret->dims[2]; x++) {
        len = kernel * kernel * kernel;
        i = 0;
        for (s32 zi = 0; zi < kernel; zi++)
          for (s32 yi = 0; yi < kernel; yi++)
            for (s32 xi = 0; xi < kernel; xi++) {
              if (z + zi > inchunk->dims[0] || y + yi > inchunk->dims[1] || x + xi > inchunk->dims[2]) {
                len--;
                continue;
              }
              data[i++] = vs_chunk_get(inchunk, z * stride + zi, y * stride + yi, x * stride + xi);
            }
        vs_chunk_set(ret, z, y, x, vs__avgfloat(data, len));
      }
  return ret;
}

chunk *vs_sumpool(chunk *inchunk, s32 kernel, s32 stride) {
  s32 dims[3] = {
    (inchunk->dims[0] + stride - 1) / stride, (inchunk->dims[1] + stride - 1) / stride,
    (inchunk->dims[2] + stride - 1) / stride
  };
  chunk *ret = vs_chunk_new(dims);
  for (s32 z = 0; z < ret->dims[0]; z++)
    for (s32 y = 0; y < ret->dims[1]; y++)
      for (s32 x = 0; x < ret->dims[2]; x++) {
        f32 sum = 0.0f;
        for (s32 zi = 0; zi < kernel; zi++)
          for (s32 yi = 0; yi < kernel; yi++)
            for (s32 xi = 0; xi < kernel; xi++) {
              if (z + zi > inchunk->dims[0] || y + yi > inchunk->dims[1] || x + xi > inchunk->dims[2]) {
                continue;
              }
              sum += vs_chunk_get(inchunk, z * stride + zi, y * stride + yi, x * stride + xi);
            }
        vs_chunk_set(ret, z, y, x, sum);
      }
  return ret;
}


static chunk *vs__create_box_kernel(s32 size) {
  int dims[3] = {size,size,size};
  chunk* kernel = vs_chunk_new(dims);
  float value = 1.0f / (size * size * size);
  for (s32 z = 0; z < size; z++) {
    for (s32 y = 0; y < size; y++) { for (s32 x = 0; x < size; x++) { vs_chunk_set(kernel, z, y, x, value); } }
  }
  return kernel;
}

static chunk* vs__convolve3d(chunk* input, chunk* kernel) {

  s32 dims[3] = {input->dims[0], input->dims[1], input->dims[2]};

  chunk* ret = vs_chunk_new(dims);
  s32 pad = kernel->dims[0] / 2;

  for (s32 z = 0; z < input->dims[0]; z++) {
    for (s32 y = 0; y < input->dims[1]; y++) {
      for (s32 x = 0; x < input->dims[2]; x++) {
        float sum = 0.0f;
        for (s32 kz = 0; kz < kernel->dims[0]; kz++) {
          for (s32 ky = 0; ky < kernel->dims[1]; ky++) {
            for (s32 kx = 0; kx < kernel->dims[2]; kx++) {
              s32 iz = z + kz - pad;
              s32 iy = y + ky - pad;
              s32 ix = x + kx - pad;
              if (iz >= 0 && iz < input->dims[0] && iy >= 0 && iy < input->dims[1] && ix >= 0 && ix < input->dims[2]) {
                float input_val = vs_chunk_get(input, iz, iy, ix);
                sum += input_val * vs_chunk_get(kernel, kz, ky, kx);
              }
            }
          }
        }
        vs_chunk_set(ret, z, y, x, sum);
      }
    }
  }
  return ret;
}

chunk* vs_unsharp_mask_3d(chunk* input, float amount, s32 kernel_size) {
  int dims[3] = {input->dims[0], input->dims[1], input->dims[2]};
  chunk* kernel = vs__create_box_kernel(kernel_size);
  chunk* blurred = vs__convolve3d(input, kernel);
  chunk* output = vs_chunk_new(dims);

  for (s32 z = 0; z < input->dims[0]; z++) {
    for (s32 y = 0; y < input->dims[1]; y++) {
      for (s32 x = 0; x < input->dims[2]; x++) {
        float original = vs_chunk_get(input, z, y, x);
        float blur = vs_chunk_get(blurred, z, y, x);
        float sharpened = original + amount * (original - blur);
        vs_chunk_set(output, z, y, x, sharpened);
      }
    }
  }

  vs_chunk_free(kernel);
  vs_chunk_free(blurred);

  return output;
}

chunk* vs_normalize_chunk(chunk* input) {
  // Create output chunk with same dimensions
  int dims[3] = {input->dims[0], input->dims[1], input->dims[2]};
  chunk* output = vs_chunk_new(dims);

  // First pass: find min and max values
  float min_val = INFINITY;
  float max_val = -INFINITY;

  for (s32 z = 0; z < input->dims[0]; z++) {
    for (s32 y = 0; y < input->dims[1]; y++) {
      for (s32 x = 0; x < input->dims[2]; x++) {
        float val = vs_chunk_get(input, z, y, x);
        min_val = vs__minfloat(min_val, val);
        max_val = vs__maxfloat(max_val, val);
      }
    }
  }

  // Handle edge case where all values are the same
  float range = max_val - min_val;
  if (range == 0.0f) {
    for (s32 z = 0; z < input->dims[0]; z++) {
      for (s32 y = 0; y < input->dims[1]; y++) {
        for (s32 x = 0; x < input->dims[2]; x++) {
          vs_chunk_set(output, z, y, x, 0.5f);
        }
      }
    }
    return output;
  }

  // Second pass: normalize values to [0.0, 1.0]
  for (s32 z = 0; z < input->dims[0]; z++) {
    for (s32 y = 0; y < input->dims[1]; y++) {
      for (s32 x = 0; x < input->dims[2]; x++) {
        float val = vs_chunk_get(input, z, y, x);
        float normalized = (val - min_val) / range;
        vs_chunk_set(output, z, y, x, normalized);
      }
    }
  }

  return output;
}

chunk* vs_transpose(chunk* input, const char* input_layout, const char* output_layout) {
    if (!input || !input_layout || !output_layout ||
        strlen(input_layout) != 3 || strlen(output_layout) != 3) {
        return NULL;
    }

    // map from input to z y x
    int input_mapping[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        switch (input_layout[i]) {
            case 'z': input_mapping[0] = i; break;
            case 'y': input_mapping[1] = i; break;
            case 'x': input_mapping[2] = i; break;
            default: return NULL;
        }
    }

    // map from z y x to output
    int output_mapping[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        switch (output_layout[i]) {
            case 'z': output_mapping[i] = 0; break;
            case 'y': output_mapping[i] = 1; break;
            case 'x': output_mapping[i] = 2; break;
            default: return NULL;
        }
    }

    // Calculate new dimensions based on output layout
    int new_dims[3];
    for (int i = 0; i < 3; i++) {
        new_dims[i] = input->dims[input_mapping[output_mapping[i]]];
    }

    chunk* output = vs_chunk_new(new_dims);
    if (!output) {
        return NULL;
    }

    // Perform the transpose
    for (int i = 0; i < new_dims[0]; i++) {
        for (int j = 0; j < new_dims[1]; j++) {
            for (int k = 0; k < new_dims[2]; k++) {
                // Current position in output order
                int current_pos[3] = {i, j, k};

                // Convert to canonical zyx order
                int canonical_pos[3];
                for (int dim = 0; dim < 3; dim++) {
                    canonical_pos[output_mapping[dim]] = current_pos[dim];
                }

                // Convert to input order
                int input_pos[3];
                for (int dim = 0; dim < 3; dim++) {
                    input_pos[dim] = canonical_pos[input_mapping[dim]];
                }

                float value = vs_chunk_get(input, input_pos[0], input_pos[1], input_pos[2]);
                vs_chunk_set(output, i, j, k, value);
            }
        }
    }

    return output;
}

chunk* vs_histogram_equalize(chunk* inchunk, s32 num_bins) {
    s32 total_voxels = inchunk->dims[0] * inchunk->dims[1] * inchunk->dims[2];

    // Create histogram
    histogram* hist = vs_chunk_histogram(inchunk->data,
                                       inchunk->dims[0],
                                       inchunk->dims[1],
                                       inchunk->dims[2],
                                       num_bins);
    if (!hist) return NULL;

    // Calculate cumulative distribution function (CDF)
    u32* cdf = calloc(num_bins, sizeof(u32));
    if (!cdf) {
        vs_histogram_free(hist);
        return NULL;
    }

    cdf[0] = hist->bins[0];
    for (s32 i = 1; i < num_bins; i++) {
        cdf[i] = cdf[i-1] + hist->bins[i];
    }

    // Find first non-zero bin
    s32 cdf_min = 0;
    for (s32 i = 0; i < num_bins; i++) {
        if (cdf[i] > 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    // Create output chunk
    chunk* ret = vs_chunk_new(inchunk->dims);
    if (!ret) {
        free(cdf);
        vs_histogram_free(hist);
        return NULL;
    }

    // Apply histogram equalization
    f32 scale = (f32)(num_bins - 1) / (total_voxels - cdf_min);

    for (s32 z = 0; z < inchunk->dims[0]; z++) {
        for (s32 y = 0; y < inchunk->dims[1]; y++) {
            for (s32 x = 0; x < inchunk->dims[2]; x++) {
                f32 val = vs_chunk_get(inchunk, z, y, x);
                s32 bin = vs__get_bin_index(hist, val);
                f32 new_val = (cdf[bin] - cdf_min) * scale;

                // Scale back to original range
                new_val = hist->min_value + (new_val / (num_bins - 1)) *
                         (hist->max_value - hist->min_value);

                vs_chunk_set(ret, z, y, x, new_val);
            }
        }
    }

    free(cdf);
    vs_histogram_free(hist);
    return ret;
}

chunk* vs_mask(chunk* inchunk, chunk* mask) {
    chunk* ret = vs_chunk_new(inchunk->dims);
    for (int z = 0; z < inchunk->dims[0]; z++) {
        for (int y = 0; y < inchunk->dims[1]; y++) {
            for (int x = 0; x < inchunk->dims[2]; x++) {
                f32 m = vs_chunk_get(mask,z,y,x);
                f32 v = vs_chunk_get(inchunk,z,y,x);
                vs_chunk_set(ret,z,y,x,m*v);
            }
        }
    }
    return ret;
}

#define LABEL_EPSILON 0.0001f
#define is_labeled(val) (fabsf((val) - 1.0f) < LABEL_EPSILON)
#define is_unlabeled(val)  (!(is_labeled(val)))


void vs__flood_fill_2d(chunk* out_chunk, chunk* in_chunk, s32 z, s32 y, s32 x, f32 label) {
    if (y < 0 || y >= in_chunk->dims[1] ||
        x < 0 || x >= in_chunk->dims[2] ||
        !is_labeled(vs_chunk_get(in_chunk, z, y, x)) ||
        !is_unlabeled(vs_chunk_get(out_chunk, z, y, x))) {
        return;
    }

    vs_chunk_set(out_chunk, z, y, x, label);

    vs__flood_fill_2d(out_chunk, in_chunk, z, y+1, x, label);
    vs__flood_fill_2d(out_chunk, in_chunk, z, y-1, x, label);
    vs__flood_fill_2d(out_chunk, in_chunk, z, y, x+1, label);
    vs__flood_fill_2d(out_chunk, in_chunk, z, y, x-1, label);
}

f32 vs__get_most_common_label(chunk* out_chunk, s32 z, s32 y, s32 x) {
    f32 labels[9];  // Store unique labels
    s32 counts[9];  // Store counts for each label
    s32 num_unique = 0;

    // Scan 3x3 neighborhood in previous slice
    for (s32 dy = -1; dy <= 1; dy++) {
        for (s32 dx = -1; dx <= 1; dx++) {
            if (y + dy < 0 || y + dy >= out_chunk->dims[1] ||
                x + dx < 0 || x + dx >= out_chunk->dims[2]) {
                continue;
            }

            f32 label = vs_chunk_get(out_chunk, z-1, y+dy, x+dx);
            if (is_unlabeled(label)) {
                continue;
            }

            // Check if we've seen this label before
            bool found = false;
            for (s32 i = 0; i < num_unique; i++) {
                if (fabsf(labels[i] - label) < LABEL_EPSILON) {
                    counts[i]++;
                    found = true;
                    break;
                }
            }

            // If new label, add it
            if (!found && num_unique < 9) {
                labels[num_unique] = label;
                counts[num_unique] = 1;
                num_unique++;
            }
        }
    }

    // Find label with highest count
    if (num_unique == 0) {
        return 0.0f;  // No labels found
    }

    s32 max_count = counts[0];
    f32 most_common = labels[0];

    for (s32 i = 1; i < num_unique; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            most_common = labels[i];
        }
    }

    return most_common;
}
chunk* vs_connected_components_3d(chunk* in_chunk) {
    chunk* out_chunk = vs_chunk_new(in_chunk->dims);
    f32 current_label = 0;

    // Process first slice
    for (s32 y = 0; y < in_chunk->dims[1]; y++) {
        for (s32 x = 0; x < in_chunk->dims[2]; x++) {
            if (is_labeled(vs_chunk_get(in_chunk, 0, y, x)) &&
                is_unlabeled(vs_chunk_get(out_chunk, 0, y, x))) {
                current_label += 1;
                vs__flood_fill_2d(out_chunk, in_chunk, 0, y, x, current_label);
                }
        }
    }

    // Process subsequent slices
    for (s32 z = 1; z < in_chunk->dims[0]; z++) {
        // First pass: propagate existing labels forward
        for (s32 y = 0; y < in_chunk->dims[1]; y++) {
            for (s32 x = 0; x < in_chunk->dims[2]; x++) {
                if (is_labeled(vs_chunk_get(in_chunk, z, y, x))) {
                    f32 prev_label = vs__get_most_common_label(out_chunk, z, y, x);
                    if (!is_unlabeled(prev_label)) {
                        vs__flood_fill_2d(out_chunk, in_chunk, z, y, x, prev_label);
                    }
                }
            }
        }

        // Second pass: create new labels for unlabeled segments
        for (s32 y = 0; y < in_chunk->dims[1]; y++) {
            for (s32 x = 0; x < in_chunk->dims[2]; x++) {
                if (is_labeled(vs_chunk_get(in_chunk, z, y, x)) &&
                    is_unlabeled(vs_chunk_get(out_chunk, z, y, x))) {
                    current_label += 1;
                    vs__flood_fill_2d(out_chunk, in_chunk, z, y, x, current_label);
                    }
            }
        }
    }

    return out_chunk;
}


s32 vs_count_labels(chunk* labeled_chunk, s32** counts) {
    // First pass: find max label
    f32 max_label = 0;
    for (s32 z = 0; z < labeled_chunk->dims[0]; z++) {
        for (s32 y = 0; y < labeled_chunk->dims[1]; y++) {
            for (s32 x = 0; x < labeled_chunk->dims[2]; x++) {
                f32 label = vs_chunk_get(labeled_chunk, z, y, x);
                if (label > max_label) {
                    max_label = label;
                }
            }
        }
    }

    // Allocate array for counts (+1 because we include 0)
    s32 num_labels = (s32)max_label + 1;
    *counts = (s32*)calloc(num_labels, sizeof(s32));

    // Count voxels for each label
    for (s32 z = 0; z < labeled_chunk->dims[0]; z++) {
        for (s32 y = 0; y < labeled_chunk->dims[1]; y++) {
            for (s32 x = 0; x < labeled_chunk->dims[2]; x++) {
                f32 label = vs_chunk_get(labeled_chunk, z, y, x);
                (*counts)[(s32)label]++;
            }
        }
    }

    return num_labels;
}

// mesh

mesh* vs_mesh_new(f32 *vertices,
                    f32 *normals, // can be NULL if no normals
                    s32 *indices,
                    u8* colors, // can be NULL if no colors
                    s32 vertex_count,
                    s32 index_count) {

    mesh* ret = malloc(sizeof(mesh));
    ret->vertices = vertices;
    ret->indices = indices;
    ret->normals = normals;
    ret->colors = colors;
    ret->vertex_count = vertex_count;
    ret->index_count = index_count;
    return ret;
}


void vs_mesh_free(mesh *mesh) {
    if (mesh) {
        free(mesh->vertices);
        free(mesh->indices);
        free(mesh->normals);
        free(mesh->colors);
        free(mesh);
    }
}

void vs_mesh_get_bounds(const mesh *m,
                    f32 *origin_z, f32 *origin_y, f32 *origin_x,
                    f32 *length_z, f32 *length_y, f32 *length_x) {
    if (!m || !m->vertices || m->vertex_count <= 0) {
        if (origin_z) *origin_z = 0.0f;
        if (origin_y) *origin_y = 0.0f;
        if (origin_x) *origin_x = 0.0f;
        if (length_z) *length_z = 0.0f;
        if (length_y) *length_y = 0.0f;
        if (length_x) *length_x = 0.0f;
        return;
    }

    f32 min_z = m->vertices[0];
    f32 max_z = m->vertices[0];
    f32 min_y = m->vertices[1];
    f32 max_y = m->vertices[1];
    f32 min_x = m->vertices[2];
    f32 max_x = m->vertices[2];

    for (s32 i = 0; i < m->vertex_count * 3; i += 3) {
        if (m->vertices[i] < min_z) min_z = m->vertices[i];
        if (m->vertices[i] > max_z) max_z = m->vertices[i];

        if (m->vertices[i + 1] < min_y) min_y = m->vertices[i + 1];
        if (m->vertices[i + 1] > max_y) max_y = m->vertices[i + 1];

        if (m->vertices[i + 2] < min_x) min_x = m->vertices[i + 2];
        if (m->vertices[i + 2] > max_x) max_x = m->vertices[i + 2];
    }

    if (origin_z) *origin_z = min_z;
    if (origin_y) *origin_y = min_y;
    if (origin_x) *origin_x = min_x;

    if (length_z) *length_z = max_z - min_z;
    if (length_y) *length_y = max_y - min_y;
    if (length_x) *length_x = max_x - min_x;
}

void vs_mesh_translate(mesh *m, f32 z, f32 y, f32 x) {
    if (!m || !m->vertices || m->vertex_count <= 0) {
        return;
    }

    for (s32 i = 0; i < m->vertex_count * 3; i += 3) {
        m->vertices[i]     += z;  // Z
        m->vertices[i + 1] += y;  // Y
        m->vertices[i + 2] += x;  // X
    }
}

void vs_mesh_scale(mesh *m, f32 scale_z, f32 scale_y, f32 scale_x) {
    if (!m || !m->vertices || m->vertex_count <= 0) {
        return;
    }

    for (s32 i = 0; i < m->vertex_count * 3; i += 3) {
        m->vertices[i]     *= scale_z;  // Z
        m->vertices[i + 1] *= scale_y;  // Y
        m->vertices[i + 2] *= scale_x;  // X
    }

    // If normals are present, we need to renormalize them after non-uniform scaling
    if (m->normals) {
        for (s32 i = 0; i < m->vertex_count * 3; i += 3) {
            // Scale the normal vector
            f32 nz = m->normals[i]     * scale_z;
            f32 ny = m->normals[i + 1] * scale_y;
            f32 nx = m->normals[i + 2] * scale_x;

            // Calculate length for normalization
            f32 length = sqrtf(nz * nz + ny * ny + nx * nx);

            // Avoid division by zero
            if (length > 0.0001f) {
                m->normals[i]     = nz / length;
                m->normals[i + 1] = ny / length;
                m->normals[i + 2] = nx / length;
            }
        }
    }
}

static void vs__interpolate_vertex_and_value(f32 isovalue,
                                    f32 v1, f32 v2,
                                    f32 x1, f32 y1, f32 z1,
                                    f32 x2, f32 y2, f32 z2,
                                    f32* out_x, f32* out_y, f32* out_z,
                                    f32* out_value) {
    if (fabs(isovalue - v1) < 0.00001f) {
        *out_x = x1;
        *out_y = y1;
        *out_z = z1;
        *out_value = v1;
        return;
    }
    if (fabs(isovalue - v2) < 0.00001f) {
        *out_x = x2;
        *out_y = y2;
        *out_z = z2;
        *out_value = v2;
        return;
    }
    if (fabs(v1 - v2) < 0.00001f) {
        *out_x = x1;
        *out_y = y1;
        *out_z = z1;
        *out_value = v1;
        return;
    }

    f32 mu = (isovalue - v1) / (v2 - v1);
    *out_x = x1 + mu * (x2 - x1);
    *out_y = y1 + mu * (y2 - y1);
    *out_z = z1 + mu * (z2 - z1);

    *out_value = v1;
}

static f32 vs__get_value(const f32* values, s32 x, s32 y, s32 z,
                            s32 dimx, s32 dimy, s32 dimz) {
    return values[z * (dimx * dimy) + y * dimx + x];
}

static void vs__process_cube(const f32* values,
                        s32 x, s32 y, s32 z,
                        s32 dimx, s32 dimy, s32 dimz,
                        f32 isovalue,
                        f32* vertices,
                        f32* colors,
                        s32* indices,
                        s32* vertex_count,
                        s32* index_count) {


static const s32 edgeTable[256]={
0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };

static const s32 triTable[256][16] =
{ {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} };

    f32 cube_values[8];
    cube_values[0] = vs__get_value(values, x, y, z, dimx, dimy, dimz);
    cube_values[1] = vs__get_value(values, x + 1, y, z, dimx, dimy, dimz);
    cube_values[2] = vs__get_value(values, x + 1, y + 1, z, dimx, dimy, dimz);
    cube_values[3] = vs__get_value(values, x, y + 1, z, dimx, dimy, dimz);
    cube_values[4] = vs__get_value(values, x, y, z + 1, dimx, dimy, dimz);
    cube_values[5] = vs__get_value(values, x + 1, y, z + 1, dimx, dimy, dimz);
    cube_values[6] = vs__get_value(values, x + 1, y + 1, z + 1, dimx, dimy, dimz);
    cube_values[7] = vs__get_value(values, x, y + 1, z + 1, dimx, dimy, dimz);

    s32 cubeindex = 0;
    for (s32 i = 0; i < 8; i++) {
        if (cube_values[i] < isovalue)
            cubeindex |= (1 << i);
    }

    if (edgeTable[cubeindex] == 0)
        return;

    f32 edge_verts[12][3];  // [x,y,z] for each possible edge vertex
    f32 edge_values[12];    // scalar field value for each edge vertex

    if (edgeTable[cubeindex] & 1)
        vs__interpolate_vertex_and_value(isovalue, cube_values[0], cube_values[1],
                         x, y, z,             // vertex 0
                         x + 1, y, z,         // vertex 1
                         &edge_verts[0][0], &edge_verts[0][1], &edge_verts[0][2],
                         &edge_values[0]);

    if (edgeTable[cubeindex] & 2)
        vs__interpolate_vertex_and_value(isovalue, cube_values[1], cube_values[2],
                         x + 1, y, z,         // vertex 1
                         x + 1, y + 1, z,     // vertex 2
                         &edge_verts[1][0], &edge_verts[1][1], &edge_verts[1][2],
                         &edge_values[1]);

    if (edgeTable[cubeindex] & 4)
        vs__interpolate_vertex_and_value(isovalue, cube_values[2], cube_values[3],
                         x + 1, y + 1, z,     // vertex 2
                         x, y + 1, z,         // vertex 3
                         &edge_verts[2][0], &edge_verts[2][1], &edge_verts[2][2],
                         &edge_values[2]);

    if (edgeTable[cubeindex] & 8)
        vs__interpolate_vertex_and_value(isovalue, cube_values[3], cube_values[0],
                         x, y + 1, z,         // vertex 3
                         x, y, z,             // vertex 0
                         &edge_verts[3][0], &edge_verts[3][1], &edge_verts[3][2],
                         &edge_values[3]);

    if (edgeTable[cubeindex] & 16)
        vs__interpolate_vertex_and_value(isovalue, cube_values[4], cube_values[5],
                         x, y, z + 1,         // vertex 4
                         x + 1, y, z + 1,     // vertex 5
                         &edge_verts[4][0], &edge_verts[4][1], &edge_verts[4][2],
                         &edge_values[4]);

    if (edgeTable[cubeindex] & 32)
        vs__interpolate_vertex_and_value(isovalue, cube_values[5], cube_values[6],
                         x + 1, y, z + 1,     // vertex 5
                         x + 1, y + 1, z + 1, // vertex 6
                         &edge_verts[5][0], &edge_verts[5][1], &edge_verts[5][2],
                         &edge_values[5]);

    if (edgeTable[cubeindex] & 64)
        vs__interpolate_vertex_and_value(isovalue, cube_values[6], cube_values[7],
                         x + 1, y + 1, z + 1, // vertex 6
                         x, y + 1, z + 1,     // vertex 7
                         &edge_verts[6][0], &edge_verts[6][1], &edge_verts[6][2],
                         &edge_values[6]);

    if (edgeTable[cubeindex] & 128)
        vs__interpolate_vertex_and_value(isovalue, cube_values[7], cube_values[4],
                         x, y + 1, z + 1,     // vertex 7
                         x, y, z + 1,         // vertex 4
                         &edge_verts[7][0], &edge_verts[7][1], &edge_verts[7][2],
                         &edge_values[7]);

    if (edgeTable[cubeindex] & 256)
        vs__interpolate_vertex_and_value(isovalue, cube_values[0], cube_values[4],
                         x, y, z,             // vertex 0
                         x, y, z + 1,         // vertex 4
                         &edge_verts[8][0], &edge_verts[8][1], &edge_verts[8][2],
                         &edge_values[8]);

    if (edgeTable[cubeindex] & 512)
        vs__interpolate_vertex_and_value(isovalue, cube_values[1], cube_values[5],
                         x + 1, y, z,         // vertex 1
                         x + 1, y, z + 1,     // vertex 5
                         &edge_verts[9][0], &edge_verts[9][1], &edge_verts[9][2],
                         &edge_values[9]);

    if (edgeTable[cubeindex] & 1024)
        vs__interpolate_vertex_and_value(isovalue, cube_values[2], cube_values[6],
                         x + 1, y + 1, z,     // vertex 2
                         x + 1, y + 1, z + 1, // vertex 6
                         &edge_verts[10][0], &edge_verts[10][1], &edge_verts[10][2],
                         &edge_values[10]);

    if (edgeTable[cubeindex] & 2048)
        vs__interpolate_vertex_and_value(isovalue, cube_values[3], cube_values[7],
                         x, y + 1, z,         // vertex 3
                         x, y + 1, z + 1,     // vertex 7
                         &edge_verts[11][0], &edge_verts[11][1], &edge_verts[11][2],
                         &edge_values[11]);

    for (s32 i = 0; triTable[cubeindex][i] != -1; i += 3) {
        for (s32 j = 0; j < 3; j++) {
            s32 edge = triTable[cubeindex][i + j];
            vertices[*vertex_count * 3] = edge_verts[edge][0];
            vertices[*vertex_count * 3 + 1] = edge_verts[edge][1];
            vertices[*vertex_count * 3 + 2] = edge_verts[edge][2];

            colors[*vertex_count] = edge_values[edge];  // Store the interpolated value

            indices[*index_count] = *vertex_count;

            (*vertex_count)++;
            (*index_count)++;
        }
    }
}

s32 vs_march_cubes(const f32* values,
                s32 dimz, s32 dimy, s32 dimx,
                f32 isovalue,
                f32** out_vertices,      //  [z,y,x,z,y,x,...]
                f32** out_colors,        //  [value, value, value, ...]
                s32** out_indices,
                s32* out_vertex_count,
                s32* out_index_count) {

    s32 max_triangles = (dimx - 1) * (dimy - 1) * (dimz - 1) * 5;

    f32* vertices = malloc(sizeof(f32) * max_triangles * 3 * 3); // 3 vertices per tri, 3 coords per vertex
    f32* colors = malloc(sizeof(f32) * max_triangles * 3);       // 1 value per vertex
    s32* indices = malloc(sizeof(s32) * max_triangles * 3);      // 3 indices per triangle

    if (!vertices || !colors || !indices) {
        free(vertices);
        free(colors);
        free(indices);
        return 1;
    }

    s32 vertex_count = 0;
    s32 index_count = 0;

    for (s32 z = 0; z < dimz - 1; z++) {
        for (s32 y = 0; y < dimy - 1; y++) {
            for (s32 x = 0; x < dimx - 1; x++) {
                vs__process_cube(values, x, y, z, dimx, dimy, dimz,
                           isovalue, vertices, colors, indices,
                           &vertex_count, &index_count);
            }
        }
    }

    // Shrink arrays to actual size
    vertices = realloc(vertices, sizeof(f32) * vertex_count * 3);
    colors = realloc(colors, sizeof(f32) * vertex_count);
    indices = realloc(indices, sizeof(s32) * index_count);

    *out_vertices = vertices;
    *out_colors = colors;
    *out_indices = indices;
    *out_vertex_count = vertex_count;
    *out_index_count = index_count;

    return 0;
}

rgb viridis_colormap(uint8_t value) {
    static const rgb viridis_colors[256] = {
        {68, 1, 84}, {68, 2, 85}, {68, 3, 87}, {69, 5, 88}, {69, 6, 90},
        {69, 8, 91}, {70, 9, 92}, {70, 11, 94}, {70, 12, 95}, {70, 14, 97},
        {71, 15, 98}, {71, 17, 99}, {71, 18, 101}, {71, 20, 102}, {71, 21, 103},
        {71, 22, 105}, {71, 24, 106}, {72, 25, 107}, {72, 26, 108}, {72, 28, 110},
        {72, 29, 111}, {72, 30, 112}, {72, 32, 113}, {72, 33, 114}, {72, 34, 115},
        {72, 35, 116}, {71, 37, 117}, {71, 38, 118}, {71, 39, 119}, {71, 40, 120},
        {71, 42, 121}, {71, 43, 122}, {71, 44, 123}, {70, 45, 124}, {70, 47, 124},
        {70, 48, 125}, {70, 49, 126}, {69, 50, 127}, {69, 52, 127}, {69, 53, 128},
        {69, 54, 129}, {68, 55, 129}, {68, 57, 130}, {67, 58, 131}, {67, 59, 131},
        {67, 60, 132}, {66, 61, 132}, {66, 62, 133}, {66, 64, 133}, {65, 65, 134},
        {65, 66, 134}, {64, 67, 135}, {64, 68, 135}, {63, 69, 135}, {63, 71, 136},
        {62, 72, 136}, {62, 73, 137}, {61, 74, 137}, {61, 75, 137}, {61, 76, 137},
        {60, 77, 138}, {60, 78, 138}, {59, 80, 138}, {59, 81, 138}, {58, 82, 139},
        {58, 83, 139}, {57, 84, 139}, {57, 85, 139}, {56, 86, 139}, {56, 87, 140},
        {55, 88, 140}, {55, 89, 140}, {54, 90, 140}, {54, 91, 140}, {53, 92, 140},
        {53, 93, 140}, {52, 94, 141}, {52, 95, 141}, {51, 96, 141}, {51, 97, 141},
        {50, 98, 141}, {50, 99, 141}, {49, 100, 141}, {49, 101, 141}, {49, 102, 141},
        {48, 103, 141}, {48, 104, 141}, {47, 105, 141}, {47, 106, 141}, {46, 107, 142},
        {46, 108, 142}, {46, 109, 142}, {45, 110, 142}, {45, 111, 142}, {44, 112, 142},
        {44, 113, 142}, {44, 114, 142}, {43, 115, 142}, {43, 116, 142}, {42, 117, 142},
        {42, 118, 142}, {42, 119, 142}, {41, 120, 142}, {41, 121, 142}, {40, 122, 142},
        {40, 122, 142}, {40, 123, 142}, {39, 124, 142}, {39, 125, 142}, {39, 126, 142},
        {38, 127, 142}, {38, 128, 142}, {38, 129, 142}, {37, 130, 142}, {37, 131, 141},
        {36, 132, 141}, {36, 133, 141}, {36, 134, 141}, {35, 135, 141}, {35, 136, 141},
        {35, 137, 141}, {34, 137, 141}, {34, 138, 141}, {34, 139, 141}, {33, 140, 141},
        {33, 141, 140}, {33, 142, 140}, {32, 143, 140}, {32, 144, 140}, {32, 145, 140},
        {31, 146, 140}, {31, 147, 139}, {31, 148, 139}, {31, 149, 139}, {31, 150, 139},
        {30, 151, 138}, {30, 152, 138}, {30, 153, 138}, {30, 153, 138}, {30, 154, 137},
        {30, 155, 137}, {30, 156, 137}, {30, 157, 136}, {30, 158, 136}, {30, 159, 136},
        {30, 160, 135}, {31, 161, 135}, {31, 162, 134}, {31, 163, 134}, {32, 164, 133},
        {32, 165, 133}, {33, 166, 133}, {33, 167, 132}, {34, 167, 132}, {35, 168, 131},
        {35, 169, 130}, {36, 170, 130}, {37, 171, 129}, {38, 172, 129}, {39, 173, 128},
        {40, 174, 127}, {41, 175, 127}, {42, 176, 126}, {43, 177, 125}, {44, 177, 125},
        {46, 178, 124}, {47, 179, 123}, {48, 180, 122}, {50, 181, 122}, {51, 182, 121},
        {53, 183, 120}, {54, 184, 119}, {56, 185, 118}, {57, 185, 118}, {59, 186, 117},
        {61, 187, 116}, {62, 188, 115}, {64, 189, 114}, {66, 190, 113}, {68, 190, 112},
        {69, 191, 111}, {71, 192, 110}, {73, 193, 109}, {75, 194, 108}, {77, 194, 107},
        {79, 195, 105}, {81, 196, 104}, {83, 197, 103}, {85, 198, 102}, {87, 198, 101},
        {89, 199, 100}, {91, 200, 98}, {94, 201, 97}, {96, 201, 96}, {98, 202, 95},
        {100, 203, 93}, {103, 204, 92}, {105, 204, 91}, {107, 205, 89}, {109, 206, 88},
        {112, 206, 86}, {114, 207, 85}, {116, 208, 84}, {119, 208, 82}, {121, 209, 81},
        {124, 210, 79}, {126, 210, 78}, {129, 211, 76}, {131, 211, 75}, {134, 212, 73},
        {136, 213, 71}, {139, 213, 70}, {141, 214, 68}, {144, 214, 67}, {146, 215, 65},
        {149, 215, 63}, {151, 216, 62}, {154, 216, 60}, {157, 217, 58}, {159, 217, 56},
        {162, 218, 55}, {165, 218, 53}, {167, 219, 51}, {170, 219, 50}, {173, 220, 48},
        {175, 220, 46}, {178, 221, 44}, {181, 221, 43}, {183, 221, 41}, {186, 222, 39},
        {189, 222, 38}, {191, 223, 36}, {194, 223, 34}, {197, 223, 33}, {199, 224, 31},
        {202, 224, 30}, {205, 224, 29}, {207, 225, 28}, {210, 225, 27}, {212, 225, 26},
        {215, 226, 25}, {218, 226, 24}, {220, 226, 24}, {223, 227, 24}, {225, 227, 24},
        {228, 227, 24}, {231, 228, 25}, {233, 228, 25}, {236, 228, 26}, {238, 229, 27},
        {241, 229, 28}, {243, 229, 30}, {246, 230, 31}, {248, 230, 33}, {250, 230, 34},
        {253, 231, 36}
    };
    return viridis_colors[value];
}

int vs_colorize(const f32* grayscale, rgb* colors, s32 vertex_count, f32 min, f32 max) {
    if (max <= min) {
        LOG_ERROR("vs_colorize max (%f) must be greater than min (%f)", max, min);
        return 1;
    }

    f32 range = max - min;
    for (s32 i = 0; i < vertex_count; i++) {
        if (grayscale[i] < min || grayscale[i] > max) {
            LOG_ERROR("vs_colorize grayscale values must be between %f and %f. encountered %f", min, max, grayscale[i]);
            return 1;
        }

        uint8_t value = (uint8_t)(((grayscale[i] - min) / range) * 255.0f);
        colors[i] = viridis_colormap(value);
    }
    return 0;
}

// nrrd
static int vs__nrrd_parse_sizes(char* value, nrrd* nrrd) {
    char* token = strtok(value, " ");
    s32 i = 0;
    while (token != NULL && i < nrrd->dimension) {
        nrrd->sizes[i] = atoi(token);
        if (nrrd->sizes[i] <= 0) {
            LOG_ERROR("Invalid size value: %s", token);
            return 1;
        }
        token = strtok(NULL, " ");
        i++;
    }
    return (i == nrrd->dimension) ? 0 : 1;
}

static int vs__nrrd_parse_space_directions(char* value, nrrd* nrrd) {
    char* token = strtok(value, ") (");
    s32 i = 0;
    while (token != NULL && i < nrrd->dimension) {
        if (strcmp(token, "none") == 0) {
            nrrd->space_directions[i][0] = 0;
            nrrd->space_directions[i][1] = 0;
            nrrd->space_directions[i][2] = 0;
        } else {
            if (sscanf(token, "%f,%f,%f",
                      &nrrd->space_directions[i][0],
                      &nrrd->space_directions[i][1],
                      &nrrd->space_directions[i][2]) != 3) {
                LOG_ERROR("Invalid space direction: %s", token);
                return 1;
            }
        }
        token = strtok(NULL, ") (");
        i++;
    }
    return 0;
}

static int vs__nrrd_parse_space_origin(char* value, nrrd* nrrd) {
    value++; // Skip first '('
    value[strlen(value)-1] = '\0'; // Remove last ')'

    if (sscanf(value, "%f,%f,%f",
               &nrrd->space_origin[0],
               &nrrd->space_origin[1],
               &nrrd->space_origin[2]) != 3) {
        LOG_ERROR("Invalid space origin: %s", value);
        return 1;
    }
    return 0;
}

static size_t vs__nrrd_get_type_size(const char* type) {
    if (strcmp(type, "uint8") == 0 || strcmp(type, "uchar") == 0) return 1;
    if (strcmp(type, "uint16") == 0) return 2;
    if (strcmp(type, "uint32") == 0) return 4;
    if (strcmp(type, "f32") == 0) return 4;
    if (strcmp(type, "double") == 0) return 8;
    return 0;
}

static int vs__nrrd_read_raw_data(FILE* fp, nrrd* nrrd) {
    size_t bytes_read = fread(nrrd->data, 1, nrrd->data_size, fp);
    if (bytes_read != nrrd->data_size) {
        LOG_ERROR("Failed to read data: expected %zu bytes, got %zu",
                nrrd->data_size, bytes_read);
        return 1;
    }
    return 0;
}

static int vs__nrrd_read_gzip_data(FILE* fp, nrrd* nrrd) {
    LOG_ERROR("reading compressed data is not supported yet for nrrd\n");
    return 1;
    #if 0
    z_stream strm = {0};
    unsigned char in[16384];
    size_t bytes_written = 0;

    if (inflateInit2(&strm,-MAX_WBITS) != Z_OK) {
        printf("Failed to initialize zlib");
        return 1;
    }

    s32 ret;
    do {
        strm.avail_in = fread(in, 1, sizeof(in), fp);
        if (ferror(fp)) {
            inflateEnd(&strm);
            printf("Error reading compressed data");
            return 1;
        }
        if (strm.avail_in == 0) break;
        strm.next_in = in;

        do {
            strm.avail_out = nrrd->data_size - bytes_written;
            strm.next_out = (unsigned char*)nrrd->data + bytes_written;
            ret = inflate(&strm, Z_NO_FLUSH);

            if (ret == Z_NEED_DICT || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
                inflateEnd(&strm);
                printf("Decompression error: %s", strm.msg);
                return 1;
            }

            bytes_written = nrrd->data_size - strm.avail_out;

        } while (strm.avail_out == 0);

    } while (ret != Z_STREAM_END);

    inflateEnd(&strm);
    return 0;
    #endif
}

nrrd* vs_nrrd_read(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        LOG_ERROR("could not open %s\n",filename);
        return NULL;
    }

    nrrd* ret = calloc(1, sizeof(nrrd));
    if (!ret) {

        LOG_ERROR("could not allocate ram for nrrd\n");
        fclose(fp);
        return NULL;
    }
    ret->is_valid = true;

    char line[MAX_LINE_LENGTH];
    if (!fgets(line, sizeof(line), fp)) {
        LOG_ERROR("Failed to read magic");
        ret->is_valid = false;
        goto cleanup;
    }
    vs__trim(line);

    if (!vs__str_starts_with(line, "NRRD")) {
        LOG_ERROR("Not a NRRD file: %s", line);
        ret->is_valid = false;
        goto cleanup;
    }

    while (fgets(line, sizeof(line), fp)) {
        vs__trim(line);

        // Empty line marks end of header
        if (strlen(line) == 0) break;

        //if we are left with just a newline after vs__trimming then we have a blank line, we are going to
        // start reading data now so we need to break
        if(line[0] == '\n') break;

        // Skip comments
        if (line[0] == '#') continue;

        char* separator = strchr(line, ':');
        if (!separator) continue;

        *separator = '\0';
        char* key = line;
        char* value = separator + 1;
        while (*value == ' ') value++;

        vs__trim(key);
        vs__trim(value);

        if (strcmp(key, "type") == 0) {
            strncpy(ret->type, value, sizeof(ret->type)-1);
        }
        else if (strcmp(key, "dimension") == 0) {
            ret->dimension = atoi(value);
            if (ret->dimension <= 0 || ret->dimension > 16) {
                LOG_ERROR("Invalid dimension: %d", ret->dimension);
                ret->is_valid = false;
                goto cleanup;
            }
        }
        else if (strcmp(key, "space") == 0) {
            strncpy(ret->space, value, sizeof(ret->space)-1);
        }
        else if (strcmp(key, "sizes") == 0) {
            if (!vs__nrrd_parse_sizes(value, ret)) {
                ret->is_valid = false;
                goto cleanup;
            }
        }
        else if (strcmp(key, "space directions") == 0) {
            if (!vs__nrrd_parse_space_directions(value, ret)) {
                ret->is_valid = false;
                goto cleanup;
            }
        }
        else if (strcmp(key, "endian") == 0) {
            strncpy(ret->endian, value, sizeof(ret->endian)-1);
        }
        else if (strcmp(key, "encoding") == 0) {
            strncpy(ret->encoding, value, sizeof(ret->encoding)-1);
        }
        else if (strcmp(key, "space origin") == 0) {
            if (!vs__nrrd_parse_space_origin(value, ret)) {
                ret->is_valid = false;
                goto cleanup;
            }
        }
    }

    size_t type_size = vs__nrrd_get_type_size(ret->type);
    if (type_size == 0) {
        LOG_ERROR("Unsupported type: %s", ret->type);
        ret->is_valid = false;
        goto cleanup;
    }

    ret->data_size = type_size;
    for (s32 i = 0; i < ret->dimension; i++) {
        ret->data_size *= ret->sizes[i];
    }

    ret->data = malloc(ret->data_size);
    if (!ret->data) {
        LOG_ERROR("Failed to allocate %zu bytes", ret->data_size);
        ret->is_valid = false;
        goto cleanup;
    }

    if (strcmp(ret->encoding, "raw") == 0) {
        if (!vs__nrrd_read_raw_data(fp, ret)) {
            ret->is_valid = false;
            goto cleanup;
        }
    }
    else if (strcmp(ret->encoding, "gzip") == 0) {
        if (!vs__nrrd_read_gzip_data(fp, ret)) {
            ret->is_valid = false;
            goto cleanup;
        }
    }
    else {
        LOG_ERROR("Unsupported encoding: %s", ret->encoding);
        ret->is_valid = false;
        goto cleanup;
    }

cleanup:
    fclose(fp);
    if (!ret->is_valid) {
        if (ret->data) free(ret->data);
        free(ret);
        return NULL;
    }
    return ret;
}

void vs_nrrd_free(nrrd* nrrd) {
    if (nrrd) {
        if (nrrd->data) free(nrrd->data);
        free(nrrd);
    }
}

// obj

s32 vs_read_obj(const char* filename,
            f32** vertices, s32** indices,
            s32* vertex_count, s32* index_count) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        return 1;
    }

    size_t vertex_capacity = 1024;
    size_t index_capacity = 1024;
    *vertices = malloc(vertex_capacity * 3 * sizeof(f32));
    *indices = malloc(index_capacity * sizeof(s32));
    *vertex_count = 0;
    *index_count = 0;

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'v' && line[1] == ' ') {
            if (*vertex_count >= vertex_capacity) {
                vertex_capacity *= 2;
                f32* new_vertices = realloc(*vertices, vertex_capacity * 3 * sizeof(f32));
                if (!new_vertices) {
                    fclose(fp);
                    return 1;
                }
                *vertices = new_vertices;
            }

            // Read vertex coordinates
            f32 x, y, z;
            if (sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3) {
                (*vertices)[(*vertex_count) * 3] = x;
                (*vertices)[(*vertex_count) * 3 + 1] = y;
                (*vertices)[(*vertex_count) * 3 + 2] = z;
                (*vertex_count)++;
            }
        }
        else if (line[0] == 'f' && line[1] == ' ') {
            // Parse face indices
            s32 v1, v2, v3, t1, t2, t3, n1, n2, n3;
            s32 matches = sscanf(line + 2, "%d/%d/%d %d/%d/%d %d/%d/%d",
                               &v1, &t1, &n1, &v2, &t2, &n2, &v3, &t3, &n3);

            if (matches != 9) {
                // Try parsing without texture/normal indices
                matches = sscanf(line + 2, "%d %d %d", &v1, &v2, &v3);
                if (matches != 3) {
                    continue;  // Skip malformed faces
                }
            }

            if (*index_count + 3 > index_capacity) {
                index_capacity *= 2;
                s32* new_indices = realloc(*indices, index_capacity * sizeof(s32));
                if (!new_indices) {
                    fclose(fp);
                    return 1;
                }
                *indices = new_indices;
            }

            // Store face indices (converting from 1-based to 0-based)
            (*indices)[(*index_count)++] = v1 - 1;
            (*indices)[(*index_count)++] = v2 - 1;
            (*indices)[(*index_count)++] = v3 - 1;
        }
    }

    // Shrink arrays to actual size
    *vertices = realloc(*vertices, (*vertex_count) * 3 * sizeof(f32));
    *indices = realloc(*indices, (*index_count) * sizeof(s32));

    fclose(fp);
    return 0;
}

s32 vs_write_obj(const char* filename,
             const f32* vertices, const s32* indices,
             s32 vertex_count, s32 index_count) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        return 1;
    }

    fprintf(fp, "# OBJ file created by minilibs/miniobj\n");

    // Write vertices
    for (s32 i = 0; i < vertex_count; i++) {
        fprintf(fp, "v %.6f %.6f %.6f\n",
                vertices[i * 3],
                vertices[i * 3 + 1],
                vertices[i * 3 + 2]);
    }

    // Write faces (converting from 0-based to 1-based indices)
    assert(index_count % 3 == 0);  // Ensure we have complete triangles
    for (s32 i = 0; i < index_count; i += 3) {
        fprintf(fp, "f %d %d %d\n",
                indices[i] + 1,
                indices[i + 1] + 1,
                indices[i + 2] + 1);
    }

    fclose(fp);
    return 0;
}

// ply

//TODO: most the ply files I come across use x y z order. should we swap the order here so they
// end up in the data as z y x?

s32 vs_ply_write(const char *filename,
                    const f32 *vertices,
                    const f32 *normals,
                    const rgb *colors,
                    const s32 *indices,
                    s32 vertex_count,
                    s32 index_count) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        return 1;
    }
    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "comment Created by minilibs\n");
    fprintf(fp, "element vertex %d\n", vertex_count);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    if (normals) {
        fprintf(fp, "property float nx\n");
        fprintf(fp, "property float ny\n");
        fprintf(fp, "property float nz\n");
    }
    if (colors) {
        fprintf(fp, "property uchar red\n");
        fprintf(fp, "property uchar green\n");
        fprintf(fp, "property uchar blue\n");
    }
    fprintf(fp, "element face %d\n", index_count / 3);
    fprintf(fp, "property list uchar int vertex_indices\n");
    fprintf(fp, "end_header\n");

    for (s32 i = 0; i < vertex_count; i++) {
        if (normals && colors) {
            fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %d %d %d\n",
                    vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2],
                    normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2],
                    colors[i].r, colors[i].g, colors[i].b);
        } else if (normals) {
            fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f\n",
                    vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2],
                    normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
        } else if (colors) {
            fprintf(fp, "%.6f %.6f %.6f %d %d %d\n",
                    vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2],
                    colors[i].r, colors[i].g, colors[i].b);
        } else {
            fprintf(fp, "%.6f %.6f %.6f\n",
                    vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
        }
    }

    for (s32 i = 0; i < index_count; i += 3) {
        fprintf(fp, "3 %d %d %d\n", indices[i], indices[i + 1], indices[i + 2]);
    }
    fclose(fp);
    return 0;
}

s32 vs_ply_read(const char *filename,
                          f32 **out_vertices,
                          f32 **out_normals,
                          s32 **out_indices,
                          s32 *out_vertex_count,
                          s32 *out_normal_count,
                          s32 *out_index_count) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    return 1;
  }

  char buffer[1024];
  if (!fgets(buffer, sizeof(buffer), fp) || strncmp(buffer, "ply", 3) != 0) {
    fclose(fp);
    return 1;
  }

  // Check format
  if (!fgets(buffer, sizeof(buffer), fp)) {
    fclose(fp);
    return 1;
  }
  s32 is_binary = (strncmp(buffer, "format binary_little_endian", 26) == 0);
  s32 is_ascii = (strncmp(buffer, "format ascii", 11) == 0);
  if (!is_binary && !is_ascii) {
    fclose(fp);
    return 1;
  }

  s32 vertex_count = 0;
  s32 face_count = 0;
  s32 has_normals = 0;
  s32 in_header = 1;
  s32 got_vertex = 0;
  s32 got_face = 0;
  s32 is_double = 0;  // Track if the file uses doubles

  // Parse header
  while (in_header && fgets(buffer, sizeof(buffer), fp)) {
    if (strncmp(buffer, "end_header", 10) == 0) {
      in_header = 0;
    } else if (strncmp(buffer, "element vertex", 13) == 0) {
      sscanf(buffer, "element vertex %d", &vertex_count);
      got_vertex = 1;
    } else if (strncmp(buffer, "element face", 12) == 0) {
      sscanf(buffer, "element face %d", &face_count);
      got_face = 1;
    } else if (strncmp(buffer, "property double", 14) == 0) {
      is_double = 1;  // File uses doubles
    } else if (strncmp(buffer, "property double nx", 17) == 0) {
      has_normals = 1;
    }
  }

  if (!got_vertex || vertex_count <= 0) {
    fclose(fp);
    return 1;
  }

  // Allocate memory for float32 output
  f32 *vertices = malloc(vertex_count * 3 * sizeof(f32));
  f32 *normals = NULL;
  s32 *indices = NULL;

  if (has_normals) {
    normals = malloc(vertex_count * 3 * sizeof(f32));
    if (!normals) {
      free(vertices);
      fclose(fp);
      return 1;
    }
  }

  if (got_face && face_count > 0) {
    indices = malloc(face_count * 3 * sizeof(s32));
    if (!indices) {
      free(vertices);
      free(normals);
      fclose(fp);
      return 1;
    }
  }

  if (!vertices) {
    free(normals);
    free(indices);
    fclose(fp);
    return 1;
  }

  // Read vertex data
  if (is_binary) {
    if (is_double) {
      // Reading doubles and converting to floats
      double temp[6];  // Temporary buffer for doubles (3 for position, 3 for normals)
      for (s32 i = 0; i < vertex_count; i++) {
        // Read position as double and convert to f32
        if (fread(temp, sizeof(double), 3, fp) != 3) {
          free(vertices);
          free(normals);
          free(indices);
          fclose(fp);
          return 1;
        }
        vertices[i * 3] = (f32)temp[0];
        vertices[i * 3 + 1] = (f32)temp[1];
        vertices[i * 3 + 2] = (f32)temp[2];

        // Read normals if present
        if (has_normals) {
          if (fread(temp, sizeof(double), 3, fp) != 3) {
            free(vertices);
            free(normals);
            free(indices);
            fclose(fp);
            return 1;
          }
          normals[i * 3] = (f32)temp[0];
          normals[i * 3 + 1] = (f32)temp[1];
          normals[i * 3 + 2] = (f32)temp[2];
        }
      }
    } else {
      // Reading floats directly
      for (s32 i = 0; i < vertex_count; i++) {
        // Read position
        if (fread(&vertices[i * 3], sizeof(f32), 3, fp) != 3) {
          free(vertices);
          free(normals);
          free(indices);
          fclose(fp);
          return 1;
        }

        // Read normals if present
        if (has_normals) {
          if (fread(&normals[i * 3], sizeof(f32), 3, fp) != 3) {
            free(vertices);
            free(normals);
            free(indices);
            fclose(fp);
            return 1;
          }
        }
      }
    }
  } else {
    // ASCII reading - read as double and convert to f32
    double temp[6];  // Temporary buffer for doubles
    for (s32 i = 0; i < vertex_count; i++) {
      if (has_normals) {
        if (fscanf(fp, "%lf %lf %lf %lf %lf %lf",
                   &temp[0], &temp[1], &temp[2],
                   &temp[3], &temp[4], &temp[5]) != 6) {
          free(vertices);
          free(normals);
          free(indices);
          fclose(fp);
          return 1;
        }
        vertices[i * 3] = (f32)temp[0];
        vertices[i * 3 + 1] = (f32)temp[1];
        vertices[i * 3 + 2] = (f32)temp[2];
        normals[i * 3] = (f32)temp[3];
        normals[i * 3 + 1] = (f32)temp[4];
        normals[i * 3 + 2] = (f32)temp[5];
      } else {
        if (fscanf(fp, "%lf %lf %lf",
                   &temp[0], &temp[1], &temp[2]) != 3) {
          free(vertices);
          free(normals);
          free(indices);
          fclose(fp);
          return 1;
        }
        vertices[i * 3] = (f32)temp[0];
        vertices[i * 3 + 1] = (f32)temp[1];
        vertices[i * 3 + 2] = (f32)temp[2];
      }
    }
  }

  // Read face data if present
  s32 index_count = 0;
  if (got_face && indices) {
    if (is_binary) {
      for (s32 i = 0; i < face_count; i++) {
        unsigned char vertex_per_face;
        if (fread(&vertex_per_face, sizeof(unsigned char), 1, fp) != 1 || vertex_per_face != 3) {
          free(vertices);
          free(normals);
          free(indices);
          fclose(fp);
          return 1;
        }

        if (fread(&indices[index_count], sizeof(s32), 3, fp) != 3) {
          free(vertices);
          free(normals);
          free(indices);
          fclose(fp);
          return 1;
        }
        index_count += 3;
      }
    } else {
      for (s32 i = 0; i < face_count; i++) {
        s32 vertex_per_face;
        if (fscanf(fp, "%d", &vertex_per_face) != 1 || vertex_per_face != 3) {
          free(vertices);
          free(normals);
          free(indices);
          fclose(fp);
          return 1;
        }

        if (fscanf(fp, "%d %d %d",
                   &indices[index_count],
                   &indices[index_count + 1],
                   &indices[index_count + 2]) != 3) {
          free(vertices);
          free(normals);
          free(indices);
          fclose(fp);
          return 1;
        }
        index_count += 3;
      }
    }
  }

  fclose(fp);

  *out_vertices = vertices;
  *out_normals = normals;
  *out_indices = indices;
  *out_vertex_count = vertex_count;
  *out_normal_count = has_normals ? vertex_count : 0;
  *out_index_count = index_count;

  return 0;
}

// ppm

ppm* vs_ppm_new(u32 width, u32 height) {
    ppm* img = malloc(sizeof(ppm));
    if (!img) {
        return NULL;
    }

    img->width = width;
    img->height = height;
    img->max_val = 255;
    img->data = calloc(width * height * 3, sizeof(u8));

    if (!img->data) {
        free(img);
        return NULL;
    }

    return img;
}

void vs_ppm_free(ppm* img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

static void vs__skip_whitespace_and_comments(FILE* fp) {
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (c == '#') {
            // Skip until end of line
            while ((c = fgetc(fp)) != EOF && c != '\n');
        } else if (!isspace(c)) {
            ungetc(c, fp);
            break;
        }
    }
}

static bool vs__ppm_read_header(FILE* fp, ppm_type* type, u32* width, u32* height, u8* max_val) {
    char magic[3];

    if (fgets(magic, sizeof(magic), fp) == NULL) {
        return false;
    }

    if (magic[0] != 'P' || (magic[1] != '3' && magic[1] != '6')) {
        return false;
    }

    *type = magic[1] == '3' ? P3 : P6;

    vs__skip_whitespace_and_comments(fp);

    if (fscanf(fp, "%u %u", width, height) != 2) {
        return false;
    }

    vs__skip_whitespace_and_comments(fp);

    unsigned int max_val_temp;
    if (fscanf(fp, "%u", &max_val_temp) != 1 || max_val_temp > 255) {
        return false;
    }
    *max_val = (u8)max_val_temp;

    fgetc(fp);

    return true;
}

ppm* vs_ppm_read(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        return NULL;
    }

    ppm_type type;
    u32 width, height;
    u8 max_val;

    if (!vs__ppm_read_header(fp, &type, &width, &height, &max_val)) {
        fclose(fp);
        return NULL;
    }

    ppm* img = vs_ppm_new(width, height);
    if (!img) {
        fclose(fp);
        return NULL;
    }

    img->max_val = max_val;
    size_t pixel_count = width * height * 3;

    if (type == P3) {
        // ASCII format
        for (size_t i = 0; i < pixel_count; i++) {
            unsigned int val;
            if (fscanf(fp, "%u", &val) != 1 || val > max_val) {
                vs_ppm_free(img);
                fclose(fp);
                return NULL;
            }
            img->data[i] = (u8)val;
        }
    } else {
        // Binary format
        if (fread(img->data, 1, pixel_count, fp) != pixel_count) {
            vs_ppm_free(img);
            fclose(fp);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    return img;
}

int vs_ppm_write(const char* filename, const ppm* img, ppm_type type) {
    if (!img || !img->data) {
        return 1;
    }

    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        return 1;
    }

    // Write header
    fprintf(fp, "P%c\n", type == P3 ? '3' : '6');
    fprintf(fp, "%u %u\n", img->width, img->height);
    fprintf(fp, "%u\n", img->max_val);

    size_t pixel_count = img->width * img->height * 3;

    if (type == P3) {
        // ASCII format
        for (size_t i = 0; i < pixel_count; i++) {
            fprintf(fp, "%u", img->data[i]);
            fprintf(fp, (i + 1) % 3 == 0 ? "\n" : " ");
        }
    } else {
        // Binary format
        if (fwrite(img->data, 1, pixel_count, fp) != pixel_count) {
            fclose(fp);
            return 1;
        }
    }

    fclose(fp);
    return 0;
}

void vs_ppm_set_pixel(ppm* img, u32 x, u32 y, u8 r, u8 g, u8 b) {
    if (!img || x >= img->width || y >= img->height) {
        return;
    }

    size_t idx = (y * img->width + x) * 3;
    img->data[idx] = r;
    img->data[idx + 1] = g;
    img->data[idx + 2] = b;
}

void vs_ppm_get_pixel(const ppm* img, u32 x, u32 y, u8* r, u8* g, u8* b) {
    if (!img || x >= img->width || y >= img->height) {
        *r = *g = *b = 0;
        return;
    }

    size_t idx = (y * img->width + x) * 3;
    *r = img->data[idx];
    *g = img->data[idx + 1];
    *b = img->data[idx + 2];
}

// tiff
static uint32_t vs__tiff_read_bytes(FILE* fp, int count, int littleEndian) {
    uint32_t value = 0;
    uint8_t byte;

    if (littleEndian) {
        for (int i = 0; i < count; i++) {
            if (fread(&byte, 1, 1, fp) != 1) return 0;
            value |= ((uint32_t)byte << (i * 8));
        }
    } else {
        for (int i = 0; i < count; i++) {
            if (fread(&byte, 1, 1, fp) != 1) return 0;
            value = (value << 8) | byte;
        }
    }

    return value;
}

static void vs__tiff_read_string(FILE* fp, char* str, uint32_t offset, uint32_t count, long currentPos) {
    long savedPos = ftell(fp);
    fseek(fp, offset, SEEK_SET);
    fread(str, 1, count - 1, fp);
    str[count - 1] = '\0';
    fseek(fp, savedPos, SEEK_SET);
}

static float vs__tiff_read_rational(FILE* fp, uint32_t offset, int littleEndian, long currentPos) {
    long savedPos = ftell(fp);
    fseek(fp, offset, SEEK_SET);
    uint32_t numerator = vs__tiff_read_bytes(fp, 4, littleEndian);
    uint32_t denominator = vs__tiff_read_bytes(fp, 4, littleEndian);
    fseek(fp, savedPos, SEEK_SET);
    return denominator ? (float)numerator / denominator : 0.0f;
}

static void vs__tiff_read_ifd_entry(FILE* fp, DirectoryInfo* dir, int littleEndian, long ifdStart) {
    uint16_t tag = vs__tiff_read_bytes(fp, 2, littleEndian);
    uint16_t type = vs__tiff_read_bytes(fp, 2, littleEndian);
    uint32_t count = vs__tiff_read_bytes(fp, 4, littleEndian);
    uint32_t valueOffset = vs__tiff_read_bytes(fp, 4, littleEndian);

    long currentPos = ftell(fp);

    switch (tag) {
        case TIFFTAG_SUBFILETYPE:
            dir->subfileType = valueOffset;
            break;
        case TIFFTAG_IMAGEWIDTH:
            dir->width = (type == TIFF_SHORT) ? (uint16_t)valueOffset : valueOffset;
            break;
        case TIFFTAG_IMAGELENGTH:
            dir->height = (type == TIFF_SHORT) ? (uint16_t)valueOffset : valueOffset;
            break;
        case TIFFTAG_BITSPERSAMPLE:
            dir->bitsPerSample = (uint16_t)valueOffset;
            break;
        case TIFFTAG_COMPRESSION:
            dir->compression = (uint16_t)valueOffset;
            break;
        case TIFFTAG_PHOTOMETRIC:
            dir->photometric = (uint16_t)valueOffset;
            break;
        case TIFFTAG_IMAGEDESCRIPTION:
            vs__tiff_read_string(fp, dir->imageDescription, valueOffset, count, currentPos);
            break;
        case TIFFTAG_SOFTWARE:
            vs__tiff_read_string(fp, dir->software, valueOffset, count, currentPos);
            break;
        case TIFFTAG_DATETIME:
            vs__tiff_read_string(fp, dir->dateTime, valueOffset, count, currentPos);
            break;
        case TIFFTAG_SAMPLESPERPIXEL:
            dir->samplesPerPixel = (uint16_t)valueOffset;
            break;
        case TIFFTAG_ROWSPERSTRIP:
            dir->rowsPerStrip = (type == TIFF_SHORT) ? (uint16_t)valueOffset : valueOffset;
            break;
        case TIFFTAG_PLANARCONFIG:
            dir->planarConfig = (uint16_t)valueOffset;
            break;
        case TIFFTAG_XRESOLUTION:
            dir->xResolution = vs__tiff_read_rational(fp, valueOffset, littleEndian, currentPos);
            break;
        case TIFFTAG_YRESOLUTION:
            dir->yResolution = vs__tiff_read_rational(fp, valueOffset, littleEndian, currentPos);
            break;
        case TIFFTAG_RESOLUTIONUNIT:
            dir->resolutionUnit = (uint16_t)valueOffset;
            break;
        case TIFFTAG_SAMPLEFORMAT:
            dir->sampleFormat = (uint16_t)valueOffset;
            break;
        case TIFFTAG_STRIPOFFSETS:
            dir->stripInfo.offset = (type == TIFF_SHORT) ? (uint16_t)valueOffset : valueOffset;
            break;
        case TIFFTAG_STRIPBYTECOUNTS:
            dir->stripInfo.byteCount = (type == TIFF_SHORT) ? (uint16_t)valueOffset : valueOffset;
            break;
        default:
            assert(false);
            break;
    }
}

static bool vs__tiff_validate_directory(DirectoryInfo* dir, TiffImage* img) {
    if (dir->width == 0 || dir->height == 0) {
        snprintf(img->errorMsg, sizeof(img->errorMsg), "Invalid dimensions");
        return false;
    }

    if (dir->bitsPerSample != 8 && dir->bitsPerSample != 16) {
        snprintf(img->errorMsg, sizeof(img->errorMsg),
                "Unsupported bits per sample: %d", dir->bitsPerSample);
        return false;
    }

    if (dir->compression != 1) {
        snprintf(img->errorMsg, sizeof(img->errorMsg),
                "Unsupported compression: %d", dir->compression);
        return false;
    }

    if (dir->samplesPerPixel != 1) {
        snprintf(img->errorMsg, sizeof(img->errorMsg),
                "Only single channel images supported");
        return false;
    }

    if (dir->planarConfig != 1) {
        snprintf(img->errorMsg, sizeof(img->errorMsg),
                "Only contiguous data supported");
        return false;
    }

    size_t expectedSize = dir->width * dir->height * (dir->bitsPerSample / 8);
    if (dir->stripInfo.byteCount != expectedSize) {
        snprintf(img->errorMsg, sizeof(img->errorMsg), "Data size mismatch");
        return false;
    }

    return true;
}

TiffImage* vs_tiff_read(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;

    TiffImage* img = calloc(1, sizeof(TiffImage));
    if (!img) {
        fclose(fp);
        return NULL;
    }

    img->isValid = true;

    uint16_t byteOrder = vs__tiff_read_bytes(fp, 2, 1);
    int littleEndian = (byteOrder == 0x4949);

    if (byteOrder != 0x4949 && byteOrder != 0x4D4D) {
        img->isValid = false;
        snprintf(img->errorMsg, sizeof(img->errorMsg), "Invalid byte order marker");
        fclose(fp);
        return img;
    }

    if (vs__tiff_read_bytes(fp, 2, littleEndian) != 42) {
        img->isValid = false;
        snprintf(img->errorMsg, sizeof(img->errorMsg), "Invalid TIFF version");
        fclose(fp);
        return img;
    }

    // First pass: count directories
    uint32_t ifdOffset = vs__tiff_read_bytes(fp, 4, littleEndian);
    img->depth = 0;
    uint32_t nextIFD = ifdOffset;

    while (nextIFD != 0) {
        img->depth++;
        fseek(fp, nextIFD, SEEK_SET);
        uint16_t numEntries = vs__tiff_read_bytes(fp, 2, littleEndian);
        fseek(fp, 12 * numEntries, SEEK_CUR);  // Skip entries
        nextIFD = vs__tiff_read_bytes(fp, 4, littleEndian);
    }

    // Allocate directory info array
    img->directories = calloc(img->depth, sizeof(DirectoryInfo));
    if (!img->directories) {
        img->isValid = false;
        snprintf(img->errorMsg, sizeof(img->errorMsg), "Memory allocation failed");
        fclose(fp);
        return img;
    }

    // Second pass: read directory information
    nextIFD = ifdOffset;
    int dirIndex = 0;

    while (nextIFD != 0 && img->isValid) {
        DirectoryInfo* currentDir = &img->directories[dirIndex];

        // Set defaults
        currentDir->samplesPerPixel = 1;
        currentDir->planarConfig = 1;
        currentDir->sampleFormat = 1;
        currentDir->compression = 1;

        fseek(fp, nextIFD, SEEK_SET);
        long ifdStart = ftell(fp);

        uint16_t numEntries = vs__tiff_read_bytes(fp, 2, littleEndian);

        for (int i = 0; i < numEntries && img->isValid; i++) {
            vs__tiff_read_ifd_entry(fp, currentDir, littleEndian, ifdStart);
        }

        if (!vs__tiff_validate_directory(currentDir, img)) {
            img->isValid = false;
            break;
        }

        nextIFD = vs__tiff_read_bytes(fp, 4, littleEndian);
        dirIndex++;
    }

    if (img->isValid) {
        DirectoryInfo* firstDir = &img->directories[0];
        size_t sliceSize = firstDir->width * firstDir->height * (firstDir->bitsPerSample / 8);
        img->dataSize = sliceSize * img->depth;
        img->data = malloc(img->dataSize);

        if (!img->data) {
            img->isValid = false;
            snprintf(img->errorMsg, sizeof(img->errorMsg), "Memory allocation failed");
        } else {
            for (int i = 0; i < img->depth && img->isValid; i++) {
                DirectoryInfo* dir = &img->directories[i];
                fseek(fp, dir->stripInfo.offset, SEEK_SET);
                size_t bytesRead = fread((uint8_t*)img->data + (i * sliceSize), 1,
                                       dir->stripInfo.byteCount, fp);
                if (bytesRead != dir->stripInfo.byteCount) {
                    img->isValid = false;
                    snprintf(img->errorMsg, sizeof(img->errorMsg),
                            "Failed to read image data for directory %d", i);
                }
            }
        }
    }

    fclose(fp);
    return img;
}

void vs_tiff_free(TiffImage* img) {
    if (img) {
        free(img->directories);
        free(img->data);
        free(img);
    }
}

const char* vs_tiff_compression_name(uint16_t compression) {
    switch (compression) {
        case 1: return "None";
        case 2: return "CCITT modified Huffman RLE";
        case 3: return "CCITT Group 3 fax encoding";
        case 4: return "CCITT Group 4 fax encoding";
        case 5: return "LZW";
        case 6: return "JPEG (old-style)";
        case 7: return "JPEG";
        case 8: return "Adobe Deflate";
        case 32773: return "PackBits compression";
        default: return "Unknown";
    }
}

const char* vs_tiff_photometric_name(uint16_t photometric) {
    switch (photometric) {
        case 0: return "min-is-white";
        case 1: return "min-is-black";
        case 2: return "RGB";
        case 3: return "palette color";
        case 4: return "transparency mask";
        case 5: return "CMYK";
        case 6: return "YCbCr";
        case 8: return "CIELab";
        default: return "Unknown";
    }
}

const char* vs_tiff_planar_config_name(uint16_t config) {
    switch (config) {
        case 1: return "single image plane";
        case 2: return "separate image planes";
        default: return "Unknown";
    }
}

const char* vs_tiff_sample_format_name(uint16_t format) {
    switch (format) {
        case 1: return "unsigned integer";
        case 2: return "signed integer";
        case 3: return "IEEE floating point";
        case 4: return "undefined";
        default: return "Unknown";
    }
}

const char* vs_tiff_resolution_unit_name(uint16_t unit) {
    switch (unit) {
        case 1: return "unitless";
        case 2: return "inches";
        case 3: return "centimeters";
        default: return "Unknown";
    }
}

void vs_tiff_print_tags(const TiffImage* img, int directory) {
    if (!img || !img->directories || directory >= img->depth) return;

    const DirectoryInfo* dir = &img->directories[directory];

    printf("\n=== TIFF directory %d ===\n", directory);
    printf("TIFF Directory %d\n", directory);

    if (dir->subfileType != 0) {
        printf("  Subfile Type: (%d = 0x%x)\n", dir->subfileType, dir->subfileType);
    }

    printf("  Image Width: %u Image Length: %u\n", dir->width, dir->height);

    if (dir->xResolution != 0 || dir->yResolution != 0) {
        printf("  Resolution: %g, %g (%s)\n",
               dir->xResolution, dir->yResolution,
               vs_tiff_resolution_unit_name(dir->resolutionUnit));
    }

    printf("  Bits/Sample: %u\n", dir->bitsPerSample);
    printf("  Sample Format: %s\n", vs_tiff_sample_format_name(dir->sampleFormat));
    printf("  Compression Scheme: %s\n", vs_tiff_compression_name(dir->compression));
    printf("  Photometric Interpretation: %s\n", vs_tiff_photometric_name(dir->photometric));
    printf("  Samples/Pixel: %u\n", dir->samplesPerPixel);

    if (dir->rowsPerStrip) {
        printf("  Rows/Strip: %u\n", dir->rowsPerStrip);
    }

    printf("  Planar Configuration: %s\n", vs_tiff_planar_config_name(dir->planarConfig));

    if (dir->imageDescription[0]) {
        printf("  ImageDescription: %s\n", dir->imageDescription);
    }
    if (dir->software[0]) {
        printf("  Software: %s\n", dir->software);
    }
    if (dir->dateTime[0]) {
        printf("  DateTime: %s\n", dir->dateTime);
    }
}

void vs_tiff_print_all_tags(const TiffImage* img) {
    if (!img) {
        LOG_ERROR("Error: NULL TIFF image\n");
        return;
    }

    if (!img->isValid) {
        LOG_ERROR("Error reading TIFF: %s\n", img->errorMsg);
        return;
    }

    for (int i = 0; i < img->depth; i++) {
        vs_tiff_print_tags(img, i);
    }
}


size_t vs_tiff_directory_size(const TiffImage* img, int directory) {
    if (!img || !img->isValid || !img->directories || directory >= img->depth) {
        return 0;
    }

    const DirectoryInfo* dir = &img->directories[directory];
    return dir->width * dir->height * (dir->bitsPerSample / 8);
}

void* vs_tiff_read_directory_data(const TiffImage* img, int directory) {

    size_t bufferSize = vs_tiff_directory_size(img, directory);
    void* buffer = malloc(bufferSize);

    if (!img || !img->isValid || !img->directories || !buffer || directory >= img->depth) {
        return NULL;
    }

    const DirectoryInfo* dir = &img->directories[directory];
    size_t sliceSize = dir->width * dir->height * (dir->bitsPerSample / 8);

    if (bufferSize < sliceSize) {
        return NULL;
    }

    size_t offset = sliceSize * directory;
    memcpy(buffer, (uint8_t*)img->data + offset, sliceSize);

    return buffer;
}

uint16_t vs_tiff_pixel16(const uint16_t* buffer, int y, int x, int width) {
    return buffer[ y * width + x];
}

uint8_t vs_tiff_pixel8(const uint8_t* buffer, int y, int x, int width) {
    return buffer[y * width + x];
}


static void vs__tiff_write_bytes(FILE* fp, uint32_t value, int count, int littleEndian) {
    if (littleEndian) {
        for (int i = 0; i < count; i++) {
            uint8_t byte = (value >> (i * 8)) & 0xFF;
            fwrite(&byte, 1, 1, fp);
        }
    } else {
        for (int i = count - 1; i >= 0; i--) {
            uint8_t byte = (value >> (i * 8)) & 0xFF;
            fwrite(&byte, 1, 1, fp);
        }
    }
}

static void vs__tiff_write_string(FILE* fp, const char* str, uint32_t offset) {
    fseek(fp, offset, SEEK_SET);
    size_t len = strlen(str);
    fwrite(str, 1, len + 1, fp);  // Include null terminator
}

static void vs__tiff_write_rational(FILE* fp, float value, uint32_t offset, int littleEndian) {
    fseek(fp, offset, SEEK_SET);
    uint32_t numerator = (uint32_t)(value * 1000);
    uint32_t denominator = 1000;
    vs__tiff_write_bytes(fp, numerator, 4, littleEndian);
    vs__tiff_write_bytes(fp, denominator, 4, littleEndian);
}

static void vs__tiff_current_date_time(char* dateTime) {
    time_t now;
    struct tm* timeinfo;
    time(&now);
    timeinfo = localtime(&now);
    strftime(dateTime, 20, "%Y:%m:%d %H:%M:%S", timeinfo);
}

static uint32_t vs__tiff_write_ifd_entry(FILE* fp, uint16_t tag, uint16_t type, uint32_t count,
                             uint32_t value, int littleEndian) {
    vs__tiff_write_bytes(fp, tag, 2, littleEndian);
    vs__tiff_write_bytes(fp, type, 2, littleEndian);
    vs__tiff_write_bytes(fp, count, 4, littleEndian);
    vs__tiff_write_bytes(fp, value, 4, littleEndian);
    return 12;  // Size of IFD entry
}

int vs_tiff_write(const char* filename, const TiffImage* img, bool littleEndian) {
    if (!img || !img->directories || !img->data || !img->isValid) return 1;

    FILE* fp = fopen(filename, "wb");
    if (!fp) return 1;

    // Write header
    vs__tiff_write_bytes(fp, littleEndian ? 0x4949 : 0x4D4D, 2, 1);  // Byte order marker
    vs__tiff_write_bytes(fp, 42, 2, littleEndian);                    // TIFF version

    uint32_t ifdOffset = 8;  // Start first IFD after header
    vs__tiff_write_bytes(fp, ifdOffset, 4, littleEndian);

    // Calculate space needed for string and rational values
    uint32_t extraDataOffset = ifdOffset;
    for (int d = 0; d < img->depth; d++) {
        extraDataOffset += 2 + (12 * 17) + 4;  // Directory entry count + entries + next IFD pointer
    }

    // Write each directory
    for (int d = 0; d < img->depth; d++) {
        const DirectoryInfo* dir = &img->directories[d];

        // Position at IFD start
        fseek(fp, ifdOffset, SEEK_SET);

        // Write number of directory entries
        vs__tiff_write_bytes(fp, 17, 2, littleEndian);  // Number of IFD entries

        // Write directory entries
        vs__tiff_write_ifd_entry(fp, TIFFTAG_SUBFILETYPE, TIFF_LONG, 1, dir->subfileType, littleEndian);
        vs__tiff_write_ifd_entry(fp, TIFFTAG_IMAGEWIDTH, TIFF_LONG, 1, dir->width, littleEndian);
        vs__tiff_write_ifd_entry(fp, TIFFTAG_IMAGELENGTH, TIFF_LONG, 1, dir->height, littleEndian);
        vs__tiff_write_ifd_entry(fp, TIFFTAG_BITSPERSAMPLE, TIFF_SHORT, 1, dir->bitsPerSample, littleEndian);
        vs__tiff_write_ifd_entry(fp, TIFFTAG_COMPRESSION, TIFF_SHORT, 1, dir->compression, littleEndian);
        vs__tiff_write_ifd_entry(fp, TIFFTAG_PHOTOMETRIC, TIFF_SHORT, 1, dir->photometric, littleEndian);
        vs__tiff_write_ifd_entry(fp, TIFFTAG_SAMPLESPERPIXEL, TIFF_SHORT, 1, dir->samplesPerPixel, littleEndian);
        vs__tiff_write_ifd_entry(fp, TIFFTAG_ROWSPERSTRIP, TIFF_LONG, 1, dir->rowsPerStrip, littleEndian);
        vs__tiff_write_ifd_entry(fp, TIFFTAG_PLANARCONFIG, TIFF_SHORT, 1, dir->planarConfig, littleEndian);
        vs__tiff_write_ifd_entry(fp, TIFFTAG_SAMPLEFORMAT, TIFF_SHORT, 1, dir->sampleFormat, littleEndian);

        // Write resolution entries
        vs__tiff_write_ifd_entry(fp, TIFFTAG_XRESOLUTION, TIFF_RATIONAL, 1, extraDataOffset, littleEndian);
        vs__tiff_write_rational(fp, dir->xResolution, extraDataOffset, littleEndian);
        extraDataOffset += 8;

        vs__tiff_write_ifd_entry(fp, TIFFTAG_YRESOLUTION, TIFF_RATIONAL, 1, extraDataOffset, littleEndian);
        vs__tiff_write_rational(fp, dir->yResolution, extraDataOffset, littleEndian);
        extraDataOffset += 8;

        vs__tiff_write_ifd_entry(fp, TIFFTAG_RESOLUTIONUNIT, TIFF_SHORT, 1, dir->resolutionUnit, littleEndian);

        // Write metadata strings if present
        if (dir->imageDescription[0]) {
            size_t len = strlen(dir->imageDescription) + 1;
            vs__tiff_write_ifd_entry(fp, TIFFTAG_IMAGEDESCRIPTION, TIFF_ASCII, len, extraDataOffset, littleEndian);
            vs__tiff_write_string(fp, dir->imageDescription, extraDataOffset);
            extraDataOffset += len;
        }

        if (dir->software[0]) {
            size_t len = strlen(dir->software) + 1;
            vs__tiff_write_ifd_entry(fp, TIFFTAG_SOFTWARE, TIFF_ASCII, len, extraDataOffset, littleEndian);
            vs__tiff_write_string(fp, dir->software, extraDataOffset);
            extraDataOffset += len;
        }

        if (dir->dateTime[0]) {
            vs__tiff_write_ifd_entry(fp, TIFFTAG_DATETIME, TIFF_ASCII, 20, extraDataOffset, littleEndian);
            vs__tiff_write_string(fp, dir->dateTime, extraDataOffset);
            extraDataOffset += 20;
        }

        // Calculate strip size and write strip information
        size_t stripSize = dir->width * dir->height * (dir->bitsPerSample / 8);
        vs__tiff_write_ifd_entry(fp, TIFFTAG_STRIPOFFSETS, TIFF_LONG, 1, extraDataOffset, littleEndian);
        vs__tiff_write_ifd_entry(fp, TIFFTAG_STRIPBYTECOUNTS, TIFF_LONG, 1, stripSize, littleEndian);

        // Write image data
        fseek(fp, extraDataOffset, SEEK_SET);
        size_t offset = stripSize * d;
        fwrite((uint8_t*)img->data + offset, 1, stripSize, fp);
        extraDataOffset += stripSize;

        // Write next IFD offset or 0 if last directory
        uint32_t nextIFD = (d < img->depth - 1) ? extraDataOffset : 0;
        vs__tiff_write_bytes(fp, nextIFD, 4, littleEndian);

        ifdOffset = nextIFD;
    }

    fclose(fp);
    return 0;
}

TiffImage* vs_tiff_create(uint32_t width, uint32_t height, uint16_t depth,
                           uint16_t bitsPerSample) {
    TiffImage* img = calloc(1, sizeof(TiffImage));
    if (!img) return NULL;

    img->depth = depth;
    img->directories = calloc(depth, sizeof(DirectoryInfo));
    if (!img->directories) {
        free(img);
        return NULL;
    }

    img->dataSize = width * height * (bitsPerSample / 8) * depth;
    img->data = calloc(1, img->dataSize);
    if (!img->data) {
        free(img->directories);
        free(img);
        return NULL;
    }

    // Initialize each directory
    for (int i = 0; i < depth; i++) {
        DirectoryInfo* dir = &img->directories[i];
        dir->width = width;
        dir->height = height;
        dir->bitsPerSample = bitsPerSample;
        dir->compression = 1;  // No compression
        dir->photometric = 1;  // min-is-black
        dir->samplesPerPixel = 1;
        dir->rowsPerStrip = height;
        dir->planarConfig = 1;
        dir->sampleFormat = 1;  // unsigned integer
        dir->xResolution = 72.0f;
        dir->yResolution = 72.0f;
        dir->resolutionUnit = 2;  // ?
        dir->subfileType = 0;

        vs__tiff_current_date_time(dir->dateTime);
    }

    img->isValid = true;
    return img;
}


// vcps

static int vs__vcps_read_binary_data(FILE* fp, void* out_data, const char* src_type, const char* dst_type, size_t count) {
    // Fast path: types match, direct read
    if (strcmp(src_type, dst_type) == 0) {
        size_t element_size = strcmp(src_type, "float") == 0 ? sizeof(f32) : sizeof(f64);
        return fread(out_data, element_size, count, fp) == count ? 0 : 1;
    }

    // Conversion path
    if (strcmp(src_type, "double") == 0 && strcmp(dst_type, "float") == 0) {
        f64* temp = malloc(count * sizeof(f64));
        if (!temp) return 1;

        int status = fread(temp, sizeof(f64), count, fp) == count ? 0 : 1;
        if (status == 0) {
            f32* out = out_data;
            for (size_t i = 0; i < count; i++) {
                out[i] = (f32)temp[i];
            }
        }
        free(temp);
        return status;
    }
    else if (strcmp(src_type, "float") == 0 && strcmp(dst_type, "double") == 0) {
        f32* temp = malloc(count * sizeof(f32));
        if (!temp) return 1;

        int status = fread(temp, sizeof(f32), count, fp) == count ? 0 : 1;
        if (status == 0) {
            f64* out = out_data;
            for (size_t i = 0; i < count; i++) {
                out[i] = (f64)temp[i];
            }
        }
        free(temp);
        return status;
    }

    return 1;
}

static int vs__vcps_write_binary_data(FILE* fp, const void* data, const char* src_type, const char* dst_type, size_t count) {
    // Fast path: types match, direct write
    if (strcmp(src_type, dst_type) == 0) {
        size_t element_size = strcmp(src_type, "float") == 0 ? sizeof(f32) : sizeof(f64);
        return fwrite(data, element_size, count, fp) == count ? 0 : 1;
    }

    // Conversion path
    if (strcmp(src_type, "float") == 0 && strcmp(dst_type, "double") == 0) {
        f64* temp = malloc(count * sizeof(f64));
        if (!temp) return 1;

        const f32* in = data;
        for (size_t i = 0; i < count; i++) {
            temp[i] = (f64)in[i];
        }

        int status = fwrite(temp, sizeof(f64), count, fp) == count ? 0 : 1;
        free(temp);
        return status;
    }
    else if (strcmp(src_type, "double") == 0 && strcmp(dst_type, "float") == 0) {
        f32* temp = malloc(count * sizeof(f32));
        if (!temp) return 1;

        const f64* in = data;
        for (size_t i = 0; i < count; i++) {
            temp[i] = (f32)in[i];
        }

        int status = fwrite(temp, sizeof(f32), count, fp) == count ? 0 : 1;
        free(temp);
        return status;
    }

    return 1;
}


int vs_vcps_read(const char* filename,
              size_t* width, size_t* height, size_t* dim,
              void* data, const char* dst_type) {
    if (!dst_type || (strcmp(dst_type, "float") != 0 && strcmp(dst_type, "double") != 0)) {
        LOG_ERROR("Error: Invalid destination type\n");
        return 1;
    }

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        LOG_ERROR("Error: Cannot open file %s\n", filename);
        return 1;
    }

    // Read header
    char line[256];
    int header_complete = 0;
    int ordered = 0;
    char src_type[32] = {0};
    int version = 0;
    *width = 0;
    *height = 0;
    *dim = 0;

    while (fgets(line, sizeof(line), fp)) {
        vs__trim(line);

        if (strcmp(line, "<>") == 0) {
            header_complete = 1;
            break;
        }

        char key[32], value[32];
        if (sscanf(line, "%31[^:]: %31s", key, value) == 2) {
            if (strcmp(key, "width") == 0) {
                *width = atoi(value);
            } else if (strcmp(key, "height") == 0) {
                *height = atoi(value);
            } else if (strcmp(key, "dim") == 0) {
                *dim = atoi(value);
            } else if (strcmp(key, "type") == 0) {
                strncpy(src_type, value, sizeof(src_type) - 1);
            } else if (strcmp(key, "version") == 0) {
                version = atoi(value);
            } else if (strcmp(key, "ordered") == 0) {
                ordered = (strcmp(value, "true") == 0);
            }
        }
    }

    if (!header_complete || *width == 0 || *height == 0 || *dim == 0 ||
        (strcmp(src_type, "float") != 0 && strcmp(src_type, "double") != 0) ||
        !ordered) {
        LOG_ERROR("Error: Invalid header (w=%zu h=%zu d=%zu t=%s o=%d)\n",
                *width, *height, *dim, src_type, ordered);
        fclose(fp);
        return 1;
    }

    size_t total_points = (*width) * (*height) * (*dim);
    int status = vs__vcps_read_binary_data(fp, data, src_type, dst_type, total_points);

    fclose(fp);
    return status;
}

int vs_vcps_write(const char* filename,
               size_t width, size_t height, size_t dim,
               const void* data, const char* src_type, const char* dst_type) {
    if (!src_type || !dst_type ||
        (strcmp(src_type, "float") != 0 && strcmp(src_type, "double") != 0) ||
        (strcmp(dst_type, "float") != 0 && strcmp(dst_type, "double") != 0)) {
        LOG_ERROR("Error: Invalid type specification\n");
        return 1;
    }

    FILE* fp = fopen(filename, "w");
    if (!fp) return 1;

    // Write header
    fprintf(fp, "width: %zu\n", width);
    fprintf(fp, "height: %zu\n", height);
    fprintf(fp, "dim: %zu\n", dim);
    fprintf(fp, "ordered: true\n");
    fprintf(fp, "type: %s\n", dst_type);
    fprintf(fp, "version: 1\n");
    fprintf(fp, "<>\n");
    fclose(fp);

    // Reopen in binary append mode for data
    fp = fopen(filename, "ab");
    if (!fp) return 1;

    size_t total_points = width * height * dim;
    int status = vs__vcps_write_binary_data(fp, data, src_type, dst_type, total_points);

    fclose(fp);
    return status;
}

// vol

volume *vs_vol_new(char *cache_dir, char *url) {
  volume *ret = malloc(sizeof(volume));
  if (ret == NULL) {
    return NULL;
  }


  if (cache_dir != NULL) {
    if (vs__mkdir_p(cache_dir)) {
      LOG_ERROR("Could not mkdir %s",cache_dir);
      return NULL;
    }
  }

  void* zarray_buf = NULL;
  if (url != NULL) {
    char zarray_url[1024] = {'\0'};
    snprintf(zarray_url,1023,"%s/.zarray",url);
    LOG_INFO("trying to read .zarray from %s",zarray_url);
    if (vs_download(zarray_url, &zarray_buf) <= 0) {
      LOG_ERROR("could not download .zarray file!");
      return NULL;
    }
  }
  zarr_metadata metadata;
  if (vs_zarr_parse_metadata(zarray_buf,&metadata)) {
    LOG_ERROR("failed to parse .zarray");
    return NULL;
  }

  strncpy(ret->url,url,sizeof(ret->url));
  strncpy(ret->cache_dir,cache_dir,sizeof(ret->cache_dir));
  ret->metadata = metadata;

  free(zarray_buf);
  return ret;
}

void vs_vol_free(volume* vol) {
    if (vol) {
        free(vol);
    }
}

chunk *vs_vol_get_chunk(volume *vol, s32 vol_start[static 3], s32 chunk_dims[static 3]) {
    //TODO: support arbitrary starts and sizes within the volume
    //for now, we will assume that the volume starts and chunk dimensions are aligned with the zarr block sizes within
    // volume because it makes the index calculations much easier

    //TODO: make sure that we aren't readng past the end of the chunk if the chunk happens to be the last chunk
    //in a given dimension

    if (vol_start[0] % vol->metadata.chunks[0] != 0) {
        LOG_ERROR("vol_start indices must be a multiple of the zrr block size %d", vol->metadata.chunks[0]);
        return NULL;
    }

    if (vol_start[1] % vol->metadata.chunks[1] != 0) {
        LOG_ERROR("vol_start indices must be a multiple of the zrr block size %d", vol->metadata.chunks[1]);
        return NULL;
    }

    if (vol_start[2] % vol->metadata.chunks[2] != 0) {
        LOG_ERROR("vol_start indices must be a multiple of the zrr block size %d", vol->metadata.chunks[2]);
        return NULL;
    }

    if (chunk_dims[0] % vol->metadata.chunks[0] != 0) {
        LOG_ERROR("chunk_dims must be a multiple of the zrr block size %d", vol->metadata.chunks[0]);
        return NULL;
    }

    if (chunk_dims[1] % vol->metadata.chunks[1] != 0) {
        LOG_ERROR("chunk_dims must be a multiple of the zrr block size %d", vol->metadata.chunks[1]);
        return NULL;
    }

    if (chunk_dims[2] % vol->metadata.chunks[2] != 0) {
        LOG_ERROR("chunk_dims must be a multiple of the zrr block size %d", vol->metadata.chunks[2]);
        return NULL;
    }

    chunk *ret = vs_chunk_new(chunk_dims);

    int zstart = vol_start[0] / vol->metadata.chunks[0];
    int ystart = vol_start[1] / vol->metadata.chunks[1];
    int xstart = vol_start[2] / vol->metadata.chunks[2];
    int zend = (vol_start[0] + chunk_dims[0]-1) / vol->metadata.chunks[0];
    int yend = (vol_start[1] + chunk_dims[1]-1) / vol->metadata.chunks[1];
    int xend = (vol_start[2] + chunk_dims[2]-1) / vol->metadata.chunks[2];


    for (int z = zstart; z <= zend; z++) {
        for (int y = ystart; y <= yend; y++) {
            for (int x = xstart; x <= xend; x++) {
                char blockpath[1024] = {'\0'};
                chunk *c = NULL;

                if (vol->metadata.dimension_separator == '/') {
                    snprintf(blockpath, 1023, "%s/%d/%d/%d", vol->cache_dir, z, y, x);
                } else {
                    snprintf(blockpath, 1023, "%s/%d.%d.%d", vol->cache_dir, z, y, x);
                }
                LOG_INFO("checking for zarr block at %s", blockpath);
                if (vs__path_exists(blockpath)) {
                    LOG_INFO("reading %s from disk", blockpath);
                    c = vs_zarr_read_chunk(blockpath, vol->metadata);
                    if (c == NULL) {
                        LOG_ERROR("failed to read zarr chunk from %s", blockpath);
                        vs_chunk_free(ret);
                        return NULL;
                    }
                } else {
                    char url[1024] = {'\0'};
                    if (vol->metadata.dimension_separator == '/') {
                        snprintf(url, 1023, "%s/%d/%d/%d", vol->url, z, y, x);
                    } else {
                        snprintf(url, 1023, "%s/%d.%d.%d", vol->url, z, y, x);
                    }
                    LOG_INFO("downloading block from %s", url);
                    c = vs_zarr_fetch_block(url, vol->metadata);
                    if (c == NULL) {
                        //NOTE: this is not necessarily an error. Some logical blocks do not exist physically because
                        //they are all zero, and zarr will by default not keep all zero chunk files. so for now we'll assume
                        //that is the case and just skip it
                        LOG_ERROR("could not download block from %s", url);
                        continue;
                    }
                    LOG_INFO("downloaded block from %s", url);
                    LOG_INFO("writing chunk to %s", blockpath);
                    if (vs_zarr_write_chunk(blockpath, vol->metadata, c)) {
                        LOG_ERROR("failed to write zarr chunk to %s", blockpath);
                        vs_chunk_free(c);
                        vs_chunk_free(ret);
                        return NULL;
                    }
                }

                s32 src_start[3] = {
                    MAX(0, vol_start[0] - z * vol->metadata.chunks[0]),
                    MAX(0, vol_start[1] - y * vol->metadata.chunks[1]),
                    MAX(0, vol_start[2] - x * vol->metadata.chunks[2])
                  };

                s32 dest_start[3] = {
                    z * vol->metadata.chunks[0] - vol_start[0],
                    y * vol->metadata.chunks[1] - vol_start[1],
                    x * vol->metadata.chunks[2] - vol_start[2]
                  };

                s32 copy_dims[3] = {
                    MIN(vol->metadata.chunks[0] - src_start[0], chunk_dims[0] - dest_start[0]),
                    MIN(vol->metadata.chunks[1] - src_start[1], chunk_dims[1] - dest_start[1]),
                    MIN(vol->metadata.chunks[2] - src_start[2], chunk_dims[2] - dest_start[2])
                  };

                if (vs_chunk_graft(ret, c, src_start, dest_start, copy_dims)) {
                    vs_chunk_free(c);
                    vs_chunk_free(ret);
                    LOG_ERROR("failed to graft chunk");
                    return NULL;
                }
                vs_chunk_free(c);
            }
        }
    }
    return ret;
}

ChunkLoadState* vs_vol_get_chunk_start(volume* vol, s32 vol_start[static 3], s32 chunk_dims[static 3]) {
    // Validate alignment with block sizes
    for (int i = 0; i < 3; i++) {
        if (vol_start[i] % vol->metadata.chunks[i] != 0 ||
            chunk_dims[i] % vol->metadata.chunks[i] != 0) {
            LOG_ERROR("Indices must be multiple of block size");
            return NULL;
        }
    }

    ChunkLoadState* state = malloc(sizeof(ChunkLoadState));
    state->vol = vol;
    memcpy(state->vol_start, vol_start, sizeof(s32) * 3);
    memcpy(state->chunk_dims, chunk_dims, sizeof(s32) * 3);
    state->ret = vs_chunk_new(chunk_dims);

    state->zstart = vol_start[0] / vol->metadata.chunks[0];
    state->ystart = vol_start[1] / vol->metadata.chunks[1];
    state->xstart = vol_start[2] / vol->metadata.chunks[2];
    state->zend = (vol_start[0] + chunk_dims[0]-1) / vol->metadata.chunks[0];
    state->yend = (vol_start[1] + chunk_dims[1]-1) / vol->metadata.chunks[1];
    state->xend = (vol_start[2] + chunk_dims[2]-1) / vol->metadata.chunks[2];

    state->z = state->zstart;
    state->y = state->ystart;
    state->x = state->xstart;
    state->downloading = false;
    state->download = NULL;

    return state;
}


static void vs_process_chunk(ChunkLoadState* state, chunk* c) {
    s32 src_start[3] = {
        MAX(0, state->vol_start[0] - state->z * state->vol->metadata.chunks[0]),
        MAX(0, state->vol_start[1] - state->y * state->vol->metadata.chunks[1]),
        MAX(0, state->vol_start[2] - state->x * state->vol->metadata.chunks[2])
    };

    s32 dest_start[3] = {
        state->z * state->vol->metadata.chunks[0] - state->vol_start[0],
        state->y * state->vol->metadata.chunks[1] - state->vol_start[1],
        state->x * state->vol->metadata.chunks[2] - state->vol_start[2]
    };

    s32 copy_dims[3] = {
        MIN(state->vol->metadata.chunks[0] - src_start[0], state->chunk_dims[0] - dest_start[0]),
        MIN(state->vol->metadata.chunks[1] - src_start[1], state->chunk_dims[1] - dest_start[1]),
        MIN(state->vol->metadata.chunks[2] - src_start[2], state->chunk_dims[2] - dest_start[2])
    };

    vs_chunk_graft(state->ret, c, src_start, dest_start, copy_dims);
}


bool vs_vol_get_chunk_poll(ChunkLoadState* state, chunk** out_chunk) {
    if (!state->downloading) {
        // Find next block to process
        while (state->z <= state->zend) {
            while (state->y <= state->yend) {
                while (state->x <= state->xend) {
                    char blockpath[1024] = {'\0'};
                    snprintf(blockpath, 1023, "%s/%d/%d/%d",
                            state->vol->cache_dir, state->z, state->y, state->x);

                    if (vs__path_exists(blockpath)) {
                        // Read from disk
                        chunk* c = vs_zarr_read_chunk(blockpath, state->vol->metadata);
                        if (c) {
                            vs_process_chunk(state, c);
                            vs_chunk_free(c);
                        }
                    } else {
                        // Start download
                        char url[1024] = {'\0'};
                        snprintf(url, 1023, "%s/%d/%d/%d",
                                state->vol->url, state->z, state->y, state->x);
                        state->download = vs_download_start(url);
                        state->downloading = true;
                        return false;
                    }
                    state->x++;
                    if (!state->downloading) continue;
                    return false;
                }
                state->x = state->xstart;
                state->y++;
            }
            state->y = state->ystart;
            state->z++;
        }

        // All done
        *out_chunk = state->ret;
        free(state);
        return true;
    }

    // Check download progress
    void* buffer;
    long size;
    if (vs_download_poll(state->download, &buffer, &size)) {
        state->downloading = false;

        if (buffer) {
            // Save downloaded block
            char blockpath[1024] = {'\0'};
            snprintf(blockpath, 1023, "%s/%d/%d/%d",
                    state->vol->cache_dir, state->z, state->y, state->x);
            chunk* c = vs_zarr_decompress_chunk(size,buffer, state->vol->metadata);
            free(buffer);

            if (c) {
                vs_zarr_write_chunk(blockpath, state->vol->metadata, c);
                vs_process_chunk(state, c);
                vs_chunk_free(c);
            }
        }

        state->x++;
        return false;
    }

    return false;
}


// zarr




chunk* vs_zarr_fetch_block(char* url, zarr_metadata metadata) {

  void* compressed_buf = NULL;
  long compressed_size;
  if ((compressed_size = vs_download(url, &compressed_buf)) <= 0) {
      free(compressed_buf);
    return NULL;
  }
  chunk* mychunk = vs_zarr_decompress_chunk(compressed_size, compressed_buf,metadata);
  free(compressed_buf);
  return mychunk;
}

static void vs__json_parse_int32_array(json_object *array_obj, int32_t output[3]) {
    size_t array_len = json_object_array_length(array_obj);
    for (size_t i = 0; i < 3 && i < array_len; i++) {
        json_object *element = json_object_array_get_idx(array_obj, i);
        output[i] = (int32_t)json_object_get_int(element);
    }
}

int vs_zarr_parse_metadata(const char *json_string, zarr_metadata *metadata) {
    json_object *root = json_tokener_parse(json_string);
    if (!root) {
        printf("Failed to parse JSON!\n");
        return 1;
    }

    json_object *shapes_value;
    if (json_object_object_get_ex(root, "shape", &shapes_value) &&
        json_object_is_type(shapes_value, json_type_array)) {
        vs__json_parse_int32_array(shapes_value, metadata->shape);
    }

    json_object *chunks_value;
    if (json_object_object_get_ex(root, "chunks", &chunks_value) &&
        json_object_is_type(chunks_value, json_type_array)) {
        vs__json_parse_int32_array(chunks_value, metadata->chunks);
    }

    json_object *compressor_value;
    if (json_object_object_get_ex(root, "compressor", &compressor_value) &&
        json_object_is_type(compressor_value, json_type_object)) {

        json_object *blocksize;
        if (json_object_object_get_ex(compressor_value, "blocksize", &blocksize)) {
            metadata->compressor.blocksize = json_object_get_int(blocksize);
        }

        json_object *clevel;
        if (json_object_object_get_ex(compressor_value, "clevel", &clevel)) {
            metadata->compressor.clevel = json_object_get_int(clevel);
        }

        json_object *cname;
        if (json_object_object_get_ex(compressor_value, "cname", &cname)) {
            const char *cname_str = json_object_get_string(cname);
            strncpy(metadata->compressor.cname, cname_str, sizeof(metadata->compressor.cname) - 1);
            metadata->compressor.cname[sizeof(metadata->compressor.cname) - 1] = '\0';
        }

        json_object *id;
        if (json_object_object_get_ex(compressor_value, "id", &id)) {
            const char *id_str = json_object_get_string(id);
            strncpy(metadata->compressor.id, id_str, sizeof(metadata->compressor.id) - 1);
            metadata->compressor.id[sizeof(metadata->compressor.id) - 1] = '\0';
        }

        json_object *shuffle;
        if (json_object_object_get_ex(compressor_value, "shuffle", &shuffle)) {
            metadata->compressor.shuffle = json_object_get_int(shuffle);
        }
    }

    json_object *dtype_value;
    if (json_object_object_get_ex(root, "dtype", &dtype_value)) {
        const char *dtype_str = json_object_get_string(dtype_value);
        strncpy(metadata->dtype, dtype_str, sizeof(metadata->dtype) - 1);
        metadata->dtype[sizeof(metadata->dtype) - 1] = '\0';
    }

    json_object *fill_value;
    if (json_object_object_get_ex(root, "fill_value", &fill_value)) {
        metadata->fill_value = json_object_get_int(fill_value);
    }

    json_object *order_value;
    if (json_object_object_get_ex(root, "order", &order_value)) {
        const char *order_str = json_object_get_string(order_value);
        if (order_str && order_str[0]) {
            metadata->order = order_str[0];
        }
    }

    json_object *dimension_separator;
    if (json_object_object_get_ex(root, "dimension_separator", &dimension_separator)) {
        const char *dimension_separator_str = json_object_get_string(dimension_separator);
        if (dimension_separator_str && dimension_separator_str[0]) {
            metadata->dimension_separator = dimension_separator_str[0];
        }
    }

    json_object *format_value;
    if (json_object_object_get_ex(root, "zarr_format", &format_value)) {
        metadata->zarr_format = json_object_get_int(format_value);
    }

    json_object_put(root);
    return 0;
}

zarr_metadata vs_zarr_parse_zarray(char *path) {
  zarr_metadata metadata = {0};

  FILE *fp = fopen(path, "rt");
  if (fp == NULL) {
    LOG_ERROR("could not open file %s\n", path);
    assert(false);
    return metadata;
  }
  s32 size;
  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char *buf = calloc(size + 1, 1);
  fread(buf, 1, size, fp);


  if (vs_zarr_parse_metadata(buf, &metadata)) {
    printf("Shape: [%d, %d, %d]\n",
           metadata.shape[0], metadata.shape[1], metadata.shape[2]);
    printf("Chunks: [%d, %d, %d]\n",
           metadata.chunks[0], metadata.chunks[1], metadata.chunks[2]);
    printf("Compressor:\n");
    printf("  blocksize: %d\n", metadata.compressor.blocksize);
    printf("  clevel: %d\n", metadata.compressor.clevel);
    printf("  cname: %s\n", metadata.compressor.cname);
    printf("  id: %s\n", metadata.compressor.id);
    printf("  shuffle: %d\n", metadata.compressor.shuffle);
    printf("dtype: %s\n", metadata.dtype);
    printf("fill_value: %d\n", metadata.fill_value);
    printf("order: %c\n", metadata.order);
    printf("zarr_format: %d\n", metadata.zarr_format);
  }

  free(buf);
  return metadata;
}

chunk* vs_zarr_read_chunk(char* path, zarr_metadata metadata) {

    FILE* fp = fopen(path, "rb");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    u8* compressed_data = malloc(size);
    fread(compressed_data,1,size,fp);

    chunk* ret= vs_zarr_decompress_chunk(size, compressed_data, metadata);
    free(compressed_data);
    return ret;
}

chunk* vs_zarr_decompress_chunk(long size, void* compressed_data, zarr_metadata metadata) {

    int z = metadata.chunks[0];
    int y = metadata.chunks[1];
    int x = metadata.chunks[2];
    int dtype_size = 0;
    if(strcmp(metadata.dtype,"|u1") == 0) {
        dtype_size = 1;
    } else if (strcmp(metadata.dtype, "|u2") == 0) {
        ASSERT(false,"16 bit zarr not currently supported");
        dtype_size = 2;
    } else {
        LOG_ERROR("unsupported zarr format. Only unsigned 8 and unsigned 16 are supported\n");
    }

    unsigned char* decompressed_data;
    int decompressed_size;
    // the data may not actually be compressed. if so, just use the compressed data
    if(strnlen(metadata.compressor.cname,32) == 0) {
        decompressed_data = compressed_data;
        decompressed_size = size;
    } else {
        decompressed_data = malloc(z * y * x * dtype_size*2);
        decompressed_size = blosc2_decompress(compressed_data, size, decompressed_data, z * y * x * dtype_size);
        if (decompressed_size < 0) {
            LOG_ERROR("Blosc2 decompression failed: %d\n", decompressed_size);
            free(decompressed_data);
            return NULL;
        }
    }
    chunk *ret = vs_chunk_new((s32[3]){z, y, x});

    for (int z = 0; z < ret->dims[0]; z++) {
        for (int y = 0; y < ret->dims[1]; y++) {
            for (int x = 0; x < ret->dims[2]; x++) {
                vs_chunk_set(ret, z, y, x, (f32) decompressed_data[z * ret->dims[1] * ret->dims[2] + y * ret->dims[2] + x]);
            }
        }
    }
    if(decompressed_data != compressed_data) {
        free(decompressed_data);
    }

    return ret;
}


int vs_zarr_write_chunk(char *path, zarr_metadata metadata, chunk* c) {
    // the directory to the file path might not exist so we will mkdir for it here
    // path should be a path to the chunk file name, e.g. 54keV_7.91um_Scroll1A.zarr/0/50/30/30 will write out
    // a file called 30 in directory 30 in 50 in 0 in 54keV...

    char* dirname = vs__basename(path);
    if (vs__mkdir_p(dirname)) {
        LOG_ERROR("failed to mkdirs to %s",dirname);
        return 1;
    }
    void* compressed_buf = NULL;
    int len = vs_zarr_compress_chunk(c,metadata,&compressed_buf);
    if (len <= 0) {
        //TODO: len == 0 is probably an error, right?
        return 1;
    }
    FILE* fp = fopen(path, "wb");
    if (fp == NULL) {
        LOG_ERROR("failed to open %s",path);
        return 1;
    }
    fwrite(compressed_buf,1,len,fp);
    LOG_INFO("wrote chunk to %s",path);
    fclose(fp);
    free(dirname);
    free(compressed_buf);
    return 0;
}

int vs_zarr_compress_chunk(chunk* c, zarr_metadata metadata, void** compressed_data) {
    if (c->dims[0] != metadata.chunks[0]) {
        LOG_ERROR("zarr block size mismatch with chunk dims");
        return 1;
    }
    if (c->dims[1] != metadata.chunks[1]) {
        LOG_ERROR("zarr block size mismatch with chunk dims");
        return 1;
    }
    if (c->dims[2] != metadata.chunks[2]) {
        LOG_ERROR("zarr block size mismatch with chunk dims");
        return 1;
    }
  int z = metadata.chunks[0];
  int y = metadata.chunks[1];
  int x = metadata.chunks[2];
  int dtype_size = 0;
  u8* decompressed_data = NULL;
  if (strcmp(metadata.dtype, "|u1") == 0) {
    dtype_size = 1;
    decompressed_data = malloc(z*y*x);
    for (int _z = 0; _z < z; _z++) {
      for (int _y = 0; _y < y; _y++) {
        for (int _x = 0; _x < x; _x++) {
          decompressed_data[_z*y*x+_y*x+_x] = (u8) vs_chunk_get(c,_z,_y,_x);
        }
      }
    }
  } else if (strcmp(metadata.dtype, "|u2") == 0) {
      LOG_ERROR("16 bit zarr not currently supported\n");
      return 1;
  } else {
    LOG_ERROR("unsupported zarr format. Only unsigned 8 is supported\n");
  }
  *compressed_data = malloc(z*y*x+BLOSC2_MAX_OVERHEAD);
  int compressed_len = blosc2_compress(metadata.compressor.clevel,metadata.compressor.shuffle,dtype_size,decompressed_data,z*y*x,*compressed_data,z*y*x*BLOSC2_MAX_OVERHEAD);

  if (compressed_len <= 0) {
    LOG_ERROR("Blosc2 compression failed: %d\n", compressed_len);
    free(compressed_data);
    free(decompressed_data);
    return -1;
  }
  return compressed_len;
}


//vesuvius specific
chunk *vs_tiff_to_chunk(const char *tiffpath) {
  TiffImage *img = vs_tiff_read(tiffpath);
  if (!img || !img->isValid) {
      LOG_ERROR("tiff is NULL or invalid");
    return NULL;
  }
  if (img->depth <= 1) {
    printf("can't load a 2d tiff as a chunk");
    return NULL;
  }

  //TODO: can we assume that all 3D tiffs have the same x,y dimensions for all slices? because we are right here
  s32 dims[3] = {img->depth, img->directories[0].height, img->directories[0].width};
  chunk *ret = vs_chunk_new(dims);
  for (s32 z = 0; z < dims[0]; z++) {
    void *buf = vs_tiff_read_directory_data(img, z);
    for (s32 y = 0; y < dims[1]; y++) {
      for (s32 x = 0; x < dims[2]; x++) {
        if (img->directories[z].bitsPerSample == 8) {
          ret->data[z * dims[1] * dims[2] + y * dims[2] + x] = vs_tiff_pixel8(
            buf, y, x, img->directories[z].width);
        } else if (img->directories[z].bitsPerSample == 16) {
          ret->data[z * dims[1] * dims[2] + y * dims[2] + x] = vs_tiff_pixel16(
            buf, y, x, img->directories[z].width);
        }
      }
    }
  }
  return ret;
}


slice *vs_tiff_to_slice(const char *tiffpath, int index) {
  TiffImage *img = vs_tiff_read(tiffpath);
  if (!img || !img->isValid) {
      LOG_ERROR("tiff is null or invalid");
      return NULL;
  }
  if (index < 0 || index >= img->depth) {
      LOG_ERROR("index %d is invalid for a tiff with depth %d",index,img->depth);
      return NULL;
  }

  s32 dims[2] = {img->directories[0].height, img->directories[0].width};
  slice *ret = vs_slice_new(dims);

  void *buf = vs_tiff_read_directory_data(img, index);
  for (s32 y = 0; y < dims[0]; y++) {
    for (s32 x = 0; x < dims[1]; x++) {
      if (img->directories[index].bitsPerSample == 8) {
        ret->data[y * dims[1] + x] = vs_tiff_pixel8(buf, y, x, img->directories[index].width);
      } else if (img->directories[index].bitsPerSample == 16) {
        ret->data[y * dims[1] + x] = vs_tiff_pixel16(buf, y, x, img->directories[index].width);
      }
    }
  }
  return ret;
}

slice* vs_slice_extract(chunk* c, int index) {
    if (index < 0 || index >= c->dims[0]) {
        LOG_ERROR("index out of bounds");
        return NULL;
    }
    slice* out = vs_slice_new((s32[2]){c->dims[1],c->dims[2]});
    for (int y = 0; y < out->dims[0];y++) {
        for (int x = 0; x < out->dims[1];x++) {
            vs_slice_set(out,y,x,vs_chunk_get(c,index,y,x));
        }
    }
    return out;
}


// Function to write a single PPM frame from the three chunks
void vs_write_ppm_frame(FILE* fp, const chunk* r_chunk, const chunk* g_chunk,
                    const chunk* b_chunk, int frame_idx) {
    int width = r_chunk->dims[2];
    int height = r_chunk->dims[1];
    int frame_size = width * height;

    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    // Allocate buffer for one frame
    unsigned char* frame_data = (unsigned char*)malloc(width * height * 3);

    // Calculate offsets for the current frame in the chunk data
    int frame_offset = frame_idx * frame_size;

    // Combine RGB channels and convert from float [0-1] to byte [0-255]
    for (int i = 0; i < frame_size; i++) {
        frame_data[i * 3 + 0] = (unsigned char)(r_chunk->data[frame_offset + i] * 255.0f);
        frame_data[i * 3 + 1] = (unsigned char)(g_chunk->data[frame_offset + i] * 255.0f);
        frame_data[i * 3 + 2] = (unsigned char)(b_chunk->data[frame_offset + i] * 255.0f);
    }

    // Write the frame data
    fwrite(frame_data, sizeof(unsigned char), width * height * 3, fp);

    free(frame_data);
}

// Function to convert chunks to video using FFmpeg
void vs_chunks_to_video(const chunk* r_chunk, const chunk* g_chunk, const chunk* b_chunk,
                    const char* output_filename, int fps) {
    char command[256];

    // Verify dimensions match
    for (int i = 0; i < 3; i++) {
        if (r_chunk->dims[i] != g_chunk->dims[i] || r_chunk->dims[i] != b_chunk->dims[i]) {
            fprintf(stderr, "Error: Chunk dimensions don't match\n");
            return;
        }
    }

    int frames = r_chunk->dims[0];

    // Create temporary directory for frames
    system("mkdir -p temp_frames");

    // Write each frame as a PPM file
    for (int f = 0; f < frames; f++) {
        char filename[64];
        sprintf(filename, "temp_frames/frame_%d.ppm", f);
        FILE* fp = fopen(filename, "wb");
        if (!fp) {
            fprintf(stderr, "Error: Cannot create frame file %s\n", filename);
            return;
        }

        vs_write_ppm_frame(fp, r_chunk, g_chunk, b_chunk, f);
        fclose(fp);
    }

    // Use FFmpeg to convert PPM frames to video
    sprintf(command, "ffmpeg -y -framerate %d -i temp_frames/frame_%%d.ppm "
            "-c:v libx264 -pix_fmt yuv420p %s", fps, output_filename);
    system(command);

    // Clean up temporary files
    system("rm -rf temp_frames");
}


// BMP Header Structures
#pragma pack(push, 1) // Ensure no padding
typedef struct {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BMPFileHeader;

typedef struct {
    uint32_t biSize;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter;
    int32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BMPInfoHeader;
#pragma pack(pop)

int vs_bmp_write(const char *filename, slice *image) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return -1;
    }

    int width = image->dims[1];
    int height = image->dims[0];

    BMPFileHeader file_header;
    BMPInfoHeader info_header;

    // Calculate the size of each row including padding
    int rowSize = (width + 3) & ~3; // Round up to the nearest multiple of 4
    int imageSize = rowSize * height;

    // BMP file header
    file_header.bfType = 0x4D42; // 'BM'
    file_header.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + 256 * 4; // File header + Info header + Palette
    file_header.bfSize = file_header.bfOffBits + imageSize;
    file_header.bfReserved1 = 0;
    file_header.bfReserved2 = 0;

    // BMP info header
    info_header.biSize = sizeof(BMPInfoHeader);
    info_header.biWidth = width;
    info_header.biHeight = -height; // Negative height to indicate top-down row order
    info_header.biPlanes = 1;
    info_header.biBitCount = 8; // 8 bits per pixel (grayscale)
    info_header.biCompression = 0;
    info_header.biSizeImage = imageSize;
    info_header.biXPelsPerMeter = 2835; // 72 DPI
    info_header.biYPelsPerMeter = 2835; // 72 DPI
    info_header.biClrUsed = 256;
    info_header.biClrImportant = 256;

    // Write BMP file header
    fwrite(&file_header, sizeof(BMPFileHeader), 1, file);

    // Write BMP info header
    fwrite(&info_header, sizeof(BMPInfoHeader), 1, file);

    // Write the grayscale palette (256 shades of gray)
    for (int i = 0; i < 256; ++i) {
        unsigned char color[4] = {i, i, i, 0}; // R, G, B, Reserved
        fwrite(color, sizeof(unsigned char), 4, file);
    }

    // Allocate buffer for one row
    unsigned char *row = (unsigned char *)malloc(rowSize);
    if (!row) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return -1;
    }

    // Write the pixel data with padding, clamping values between 0 and 255
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float pixel_value = vs_slice_get(image, y, x);
            if (pixel_value < 0) pixel_value = 0;
            if (pixel_value > 255) pixel_value = 255;
            row[x] = (unsigned char)pixel_value;
        }

        // Clear padding bytes
        memset(row + width, 0, rowSize - width);

        // Write the padded row
        fwrite(row, sizeof(unsigned char), rowSize, file);
    }

    free(row);
    fclose(file);
    return 0;
}

#endif // defined(VESUVIUS_IMPL)
#endif // VESUVIUS_H
