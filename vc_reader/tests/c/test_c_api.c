/* The C API from C99: open a two-level volume, choose levels for a view, draw a
 * progressive frame over a plane and a surface, composite layers and map them to
 * display pixels, and report errors through statuses. */
#define _POSIX_C_SOURCE 200809L
#include <vc_reader/vc_reader_c.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(condition)                                                              \
    do {                                                                              \
        if (!(condition)) {                                                           \
            fprintf(stderr, "%s:%d: CHECK(%s) failed (%s)\n", __FILE__, __LINE__,     \
                    #condition, vcr_last_error());                                    \
            ++failures;                                                               \
        }                                                                             \
    } while (0)

static char root[256];

static void path_of(char* out, size_t size, const char* relative)
{
    snprintf(out, size, "%s/%s", root, relative);
}

static void write_file(const char* relative, const void* data, size_t size)
{
    char path[512];
    FILE* f;
    path_of(path, sizeof(path), relative);
    f = fopen(path, "wb");
    if (!f) {
        perror(path);
        exit(1);
    }
    fwrite(data, 1, size, f);
    fclose(f);
}

static void write_text(const char* relative, const char* text) { write_file(relative, text, strlen(text)); }

static float value_at(float x, float y, float z) { return x + 2 * y + 3 * z; }

/* Level 0: 16^3 uint8 with value x + 2y + 3z in 8^3 chunks; level 1: 8^3 of 200. */
static void write_volume(void)
{
    static const char* zarray16 = "{\"zarr_format\":2,\"shape\":[16,16,16],\"chunks\":[8,8,8],\"dtype\":\"|u1\","
                                  "\"compressor\":null,\"fill_value\":0,\"order\":\"C\",\"filters\":null}";
    static const char* zarray8 = "{\"zarr_format\":2,\"shape\":[8,8,8],\"chunks\":[8,8,8],\"dtype\":\"|u1\","
                                 "\"compressor\":null,\"fill_value\":0,\"order\":\"C\",\"filters\":null}";
    unsigned char chunk[512];
    char path[512], name[64];
    int cz, cy, cx, z, y, x;
    path_of(path, sizeof(path), "0");
    mkdir(path, 0700);
    path_of(path, sizeof(path), "1");
    mkdir(path, 0700);
    write_text(".zgroup", "{\"zarr_format\":2}");
    write_text("0/.zarray", zarray16);
    write_text("1/.zarray", zarray8);
    for (cz = 0; cz < 2; ++cz)
        for (cy = 0; cy < 2; ++cy)
            for (cx = 0; cx < 2; ++cx) {
                for (z = 0; z < 8; ++z)
                    for (y = 0; y < 8; ++y)
                        for (x = 0; x < 8; ++x)
                            chunk[(z * 8 + y) * 8 + x] =
                                (unsigned char)value_at((float)(cx * 8 + x), (float)(cy * 8 + y), (float)(cz * 8 + z));
                snprintf(name, sizeof(name), "0/%d.%d.%d", cz, cy, cx);
                write_file(name, chunk, sizeof(chunk));
            }
    memset(chunk, 200, sizeof(chunk));
    write_file("1/0.0.0", chunk, sizeof(chunk));
}

static void remove_volume(void)
{
    char path[512], name[64];
    int cz, cy, cx;
    for (cz = 0; cz < 2; ++cz)
        for (cy = 0; cy < 2; ++cy)
            for (cx = 0; cx < 2; ++cx) {
                snprintf(name, sizeof(name), "0/%d.%d.%d", cz, cy, cx);
                path_of(path, sizeof(path), name);
                remove(path);
            }
    path_of(path, sizeof(path), "1/0.0.0"), remove(path);
    path_of(path, sizeof(path), "0/.zarray"), remove(path);
    path_of(path, sizeof(path), "1/.zarray"), remove(path);
    path_of(path, sizeof(path), ".zgroup"), remove(path);
    path_of(path, sizeof(path), "0"), rmdir(path);
    path_of(path, sizeof(path), "1"), rmdir(path);
    rmdir(root);
}

static int parts_read = 0;
static int count_part(void* user)
{
    (void)user;
    ++parts_read;
    return 1;
}

static size_t prefill_calls = 0;
static int prefill_progress(void* user, size_t resolved, size_t total)
{
    (void)user, (void)resolved, (void)total;
    ++prefill_calls;
    return 0;
}

int main(void)
{
    const char* tmp = getenv("TMPDIR");
    vcr_open_options options;
    vcr_volume* volume = NULL;
    vcr_level_info info;
    vcr_plane plane;
    vcr_frame frame;
    vcr_request request;
    vcr_result result;
    vcr_pending* pending = NULL;
    vcr_composite max;
    vcr_surface* surface = NULL;
    vcr_raster raster;
    float values[64], grid[4 * 6 * 3], offsets[2] = {0.0f, 1.0f};
    int8_t levels[32];
    uint8_t bytes[32], coverage[32];
    uint32_t lut[256], pixels[32];
    int q, r, c, rows = 0, cols = 0;

    snprintf(root, sizeof(root), "%s/vcr-c-api-XXXXXX", tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(root)) {
        perror("mkdtemp");
        return 1;
    }
    write_volume();

    CHECK(vcr_abi_version() == VCR_C_ABI_VERSION);
    CHECK(vcr_has_cuda() == 0 || vcr_has_cuda() == 1);

    vcr_open_options_init(&options);
    options.memory_cache_bytes = 1 << 20;
    CHECK(vcr_volume_open(root, &options, &volume) == VCR_OK);
    if (!volume) {
        remove_volume();
        return 1;
    }
    CHECK(vcr_volume_level_count(volume) == 2);
    vcr_level_info_init(&info);
    CHECK(vcr_volume_level_info(volume, 1, &info) == VCR_OK);
    CHECK(info.present && info.shape[0] == 8 && info.chunks[2] == 8 && strcmp(info.dtype, "|u1") == 0);
    CHECK(info.voxel_extent[0] == 2.0 && info.voxel_extent[2] == 2.0);
    CHECK(vcr_level_for_view(volume, 1.0f, 1.0f) == 0);
    CHECK(vcr_level_for_view(volume, 0.25f, 1.0f) == 1);
    CHECK(vcr_fallback_levels(volume, 0, 16, 16, 1.0f, 5) == 1);

    /* A z plane at z = 4 over x 2..9, y 3..6. */
    memset(&plane, 0, sizeof(plane));
    plane.origin[0] = 2, plane.origin[1] = 3, plane.origin[2] = 4;
    plane.dx[0] = 1, plane.dy[1] = 1, plane.normal[2] = 1;
    plane.width = 8, plane.height = 4;
    memset(&frame, 0, sizeof(frame));
    frame.plane = &plane;
    vcr_request_init(&request);
    request.volume = volume;
    request.start_level = 0;
    request.fallback_levels = 1;
    vcr_result_init(&result);
    result.values = values, result.values_capacity = 64;
    result.levels = levels, result.levels_capacity = 32;
    CHECK(vcr_request_value_count(&frame, &request) == 32);

    /* Progressive: nothing is cached; the pending reads complete the frame. */
    CHECK(vcr_pending_create(&pending) == VCR_OK);
    CHECK(vcr_sample_cached(&frame, &request, pending, &result) == VCR_OK);
    CHECK(result.width == 8 && result.height == 4 && result.layers == 1);
    for (q = 0; q < 32; ++q)
        CHECK(levels[q] == VCR_LEVEL_MISSING);
    CHECK(!vcr_pending_empty(pending));
    CHECK(vcr_pending_load(pending, count_part, NULL) == VCR_OK);
    CHECK(parts_read == 2);  /* the coarse preview, then the start level */
    vcr_pending_clear(pending);
    CHECK(vcr_sample_cached(&frame, &request, pending, &result) == VCR_OK);
    CHECK(vcr_pending_empty(pending));
    for (q = 0; q < 32; ++q)
        CHECK(levels[q] == 0 && values[q] == value_at((float)(2 + q % 8), (float)(3 + q / 8), 4));

    /* Blocking, two layers reduced by max; then the bytes through a window/level table. */
    vcr_composite_init(&max, VCR_COMPOSITE_MAX);
    request.offsets = offsets, request.offset_count = 2;
    request.composite = &max;
    CHECK(vcr_sample(&frame, &request, &result) == VCR_OK);
    CHECK(result.layers == 1 && values[9] == value_at(3, 4, 5));
    vcr_to_uint8(values, 32, bytes, coverage);
    CHECK(bytes[9] == (uint8_t)value_at(3, 4, 5) && coverage[9] == 1);
    vcr_window_level_lut(lut, 0.0f, 255.0f);
    vcr_apply_lut(bytes, 4, 8, 8, lut, pixels, 8);
    CHECK(pixels[0] == lut[bytes[0]]);

    /* A flat surface: vertices every 2 voxels in x and 3 in y at z = 10. */
    for (r = 0; r < 4; ++r)
        for (c = 0; c < 6; ++c) {
            grid[(r * 6 + c) * 3 + 0] = 2.0f * (float)c;
            grid[(r * 6 + c) * 3 + 1] = 3.0f * (float)r;
            grid[(r * 6 + c) * 3 + 2] = 10.0f;
        }
    CHECK(vcr_surface_create(grid, 4, 6, 0, NULL, NULL, &surface) == VCR_OK);
    vcr_surface_shape(surface, &rows, &cols);
    CHECK(rows == 4 && cols == 6);
    memset(&raster, 0, sizeof(raster));
    raster.x0 = 0.5, raster.y0 = 0.5, raster.sx = 0.25, raster.sy = 0.25;
    raster.width = 8, raster.height = 4;
    CHECK(fabsf(vcr_surface_pixels_per_voxel(surface, &raster) - 2.0f) < 1e-4f);
    frame.plane = NULL;
    frame.surface = surface;
    frame.raster = raster;
    request.offsets = NULL, request.offset_count = 0, request.composite = NULL;
    CHECK(vcr_sample(&frame, &request, &result) == VCR_OK);
    CHECK(fabsf(values[0] - value_at(1, 1.5f, 10)) < 1e-3f && levels[0] == 0);

    /* Errors: statuses and messages, never exceptions. */
    request.start_level = 7;
    CHECK(vcr_sample(&frame, &request, &result) == VCR_ERROR_INVALID_ARGUMENT && strlen(vcr_last_error()) > 0);
    request.start_level = 0;
    result.values_capacity = 3;
    CHECK(vcr_sample(&frame, &request, &result) == VCR_ERROR_BUFFER_TOO_SMALL);
    {
        vcr_volume* missing = NULL;
        CHECK(vcr_volume_open("/nonexistent/volume.zarr", NULL, &missing) != VCR_OK && missing == NULL);
    }

    CHECK(vcr_volume_prefill(volume, 0, prefill_progress, NULL) == VCR_OK && prefill_calls > 0);

    /* On a GPU: the same progressive frame from resident bricks gives the CPU's values. */
    if (vcr_has_cuda()) {
        vcr_volume* gpu = NULL;
        float cpu[32];
        result.values_capacity = 64;
        frame.plane = &plane, frame.surface = NULL;
        CHECK(vcr_sample(&frame, &request, &result) == VCR_OK);
        memcpy(cpu, values, sizeof(cpu));
        options.gpu_cache_bytes = 64 << 20;
        CHECK(vcr_volume_open(root, &options, &gpu) == VCR_OK);
        vcr_level_info_init(&info);
        CHECK(gpu && vcr_volume_level_info(gpu, 0, &info) == VCR_OK && info.gpu);
        {
            /* One budget split over the levels: level 1 (512 B) whole, level 0 the rest. */
            vcr_level_info coarse;
            vcr_level_info_init(&coarse);
            CHECK(vcr_volume_level_info(gpu, 1, &coarse) == VCR_OK && coarse.gpu_cache_bytes == 512);
            CHECK(info.gpu_cache_bytes == 4096);  /* all of level 0 fits in what is left */
            CHECK(info.gpu_cache_bytes + coarse.gpu_cache_bytes <= options.gpu_cache_bytes);
        }
        vcr_volume_release(gpu);
        gpu = NULL;
        options.gpu_cache_bytes = 2048;  /* less than both levels need: shares stay within it */
        CHECK(vcr_volume_open(root, &options, &gpu) == VCR_OK);
        {
            vcr_level_info fine, coarse;
            vcr_level_info_init(&fine);
            vcr_level_info_init(&coarse);
            CHECK(vcr_volume_level_info(gpu, 0, &fine) == VCR_OK && vcr_volume_level_info(gpu, 1, &coarse) == VCR_OK);
            CHECK(coarse.gpu_cache_bytes == 512 && fine.gpu_cache_bytes == 1536);
        }
        vcr_volume_release(gpu);
        gpu = NULL;
        options.gpu_cache_bytes = 64 << 20;
        CHECK(vcr_volume_open(root, &options, &gpu) == VCR_OK);
        request.volume = gpu;
        vcr_pending_clear(pending);
        CHECK(vcr_sample_cached(&frame, &request, pending, &result) == VCR_OK && !result.gpu_failed);
        CHECK(vcr_pending_load(pending, NULL, NULL) == VCR_OK);
        vcr_pending_clear(pending);
        CHECK(vcr_sample_cached(&frame, &request, pending, &result) == VCR_OK && !result.gpu_failed);
        CHECK(vcr_pending_empty(pending));
        for (q = 0; q < 32; ++q)
            CHECK(levels[q] == 0 && fabsf(values[q] - cpu[q]) <= 1e-3f);
        request.volume = volume;
        vcr_volume_release(gpu);
    }

    vcr_pending_release(pending);
    vcr_surface_release(surface);
    vcr_volume_release(volume);
    remove_volume();
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("C API passed\n");
    return 0;
}
