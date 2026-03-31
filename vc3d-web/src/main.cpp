// WASM entry point for vc3d-web.
// Exports C functions callable from JavaScript via Emscripten.
// Uses the RenderCore C API (same interface as native vc3d).

#include <cstdint>
#include <emscripten/emscripten.h>

// RenderCore C API (implemented in RenderCore_wasm.cpp)
extern "C" {
    void*           vc3d_create(int width, int height);
    void            vc3d_destroy(void* ctx);
    void            vc3d_resize(void* ctx, int width, int height);
    void            vc3d_set_plane(void* ctx, float ox, float oy, float oz,
                                   float nx, float ny, float nz);
    void            vc3d_pan(void* ctx, float dx, float dy);
    void            vc3d_zoom(void* ctx, float factor);
    void            vc3d_scroll(void* ctx, float dz);
    void            vc3d_set_window_level(void* ctx, float window, float level);
    int             vc3d_render(void* ctx);
    const uint32_t* vc3d_pixels(void* ctx);
    int             vc3d_width(void* ctx);
    int             vc3d_height(void* ctx);

    // WASM-specific chunk API
    void  vc3d_add_level(void* ctx, int sz, int sy, int sx,
                         int cz, int cy, int cx);
    void  vc3d_reset_levels(void* ctx);
    void  vc3d_set_cache_max(void* ctx, int maxBytes);
    void  vc3d_load_chunk(void* ctx, int level, int iz, int iy, int ix,
                          const uint8_t* data, int size);
    int   vc3d_cache_count(void* ctx);
    int   vc3d_cache_bytes(void* ctx);
    float vc3d_scale(void* ctx);
    float vc3d_z_offset(void* ctx);
    int   vc3d_pyramid_level(void* ctx);
    int   vc3d_num_levels(void* ctx);
    float vc3d_origin_x(void* ctx);
    float vc3d_origin_y(void* ctx);
    float vc3d_origin_z(void* ctx);
}

// Single global context (one viewer per page)
static void* g_ctx = nullptr;

// ============================================================================
// Emscripten exports -- thin wrappers around the C API with the global context
// ============================================================================

extern "C" {

EMSCRIPTEN_KEEPALIVE
void wasm_init(int width, int height) {
    if (g_ctx) vc3d_destroy(g_ctx);
    g_ctx = vc3d_create(width, height);
}

EMSCRIPTEN_KEEPALIVE
void wasm_resize(int width, int height) {
    if (g_ctx) vc3d_resize(g_ctx, width, height);
}

EMSCRIPTEN_KEEPALIVE
void wasm_add_level(int sz, int sy, int sx, int cz, int cy, int cx) {
    if (g_ctx) vc3d_add_level(g_ctx, sz, sy, sx, cz, cy, cx);
}

EMSCRIPTEN_KEEPALIVE
void wasm_reset_levels() {
    if (g_ctx) vc3d_reset_levels(g_ctx);
}

EMSCRIPTEN_KEEPALIVE
void wasm_set_plane(float ox, float oy, float oz,
                    float nx, float ny, float nz) {
    if (g_ctx) vc3d_set_plane(g_ctx, ox, oy, oz, nx, ny, nz);
}

EMSCRIPTEN_KEEPALIVE
void wasm_pan(float dx, float dy) {
    if (g_ctx) vc3d_pan(g_ctx, dx, dy);
}

EMSCRIPTEN_KEEPALIVE
void wasm_zoom(float factor) {
    if (g_ctx) vc3d_zoom(g_ctx, factor);
}

EMSCRIPTEN_KEEPALIVE
void wasm_scroll(float dz) {
    if (g_ctx) vc3d_scroll(g_ctx, dz);
}

EMSCRIPTEN_KEEPALIVE
void wasm_set_window_level(float window, float level) {
    if (g_ctx) vc3d_set_window_level(g_ctx, window, level);
}

EMSCRIPTEN_KEEPALIVE
void wasm_set_cache_max(int maxBytes) {
    if (g_ctx) vc3d_set_cache_max(g_ctx, maxBytes);
}

EMSCRIPTEN_KEEPALIVE
void wasm_load_chunk(int level, int iz, int iy, int ix,
                     const uint8_t* data, int size) {
    if (g_ctx) vc3d_load_chunk(g_ctx, level, iz, iy, ix, data, size);
}

EMSCRIPTEN_KEEPALIVE
int wasm_render() {
    if (!g_ctx) return 0;
    return vc3d_render(g_ctx);
}

// Returns pointer to ARGB32 pixels (width * height * 4 bytes).
EMSCRIPTEN_KEEPALIVE
const uint32_t* wasm_pixels() {
    if (!g_ctx) return nullptr;
    return vc3d_pixels(g_ctx);
}

EMSCRIPTEN_KEEPALIVE
int wasm_width() { return g_ctx ? vc3d_width(g_ctx) : 0; }

EMSCRIPTEN_KEEPALIVE
int wasm_height() { return g_ctx ? vc3d_height(g_ctx) : 0; }

EMSCRIPTEN_KEEPALIVE
int wasm_cache_count() { return g_ctx ? vc3d_cache_count(g_ctx) : 0; }

EMSCRIPTEN_KEEPALIVE
int wasm_cache_bytes() { return g_ctx ? vc3d_cache_bytes(g_ctx) : 0; }

EMSCRIPTEN_KEEPALIVE
float wasm_scale() { return g_ctx ? vc3d_scale(g_ctx) : 1.0f; }

EMSCRIPTEN_KEEPALIVE
float wasm_z_offset() { return g_ctx ? vc3d_z_offset(g_ctx) : 0.0f; }

EMSCRIPTEN_KEEPALIVE
int wasm_pyramid_level() { return g_ctx ? vc3d_pyramid_level(g_ctx) : 0; }

EMSCRIPTEN_KEEPALIVE
int wasm_num_levels() { return g_ctx ? vc3d_num_levels(g_ctx) : 0; }

EMSCRIPTEN_KEEPALIVE
float wasm_origin_x() { return g_ctx ? vc3d_origin_x(g_ctx) : 0.0f; }

EMSCRIPTEN_KEEPALIVE
float wasm_origin_y() { return g_ctx ? vc3d_origin_y(g_ctx) : 0.0f; }

EMSCRIPTEN_KEEPALIVE
float wasm_origin_z() { return g_ctx ? vc3d_origin_z(g_ctx) : 0.0f; }

} // extern "C"
