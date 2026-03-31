// WASM entry point for vc3d-web.
// Exports C functions callable from JavaScript via Emscripten.

#include "renderer.hpp"
#include "chunk_cache.hpp"

#include <emscripten/emscripten.h>

static Renderer g_renderer;
static ChunkCache g_cache;
static std::vector<LevelMeta> g_levels;

extern "C" {

// Initialize the renderer with given canvas dimensions.
EMSCRIPTEN_KEEPALIVE
void vc3d_init(int width, int height) {
    g_renderer.init(width, height);
}

// Resize the renderer (e.g., on window resize).
EMSCRIPTEN_KEEPALIVE
void vc3d_resize(int width, int height) {
    g_renderer.resize(width, height);
}

// Configure volume metadata: add a pyramid level.
// shape_z/y/x = volume dimensions at this level.
// chunk_z/y/x = chunk dimensions at this level.
EMSCRIPTEN_KEEPALIVE
void vc3d_add_level(int shape_z, int shape_y, int shape_x,
                    int chunk_z, int chunk_y, int chunk_x) {
    g_levels.push_back(LevelMeta{{shape_z, shape_y, shape_x},
                                 {chunk_z, chunk_y, chunk_x}});
    g_cache.setLevels(g_levels);
}

// Reset level metadata (call before re-adding levels for a new volume).
EMSCRIPTEN_KEEPALIVE
void vc3d_reset_levels() {
    g_levels.clear();
    g_cache.setLevels({});
}

// Set the slice plane with origin and two basis vectors.
// All coordinates in voxel space at the current pyramid level.
EMSCRIPTEN_KEEPALIVE
void vc3d_set_plane(float ox, float oy, float oz,
                    float vx_x, float vx_y, float vx_z,
                    float vy_x, float vy_y, float vy_z) {
    g_renderer.setPlane(ox, oy, oz, vx_x, vx_y, vx_z, vy_x, vy_y, vy_z);
}

// Convenience: set an axis-aligned XY slice at given Z depth.
// scale = zoom factor (pixels per voxel), panX/panY in pixels.
EMSCRIPTEN_KEEPALIVE
void vc3d_set_slice(float z, float scale, float panX, float panY) {
    g_renderer.setAxisAlignedSlice(z, scale, panX, panY);
}

// Set the pyramid level to render from.
EMSCRIPTEN_KEEPALIVE
void vc3d_set_level(int level) {
    g_renderer.setLevel(level);
}

// Set maximum cache size in bytes.
EMSCRIPTEN_KEEPALIVE
void vc3d_set_cache_max(int maxBytes) {
    g_cache.setMaxBytes(static_cast<size_t>(maxBytes));
}

// Load a decompressed chunk into the cache.
// data = pointer to raw uint8 voxel data in ZYX row-major order.
// size = byte count (should be chunk_z * chunk_y * chunk_x).
EMSCRIPTEN_KEEPALIVE
void vc3d_load_chunk(int level, int iz, int iy, int ix,
                     const uint8_t* data, int size) {
    g_cache.loadChunk(level, iz, iy, ix, data, static_cast<size_t>(size));
}

// Render the current view. Returns pointer to RGBA pixel data
// (width * height * 4 bytes). Valid until next render() call.
EMSCRIPTEN_KEEPALIVE
const uint8_t* vc3d_render() {
    return g_renderer.render(g_cache);
}

// Query current canvas dimensions.
EMSCRIPTEN_KEEPALIVE
int vc3d_width() { return g_renderer.width(); }

EMSCRIPTEN_KEEPALIVE
int vc3d_height() { return g_renderer.height(); }

// Query cache stats.
EMSCRIPTEN_KEEPALIVE
int vc3d_cache_count() { return static_cast<int>(g_cache.count()); }

EMSCRIPTEN_KEEPALIVE
int vc3d_cache_bytes() { return static_cast<int>(g_cache.totalBytes()); }

} // extern "C"
