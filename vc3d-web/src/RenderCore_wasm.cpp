// WASM implementation of the RenderCore C API (RenderCore_C.h).
// Same function signatures as the native RenderCore, but backed by the
// lightweight ChunkCache instead of the full Volume class (which requires
// OpenCV, xtensor, filesystem, networking, etc.).
//
// This gives the WASM viewer the same API as native vc3d:
// - vc3d_create / vc3d_destroy
// - vc3d_set_plane / vc3d_pan / vc3d_zoom / vc3d_scroll
// - vc3d_set_window_level
// - vc3d_render / vc3d_pixels
//
// Plus WASM-specific exports for feeding chunk data from JavaScript:
// - vc3d_add_level / vc3d_reset_levels / vc3d_load_chunk / vc3d_set_cache_max

#include "chunk_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

// ============================================================================
// Basis computation (matches native RenderCore)
// ============================================================================

static void computeBasis(const float n[3], float vx[3], float vy[3])
{
    float ax = std::abs(n[0]), ay = std::abs(n[1]), az = std::abs(n[2]);
    float up[3];
    if (ax <= ay && ax <= az)
        up[0] = 1, up[1] = 0, up[2] = 0;
    else if (ay <= az)
        up[0] = 0, up[1] = 1, up[2] = 0;
    else
        up[0] = 0, up[1] = 0, up[2] = 1;

    vx[0] = up[1]*n[2] - up[2]*n[1];
    vx[1] = up[2]*n[0] - up[0]*n[2];
    vx[2] = up[0]*n[1] - up[1]*n[0];
    float len = std::sqrt(vx[0]*vx[0] + vx[1]*vx[1] + vx[2]*vx[2]);
    if (len > 0) { vx[0] /= len; vx[1] /= len; vx[2] /= len; }

    vy[0] = n[1]*vx[2] - n[2]*vx[1];
    vy[1] = n[2]*vx[0] - n[0]*vx[2];
    vy[2] = n[0]*vx[1] - n[1]*vx[0];
    len = std::sqrt(vy[0]*vy[0] + vy[1]*vy[1] + vy[2]*vy[2]);
    if (len > 0) { vy[0] /= len; vy[1] /= len; vy[2] /= len; }

    if (vx[0] < 0) { vx[0] = -vx[0]; vx[1] = -vx[1]; vx[2] = -vx[2]; }
    if (vy[1] < 0) { vy[0] = -vy[0]; vy[1] = -vy[1]; vy[2] = -vy[2]; }
}

// ============================================================================
// RenderCore (WASM)
// ============================================================================

struct RenderCoreWasm {
    // Framebuffer (ARGB32, matching native RenderCore)
    std::unique_ptr<uint32_t[]> pixels;
    int width = 0;
    int height = 0;

    // Chunk cache (owned, receives data from JS)
    ChunkCache cache;
    std::vector<LevelMeta> levels;

    // Camera: plane definition (world/level-0 coordinates)
    float origin[3] = {0, 0, 0};
    float normal[3] = {0, 0, 1};
    float vx[3] = {1, 0, 0};
    float vy[3] = {0, 1, 0};
    float scale = 0.5f;    // pixels per voxel
    float zOff = 0.0f;     // slice offset along normal

    // Pyramid level (derived from scale)
    int dsLevel = 0;

    // Window/level LUT
    float windowLow = 0.0f;
    float windowHigh = 255.0f;
    uint32_t lut[256];

    RenderCoreWasm(int w, int h) {
        rebuildLut();
        recalcBasis();
        resize(w, h);
    }

    void resize(int w, int h) {
        if (w <= 0 || h <= 0) return;
        width = w;
        height = h;
        pixels = std::make_unique<uint32_t[]>(w * h);
        std::memset(pixels.get(), 0, sizeof(uint32_t) * w * h);
    }

    void rebuildLut() {
        float lo = std::clamp(windowLow, 0.0f, 255.0f);
        float hi = std::clamp(windowHigh, lo + 1.0f, 255.0f);
        float span = std::max(1.0f, hi - lo);
        for (int i = 0; i < 256; ++i) {
            uint8_t v = static_cast<uint8_t>(
                std::clamp((static_cast<float>(i) - lo) / span * 255.0f, 0.0f, 255.0f));
            lut[i] = 0xFF000000u | (uint32_t(v) << 16) | (uint32_t(v) << 8) | uint32_t(v);
        }
    }

    void recalcBasis() {
        computeBasis(normal, vx, vy);
    }

    void recalcPyramidLevel() {
        int nScales = static_cast<int>(levels.size());
        if (nScales == 0) { dsLevel = 0; return; }
        float invScale = 1.0f / scale;
        int lvl = 0;
        while (lvl + 1 < nScales && (1 << (lvl + 1)) <= invScale)
            ++lvl;
        dsLevel = lvl;
    }

    // Sample a single voxel (trilinear) at a given pyramid level.
    // Coordinates are in level-space (already scaled by 1/2^level).
    float sampleTrilinear(int level, float vz, float vy, float vx) {
        auto shape = cache.levelShape(level);
        int sz = shape[0], sy = shape[1], sx = shape[2];
        if (vx < 0 || vy < 0 || vz < 0 || vx >= sx - 1 || vy >= sy - 1 || vz >= sz - 1)
            return sampleNearest(level, vz, vy, vx);

        int ix = static_cast<int>(vx);
        int iy = static_cast<int>(vy);
        int iz = static_cast<int>(vz);
        float fx = vx - ix, fy = vy - iy, fz = vz - iz;

        auto cs = cache.chunkShape(level);
        int cz = cs[0], cy = cs[1], cx = cs[2];

        auto s = [&](int z, int y, int x) -> float {
            int ciz = z / cz, ciy = y / cy, cix = x / cx;
            int lz = z % cz, ly = y % cy, lx = x % cx;
            auto chunk = cache.get(ChunkKey{level, ciz, ciy, cix});
            if (!chunk) return 0;
            size_t off = static_cast<size_t>(lz) * cy * cx +
                         static_cast<size_t>(ly) * cx + lx;
            if (off >= chunk->bytes.size()) return 0;
            return chunk->data()[off];
        };

        float c000 = s(iz, iy, ix),     c001 = s(iz, iy, ix+1);
        float c010 = s(iz, iy+1, ix),   c011 = s(iz, iy+1, ix+1);
        float c100 = s(iz+1, iy, ix),   c101 = s(iz+1, iy, ix+1);
        float c110 = s(iz+1, iy+1, ix), c111 = s(iz+1, iy+1, ix+1);

        float c00 = c000 + fx * (c001 - c000);
        float c01 = c010 + fx * (c011 - c010);
        float c10 = c100 + fx * (c101 - c100);
        float c11 = c110 + fx * (c111 - c110);
        float c0 = c00 + fy * (c01 - c00);
        float c1 = c10 + fy * (c11 - c10);
        return c0 + fz * (c1 - c0);
    }

    float sampleNearest(int level, float vz, float vy, float vx) {
        auto shape = cache.levelShape(level);
        int sz = shape[0], sy = shape[1], sx = shape[2];
        if (vx < 0 || vy < 0 || vz < 0 || vx >= sx || vy >= sy || vz >= sz)
            return 0;

        int ix = static_cast<int>(vx + 0.5f);
        int iy = static_cast<int>(vy + 0.5f);
        int iz = static_cast<int>(vz + 0.5f);
        if (ix >= sx) ix = sx - 1;
        if (iy >= sy) iy = sy - 1;
        if (iz >= sz) iz = sz - 1;

        auto cs = cache.chunkShape(level);
        int cz = cs[0], cy = cs[1], cx = cs[2];
        int ciz = iz / cz, ciy = iy / cy, cix = ix / cx;
        int lz = iz % cz, ly = iy % cy, lx = ix % cx;

        auto chunk = cache.get(ChunkKey{level, ciz, ciy, cix});
        if (!chunk) return 0;
        size_t offset = static_cast<size_t>(lz) * cy * cx +
                        static_cast<size_t>(ly) * cx + lx;
        if (offset >= chunk->bytes.size()) return 0;
        return chunk->data()[offset];
    }

    int render() {
        if (!pixels || width <= 0 || height <= 0 || levels.empty())
            return 0;

        int level = dsLevel;

        // Scale factor from world (level-0) to this pyramid level
        float levelScale = (level > 0) ? (1.0f / static_cast<float>(1 << level)) : 1.0f;

        // Plane origin shifted by zOff along normal
        float po[3] = {
            origin[0] + normal[0] * zOff,
            origin[1] + normal[1] * zOff,
            origin[2] + normal[2] * zOff
        };

        // Voxel step per pixel (in world coords)
        float m = 1.0f / scale;
        float vxStep[3] = { vx[0] * m, vx[1] * m, vx[2] * m };
        float vyStep[3] = { vy[0] * m, vy[1] * m, vy[2] * m };

        // Top-left corner (center the view)
        float cx = width * 0.5f;
        float cy = height * 0.5f;
        float tl[3] = {
            po[0] - vxStep[0] * cx - vyStep[0] * cy,
            po[1] - vxStep[1] * cx - vyStep[1] * cy,
            po[2] - vxStep[2] * cx - vyStep[2] * cy
        };

        // Scale to level coordinates
        float tlL[3] = { tl[0] * levelScale, tl[1] * levelScale, tl[2] * levelScale };
        float vxL[3] = { vxStep[0] * levelScale, vxStep[1] * levelScale, vxStep[2] * levelScale };
        float vyL[3] = { vyStep[0] * levelScale, vyStep[1] * levelScale, vyStep[2] * levelScale };

        bool useNearest = (level >= 3);

        for (int py = 0; py < height; py++) {
            float rowX = tlL[0] + vyL[0] * py;
            float rowY = tlL[1] + vyL[1] * py;
            float rowZ = tlL[2] + vyL[2] * py;
            uint32_t* dst = pixels.get() + py * width;

            for (int px = 0; px < width; px++) {
                // voxel coords at this pyramid level: (x, y, z)
                float vvx = rowX + vxL[0] * px;
                float vvy = rowY + vxL[1] * px;
                float vvz = rowZ + vxL[2] * px;

                float val;
                if (useNearest)
                    val = sampleNearest(level, vvz, vvy, vvx);
                else
                    val = sampleTrilinear(level, vvz, vvy, vvx);

                uint8_t v = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
                dst[px] = lut[v];
            }
        }

        return 1;
    }
};

// ============================================================================
// C API — matches RenderCore_C.h signatures
// ============================================================================

#define RC(ctx) static_cast<RenderCoreWasm*>(ctx)

extern "C" {

void* vc3d_create(int width, int height)
{
    return new RenderCoreWasm(width, height);
}

void vc3d_destroy(void* ctx)
{
    delete RC(ctx);
}

void vc3d_resize(void* ctx, int width, int height)
{
    RC(ctx)->resize(width, height);
}

void vc3d_set_volume(void* ctx, void* /*volume*/)
{
    // No-op for WASM — volume data is fed via vc3d_load_chunk
}

void vc3d_set_plane(void* ctx,
                    float ox, float oy, float oz,
                    float nx, float ny, float nz)
{
    auto* rc = RC(ctx);
    rc->origin[0] = ox; rc->origin[1] = oy; rc->origin[2] = oz;
    float len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len > 0) { nx /= len; ny /= len; nz /= len; }
    rc->normal[0] = nx; rc->normal[1] = ny; rc->normal[2] = nz;
    rc->recalcBasis();
}

void vc3d_pan(void* ctx, float dx, float dy)
{
    auto* rc = RC(ctx);
    float m = 1.0f / rc->scale;
    rc->origin[0] += rc->vx[0] * dx * m + rc->vy[0] * dy * m;
    rc->origin[1] += rc->vx[1] * dx * m + rc->vy[1] * dy * m;
    rc->origin[2] += rc->vx[2] * dx * m + rc->vy[2] * dy * m;
}

void vc3d_zoom(void* ctx, float factor)
{
    auto* rc = RC(ctx);
    rc->scale *= factor;
    rc->scale = std::clamp(rc->scale, 0.01f, 10.0f);
    rc->recalcPyramidLevel();
}

void vc3d_scroll(void* ctx, float dz)
{
    RC(ctx)->zOff += dz;
}

void vc3d_set_window_level(void* ctx, float window, float level)
{
    auto* rc = RC(ctx);
    rc->windowLow = level - window * 0.5f;
    rc->windowHigh = level + window * 0.5f;
    rc->rebuildLut();
}

int vc3d_render(void* ctx)
{
    return RC(ctx)->render();
}

const uint32_t* vc3d_pixels(void* ctx)
{
    return RC(ctx)->pixels.get();
}

int vc3d_width(void* ctx)
{
    return RC(ctx)->width;
}

int vc3d_height(void* ctx)
{
    return RC(ctx)->height;
}

// ============================================================================
// WASM-specific exports (chunk data from JavaScript)
// ============================================================================

void vc3d_add_level(void* ctx, int shape_z, int shape_y, int shape_x,
                    int chunk_z, int chunk_y, int chunk_x)
{
    auto* rc = RC(ctx);
    rc->levels.push_back(LevelMeta{{shape_z, shape_y, shape_x},
                                   {chunk_z, chunk_y, chunk_x}});
    rc->cache.setLevels(rc->levels);
    rc->recalcPyramidLevel();
}

void vc3d_reset_levels(void* ctx)
{
    auto* rc = RC(ctx);
    rc->levels.clear();
    rc->cache.setLevels({});
    rc->dsLevel = 0;
}

void vc3d_set_cache_max(void* ctx, int maxBytes)
{
    RC(ctx)->cache.setMaxBytes(static_cast<size_t>(maxBytes));
}

void vc3d_load_chunk(void* ctx, int level, int iz, int iy, int ix,
                     const uint8_t* data, int size)
{
    RC(ctx)->cache.loadChunk(level, iz, iy, ix, data, static_cast<size_t>(size));
}

int vc3d_cache_count(void* ctx)
{
    return static_cast<int>(RC(ctx)->cache.count());
}

int vc3d_cache_bytes(void* ctx)
{
    return static_cast<int>(RC(ctx)->cache.totalBytes());
}

float vc3d_scale(void* ctx)
{
    return RC(ctx)->scale;
}

float vc3d_z_offset(void* ctx)
{
    return RC(ctx)->zOff;
}

int vc3d_pyramid_level(void* ctx)
{
    return RC(ctx)->dsLevel;
}

int vc3d_num_levels(void* ctx)
{
    return static_cast<int>(RC(ctx)->levels.size());
}

float vc3d_origin_x(void* ctx) { return RC(ctx)->origin[0]; }
float vc3d_origin_y(void* ctx) { return RC(ctx)->origin[1]; }
float vc3d_origin_z(void* ctx) { return RC(ctx)->origin[2]; }

} // extern "C"
