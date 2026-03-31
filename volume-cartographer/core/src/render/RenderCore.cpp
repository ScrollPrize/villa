#include "vc/core/render/RenderCore.hpp"
#include "vc/core/render/RenderCore_C.h"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/SampleParams.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

// Non-temporal store for write-only ARGB32 output — bypasses cache,
// freeing cache lines for read-heavy LUT/chunk data.
static inline void nt_store_u32(uint32_t* dst, uint32_t val) {
#if defined(__x86_64__)
    _mm_stream_si32(reinterpret_cast<int*>(dst), static_cast<int>(val));
#elif defined(__aarch64__) && __has_builtin(__builtin_nontemporal_store)
    __builtin_nontemporal_store(val, dst);
#else
    *dst = val;
#endif
}

namespace vc::render {

// ============================================================================
// Basis computation (same logic as PlaneSurface)
// ============================================================================

static void computeBasis(const float n[3], float vx[3], float vy[3])
{
    // Choose the axis least aligned with the normal
    float ax = std::abs(n[0]), ay = std::abs(n[1]), az = std::abs(n[2]);
    float up[3];
    if (ax <= ay && ax <= az)
        up[0] = 1, up[1] = 0, up[2] = 0;
    else if (ay <= az)
        up[0] = 0, up[1] = 1, up[2] = 0;
    else
        up[0] = 0, up[1] = 0, up[2] = 1;

    // vx = normalize(cross(up, n))
    vx[0] = up[1]*n[2] - up[2]*n[1];
    vx[1] = up[2]*n[0] - up[0]*n[2];
    vx[2] = up[0]*n[1] - up[1]*n[0];
    float len = std::sqrt(vx[0]*vx[0] + vx[1]*vx[1] + vx[2]*vx[2]);
    if (len > 0) { vx[0] /= len; vx[1] /= len; vx[2] /= len; }

    // vy = cross(n, vx)
    vy[0] = n[1]*vx[2] - n[2]*vx[1];
    vy[1] = n[2]*vx[0] - n[0]*vx[2];
    vy[2] = n[0]*vx[1] - n[1]*vx[0];
    len = std::sqrt(vy[0]*vy[0] + vy[1]*vy[1] + vy[2]*vy[2]);
    if (len > 0) { vy[0] /= len; vy[1] /= len; vy[2] /= len; }

    // Convention: vx should have positive x component, vy positive y
    if (vx[0] < 0) { vx[0] = -vx[0]; vx[1] = -vx[1]; vx[2] = -vx[2]; }
    if (vy[1] < 0) { vy[0] = -vy[0]; vy[1] = -vy[1]; vy[2] = -vy[2]; }
}

// ============================================================================
// RenderCore
// ============================================================================

RenderCore::RenderCore(int width, int height)
{
    rebuildLut();
    recalcBasis();
    resize(width, height);
}

RenderCore::~RenderCore() = default;

void RenderCore::resize(int width, int height)
{
    if (width <= 0 || height <= 0) return;
    width_ = width;
    height_ = height;
    pixels_ = std::make_unique<uint32_t[]>(width * height);
    std::memset(pixels_.get(), 0, sizeof(uint32_t) * width * height);
}

void RenderCore::setVolume(Volume* vol)
{
    volume_ = vol;
    recalcPyramidLevel();
}

void RenderCore::setPlane(float ox, float oy, float oz,
                          float nx, float ny, float nz)
{
    origin_[0] = ox; origin_[1] = oy; origin_[2] = oz;

    // Normalize
    float len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len > 0) { nx /= len; ny /= len; nz /= len; }
    normal_[0] = nx; normal_[1] = ny; normal_[2] = nz;

    recalcBasis();
}

void RenderCore::pan(float dx, float dy)
{
    // dx, dy are in screen pixels; convert to world units
    float m = 1.0f / scale_;
    origin_[0] += vx_[0] * dx * m + vy_[0] * dy * m;
    origin_[1] += vx_[1] * dx * m + vy_[1] * dy * m;
    origin_[2] += vx_[2] * dx * m + vy_[2] * dy * m;
}

void RenderCore::zoom(float factor)
{
    scale_ *= factor;
    scale_ = std::clamp(scale_, 0.01f, 10.0f);
    recalcPyramidLevel();
}

void RenderCore::scrollSlice(float dz)
{
    zOff_ += dz;
}

void RenderCore::setWindowLevel(float window, float level)
{
    windowLow_ = level - window * 0.5f;
    windowHigh_ = level + window * 0.5f;
    rebuildLut();
}

void RenderCore::setTileSize(int px) { tilePx_ = std::max(64, px); }
void RenderCore::setThreadCount(int n) { threadCount_ = std::max(0, n); }

void RenderCore::rebuildLut()
{
    float lo = std::clamp(windowLow_, 0.0f, 255.0f);
    float hi = std::clamp(windowHigh_, lo + 1.0f, 255.0f);
    float span = std::max(1.0f, hi - lo);
    for (int i = 0; i < 256; ++i) {
        uint8_t v = static_cast<uint8_t>(
            std::clamp((static_cast<float>(i) - lo) / span * 255.0f, 0.0f, 255.0f));
        lut_[i] = 0xFF000000u | (uint32_t(v) << 16) | (uint32_t(v) << 8) | uint32_t(v);
    }
}

void RenderCore::recalcBasis()
{
    computeBasis(normal_, vx_, vy_);
}

void RenderCore::recalcPyramidLevel()
{
    if (!volume_) { dsScaleIdx_ = 0; return; }
    int nScales = static_cast<int>(volume_->numScales());
    // Pick the pyramid level where one pixel ~ one voxel at that level
    // level L has voxelSize = 2^L, so we want 2^L <= 1/scale
    float invScale = 1.0f / scale_;
    int lvl = 0;
    while (lvl + 1 < nScales && (1 << (lvl + 1)) <= invScale)
        ++lvl;
    dsScaleIdx_ = lvl;
}

// ============================================================================
// Render
// ============================================================================

int RenderCore::render()
{
    if (!volume_ || !pixels_ || width_ <= 0 || height_ <= 0)
        return 0;

    // Compute the world-space plane origin (shifted by zOff along normal)
    cv::Vec3f planeOrigin(
        origin_[0] + normal_[0] * zOff_,
        origin_[1] + normal_[1] * zOff_,
        origin_[2] + normal_[2] * zOff_);

    cv::Vec3f bx(vx_[0], vx_[1], vx_[2]);
    cv::Vec3f by(vy_[0], vy_[1], vy_[2]);

    float m = 1.0f / scale_;
    cv::Vec3f vxStep = bx * m;
    cv::Vec3f vyStep = by * m;

    // Plane origin for the top-left pixel
    // Center the view: the plane origin is at the center of the framebuffer
    float cx = width_ * 0.5f;
    float cy = height_ * 0.5f;
    cv::Vec3f topLeft = planeOrigin - vxStep * cx - vyStep * cy;

    // Tile the framebuffer
    int tilesX = (width_ + tilePx_ - 1) / tilePx_;
    int tilesY = (height_ + tilePx_ - 1) / tilePx_;
    int totalTiles = tilesX * tilesY;

    int nThreads = threadCount_;
    if (nThreads <= 0)
        nThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    nThreads = std::min(nThreads, totalTiles);

    // Prepare sampling params
    vc::SampleParams sp;
    sp.level = dsScaleIdx_;
    sp.method = (dsScaleIdx_ >= 3) ? vc::Sampling::Nearest : vc::Sampling::Trilinear;

    // Tile render lambda
    auto renderTile = [&](int tileIdx) {
        int tx = tileIdx % tilesX;
        int ty = tileIdx / tilesX;
        int x0 = tx * tilePx_;
        int y0 = ty * tilePx_;
        int tw = std::min(tilePx_, width_ - x0);
        int th = std::min(tilePx_, height_ - y0);

        // Compute plane origin for this tile's top-left corner
        cv::Vec3f tileOrigin = topLeft + vxStep * static_cast<float>(x0)
                                       + vyStep * static_cast<float>(y0);

        // Sample via the fused plane path
        cv::Mat_<uint8_t> gray;
        volume_->samplePlaneBestEffort(gray, tileOrigin, vxStep, vyStep, tw, th, sp);

        // Write ARGB32 into framebuffer via the fused LUT.
        // Non-temporal stores bypass the cache for write-only output.
        if (!gray.empty()) {
            for (int row = 0; row < th; ++row) {
                const uint8_t* src = gray.ptr<uint8_t>(row);
                uint32_t* dst = pixels_.get() + (y0 + row) * width_ + x0;
                for (int col = 0; col < tw; ++col) {
                    nt_store_u32(&dst[col], lut_[src[col]]);
                }
            }
        } else {
            // No data: fill with black
            for (int row = 0; row < th; ++row) {
                uint32_t* dst = pixels_.get() + (y0 + row) * width_ + x0;
                std::memset(dst, 0, sizeof(uint32_t) * tw);
            }
        }
    };

    if (nThreads <= 1) {
        // Single-threaded
        for (int i = 0; i < totalTiles; ++i)
            renderTile(i);
    } else {
        // Multi-threaded using std::jthread
        std::atomic<int> nextTile{0};
        std::vector<std::jthread> workers;
        workers.reserve(nThreads);
        for (int t = 0; t < nThreads; ++t) {
            workers.emplace_back([&](std::stop_token) {
                while (true) {
                    int idx = nextTile.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= totalTiles) break;
                    renderTile(idx);
                }
            });
        }
        // jthread destructors join automatically
    }

    return totalTiles;
}

}  // namespace vc::render

// ============================================================================
// C API
// ============================================================================

extern "C" {

void* vc3d_create(int width, int height)
{
    return new vc::render::RenderCore(width, height);
}

void vc3d_destroy(void* ctx)
{
    delete static_cast<vc::render::RenderCore*>(ctx);
}

void vc3d_resize(void* ctx, int width, int height)
{
    static_cast<vc::render::RenderCore*>(ctx)->resize(width, height);
}

void vc3d_set_volume(void* ctx, void* volume)
{
    static_cast<vc::render::RenderCore*>(ctx)->setVolume(
        static_cast<Volume*>(volume));
}

void vc3d_set_plane(void* ctx,
                    float ox, float oy, float oz,
                    float nx, float ny, float nz)
{
    static_cast<vc::render::RenderCore*>(ctx)->setPlane(ox, oy, oz, nx, ny, nz);
}

void vc3d_pan(void* ctx, float dx, float dy)
{
    static_cast<vc::render::RenderCore*>(ctx)->pan(dx, dy);
}

void vc3d_zoom(void* ctx, float factor)
{
    static_cast<vc::render::RenderCore*>(ctx)->zoom(factor);
}

void vc3d_scroll(void* ctx, float dz)
{
    static_cast<vc::render::RenderCore*>(ctx)->scrollSlice(dz);
}

void vc3d_set_window_level(void* ctx, float window, float level)
{
    static_cast<vc::render::RenderCore*>(ctx)->setWindowLevel(window, level);
}

int vc3d_render(void* ctx)
{
    return static_cast<vc::render::RenderCore*>(ctx)->render();
}

const uint32_t* vc3d_pixels(void* ctx)
{
    return static_cast<vc::render::RenderCore*>(ctx)->pixels();
}

int vc3d_width(void* ctx)
{
    return static_cast<vc::render::RenderCore*>(ctx)->width();
}

int vc3d_height(void* ctx)
{
    return static_cast<vc::render::RenderCore*>(ctx)->height();
}

}  // extern "C"
