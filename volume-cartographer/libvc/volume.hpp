#pragma once
// Volume access: open local/remote zarr, sample planes, prefetch.
// No OpenCV. Output to raw uint8_t* or uint32_t* ARGB32 buffers.

#include "cache.hpp"
#include "json.hpp"
#include "shard.hpp"
#include "types.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace vc {

// Sampling interpolation method
enum class Sampling { Nearest, Trilinear };

class Volume {
    VolumeMeta meta_;
    std::unique_ptr<ChunkCache> cache_;
    std::filesystem::path root_;

public:
    // -- Open ---------------------------------------------------------------

    static Volume open(const std::filesystem::path& zarr_root, ChunkCache::Config cache_cfg) {
        Volume v;
        v.root_ = zarr_root;
        auto m = Json::parse_file(zarr_root / "meta.json");
        auto shape = m["shape"];
        v.meta_.shape = {shape[size_t(0)].i32(), shape[size_t(1)].i32(), shape[size_t(2)].i32()};
        v.meta_.voxel_size = m.f64_or("voxel_size", 1.0);
        v.meta_.levels = m.i32_or("levels", 1);
        cache_cfg.decode = cache_cfg.decode;  // caller provides H265 decoder
        v.cache_ = std::make_unique<ChunkCache>(std::move(cache_cfg), v.meta_);
        return v;
    }

    // -- Metadata -----------------------------------------------------------

    Vec3i shape(int level = 0) const { return meta_.shape_at_level(level); }
    int num_levels() const { return meta_.levels; }
    float voxel_size() const { return meta_.voxel_size; }
    const VolumeMeta& meta() const { return meta_; }
    const std::filesystem::path& path() const { return root_; }

    Box3f data_bounds() const {
        return {{0, 0, 0},
                {float(meta_.shape.x), float(meta_.shape.y), float(meta_.shape.z)}};
    }

    // -- Cache access -------------------------------------------------------

    ChunkCache& cache() { return *cache_; }

    void set_on_chunk_ready(std::function<void()> cb) {
        cache_->set_on_chunk_ready(std::move(cb));
    }

    void prefetch(Box3f bbox, int level) { cache_->prefetch(bbox, level); }

    // -- Sample a single voxel (nearest, from hot cache) --------------------

    uint8_t sample_nearest(Vec3f coord, int level = 0) const {
        int s = 1 << level;
        int gx = int(coord.x) / s, gy = int(coord.y) / s, gz = int(coord.z) / s;
        int sx = gx / SHARD_DIM, sy = gy / SHARD_DIM, sz = gz / SHARD_DIM;
        int cx = (gx % SHARD_DIM) / CHUNK_DIM;
        int cy = (gy % SHARD_DIM) / CHUNK_DIM;
        int cz = (gz % SHARD_DIM) / CHUNK_DIM;
        int lx = gx % CHUNK_DIM, ly = gy % CHUNK_DIM, lz = gz % CHUNK_DIM;

        ChunkKey key{level, sz, sy, sx, cz, cy, cx};
        auto chunk = cache_->get(key);
        if (!chunk) return 0;
        return chunk->data<uint8_t>()[lz * CHUNK_DIM * CHUNK_DIM + ly * CHUNK_DIM + lx];
    }

    // -- Sample a plane → ARGB32 framebuffer (the critical rendering path) --
    //
    // Samples a 2D plane through the volume defined by origin + per-pixel
    // steps vx/vy. Applies LUT (window/level baked in) to produce ARGB32.
    // Returns number of pixels that had data (vs fell back to black).
    //
    // This is best-effort: uses finest available cached level, falls back to
    // coarser levels per-pixel if fine data isn't loaded yet.

    int sample_plane_argb32(uint32_t* out, int out_stride,
                            Vec3f origin, Vec3f vx, Vec3f vy,
                            int width, int height,
                            const uint32_t lut[256],
                            int preferred_level = 0) const {
        int filled = 0;
        int s = 1 << preferred_level;

        for (int py = 0; py < height; ++py) {
            auto* row = out + py * out_stride;
            for (int px = 0; px < width; ++px) {
                Vec3f coord = origin + vx * float(px) + vy * float(py);

                // Convert to voxel coordinates at this level
                int gx = int(coord.x) / s, gy = int(coord.y) / s, gz = int(coord.z) / s;
                int sx = gx / SHARD_DIM, sy = gy / SHARD_DIM, sz = gz / SHARD_DIM;
                int cx = (gx % SHARD_DIM) / CHUNK_DIM;
                int cy = (gy % SHARD_DIM) / CHUNK_DIM;
                int cz = (gz % SHARD_DIM) / CHUNK_DIM;
                int lx = gx % CHUNK_DIM, ly = gy % CHUNK_DIM, lz = gz % CHUNK_DIM;

                ChunkKey key{preferred_level, sz, sy, sx, cz, cy, cx};
                auto chunk = cache_->get_best(key);

                if (chunk) {
                    uint8_t val = chunk->data<uint8_t>()[
                        lz * CHUNK_DIM * CHUNK_DIM + ly * CHUNK_DIM + lx];
                    row[px] = lut[val];
                    ++filled;
                } else {
                    row[px] = 0xFF000000;  // opaque black
                }
            }
        }
        return filled;
    }

    // -- Sample coordinates → uint8 buffer ----------------------------------
    // For surface-based rendering where coords come from a QuadSurface.

    int sample_coords(uint8_t* out, const Vec3f* coords, int count,
                      Sampling method = Sampling::Nearest,
                      int level = 0) const {
        int filled = 0;
        for (int i = 0; i < count; ++i) {
            out[i] = sample_nearest(coords[i], level);
            if (out[i]) ++filled;
        }
        return filled;
    }

private:
    Volume() = default;
};

// -- LUT building -----------------------------------------------------------

inline void build_lut(uint32_t lut[256], float lo, float hi,
                      float brightness = 1.0f) {
    float range = hi - lo;
    if (range <= 0) range = 1;
    float inv = 255.0f / range;
    for (int i = 0; i < 256; ++i) {
        float v = float(i - lo) * inv * brightness;
        int b = int(v);
        if (b < 0) b = 0;
        if (b > 255) b = 255;
        uint8_t g = uint8_t(b);
        lut[i] = 0xFF000000u | (g << 16) | (g << 8) | g;
    }
}

} // namespace vc
