#pragma once
// Volume: open a volume, sample planes, prefetch frames.
// Backed by FrameCache (LRU of decoded 1024x1024 frames).

#include "cache.hpp"
#include "json.hpp"
#include "shard.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace vc {

enum class Sampling { Nearest, Trilinear };

class Volume {
    VolumeMeta meta_;
    std::unique_ptr<FrameCache> cache_;
    std::filesystem::path root_;

public:
    static Volume open(const std::filesystem::path& root, FrameCache::Config cfg) {
        Volume v;
        v.root_ = root;
        auto m = Json::parse_file(root / "meta.json");
        auto shape = m["shape"];
        v.meta_.shape = {shape[size_t(0)].i32(), shape[size_t(1)].i32(), shape[size_t(2)].i32()};
        v.meta_.voxel_size = m.f64_or("voxel_size", 1.0);
        v.meta_.levels = m.i32_or("levels", 1);
        cfg.volume_root = root;
        v.cache_ = std::make_unique<FrameCache>(std::move(cfg), v.meta_);
        return v;
    }

    Vec3i shape(int level = 0) const { return meta_.shape_at_level(level); }
    int num_levels() const { return meta_.levels; }
    float voxel_size() const { return meta_.voxel_size; }
    const VolumeMeta& meta() const { return meta_; }
    const std::filesystem::path& path() const { return root_; }
    FrameCache& cache() { return *cache_; }

    Box3f bounds() const {
        return {{0,0,0}, {float(meta_.shape.x), float(meta_.shape.y), float(meta_.shape.z)}};
    }

    void set_on_chunk_ready(std::function<void()> cb) {
        cache_->set_on_ready(std::move(cb));
    }

    // Sample a single voxel (nearest neighbor)
    uint8_t sample(Vec3f coord, int level = 0) const {
        int s = 1 << level;
        int x = std::clamp(int(coord.x) / s, 0, meta_.shape_at_level(level).x - 1);
        int y = std::clamp(int(coord.y) / s, 0, meta_.shape_at_level(level).y - 1);
        int z = std::clamp(int(coord.z) / s, 0, meta_.shape_at_level(level).z - 1);

        auto frame = cache_->get({level, z});
        if (!frame) return 0;
        return frame->at(y, x);
    }

    // Sample a 2D plane → ARGB32 framebuffer.
    // origin + vx*px + vy*py defines each pixel's world coordinate.
    // Applies LUT for window/level. Returns count of filled pixels.
    int sample_plane_argb32(uint32_t* out, int out_stride,
                            Vec3f origin, Vec3f vx, Vec3f vy,
                            int width, int height,
                            const uint32_t lut[256],
                            int level = 0) const {
        int filled = 0;
        auto sh = meta_.shape_at_level(level);
        int s = 1 << level;

        // Cache the last frame to avoid repeated LRU lookups.
        // For axis-aligned XY planes, every pixel shares the same z → same frame.
        int last_z = -1;
        std::shared_ptr<Frame> last_frame;

        for (int py = 0; py < height; ++py) {
            auto* row = out + py * out_stride;
            for (int px = 0; px < width; ++px) {
                Vec3f c = origin + vx * float(px) + vy * float(py);
                int x = std::clamp(int(c.x) / s, 0, sh.x - 1);
                int y = std::clamp(int(c.y) / s, 0, sh.y - 1);
                int z = std::clamp(int(c.z) / s, 0, sh.z - 1);

                if (z != last_z) {
                    last_z = z;
                    last_frame = cache_->get_best({level, z}, meta_.levels);
                }

                if (last_frame) {
                    row[px] = lut[last_frame->at(y, x)];
                    ++filled;
                } else {
                    row[px] = 0xFF000000;
                }
            }
        }
        return filled;
    }

    // Sample coordinates → uint8 buffer (for surface rendering)
    int sample_coords(uint8_t* out, const Vec3f* coords, int count,
                      int level = 0) const {
        int filled = 0;
        auto sh = meta_.shape_at_level(level);
        int s = 1 << level;
        int last_z = -1;
        std::shared_ptr<Frame> last_frame;

        for (int i = 0; i < count; ++i) {
            int x = std::clamp(int(coords[i].x) / s, 0, sh.x - 1);
            int y = std::clamp(int(coords[i].y) / s, 0, sh.y - 1);
            int z = std::clamp(int(coords[i].z) / s, 0, sh.z - 1);

            if (z != last_z) {
                last_z = z;
                last_frame = cache_->get({level, z});
            }
            if (last_frame) {
                out[i] = last_frame->at(y, x);
                ++filled;
            } else {
                out[i] = 0;
            }
        }
        return filled;
    }

private:
    Volume() = default;
};

inline void build_lut(uint32_t lut[256], float lo, float hi, float brightness = 1.0f) {
    float range = hi - lo;
    if (range <= 0) range = 1;
    float inv = 255.0f / range;
    for (int i = 0; i < 256; ++i) {
        float v = float(float(i) - lo) * inv * brightness;
        int b = std::clamp(int(v), 0, 255);
        auto g = uint8_t(b);
        lut[i] = 0xFF000000u | (uint32_t(g) << 16) | (uint32_t(g) << 8) | g;
    }
}

} // namespace vc
