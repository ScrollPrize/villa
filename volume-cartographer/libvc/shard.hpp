#pragma once
// Canonical shard format for vc volumes.
//
// Volume = pyramid of levels, each level = grid of shards.
// Shard  = 1024^3 voxels = 8 z-slabs of 1024x1024 × 128 frames H.265 video.
// Each slab is one video: 128 XY frames stacked along Z.
// Volumes are always padded to 1024 multiples. No partial shards.
//
// Shard file layout:
//   [slab_0 H.265 bitstream][slab_1 bitstream]...[slab_7 bitstream][INDEX]
//
// INDEX = 8 entries of { uint32_t offset, uint32_t length }, 64 bytes.
//   slab i covers z-range [i*128, (i+1)*128) within the shard.
//   offset=0 && length=0 means slab is empty (all zeros).
//
// On-disk layout:
//   volume/
//     meta.json                  # {shape:[Z,Y,X], voxel_size:f, levels:N}
//     0/                         # level 0 (full res)
//       0.0.0.shard              # shard (sz=0, sy=0, sx=0)
//       ...

#include "types.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <span>
#include <vector>

namespace vc {

inline constexpr int SHARD_DIM  = 1024;
inline constexpr int SLAB_FRAMES = 128;   // frames per slab video
inline constexpr int SLABS_PER  = SHARD_DIM / SLAB_FRAMES;  // 8
inline constexpr int SLAB_INDEX_COUNT = SLABS_PER;  // 8
inline constexpr int SLAB_INDEX_BYTES = SLAB_INDEX_COUNT * 8;  // 64

struct SlabIndex {
    struct Entry { uint32_t offset, length; };
    std::array<Entry, SLABS_PER> entries{};
};

// Keys
struct ShardKey {
    int level, sz, sy, sx;
    bool operator==(const ShardKey&) const = default;
};

struct FrameKey {
    int level, z;  // global z coordinate at this level
    bool operator==(const FrameKey&) const = default;
};

struct FrameKeyHash {
    size_t operator()(const FrameKey& k) const noexcept {
        return std::hash<uint64_t>{}(uint64_t(k.level) << 32 | uint64_t(k.z));
    }
};

struct ShardKeyHash {
    size_t operator()(const ShardKey& k) const noexcept {
        auto h = uint64_t(k.level) << 36 | uint64_t(k.sz) << 24 |
                 uint64_t(k.sy) << 12 | uint64_t(k.sx);
        return std::hash<uint64_t>{}(h);
    }
};

// Volume metadata
struct VolumeMeta {
    Vec3i shape;        // level-0 shape {z, y, x}, always multiple of 1024
    float voxel_size;
    int levels;

    Vec3i shape_at_level(int level) const {
        int s = 1 << level;
        return {shape.z / s, shape.y / s, shape.x / s};
    }
};

// A decoded frame: 1024x1024 u8 pixels.
struct Frame {
    std::vector<uint8_t> pixels;  // row-major, 1024*1024

    Frame() : pixels(SHARD_DIM * SHARD_DIM, 0) {}

    uint8_t at(int y, int x) const { return pixels[y * SHARD_DIM + x]; }
    uint8_t* data() { return pixels.data(); }
    const uint8_t* data() const { return pixels.data(); }
    size_t byte_size() const { return pixels.size(); }
};

// -- File path helpers ------------------------------------------------------

inline std::filesystem::path shard_path(const std::filesystem::path& root,
                                         int level, int sz, int sy, int sx) {
    return root / std::to_string(level) / std::format("{}.{}.{}.shard", sz, sy, sx);
}

// -- Read slab index from shard file ----------------------------------------

inline SlabIndex read_slab_index(const std::filesystem::path& path) {
    SlabIndex idx{};
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return idx;
    fseek(f, -SLAB_INDEX_BYTES, SEEK_END);
    [[maybe_unused]] auto n = fread(idx.entries.data(), 1, SLAB_INDEX_BYTES, f);
    fclose(f);
    return idx;
}

// -- Read a slab's compressed bitstream from a shard file -------------------

inline std::vector<uint8_t> read_slab_compressed(const std::filesystem::path& path,
                                                   int slab_idx) {
    auto idx = read_slab_index(path);
    auto e = idx.entries[slab_idx];
    if (e.length == 0) return {};

    std::vector<uint8_t> buf(e.length);
    FILE* f = fopen(path.c_str(), "rb");
    fseek(f, e.offset, SEEK_SET);
    [[maybe_unused]] auto n = fread(buf.data(), 1, e.length, f);
    fclose(f);
    return buf;
}

// -- Write a shard from 8 compressed slab bitstreams ------------------------

inline void write_shard(const std::filesystem::path& path,
                         std::span<const std::vector<uint8_t>, SLABS_PER> slabs) {
    std::filesystem::create_directories(path.parent_path());

    SlabIndex idx{};
    uint32_t offset = 0;
    for (int i = 0; i < SLABS_PER; ++i) {
        if (slabs[i].empty()) {
            idx.entries[i] = {0, 0};
        } else {
            idx.entries[i] = {offset, uint32_t(slabs[i].size())};
            offset += uint32_t(slabs[i].size());
        }
    }

    FILE* f = fopen(path.c_str(), "wb");
    for (int i = 0; i < SLABS_PER; ++i)
        if (!slabs[i].empty())
            fwrite(slabs[i].data(), 1, slabs[i].size(), f);
    fwrite(idx.entries.data(), 1, SLAB_INDEX_BYTES, f);
    fclose(f);
}

// -- Read entire shard file into memory -------------------------------------

inline std::vector<uint8_t> read_shard_file(const std::filesystem::path& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    auto sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    [[maybe_unused]] auto n = fread(buf.data(), 1, sz, f);
    fclose(f);
    return buf;
}

} // namespace vc
