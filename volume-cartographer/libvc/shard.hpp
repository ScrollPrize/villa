#pragma once
// Canonical shard format for vc volumes.
//
// Volume = pyramid of levels, each level = grid of shards.
// Shard  = 1024^3 voxels = 8^3 chunks of 128^3 u8 voxels, H265 encoded.
// Volumes are always padded to 1024 multiples. No partial shards/chunks.
//
// Shard file layout (append-only):
//   [chunk_0 H265 data][chunk_1 H265 data]...[chunk_511 H265 data][INDEX]
//
// INDEX = 512 entries of { uint32_t offset, uint32_t length }, 4096 bytes.
//   Linear index: cz * 64 + cy * 8 + cx  (0 <= cz,cy,cx < 8)
//   offset=0 && length=0 means chunk is empty (all zeros).
//
// On-disk layout:
//   volume.zarr/
//     meta.json                  # {shape:[Z,Y,X], voxel_size:f, levels:N}
//     0/                         # level 0 (full res)
//       0.0.0.shard              # shard (sz=0, sy=0, sx=0)
//       0.0.1.shard
//       ...
//     1/                         # level 1 (2x downsampled)
//       ...

#include "types.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <span>
#include <vector>

namespace vc {

inline constexpr int SHARD_DIM   = 1024;
inline constexpr int CHUNK_DIM   = 128;
inline constexpr int CHUNKS_PER  = SHARD_DIM / CHUNK_DIM;  // 8
inline constexpr int INDEX_COUNT = CHUNKS_PER * CHUNKS_PER * CHUNKS_PER;  // 512
inline constexpr int INDEX_BYTES = INDEX_COUNT * 8;  // 4096

struct ShardIndex {
    struct Entry { uint32_t offset, length; };
    std::array<Entry, INDEX_COUNT> entries{};

    static int linear(int cz, int cy, int cx) {
        return cz * CHUNKS_PER * CHUNKS_PER + cy * CHUNKS_PER + cx;
    }

    Entry& at(int cz, int cy, int cx) { return entries[linear(cz, cy, cx)]; }
    const Entry& at(int cz, int cy, int cx) const { return entries[linear(cz, cy, cx)]; }
};

// Keys for addressing chunks and shards within a volume pyramid.
struct ShardKey {
    int level, sz, sy, sx;
    bool operator==(const ShardKey&) const = default;
};

struct ChunkKey {
    int level, sz, sy, sx, cz, cy, cx;
    bool operator==(const ChunkKey&) const = default;

    ShardKey shard() const { return {level, sz, sy, sx}; }
};

struct ChunkKeyHash {
    size_t operator()(const ChunkKey& k) const noexcept {
        // Pack into 64 bits: level(4) | sz(12) | sy(12) | sx(12) | cz(3) | cy(3) | cx(3)
        auto h = uint64_t(k.level) << 45 | uint64_t(k.sz) << 33 |
                 uint64_t(k.sy) << 21 | uint64_t(k.sx) << 9 |
                 uint64_t(k.cz) << 6 | uint64_t(k.cy) << 3 | uint64_t(k.cx);
        return std::hash<uint64_t>{}(h);
    }
};

struct ShardKeyHash {
    size_t operator()(const ShardKey& k) const noexcept {
        auto h = uint64_t(k.level) << 36 | uint64_t(k.sz) << 24 |
                 uint64_t(k.sy) << 12 | uint64_t(k.sx);
        return std::hash<uint64_t>{}(h);
    }
};

// Volume metadata (from meta.json)
struct VolumeMeta {
    Vec3i shape;        // level-0 shape {z, y, x}, always multiple of 1024
    float voxel_size;
    int levels;         // number of pyramid levels

    Vec3i shards_per_level(int level) const {
        int s = 1 << level;
        return {shape.z / (SHARD_DIM * s), shape.y / (SHARD_DIM * s), shape.x / (SHARD_DIM * s)};
    }

    Vec3i shape_at_level(int level) const {
        int s = 1 << level;
        return {shape.z / s, shape.y / s, shape.x / s};
    }
};

// -- Shard file path helpers ------------------------------------------------

inline std::filesystem::path shard_path(const std::filesystem::path& root,
                                         int level, int sz, int sy, int sx) {
    return root / std::to_string(level) / std::format("{}.{}.{}.shard", sz, sy, sx);
}

// -- Read a shard index from a shard file or buffer -------------------------

inline ShardIndex read_shard_index(const std::filesystem::path& path) {
    ShardIndex idx{};
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return idx;  // all zeros = all empty
    fseek(f, -INDEX_BYTES, SEEK_END);
    fread(idx.entries.data(), 1, INDEX_BYTES, f);
    fclose(f);
    return idx;
}

inline ShardIndex read_shard_index(std::span<const uint8_t> shard_data) {
    ShardIndex idx{};
    if (shard_data.size() < INDEX_BYTES) return idx;
    memcpy(idx.entries.data(), shard_data.data() + shard_data.size() - INDEX_BYTES, INDEX_BYTES);
    return idx;
}

// -- Read a single chunk from a shard file ----------------------------------
// Returns H265 compressed bytes. Caller decodes with video_decode().

inline std::vector<uint8_t> read_chunk_compressed(const std::filesystem::path& path,
                                                    int cz, int cy, int cx) {
    auto idx = read_shard_index(path);
    auto e = idx.at(cz, cy, cx);
    if (e.length == 0) return {};

    std::vector<uint8_t> buf(e.length);
    FILE* f = fopen(path.c_str(), "rb");
    fseek(f, e.offset, SEEK_SET);
    fread(buf.data(), 1, e.length, f);
    fclose(f);
    return buf;
}

// Read a single chunk from an in-memory shard buffer.
inline std::span<const uint8_t> read_chunk_compressed(std::span<const uint8_t> shard_data,
                                                        int cz, int cy, int cx) {
    auto idx = read_shard_index(shard_data);
    auto e = idx.at(cz, cy, cx);
    if (e.length == 0) return {};
    return shard_data.subspan(e.offset, e.length);
}

// -- Write a complete shard from 512 compressed chunks ----------------------
// chunks[i] may be empty (length 0) for all-zero chunks.

inline void write_shard(const std::filesystem::path& path,
                         std::span<const std::vector<uint8_t>, INDEX_COUNT> chunks) {
    std::filesystem::create_directories(path.parent_path());

    ShardIndex idx{};
    uint32_t offset = 0;
    for (int i = 0; i < INDEX_COUNT; ++i) {
        if (chunks[i].empty()) {
            idx.entries[i] = {0, 0};
        } else {
            idx.entries[i] = {offset, uint32_t(chunks[i].size())};
            offset += uint32_t(chunks[i].size());
        }
    }

    FILE* f = fopen(path.c_str(), "wb");
    for (int i = 0; i < INDEX_COUNT; ++i) {
        if (!chunks[i].empty())
            fwrite(chunks[i].data(), 1, chunks[i].size(), f);
    }
    fwrite(idx.entries.data(), 1, INDEX_BYTES, f);
    fclose(f);
}

// -- Read an entire shard file into memory ----------------------------------

inline std::vector<uint8_t> read_shard_file(const std::filesystem::path& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    auto sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    fread(buf.data(), 1, sz, f);
    fclose(f);
    return buf;
}

} // namespace vc
