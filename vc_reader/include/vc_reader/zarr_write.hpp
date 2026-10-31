#pragma once
// Writing local Zarr v2/v3 arrays and multiscale pyramids.

#include "vc_reader/zarr.hpp"

namespace vcr::zarr {

// ---- writing --------------------------------------------------------------------------

struct CreateOptions {
    Index shape, chunks;
    std::string dtype = "|u1";       // numpy typestr: |u1, <u2, <f4, ...
    std::string compressor = "blosc";  // "" or "none", blosc, zstd, zlib, gzip, lz4, vc-delta3d, volcomp
    // blosc clevel, zstd/zlib/gzip level, lz4 acceleration; volcomp's quantiser step q
    // (0 lossless). Values <= 0 choose the codec default (blosc 3, zstd 3, volcomp q 8).
    int level = -1;
    std::string cname = "lz4";  // blosc inner codec: lz4, zstd, zlib, ...
    int shuffle = 1;            // blosc: 0 none, 1 byte, 2 bit
    bool overwrite = false;     // replace whatever is at the path, stored chunks included
    std::string separator = ".";
    Json fillValue = 0;
    int version = 2;               // Zarr format 2 or 3
    Index shardShape;               // v3 outer shape in elements; empty: unsharded
    bool shardIndexEnd = true, shardIndexCrc = true;
};

// A local Zarr v2/v3 array opened for writing. Chunk files are replaced atomically,
// so readers never see a partial chunk. Thread-safe for distinct chunks in one
// process, including chunks sharing a shard. Separate processes must coordinate
// updates to the same shard. Partial writes to the same chunk require coordination.
class Writer {
public:
    // Writes array metadata and a matching parent group marker when absent.
    // Refuses to replace a node of a different Zarr format.
    static Writer create(const std::filesystem::path& path, const CreateOptions& options);
    // An existing local array.
    static Writer open(const std::filesystem::path& path);
    Writer(Writer&&) noexcept;
    Writer& operator=(Writer&&) noexcept;
    ~Writer();

    const ArrayMeta& meta() const;
    const std::filesystem::path& path() const;
    // Reads the stored array (no memory cache: always what is on disk).
    Array& reader();

    // Change the fill used for absent chunks, preserving stored chunks and all
    // other metadata. Invalidates references returned by reader(); separately
    // opened arrays must be reopened. Call only with no concurrent writer users.
    void setFillValue(const Json& value);

    // The containing file (a shard for sharded arrays). chunkExists tests the
    // individual chunk's index entry, not merely the presence of this file.
    std::filesystem::path chunkPath(const Index& chunk) const;
    bool chunkExists(const Index& chunk) const;
    // `data` is one decoded chunk, exactly meta().chunkBytes() bytes in C order.
    void writeChunk(const Index& chunk, const std::uint8_t* data, std::size_t size);
    // Skips a chunk that is entirely the fill value, removing any stored one.
    // Returns whether the chunk was written.
    bool writeChunkSkipEmpty(const Index& chunk, const std::uint8_t* data, std::size_t size);
    bool removeChunk(const Index& chunk);
    // Writes a C-order region; chunks it covers partially are read, patched and
    // rewritten. The region must lie inside the array. `skipEmpty`: a chunk that comes
    // out entirely fill is not stored, and any stored one is removed.
    void writeRegion(const Index& start, const Index& shape, const std::uint8_t* data, bool skipEmpty = false);

private:
    struct Impl;
    explicit Writer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// A local group (marker and attributes), kept when it already exists; `attributes` replace
// its attributes when given. Parent directories are created.
void createGroup(const std::filesystem::path& path, const std::optional<Json>& attributes = std::nullopt, int version = 2);

// Removes a local array or group with everything in it; whether anything was there.
bool removeNode(const std::filesystem::path& path);
// Names of the arrays and groups directly in a local directory, sorted.
std::vector<std::string> nodeChildren(const std::filesystem::path& path);

// Attributes of a local v2/v3 group or array (an empty object when absent).
Json readAttributes(const std::filesystem::path& node);
void writeAttributes(const std::filesystem::path& node, const Json& attributes);

// ---- writing pyramids -----------------------------------------------------------------

// Numbered levels <root>/0, /1, ...: each level is the block mean of the level below,
// reduced by `factors` (ZYX) per level.
struct PyramidOptions {
    int levels = 5;  // levels above level 0
    std::array<int, 3> factors{2, 2, 2};
    std::size_t threads = 0;  // output-chunk workers; 0: all cores
};
// Creates the arrays of levels 1..levels (metadata only) to go with a level 0 created
// from `level0`: the same dtype, codec and chunks, chunks capped to each level's shape.
void createPyramidLevels(const std::filesystem::path& root, const CreateOptions& level0, const PyramidOptions& options);
// Output chunks [begin, end), clamped to the level's chunk grid.
struct ChunkRange {
    Index begin, end;
};
// Writes `level` from level - 1 on disk (uint8, uint16, float32). Chunks that come out
// entirely fill are not stored. Separate processes may write disjoint ranges. Integer
// means round to nearest, ties to even. `progress(done, total)` may run on any worker.
void writePyramidLevel(const std::filesystem::path& root, int level, const PyramidOptions& options,
                       const std::optional<ChunkRange>& range = std::nullopt,
                       const std::function<void(std::size_t, std::size_t)>& progress = {});

struct MultiscalesOptions {
    std::string name = "image";
    int levels = 5;  // levels above level 0
    std::array<int, 3> factors{2, 2, 2};
    std::array<double, 3> scale{1, 1, 1};  // level 0's voxel size per axis (ZYX); level factors without a unit
    std::string unit;                      // axis unit of a physical scale; empty declares none
};
// One OME-NGFF 0.4 "multiscales" entry for such a pyramid, with each level's scale and
// the translation that centres its voxels on the blocks they reduce.
Json omeMultiscales(const MultiscalesOptions& options);

}  // namespace vcr::zarr
