#pragma once
// Writing Zarr v2/v3 arrays, local or on S3, and local multiscale pyramids.

#include "vc_reader/zarr.hpp"

#include <span>
#include <string_view>
#include <vector>

namespace vcr::zarr {

// ---- stores ---------------------------------------------------------------------------

// The objects below a local directory or an s3:// prefix. Replacing an object is atomic:
// a local file is renamed into place, an S3 object is one PUT. Remote stores read through
// `access`; writing to them needs signed S3 access (Access::Kind::S3). AWS keys from the
// environment are read again once when they stop working. Thread-safe.
class Store {
public:
    explicit Store(const std::filesystem::path& location, const Access& access = {}, int connections = 16);
    Store(Store&&) noexcept;
    Store& operator=(Store&&) noexcept;
    ~Store();

    const std::string& location() const;
    const Access& access() const;
    bool remote() const;

    std::optional<std::string> read(const std::string& name) const;
    // Locally an atomicWrite with the given durability; remotely a PUT.
    void write(const std::string& name, std::string_view data, Durability durability = Durability::Fast) const;
    // A file as an object, without holding it in memory: on S3 in parts above 64 MiB, so past
    // S3's 5 GiB limit on one request.
    void writeFile(const std::string& name, const std::filesystem::path& file) const;
    // An object to a file through a temporary beside it, without holding it in memory; whether it existed.
    bool readFile(const std::string& name, const std::filesystem::path& file) const;
    bool exists(const std::string& name) const;
    // Whether the object was there. On S3 that costs a request `discard` does not make.
    bool remove(const std::string& name) const;
    void discard(const std::string& name) const;
    // Everything below the location; whether anything was there.
    bool removeAll() const;
    // Every object below the location whose relative key starts with `keyPrefix`: keys
    // relative to the location, in key order.
    std::vector<s3::Object> list(const std::string& keyPrefix = {}) const;
    // The immediate subdirectories (prefixes), sorted.
    std::vector<std::string> children() const;
    // One page of the objects and subdirectories directly below the location whose relative
    // keys start with `keyPrefix` (vcr::zarr::listPage); locally, a directory listing paged the same way.
    Listing listPage(const std::string& keyPrefix = {}, const std::string& continuationToken = {}, int maxKeys = 1000) const;
    // Whether the location holds Zarr v2 or v3 node metadata / is an array.
    bool isNode() const;
    bool isArray() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---- codecs on plain bytes -------------------------------------------------------------

// `data` encoded by a numcodecs compressor config ({"id": "blosc", "cname": ...}; null copies),
// as a chunk of `itemsize`-byte elements. `shape` is only read by vc-delta3d and volcomp.
std::vector<std::uint8_t> compressBytes(const Json& compressor, std::span<const std::uint8_t> data, std::size_t itemsize = 1,
                                        const Index& shape = {});
// The reverse, into exactly `out.size()` bytes; throws on any other decoded size.
void decompressBytes(const Json& compressor, std::span<const std::uint8_t> data, std::span<std::uint8_t> out);

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
    bool packbits = false;          // bool only: elements stored as bits, before the compressor
};

// What creates an array like `meta`: its shape, chunks, dtype, codec, separator, fill,
// format, shards and packbits. Packbits is the only filter Writer stores.
CreateOptions createOptionsLike(const ArrayMeta& meta);

// A Zarr v2/v3 array, in a local directory or at an s3:// location, opened for writing.
// Chunks are replaced atomically, so readers never see a partial chunk. Thread-safe for
// distinct chunks in one process, including chunks sharing a shard. Separate processes
// must coordinate updates to the same shard. Partial writes to the same chunk require
// coordination. Each update of an S3 shard downloads and uploads the whole shard.
class Writer {
public:
    // Writes array metadata and a matching parent group marker when absent.
    // Refuses to replace a node of a different Zarr format.
    static Writer create(const std::filesystem::path& path, const CreateOptions& options, const Access& access = {});
    // An existing array.
    static Writer open(const std::filesystem::path& path, const Access& access = {});
    Writer(Writer&&) noexcept;
    Writer& operator=(Writer&&) noexcept;
    ~Writer();

    const ArrayMeta& meta() const;
    const std::filesystem::path& path() const;
    // Reads the stored array (no memory cache: always what is stored).
    Array& reader();

    // Change the fill used for absent chunks, preserving stored chunks and all
    // other metadata. Invalidates references returned by reader(); separately
    // opened arrays must be reopened. Call only with no concurrent writer users.
    void setFillValue(const Json& value);

    // The containing file (a shard for sharded arrays). chunkExists tests the
    // individual chunk's index entry, not merely the presence of this file.
    std::filesystem::path chunkPath(const Index& chunk) const;
    bool chunkExists(const Index& chunk) const;
    // Every stored chunk, in index order, from a listing of the store: the grid is not scanned,
    // except inside a shard, whose index is read per chunk.
    std::vector<Index> storedChunks() const;
    // `data` is one decoded chunk, exactly meta().chunkBytes() bytes in C order.
    void writeChunk(const Index& chunk, const std::uint8_t* data, std::size_t size);
    // Skips a chunk that is entirely the fill value, removing any stored one. A
    // skipped chunk on S3 is still one DELETE.
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

// A group (marker and attributes), kept when it already exists; `attributes` replace
// its attributes when given. Local parent directories are created.
void createGroup(const std::filesystem::path& path, const std::optional<Json>& attributes = std::nullopt, int version = 2,
                 const Access& access = {}, Durability durability = Durability::Fast);

// Removes an array or group with everything in it; whether anything was there.
bool removeNode(const std::filesystem::path& path, const Access& access = {});
// Names of the arrays and groups directly in a location, sorted.
std::vector<std::string> nodeChildren(const std::filesystem::path& path, const Access& access = {});

// Attributes of a v2/v3 group or array (an empty object when absent).
Json readAttributes(const std::filesystem::path& node, const Access& access = {});
void writeAttributes(const std::filesystem::path& node, const Json& attributes, const Access& access = {},
                     Durability durability = Durability::Fast);

// ---- writing pyramids -----------------------------------------------------------------

// How a block of the level below becomes one voxel. Mean: uint8, uint16, float32. Nearest:
// the block's first voxel, any dtype. Mode: the most common nonzero value, the smallest on
// a tie, zero when the block holds nothing else; any integer dtype.
enum class Reduction { Mean, Nearest, Mode };
// "mean", "nearest" or "mode".
Reduction parseReduction(const std::string& name);

// Numbered levels <root>/0, /1, ...: each level reduces the level below by `factors` (ZYX).
struct PyramidOptions {
    int levels = 5;  // levels above level 0
    std::array<int, 3> factors{2, 2, 2};
    std::size_t threads = 0;  // output-chunk workers; 0: all cores
    Reduction reduction = Reduction::Mean;
};
// Creates the arrays of levels 1..levels (metadata only) to go with a level 0 created
// from `level0`: the same dtype, codec and chunks, chunks capped to each level's shape.
void createPyramidLevels(const std::filesystem::path& root, const CreateOptions& level0, const PyramidOptions& options);
// The same for the level 0 already stored at <root>/0. `overwrite` replaces existing levels.
void createPyramidLevels(const std::filesystem::path& root, const PyramidOptions& options, bool overwrite = false);
// Output chunks [begin, end), clamped to the level's chunk grid.
struct ChunkRange {
    Index begin, end;
};
// Writes `level` from level - 1 on disk. Chunks that come out entirely fill are not
// stored. Separate processes may write disjoint ranges. Integer means round to nearest,
// ties to even. `progress(done, total)` may run on any worker.
void writePyramidLevel(const std::filesystem::path& root, int level, const PyramidOptions& options,
                       const std::optional<ChunkRange>& range = std::nullopt,
                       const std::function<void(std::size_t, std::size_t)>& progress = {});

// Creates and writes every level above the level 0 stored at <root>/0.
void writePyramid(const std::filesystem::path& root, const PyramidOptions& options, bool overwrite = false);

struct MultiscalesOptions {
    std::string name = "image";
    int levels = 5;  // levels above level 0
    std::array<int, 3> factors{2, 2, 2};
    std::array<double, 3> scale{1, 1, 1};  // level 0's voxel size per axis (ZYX); level factors without a unit
    std::string unit;                      // axis unit of a physical scale; empty declares none
    Reduction reduction = Reduction::Mean;
    std::array<std::string, 3> axes{"z", "y", "x"};
};
// One OME-NGFF 0.4 "multiscales" entry for such a pyramid, with each level's scale and
// the translation that centres its voxels on the blocks they reduce. A Nearest voxel is
// its block's first voxel, so it has no translation.
Json omeMultiscales(const MultiscalesOptions& options);

}  // namespace vcr::zarr
