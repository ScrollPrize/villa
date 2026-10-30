#pragma once
// Opening Zarr v2/v3 arrays and groups, local or remote (http(s)://, s3://,
// sftp://), with the disk caches other tools already wrote: vc_reader/vesuvius
// decoded caches and VC3D mirror and Delta3D caches. The same implementation
// backs the Python reader.

#include "vc_reader/native/array.hpp"
#include "vc_reader/native/disk_cache.hpp"
#include "vc_reader/native/element_type.hpp"
#include "vc_reader/native/http.hpp"
#include "vc_reader/native/memory_cache.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vcr::zarr {
using Json = nlohmann::json;

// http(s)://, s3://, s3+REGION:// and sftp:// / ssh:// locations.
bool isRemote(const std::string& location);
// The https:// URL of an s3:// location (public buckets only), otherwise unchanged.
std::string httpUrl(const std::string& location);

// ---- metadata ------------------------------------------------------------------------

struct ArrayMeta {
    Index shape, chunks;
    std::string dtype;               // numpy typestr, e.g. "|u1", "<u2", "<f4"
    ElementType type;                // how stored elements read as numbers
    std::vector<std::uint8_t> fill;  // one element, in stored byte order
    std::string separator = ".";
    Json compressor;                 // v2 compressor object, or null
    Json filters;                    // v2 filters, or null
    std::string order = "C";
    struct V3 {
        std::string prefix;          // chunk key prefix ("c/" or "c.")
        Index shards;                // chunks per shard along each axis; empty when unsharded
        bool indexEnd = true, indexCrc = false;
        Json metadata;               // the zarr.json as stored
    };
    std::optional<V3> v3;
    Json raw;                        // the metadata as a v2 .zarray

    std::size_t itemsize() const { return type.itemsize; }
    // The reader's codec id ("" uncompressed); throws when the chunks cannot be decoded.
    std::string codec() const;
    std::size_t chunkBytes() const;
    // .zarray of the same array stored uncompressed (the decoded-cache layout).
    Json decodedZarray() const;
    // .zarray of the decoded chunks stored with lossless Delta3D (as VC3D writes them).
    Json delta3dZarray() const;
};

// Parses a .zarray or v3 zarr.json. Throws for unsupported layouts and dtypes;
// whether the chunks can be decoded is checked by codec().
ArrayMeta parseArrayMeta(const Json& metadata);

// ---- access --------------------------------------------------------------------------

// How requests under one root are made: anonymous, HTTP basic or signed S3.
struct Access {
    enum class Kind { Anonymous, Basic, S3 };
    Kind kind = Kind::Anonymous;
    Auth auth;             // the credentials handed to the HTTP client
    std::string endpoint;  // S3-compatible server (path-style URLs); empty for AWS

    // The URL to GET for a remote location.
    std::string url(const std::string& location) const;
    // Canonical cache identity of a location, independent of credentials.
    std::string cacheUrl(const std::string& location) const;
};

enum class AuthMode {
    Auto,       // s3:// buckets outside the public channels use AWS credentials when there are any
    Anonymous,
    Aws,        // AWS credentials from the environment and ~/.aws; an error when there are none
};
Access resolveAccess(const std::string& location, AuthMode mode = AuthMode::Auto, bool offline = false);
// Fixed S3 keys (auth.kind "s3"). An empty region asks the bucket, then AWS_REGION.
Access s3Access(const std::string& location, Auth keys, std::string endpoint = {});
Access basicAccess(std::string user, std::string password);
// AWS keys from AWS_ACCESS_KEY_ID/..., then the AWS_PROFILE (or default) profile of
// the shared credentials and config files. kind is "s3"; region is the profile's.
std::optional<Auth> awsCredentials(const std::string& profile = {});
// A bucket's region from S3's x-amz-bucket-region header (cached); nullopt if unknown.
std::optional<std::string> bucketRegion(const std::string& bucket);

// Reads one metadata object (e.g. ".zarray") of a node. nullopt only when it does not exist.
using MetadataReader = std::function<std::optional<std::string>(const std::string& location, const std::string& name)>;
MetadataReader metadataReader(const Access& access, int connections = 4);

// ---- disk caches ---------------------------------------------------------------------

// VC3D's cache directory name for a remote volume: <last path part>-<fnv1a64 hex>.
std::string vc3dVolumeId(std::string url);
// VC3D's remote cache root as VC3D resolves it; nullopt if it does not exist.
std::optional<std::filesystem::path> vc3dCacheRoot();
// A new VC3D cache directory for a volume (opened through OpenOptions::vc3dVolumeDir),
// storing chunks as Delta3D when asked. An existing directory keeps its format.
void createVc3dVolumeDir(const std::filesystem::path& volumeDir, bool delta3d);
// cache_dir/<scheme>/<host or bucket>/<path>; public buckets are keyed by their s3:// form.
std::filesystem::path decodedCachePath(const std::filesystem::path& cacheDir, const std::string& url);
// One SQLite eviction index per directory and process, shared by writers.
std::shared_ptr<DiskCache> diskBudget(const std::string& root, std::optional<std::int64_t> maxBytes);

// VC3D's shared lease on one of its cache directories (flock LOCK_SH).
class Vc3dLease {
public:
    explicit Vc3dLease(const std::filesystem::path& volumeDir);
    ~Vc3dLease();
    Vc3dLease(const Vc3dLease&) = delete;
    Vc3dLease& operator=(const Vc3dLease&) = delete;
    bool acquire();
    void release();

private:
    std::filesystem::path path_;
    int fd_ = -1;
};

// One cache directory holding chunks of one array, in one format.
struct CacheLayer {
    std::string kind;          // "decoded", "vc3d-mirror" or "vc3d-delta3d"
    std::string root;          // directory of the level's chunk files
    ArrayMeta stored;          // how the layer stores chunks
    std::string emptyRoot;     // vc3d-delta3d: where missing-chunk markers live
    bool allowLossy = false;
    std::shared_ptr<Vc3dLease> lease;
    std::string budgetRoot;    // the selected writer's shared cache root when a byte limit applies
    LayerSpec spec() const;
};

struct LayerOptions {
    std::optional<std::string> groupUrl, levelKey;  // OME-Zarr group and level path; finds VC3D caches
    std::optional<std::filesystem::path> vc3dVolumeDir;  // a VC3D cache directory named by its owner
    bool writable = true, allowLossy = false, compress = false;
    std::optional<std::int64_t> maxBytes;
};

// Every existing cache of `url` under `cacheDirs` (in order), and the index of the
// layer to append fetched chunks to (-1: none). "vc3d" in cacheDirs stands for
// VC3D's cache root. Appends go to the first layer that holds the array; if none
// does and writing is allowed, a new decoded cache is created under the first directory.
struct Layers {
    std::vector<CacheLayer> layers;
    int writer = -1;
    std::vector<std::string> busy;  // VC3D caches skipped: another process is replacing them
};
Layers openLayers(const std::vector<std::string>& cacheDirs, const std::string& url, const ArrayMeta& source,
                  const LayerOptions& options);
// The source metadata of `url` as the caches record it, for opening offline.
std::optional<Json> cachedSourceZarray(const std::vector<std::string>& cacheDirs, const std::string& url,
                                       const std::optional<std::string>& groupUrl,
                                       const std::optional<std::string>& levelKey,
                                       const std::optional<std::filesystem::path>& vc3dVolumeDir = {});
// Every cache directory that mirrors the remote node (group or array) at `url`.
std::vector<std::filesystem::path> cachedNodeDirs(const std::vector<std::string>& cacheDirs, const std::string& url);
std::vector<std::filesystem::path> cacheRoots(const std::vector<std::string>& cacheDirs);
void atomicWrite(const std::filesystem::path& path, const std::string& data);

// ---- arrays and groups ---------------------------------------------------------------

struct OpenOptions {
    Access access;
    std::vector<std::string> cacheDirs;
    std::optional<std::filesystem::path> vc3dVolumeDir;
    bool cacheWrites = true;
    std::optional<std::int64_t> cacheMaxBytes;
    std::optional<std::int64_t> cacheMinFreeBytes;  // with cacheMaxBytes: below this much free disk, stop caching (nothing is deleted)
    bool compressCache = false, allowLossyCache = false;
    std::shared_ptr<MemoryCache> memoryCache;
    std::size_t threads = 0;  // 0: 32 remote, min(16, cores) local
    int connections = 16;
    std::string readMode = "mmap";  // "mmap" or "slab"
    bool offline = false;           // remote arrays: metadata and chunks only from the caches
    MetadataReader metadata;        // replaces direct metadata requests (e.g. a caller's metadata cache)
};

// Where a pyramid level's voxels sit in level-0 voxel coordinates, per spatial axis
// in array order (ZYX): level0 = scale * index + offset. OME dataset scale and
// translation define it relative to the first (finest) dataset. A level declaring
// no translation is taken as block-reduced from it (every Open Data CT pyramid is a
// 2x2x2 mean), so its voxels are centred at offset = (scale - 1) / 2. Numbered
// levels without OME metadata are level n of a 2x block-reduced pyramid. Arrays not
// opened as pyramid levels have the identity placement.
struct LevelPlacement {
    std::array<double, 3> scale{1, 1, 1}, offset{0, 0, 0};
    double toLevel0(int axis, double index) const { return scale[axis] * index + offset[axis]; }
    double fromLevel0(int axis, double level0) const { return (level0 - offset[axis]) / scale[axis]; }
    // The map from indices of the level `other` places to this level's indices:
    // index = scale[axis] * other_index + offset[axis] (ZYX axes, as vcr::grid::Level).
    LevelPlacement indexMapFrom(const LevelPlacement& other) const
    {
        LevelPlacement m;
        for (std::size_t a = 0; a < 3; ++a) {
            m.scale[a] = other.scale[a] / scale[a];
            m.offset[a] = (other.offset[a] - offset[a]) / scale[a];
        }
        return m;
    }
};

struct OpenedArray {
    std::string location;
    ArrayMeta meta;
    Json attributes;  // v3 attributes (with _ARRAY_DIMENSIONS); v2 arrays: their .zattrs when read
    std::vector<CacheLayer> layers;
    int writer = -1;
    LevelPlacement placement;  // set by Group::array and openPyramidLevel
    std::unique_ptr<Array> array;
};

struct ArrayHints {
    std::optional<std::string> group, levelKey;  // inferred from a digits-only last path part when absent
    std::optional<Json> metadata;                // already-read .zarray/zarr.json
};
OpenedArray openArray(const std::string& location, const OpenOptions& options, ArrayHints hints = {});

// Select one physical pyramid level without opening the other arrays. A root
// array is level 0. Numeric dataset paths retain their numbers (including gaps);
// other OME dataset paths use their advertised order. Without OME metadata,
// discovers numbered arrays locally or probes remote levels [0, probeLevels).
// Missing levels throw; no implicit renumbering or downsampling is performed.
OpenedArray openPyramidLevel(const std::string& location, int level, const OpenOptions& options,
                             int probeLevels = 32);

// The metadata of every physical pyramid level, numbered as openPyramidLevel numbers
// them, without opening arrays or caches. Sorted by level.
struct PyramidLevelInfo {
    int level = 0;
    std::string path;  // relative to the location; empty for a root array
    ArrayMeta meta;
};
std::vector<PyramidLevelInfo> pyramidLevels(const std::string& location, const OpenOptions& options,
                                            int probeLevels = 32);

class Group {
public:
    static Group open(const std::string& location, OpenOptions options);

    const std::string& location() const { return location_; }
    const Json& attributes() const { return attributes_; }
    int version() const { return version_; }
    const OpenOptions& options() const { return options_; }
    // OME-NGFF multiscales (empty without them).
    Json multiscales() const;
    // Paths from the first multiscales' datasets, otherwise numbered arrays.
    // Local and consolidated groups include gaps, sorted numerically. Remote
    // groups without an index stop at the first absent level unless probeLevels
    // is positive, in which case all levels [0, probeLevels) are checked.
    std::vector<std::string> levels(int probeLevels = 0);
    // The placement of the level at a path (see LevelPlacement); the identity for
    // paths that are not pyramid levels.
    LevelPlacement placement(const std::string& path) const;

    enum class NodeKind { Array, Group };
    // The child at a path, its metadata when it is an array; nullopt when absent.
    std::optional<std::pair<NodeKind, Json>> node(const std::string& key);
    OpenedArray array(const std::string& key);
    Group group(const std::string& key);

private:
    Group(std::string location, OpenOptions options, Json attributes, int version);
    std::optional<Json> consolidated();

    std::string location_;
    OpenOptions options_;
    Json attributes_;
    int version_ = 2;
    std::map<std::string, std::optional<std::pair<NodeKind, Json>>> nodes_;
    std::optional<std::optional<Json>> consolidated_;
};

// ---- writing --------------------------------------------------------------------------

struct CreateOptions {
    Index shape, chunks;
    std::string dtype = "|u1";       // numpy typestr: |u1, <u2, <f4, ...
    std::string compressor = "blosc";  // "" or "none", blosc, zstd, zlib, gzip, lz4, vc-delta3d, volcomp
    // blosc clevel, zstd/zlib/gzip level, lz4 acceleration; volcomp's quantiser step q
    // (0 lossless). Values <= 0 choose the codec default (blosc 3, zstd 3, volcomp q 8).
    int level = -1;
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
    // rewritten. The region must lie inside the array.
    void writeRegion(const Index& start, const Index& shape, const std::uint8_t* data);

private:
    struct Impl;
    explicit Writer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Attributes of a local v2/v3 group or array (an empty object when absent).
Json readAttributes(const std::filesystem::path& node);
void writeAttributes(const std::filesystem::path& node, const Json& attributes);

}  // namespace vcr::zarr
