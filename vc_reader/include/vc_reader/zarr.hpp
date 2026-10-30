#pragma once
// Opening Zarr v2/v3 arrays and groups, local or remote (http(s)://, s3://,
// sftp://), with the disk caches other tools already wrote: vc_reader/vesuvius
// decoded caches and VC3D mirror and Delta3D caches. The same implementation
// backs the Python reader. Reading an opened array needs vc_reader/array.hpp, writing
// vc_reader/zarr_write.hpp and a volume's metadata vc_reader/zarr_metadata.hpp.

#include "vc_reader/native/memory_cache.hpp"
#include "vc_reader/native/reader_types.hpp"
#include "vc_reader/zarr_array_meta.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vcr {
class Array;
class DiskCache;
struct LayerSpec;
}  // namespace vcr

namespace vcr::zarr {

// There is no Zarr array or group at a location.
struct NotFound : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// http(s)://, s3://, s3+REGION:// and sftp:// / ssh:// locations.
bool isRemote(const std::string& location);
// Whether a local directory holds any Zarr v2 or v3 node metadata.
bool isLocalZarrNode(const std::filesystem::path& directory);
// Whether a local directory is a Zarr array (v2 .zarray, or a v3 array's zarr.json).
bool isLocalZarrArray(const std::filesystem::path& directory);
// The https:// URL of an s3:// location (public buckets only), otherwise unchanged.
std::string httpUrl(const std::string& location);

// ---- access --------------------------------------------------------------------------

// How requests under one root are made: anonymous, HTTP basic or signed S3.
struct Access {
    enum class Kind { Anonymous, Basic, S3 };
    Kind kind = Kind::Anonymous;
    Auth auth;             // the credentials handed to the HTTP client
    std::string endpoint;  // S3-compatible server (path-style URLs); empty for AWS
    // The keys came from the environment: a copy in another process reads them again
    // (see reloadEnvironmentKeys) rather than carrying them.
    bool fromEnvironment = false;

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
// Offline needs neither signing nor a bucket region: anonymous.
Access s3Access(const std::string& location, Auth keys, std::string endpoint = {}, bool offline = false);
Access basicAccess(std::string user, std::string password);
// Reads the keys of an Access resolved from the environment again, keeping its region.
// HttpDenied when the environment no longer has any.
void reloadEnvironmentKeys(Access& access);
// (user, password) from a JSON file {"username": ..., "password": ...}: the fields may
// sit under an "auth" object, and "user" is accepted for "username".
std::pair<std::string, std::string> loadBasicAuth(const std::filesystem::path& path);
// Whether a location is in a public channel of the Open Data bucket.
bool isPublicLocation(const std::string& location);
// The immediate children of an s3:// location or one in a public channel (ListObjectsV2).
// Throws std::invalid_argument for anything else: plain HTTP has no listing.
std::vector<std::string> listChildren(const Access& access, const std::string& location);
// AWS keys from AWS_ACCESS_KEY_ID/..., then the AWS_PROFILE (or default) profile of
// the shared credentials and config files. kind is "s3"; region is the profile's.
std::optional<Auth> awsCredentials(const std::string& profile = {});
// A bucket's region from S3's x-amz-bucket-region header (cached); nullopt if unknown.
std::optional<std::string> bucketRegion(const std::string& bucket);

// Reads one metadata object (e.g. ".zarray") of a node. nullopt only when it does not
// exist; any other failure throws (std::filesystem::filesystem_error for a local one).
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
    std::string store;         // the group holding this level, else the array's own location
    std::string levelKey;      // the array's path in `store`; empty for a root array
    std::vector<std::string> busy;  // VC3D caches skipped: another process is replacing them
    bool attributesRead = false;    // `attributes` holds what the array's .zattrs say

    OpenedArray();
    ~OpenedArray();
    OpenedArray(OpenedArray&&) noexcept;
    OpenedArray& operator=(OpenedArray&&) noexcept;

    // Finishes queued cache writes and releases VC3D's leases. Reads must have ended.
    void close();
};
// The array's attributes, read on first use (and then kept with a writable cache for
// opening offline). CacheMiss offline when no cache holds them.
const Json& arrayAttributes(OpenedArray& array, const OpenOptions& options);

struct ArrayHints {
    std::optional<std::string> group, levelKey;  // inferred from a digits-only last path part when absent
    std::optional<Json> metadata;                // already-read .zarray/zarr.json
    std::optional<Json> attributes;              // already-read array attributes
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

// Where the level at `path` sits (see LevelPlacement), from the attributes of the group
// holding it; the identity for paths that are not pyramid levels. Throws
// std::invalid_argument for OME metadata that cannot place it.
LevelPlacement levelPlacement(const Json& groupAttributes, const std::string& path);

struct StoreMetadata;
// Drops the metadata kept with a remote volume's caches (see Group::open, volumeMetadata), so
// its next open reads it as published.
void forgetKeptMetadata(const std::string& location, const OpenOptions& options);

class Group {
public:
    // A group in a public channel with caches keeps its state() there once its pyramid
    // levels are known; a later open trusts it like a chunk and makes no request.
    static Group open(const std::string& location, OpenOptions options);
    // A group as state() saved it, without any request.
    static Group fromState(const Json& state, OpenOptions options);
    // What fromState needs: the attributes and every lookup made so far. `inherited`:
    // called in a forked process, where another thread of the parent may have held the
    // lock when it forked (no other thread can be using the group here).
    Json state(bool inherited = false) const;

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
    // The placement of the level at a path (see levelPlacement).
    LevelPlacement placement(const std::string& path) const;
    // Names of the arrays and groups directly in this group, sorted: from the directory,
    // consolidated metadata or the caches when offline, else by listing the S3 prefix.
    // Throws std::invalid_argument for plain HTTP, which has no listing.
    std::vector<std::string> keys();
    // The volume's metadata (see volumeMetadata), memoized with the group.
    const StoreMetadata& metadata(bool refresh = false);

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
    std::shared_ptr<const StoreMetadata> metadata_;
    std::shared_ptr<std::recursive_mutex> mutex_ = std::make_shared<std::recursive_mutex>();  // lookups may come from any thread
};

}  // namespace vcr::zarr
