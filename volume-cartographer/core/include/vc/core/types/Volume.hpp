#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <xtensor/containers/xtensor.hpp>

#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/SampleParams.hpp"
#include "vc/core/cache/HttpMetadataFetcher.hpp"  // HttpAuth
#include "vc/core/util/NetworkFilesystem.hpp"

// Forward declarations
namespace vc { class VcDataset; }

namespace vc::cache {
    class TieredChunkCache;
    class DiskStore;
}

struct CompositeParams;

class Volume
{
public:
    // Bounding box of non-zero data in level-0 voxel coordinates (inclusive).
    // Derived from the coarsest pyramid level.
    struct DataBounds {
        int minX = 0, maxX = 0;  // level-0 voxel coords, inclusive
        int minY = 0, maxY = 0;
        int minZ = 0, maxZ = 0;
        bool valid = false;
    };

    // Static flag to skip zarr shape validation against meta.json
    static inline thread_local bool skipShapeCheck = false;

    Volume() = delete;

    explicit Volume(std::filesystem::path path);

    Volume(std::filesystem::path path, std::string uuid, std::string name);

    ~Volume() noexcept;


    static std::shared_ptr<Volume> New(std::filesystem::path path);

    static std::shared_ptr<Volume> New(std::filesystem::path path, std::string uuid, std::string name);

    // Create a Volume backed by a remote zarr store over HTTP.
    // Downloads metadata (.zarray files) to a local staging dir, then
    // fetches chunk data on demand via HttpChunkSource.
    // If auth is provided, it is used as-is; otherwise credentials are
    // read from environment variables.
    static std::shared_ptr<Volume> NewFromUrl(
        const std::string& url,
        const std::filesystem::path& cacheRoot = {},
        const vc::cache::HttpAuth& auth = {});

    [[nodiscard]] bool isRemote() const noexcept { return isRemote_; }
    [[nodiscard]] std::string id() const;
    [[nodiscard]] std::string name() const;
    void setName(const std::string& n);
    [[nodiscard]] std::filesystem::path path() const noexcept { return path_; }
    void saveMetadata();

    [[nodiscard]] int sliceWidth() const noexcept;
    [[nodiscard]] int sliceHeight() const noexcept;
    [[nodiscard]] int numSlices() const noexcept;
    [[nodiscard]] std::array<int, 3> shape() const noexcept;
    [[nodiscard]] double voxelSize() const;

    [[nodiscard]] vc::VcDataset *zarrDataset(int level = 0) const;
    [[nodiscard]] size_t numScales() const noexcept;

    // Create a TieredChunkCache backed by this volume's zarr data.
    // diskStore: optional shared disk cache (nullptr to disable cold tier).
    [[nodiscard]] std::unique_ptr<vc::cache::TieredChunkCache> createTieredCache(
        std::shared_ptr<vc::cache::DiskStore> diskStore = nullptr) const;

    // --- Cache management ---

    // Lazily create and return the tiered chunk cache for this volume.
    // Thread-safe: creates on first call, returns same cache thereafter.
    [[nodiscard]] vc::cache::TieredChunkCache* tieredCache();

    // Set cache budget (must be called before first tieredCache() access).
    void setCacheBudget(size_t hotBytes, size_t warmBytes = 0);

    // Inject a shared DiskStore for the cold cache tier.
    // Must be called before first tieredCache() access.
    void setDiskStore(std::shared_ptr<vc::cache::DiskStore> store);

    // Set the maximum size for the auto-created disk cache (remote volumes).
    // Must be called before first tieredCache() access.
    void setDiskCacheMaxBytes(size_t bytes);

    // --- Sampling API ---

    // Single-slice blocking sample (uint8)
    void sample(cv::Mat_<uint8_t>& out,
                const cv::Mat_<cv::Vec3f>& coords,
                const vc::SampleParams& params);

    // Single-slice blocking sample (uint16)
    void sample(cv::Mat_<uint16_t>& out,
                const cv::Mat_<cv::Vec3f>& coords,
                const vc::SampleParams& params);

    // Single-slice non-blocking (returns actual level used)
    int sampleBestEffort(cv::Mat_<uint8_t>& out,
                         const cv::Mat_<cv::Vec3f>& coords,
                         const vc::SampleParams& params);

    // Composite non-blocking (returns actual level used)
    int sampleCompositeBestEffort(cv::Mat_<uint8_t>& out,
                                  const cv::Mat_<cv::Vec3f>& coords,
                                  const cv::Mat_<cv::Vec3f>& normals,
                                  const vc::SampleParams& params);

    // Pin the coarsest pyramid level in the hot tier (never evicted).
    // Guarantees sampleBestEffort() always returns data immediately.
    void pinCoarsestLevel(bool blocking = false);

    // Prefetch all chunks overlapping a world-space axis-aligned bounding box.
    // lo/hi are in world (level 0) coordinates, (x, y, z).
    void prefetchWorldBBox(const cv::Vec3f& lo, const cv::Vec3f& hi, int level);

    // Cancel all pending (not in-flight) async prefetch tasks.
    void cancelPendingPrefetch();

    // --- Data bounds ---

    // Return the bounding box of non-zero data (level-0 voxel coords).
    // Computed lazily; retries if previous attempt found no data.
    [[nodiscard]] const DataBounds& dataBounds() const;

    // Scan the coarsest pyramid level to find non-zero data extent.
    void computeDataBounds();

    [[nodiscard]] static bool checkDir(std::filesystem::path path);

protected:
    std::filesystem::path path_;
    nlohmann::json metadata_;
    bool metadataAutoGenerated_{false};

    int _width{0};
    int _height{0};
    int _slices{0};

    std::vector<std::unique_ptr<vc::VcDataset>> zarrDs_;
    void zarrOpen();

    // Cache ownership
    mutable std::unique_ptr<vc::cache::TieredChunkCache> tieredCache_;
    mutable std::once_flag cacheOnce_;
    size_t cacheBudgetHot_ = 8ULL << 30;   // 8 GB default
    size_t cacheBudgetWarm_ = 2ULL << 30;   // 2 GB default
    size_t diskCacheMaxBytes_ = 100ULL << 30; // 100 GB default
    std::shared_ptr<vc::cache::DiskStore> pendingDiskStore_;

    void ensureTieredCache() const;

    // Data bounds (lazy-computed, retryable if invalid)
    mutable DataBounds dataBounds_;
    mutable std::atomic<bool> boundsComputed_{false};
    mutable std::mutex boundsMutex_;

    // Bounding box of coords in chunk index space (helper for allChunksCached/prefetch)
    struct ChunkBBox {
        int minIx, maxIx, minIy, maxIy, minIz, maxIz;
    };
    // World-space bounding box (level-0 coordinates, no scaling applied)
    struct WorldBBox {
        float loX, loY, loZ, hiX, hiY, hiZ;
    };
    ChunkBBox coordsToChunkBBox(const cv::Mat_<cv::Vec3f>& coords, int level) const;
    // Compute world-space bbox once, then derive chunk bbox for any level cheaply
    WorldBBox coordsWorldBBox(const cv::Mat_<cv::Vec3f>& coords) const;
    ChunkBBox worldBBoxToChunkBBox(const WorldBBox& wb, int level) const;
    bool allChunksCached(const cv::Mat_<cv::Vec3f>& coords, int level) const;
    bool allChunksCachedFast(const WorldBBox& wb, int level) const;
    void prefetchChunks(const cv::Mat_<cv::Vec3f>& coords, int level);

    void sampleComposite(cv::Mat_<uint8_t>& out,
                         const cv::Mat_<cv::Vec3f>& coords,
                         const cv::Mat_<cv::Vec3f>& normals,
                         const vc::SampleParams& params);

    // Composite-aware chunk helpers: expand bbox by normal offsets
    ChunkBBox compositeChunkBBox(const cv::Mat_<cv::Vec3f>& coords,
                                 const cv::Mat_<cv::Vec3f>& normals,
                                 int zStart, int zEnd, int level) const;
    bool allCompositeChunksCached(const cv::Mat_<cv::Vec3f>& coords,
                                  const cv::Mat_<cv::Vec3f>& normals,
                                  int zStart, int zEnd, int level) const;
    void prefetchCompositeChunks(const cv::Mat_<cv::Vec3f>& coords,
                                 const cv::Mat_<cv::Vec3f>& normals,
                                 int zStart, int zEnd, int level);

    void loadMetadata();

    // Filesystem mount info (detected once at construction)
    vc::NetworkMountInfo mountInfo_;

    // Remote volume state
    bool isRemote_ = false;
    std::string remoteUrl_;
    std::string remoteDelimiter_ = ".";
    vc::cache::HttpAuth remoteAuth_;
};
