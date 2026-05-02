#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "utils/Json.hpp"
#include <opencv2/core.hpp>


#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/SampleParams.hpp"
#include "vc/core/cache/HttpMetadataFetcher.hpp"  // HttpAuth
#include "vc/core/util/NetworkFilesystem.hpp"
#include "utils/c3d_codec.hpp"

// Forward declarations
namespace vc { class VcDataset; }

namespace vc::cache { class BlockPipeline; class BlockCache; }
namespace utils { class ZarrArray; }

struct CompositeParams;

class Volume
{
public:
    // Bounding box of the physical volume in level-0 voxel coordinates (inclusive).
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

    ~Volume() noexcept;


    static std::shared_ptr<Volume> New(std::filesystem::path path);

    // Create a Volume backed by a remote zarr store over HTTP.
    // Downloads metadata (.zarray files) to a local staging dir, then
    // fetches chunk data on demand via HttpSource.
    // If auth is provided, it is used as-is; otherwise credentials are
    // read from environment variables.
    static std::shared_ptr<Volume> NewFromUrl(
        const std::string& url,
        const std::filesystem::path& cacheRoot = {},
        const vc::cache::HttpAuth& auth = {});

    [[nodiscard]] bool isRemote() const noexcept { return isRemote_; }
    [[nodiscard]] std::string id() const;
    [[nodiscard]] std::string name() const;
    [[nodiscard]] const std::string& remoteUrl() const noexcept { return remoteUrl_; }
    [[nodiscard]] const vc::cache::HttpAuth& remoteAuth() const noexcept { return remoteAuth_; }
    [[nodiscard]] const std::string& remoteDelimiter() const noexcept { return remoteDelimiter_; }
    [[nodiscard]] const vc::cache::ShardConfig& remoteShardConfig() const noexcept { return remoteShardConfig_; }
    [[nodiscard]] std::filesystem::path path() const noexcept { return path_; }

    [[nodiscard]] int sliceWidth() const noexcept;
    [[nodiscard]] int sliceHeight() const noexcept;
    [[nodiscard]] int numSlices() const noexcept;
    [[nodiscard]] std::array<int, 3> shape() const noexcept;
    [[nodiscard]] double voxelSize() const;

    [[nodiscard]] vc::VcDataset *zarrDataset(int level = 0) const;
    [[nodiscard]] size_t numScales() const noexcept;
    [[nodiscard]] int baseScaleLevel() const noexcept { return 0; }

    // Create a BlockPipeline backed by this volume's zarr data.
    [[nodiscard]] std::unique_ptr<vc::cache::BlockPipeline> createTieredCache() const;

    // --- Cache management ---

    // Lazily create and return the tiered chunk cache for this volume.
    // Thread-safe: creates on first call, returns same cache thereafter.
    // After resetTieredCache(), the next call re-creates the pipeline.
    [[nodiscard]] vc::cache::BlockPipeline* tieredCache();

    // Destroy the current pipeline so the next tieredCache() call creates
    // a fresh one.  Call after shutdown() on a volume that may be re-used.
    void resetTieredCache();

    // Set cache budget (must be called before first tieredCache() access).
    void setCacheBudget(size_t hotBytes);
    void setBlockCache(vc::cache::BlockCache* bc);


    // Inject a local zarr array for the cold cache tier.
    // Must be called before first tieredCache() access.
    // Set the number of background IO threads for chunk fetching.
    // Must be called before first tieredCache() access.
    void setIOThreads(int count);

    // Override the c3d encode params (target_ratio) used when re-encoding
    // non-canonical source chunks into the canonical disk cache.
    // depth/height/width are filled per-chunk. Must be called before
    // first tieredCache().
    void setEncodeParams(const utils::C3dCodecParams& params);

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

    // Fused plane sampling: generates coordinates inline during sampling,
    // eliminating the intermediate coords Mat. origin/vx_step/vy_step are
    // in world (level-0) coordinates. Returns actual pyramid level used.
    int samplePlaneBestEffort(cv::Mat_<uint8_t>& out,
                              const cv::Vec3f& origin,
                              const cv::Vec3f& vx_step,
                              const cv::Vec3f& vy_step,
                              int width, int height,
                              const vc::SampleParams& params);

    // Fused plane composite: nearest-neighbor per layer + composite + LUT → ARGB32.
    // No coord matrix. For PlaneSurface composite rendering.
    int samplePlaneCompositeBestEffortARGB32(
        uint32_t* outBuf, int outStride,
        const cv::Vec3f& origin, const cv::Vec3f& vx_step, const cv::Vec3f& vy_step,
        const cv::Vec3f& normal, float zStep, int zStart, int numLayers,
        int width, int height,
        const std::string& compositeMethod,
        const uint32_t lut[256]);

    // --- Data bounds ---
    [[nodiscard]] const DataBounds& dataBounds() const;
    void computeDataBounds();

    [[nodiscard]] static bool checkDir(const std::filesystem::path& path);

protected:
    std::filesystem::path path_;
    utils::Json metadata_;
    bool metadataAutoGenerated_{false};

    int _width{0};
    int _height{0};
    int _slices{0};

    std::vector<std::unique_ptr<vc::VcDataset>> zarrDs_;
    void zarrOpen();

    // Cache ownership
    mutable std::unique_ptr<vc::cache::BlockPipeline> tieredCache_;
    mutable std::mutex cacheMutex_;
    vc::cache::BlockCache* sharedBlockCache_ = nullptr;
    size_t cacheBudgetHot_ = 8ULL << 30;   // 8 GB default
    int ioThreads_ = 0;  // 0 = use default
    utils::C3dCodecParams encodeParams_ = {};

    void ensureTieredCache() const;

    // Data bounds (lazy-computed from volume shape)
    mutable DataBounds dataBounds_;
    mutable std::once_flag boundsOnce_;

    void sampleComposite(cv::Mat_<uint8_t>& out,
                         const cv::Mat_<cv::Vec3f>& coords,
                         const cv::Mat_<cv::Vec3f>& normals,
                         const vc::SampleParams& params);

    void loadMetadata();

    // Filesystem mount info (detected once at construction)
    vc::NetworkMountInfo mountInfo_;

    // Remote volume state
    bool isRemote_ = false;
    std::string remoteUrl_;
    std::string remoteDelimiter_ = ".";
    vc::cache::HttpAuth remoteAuth_;
    vc::cache::ShardConfig remoteShardConfig_;
};
