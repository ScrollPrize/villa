#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include "utils/Json.hpp"
#include <opencv2/core/mat.hpp>

#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/SampleParams.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/util/RemoteAuth.hpp"

namespace vc::render {
class IChunkedArray;
struct OpenedChunkedZarr;
}
namespace utils { class ZarrArray; }
namespace vcr::zarr { struct OpenOptions; }

struct CompositeParams;

class Volume
{
public:
    // Static flag to skip zarr shape validation against meta.json
    static inline thread_local bool skipShapeCheck = false;

    Volume() = delete;

    explicit Volume(std::filesystem::path path);
    struct RemoteConstructTag {};
    Volume(std::filesystem::path path, RemoteConstructTag);

    ~Volume() noexcept;


    static std::shared_ptr<Volume> New(std::filesystem::path path);

    // Open an existing local volume and apply metadata (uuid, voxelsize, name)
    // over its meta.json, as NewFromUrl does for a remote volume.
    static std::shared_ptr<Volume> New(std::filesystem::path path,
                                       const utils::Json& metadata);

    // Create a Volume backed by a remote zarr store over HTTP.
    // If auth is provided, it is used as-is. Otherwise credentials are read
    // from the ambient AWS configuration only when discoverAwsCredentials is
    // true; false forces an anonymous request. Remote chunks are always
    // persisted beneath the process-wide remote cache root.
    static std::shared_ptr<Volume> NewFromUrl(
        const std::string& url,
        const vc::HttpAuth& auth = {},
        const utils::Json& metadata = {},
        bool discoverAwsCredentials = true);

    [[nodiscard]] bool isRemote() const noexcept { return isRemote_; }
    [[nodiscard]] std::string id() const;
    [[nodiscard]] std::string name() const;
    [[nodiscard]] const utils::Json& metadata() const noexcept { return metadata_; }
    [[nodiscard]] utils::Json rootAttributes() const;
    [[nodiscard]] const std::string& remoteUrl() const noexcept { return remoteUrl_; }
    [[nodiscard]] const std::string& remoteLocator() const noexcept { return remoteLocator_; }
    [[nodiscard]] const vc::HttpAuth& remoteAuth() const noexcept { return remoteAuth_; }
    [[nodiscard]] std::filesystem::path remoteCacheRoot() const noexcept { return remoteCacheRoot_; }
    // Source identity before project/catalog metadata overrides, with the
    // logical base level included. Annotation IDs do not rename disk caches.
    [[nodiscard]] std::filesystem::path remotePersistentCachePath() const;
    [[nodiscard]] std::filesystem::path path() const noexcept { return path_; }

    // Where vc_reader opens this volume's source: the local path or remote URL
    // (s3:// for signed S3), with the credentials and the VC3D cache directory to
    // open it with. Working level L is physical pyramid level L + baseScaleLevel().
    [[nodiscard]] std::string sourceLocation() const;
    [[nodiscard]] vcr::zarr::OpenOptions sourceOpenOptions() const;

    [[nodiscard]] int sliceWidth() const noexcept;
    [[nodiscard]] int sliceHeight() const noexcept;
    [[nodiscard]] int numSlices() const noexcept;
    // Zarr/storage order: [z, y, x] = [slices, height, width].
    [[nodiscard]] std::array<int, 3> shape() const noexcept;
    [[nodiscard]] std::array<int, 3> shape(int level) const;
    [[nodiscard]] std::array<int, 3> levelShape(int level) const;
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const;
    [[nodiscard]] std::array<int, 3> chunkGridShape(int level) const;
    [[nodiscard]] size_t chunkCount(int level) const;
    // Coordinate/UI order: [x, y, z] = [width, height, slices].
    [[nodiscard]] std::array<int, 3> shapeXyz() const noexcept;
    [[nodiscard]] double voxelSize() const;
    [[nodiscard]] vc::render::ChunkDtype dtype() const noexcept { return zarrDtype_; }
    [[nodiscard]] size_t dtypeSize() const noexcept;
    [[nodiscard]] double fillValue() const noexcept { return zarrFillValue_; }

    [[nodiscard]] size_t numScales() const noexcept;
    // The working level: the source pyramid level this volume opened as its
    // level 0 (a remote locator's #vc-base-scale=N), so one voxel is 2^N
    // source voxels. The volume keeps its ID at every working level.
    [[nodiscard]] int baseScaleLevel() const noexcept { return baseScaleLevel_; }
    [[nodiscard]] bool hasExplicitVoxelSizeOverride() const noexcept
    {
        return hasExplicitVoxelSizeOverride_;
    }
    [[nodiscard]] bool hasScaleLevel(int level) const noexcept;
    [[nodiscard]] std::vector<int> presentScaleLevels() const;
    [[nodiscard]] int firstPresentScaleLevel() const;
    [[nodiscard]] int finestPresentScaleLevelAtOrBelow(int level) const;

    enum class MissingScaleLevelPolicy {
        Error,
        AllFill,
        Empty,
        VirtualDownsample,
    };

    // --- Cache management ---

    [[nodiscard]] vc::render::IChunkedArray* chunkedCache();
    [[nodiscard]] std::shared_ptr<vc::render::ChunkCache> sharedChunkCache();

    // Drop decoded/read cache state. Writes call this automatically.
    void invalidateCache();

    // --- Sampling API ---

    // Single-slice blocking sample (uint8)
    void sample(cv::Mat_<uint8_t>& out,
                const cv::Mat_<cv::Vec3f>& coords,
                const vc::SampleParams& params);

    // Single-slice blocking sample (uint16)
    void sample(cv::Mat_<uint16_t>& out,
                const cv::Mat_<cv::Vec3f>& coords,
                const vc::SampleParams& params);

    // Blocking integer reads through the chunked local/remote cache.
    // ZYX order matches zarr storage and numpy-style volume indexing:
    //   offset = [z, y, x], shape = [z, y, x].
    bool readZYX(Array3D<uint8_t>& out,
                 const std::array<int, 3>& offsetZYX,
                 int level = 0,
                 MissingScaleLevelPolicy missingPolicy = MissingScaleLevelPolicy::Error);
    bool readZYX(Array3D<uint16_t>& out,
                 const std::array<int, 3>& offsetZYX,
                 int level = 0,
                 MissingScaleLevelPolicy missingPolicy = MissingScaleLevelPolicy::Error);
    static void readZYX(Array3D<uint8_t>& out,
                        const std::array<int, 3>& offsetZYX,
                        vc::render::IChunkedArray& array,
                        int level = 0);
    static void readZYX(Array3D<uint16_t>& out,
                        const std::array<int, 3>& offsetZYX,
                        vc::render::IChunkedArray& array,
                        int level = 0);

    // UI/coordinate-order convenience:
    //   offset = [x, y, z], shape = [x, y, z].
    // Returned Array3D is still stored/indexed as [z, y, x].
    bool readXYZ(Array3D<uint8_t>& out,
                 const std::array<int, 3>& offsetXYZ,
                 int level = 0,
                 MissingScaleLevelPolicy missingPolicy = MissingScaleLevelPolicy::Error);
    bool readXYZ(Array3D<uint16_t>& out,
                 const std::array<int, 3>& offsetXYZ,
                 int level = 0,
                 MissingScaleLevelPolicy missingPolicy = MissingScaleLevelPolicy::Error);

    // Local zarr chunk reads. Chunk coordinates are [z, y, x]. Data is raw,
    // uncompressed chunk payload with exactly chunk_elems * dtype_size bytes.
    [[nodiscard]] std::optional<std::vector<std::byte>> readChunk(
        int level,
        const std::array<size_t, 3>& chunkZYX) const;
    // Byte size of one decompressed chunk at the given level.
    [[nodiscard]] size_t chunkByteSize(int level) const;
    [[nodiscard]] std::vector<std::byte> readChunkOrFill(
        int level,
        const std::array<size_t, 3>& chunkZYX) const;
    [[nodiscard]] bool chunkExists(
        int level,
        const std::array<size_t, 3>& chunkZYX) const;
protected:
    std::filesystem::path path_;
    utils::Json metadata_;
    bool metadataAutoGenerated_{false};

    int _width{0};
    int _height{0};
    int _slices{0};

    std::vector<std::array<int, 3>> zarrLevelShapes_;
    std::vector<std::array<int, 3>> zarrLevelChunkShapes_;
    std::vector<std::array<int, 3>> zarrLevelStorageChunkShapes_;
    vc::render::ChunkDtype zarrDtype_ = vc::render::ChunkDtype::UInt8;
    double zarrFillValue_ = 0.0;
    void zarrOpen();

    // Cache ownership
    mutable std::shared_ptr<vc::render::ChunkCache> chunkedCache_;
    mutable std::mutex cacheMutex_;

    // Per-level read-side ZarrArray cache. Avoids reparsing .zarray/zarr.json
    // and rebuilding the codec registry on every chunk read. ZarrArray is
    // read-only after open and serialises its own shard I/O internally, so a
    // shared instance is safe to use concurrently from OMP workers.
    mutable std::vector<std::shared_ptr<utils::ZarrArray>> readArrayCache_;
    mutable std::mutex readArrayCacheMutex_;
    std::shared_ptr<utils::ZarrArray> cachedZarrArrayForRead(int level) const;
    std::shared_ptr<vc::render::ChunkCache> createChunkCacheConfigured(
        vc::render::ChunkCache::Options options) const;
    [[nodiscard]] std::string chunkCacheSourceIdentity() const;

    void loadMetadata();

    // Remote volume state
    bool isRemote_ = false;
    std::string remoteUrl_;
    std::string remoteLocator_;
    std::string remoteCacheId_;
    std::string remoteSourceCacheId_;  // the cache of the physical (not rebased) pyramid
    int baseScaleLevel_ = 0;
    bool hasExplicitVoxelSizeOverride_ = false;
    vc::HttpAuth remoteAuth_;
    std::filesystem::path remoteCacheRoot_;
    size_t remoteNumScales_ = 0;
};
