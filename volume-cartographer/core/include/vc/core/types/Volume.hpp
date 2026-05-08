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
#include <opencv2/core.hpp>

#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/SampleParams.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/util/RemoteAuth.hpp"

namespace vc::render { class IChunkedArray; }

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

    struct PyramidPolicy {
        enum class Reduction {
            Mean,
            Max,
            BinaryOr,
        };

        // Per-level downsample from one level to the next in zarr/storage order
        // [z, y, x]. The default creates a conventional 2x pyramid in all dims.
        std::array<double, 3> downsampleZYX{2.0, 2.0, 2.0};
        Reduction reduction = Reduction::Mean;
    };

    struct ZarrCreateOptions {
        // Base level shape in zarr/storage order: [z, y, x].
        std::array<size_t, 3> shapeZYX{};
        std::array<size_t, 3> chunkShapeZYX{64, 64, 64};
        vc::render::ChunkDtype dtype = vc::render::ChunkDtype::UInt8;
        size_t numLevels = 1;
        PyramidPolicy pyramid;
        double fillValue = 0.0;
        double voxelSize = 1.0;
        std::string voxelUnit;
        std::string uuid;
        std::string name;
        std::string compressor = "blosc";
        int compressionLevel = 3;
        bool overwriteExisting = false;
    };

    // Open an existing local volume, or create a local OME-Zarr pyramid when
    // the path does not already contain a volume/zarr. Existing volumes are
    // opened as-is unless overwriteExisting is set.
    static std::shared_ptr<Volume> New(std::filesystem::path path,
                                       const ZarrCreateOptions& options);

    // Create a Volume backed by a remote zarr store over HTTP.
    // If auth is provided, it is used as-is; otherwise credentials are read
    // from environment variables. cacheRoot is currently ignored.
    static std::shared_ptr<Volume> NewFromUrl(
        const std::string& url,
        const std::filesystem::path& cacheRoot = {},
        const vc::HttpAuth& auth = {});

    [[nodiscard]] bool isRemote() const noexcept { return isRemote_; }
    [[nodiscard]] std::string id() const;
    [[nodiscard]] std::string name() const;
    [[nodiscard]] const utils::Json& metadata() const noexcept { return metadata_; }
    [[nodiscard]] utils::Json rootAttributes() const;
    void writeRootAttributes(const utils::Json& attrs);
    void updateRootAttributes(const utils::Json& attrs);
    void writeMetadata(const utils::Json& metadata);
    void updateMetadata(const utils::Json& metadata);
    [[nodiscard]] const std::string& remoteUrl() const noexcept { return remoteUrl_; }
    [[nodiscard]] const vc::HttpAuth& remoteAuth() const noexcept { return remoteAuth_; }
    [[nodiscard]] std::filesystem::path path() const noexcept { return path_; }

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
    [[nodiscard]] int baseScaleLevel() const noexcept { return 0; }
    [[nodiscard]] bool hasScaleLevel(int level) const noexcept;
    [[nodiscard]] std::vector<int> presentScaleLevels() const;
    [[nodiscard]] int firstPresentScaleLevel() const;
    [[nodiscard]] int finestPresentScaleLevelAtOrBelow(int level) const;
    [[nodiscard]] PyramidPolicy::Reduction pyramidReduction() const noexcept
    {
        return pyramidReduction_;
    }

    enum class MissingScaleLevelPolicy {
        Error,
        AllFill,
        Empty,
        VirtualDownsample,
    };

    // --- Cache management ---

    [[nodiscard]] vc::render::IChunkedArray* chunkedCache();
    [[nodiscard]] std::shared_ptr<vc::render::ChunkCache> createChunkCache(
        vc::render::ChunkCache::Options options) const;

    // Set cache budget for the chunked sampling cache.
    void setCacheBudget(size_t hotBytes);

    // Set the number of background IO threads for chunk fetching.
    void setIOThreads(int count);

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

    // Local zarr region writes. Input arrays are indexed/stored as [z, y, x].
    // Writes update coarser pyramid levels by mean downsampling using each
    // adjacent level's shape ratio.
    void writeZYX(const Array3D<uint8_t>& data,
                  const std::array<int, 3>& offsetZYX,
                  int level = 0);
    void writeZYX(const Array3D<uint16_t>& data,
                  const std::array<int, 3>& offsetZYX,
                  int level = 0);
    void writeXYZ(const Array3D<uint8_t>& data,
                  const std::array<int, 3>& offsetXYZ,
                  int level = 0);
    void writeXYZ(const Array3D<uint16_t>& data,
                  const std::array<int, 3>& offsetXYZ,
                  int level = 0);

    // Local zarr chunk I/O. Chunk coordinates are [z, y, x]. Data is raw,
    // uncompressed chunk payload with exactly chunk_elems * dtype_size bytes.
    [[nodiscard]] std::optional<std::vector<std::byte>> readChunk(
        int level,
        const std::array<size_t, 3>& chunkZYX) const;
    // Decompress directly into a caller-owned scratch buffer (must be
    // sized to chunkByteSize(level)). Returns false when the chunk is
    // missing on disk; otherwise writes exactly chunkByteSize(level)
    // bytes into `output` and returns true. Avoids the per-call heap
    // allocation that readChunk() performs.
    [[nodiscard]] bool readChunkInto(
        int level,
        const std::array<size_t, 3>& chunkZYX,
        std::span<std::byte> output) const;
    // Byte size of one decompressed chunk at the given level (matches
    // what readChunk returns and what readChunkInto requires).
    [[nodiscard]] size_t chunkByteSize(int level) const;
    [[nodiscard]] std::vector<std::byte> readChunkOrFill(
        int level,
        const std::array<size_t, 3>& chunkZYX) const;
    [[nodiscard]] bool chunkExists(
        int level,
        const std::array<size_t, 3>& chunkZYX) const;
    struct ChunkWriteOptions {
        // When false, chunks containing only the zarr fill value are removed
        // instead of written, matching zarr's write_empty_chunks=false behavior.
        bool writeEmptyChunks = true;
    };
    void writeChunk(int level,
                    const std::array<size_t, 3>& chunkZYX,
                    std::span<const std::byte> data);
    void writeChunk(int level,
                    const std::array<size_t, 3>& chunkZYX,
                    std::span<const std::byte> data,
                    ChunkWriteOptions options);
    bool removeChunk(int level,
                     const std::array<size_t, 3>& chunkZYX);

    [[nodiscard]] static bool checkDir(const std::filesystem::path& path);

protected:
    std::filesystem::path path_;
    utils::Json metadata_;
    bool metadataAutoGenerated_{false};

    int _width{0};
    int _height{0};
    int _slices{0};

    std::vector<std::array<int, 3>> zarrLevelShapes_;
    std::vector<std::array<int, 3>> zarrLevelChunkShapes_;
    vc::render::ChunkDtype zarrDtype_ = vc::render::ChunkDtype::UInt8;
    double zarrFillValue_ = 0.0;
    PyramidPolicy::Reduction pyramidReduction_ = PyramidPolicy::Reduction::Mean;
    void zarrOpen();

    // Cache ownership
    mutable std::shared_ptr<vc::render::ChunkCache> chunkedCache_;
    mutable std::mutex cacheMutex_;
    size_t cacheBudgetHot_ = 8ULL << 30;   // 8 GB default
    int ioThreads_ = 0;  // 0 = use default

    void loadMetadata();

    // Remote volume state
    bool isRemote_ = false;
    std::string remoteUrl_;
    vc::HttpAuth remoteAuth_;
    size_t remoteNumScales_ = 0;
};
