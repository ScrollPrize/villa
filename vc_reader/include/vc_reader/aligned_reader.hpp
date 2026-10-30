#pragma once
// Aligned, bounded CPU/CUDA reads in a caller-selected reference array grid, flattened
// image/label pairs, and surfaces rendered or projected into either.
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vc_reader/browsing.hpp"
#include "vc_reader/source_reader.hpp"

namespace vcr::sources {

// An array on the host or a CUDA device (without CUDA, always the host).
struct SampledArray {
    std::vector<std::int64_t> shape;
    std::string dtype;  // typestr
    std::shared_ptr<std::uint8_t[]> host;
    std::shared_ptr<void> device;  // vcr::device::Buffer
    int deviceId = -1;
    [[nodiscard]] bool onDevice() const { return device != nullptr; }
    [[nodiscard]] std::int64_t count() const;
    // The same elements in another shape; never a copy.
    [[nodiscard]] SampledArray reshaped(std::vector<std::int64_t> shape) const;
};

struct TargetResult {
    SampledArray data, valid;  // CZYX, one channel for valid
    std::vector<std::pair<std::string, SampledArray>> masks;
    [[nodiscard]] const SampledArray& mask(const std::string& name) const;
};
struct SampleResult {
    std::vector<std::pair<std::string, TargetResult>> targets;
    nlohmann::json metadata;  // null when there is none
    [[nodiscard]] const TargetResult& operator[](const std::string& name) const;
};

// What an opened source's grid says: ResolutionCandidate::verifiedAgainst checks a candidate with it.
[[nodiscard]] catalog::OpenedGrid openedGrid(SourceReader& reader);

struct AlignedOptions {
    std::string backend = "cpu";  // or "cuda"
    std::int64_t device = 0;
    std::size_t samplingCacheBytes = std::size_t(64) << 20;  // retained CPU or device tiles
    std::size_t samplingWorkBytes = std::size_t(64) << 20;   // tiles held by one request
    std::size_t planCacheBytes = std::size_t(8) << 20;       // reusable plans
    std::size_t surfaceOutputBytes = std::size_t(512) << 20; // dense surface results
    SourceReader::Settings reader;                           // how sources open
};

struct AlignedStats {
    std::size_t tileCacheBytes = 0, tileCacheHits = 0, planCacheBytes = 0, planCacheHits = 0;
    std::size_t tileReads = 0, uploadedBytes = 0, peakTileWorkBytes = 0;
    std::vector<std::pair<std::string, SourceReader::Stats>> sources, masks;
    bool hasSurface = false;
    std::size_t geometryReadBytes = 0, candidateUploadBytes = 0, largestGeometryCells = 0;
    double preparationSeconds = 0, candidateIndexSeconds = 0, rasterCallSeconds = 0, outputAllocationSeconds = 0;
};

class AlignedEngine {
public:
    struct Impl;
    virtual ~AlignedEngine();
    AlignedEngine(const AlignedEngine&) = delete;
    AlignedEngine& operator=(const AlignedEngine&) = delete;
    [[nodiscard]] virtual AlignedStats stats();
    // Channels of a named target.
    [[nodiscard]] std::int64_t targetChannels(const std::string& name) const;
    [[nodiscard]] std::vector<std::string> targetNames() const;
    [[nodiscard]] const std::string& backend() const;
    // What every read's result carries; null when nothing.
    [[nodiscard]] const nlohmann::json& metadata() const;
    [[nodiscard]] std::int64_t device() const;
    void close();
    [[nodiscard]] bool closed() const;
protected:
    explicit AlignedEngine(AlignedOptions options);
    std::unique_ptr<Impl> impl_;
};

// One iterable projection of a surface: the crop a label patch selects, and bounded reads of it.
class SurfaceProjection;

// Reads in the shared ZYX grid of a reference volume. Values are unbatched CZYX. Reads of a
// CUDA backend leave their arrays on the device.
class SampleReader : public AlignedEngine {
public:
    // `reference`: a registered 3-D image. `catalog` relates volumes when sources do not.
    SampleReader(Source reference, std::vector<std::pair<std::string, Source>> targets, std::optional<catalog::Catalog> catalog, AlignedOptions options);
    // An explicit catalog must be the one the level belongs to.
    SampleReader(const catalog::VolumeLevel& reference, std::vector<std::pair<std::string, Source>> targets, std::optional<catalog::Catalog> catalog, AlignedOptions options);
    ~SampleReader() override;
    [[nodiscard]] std::array<std::int64_t, 3> shape() const;
    [[nodiscard]] const VolumeTransform& referenceMap() const;
    [[nodiscard]] SampleResult read(const std::array<std::int64_t, 3>& start, const std::array<std::int64_t, 3>& size);
    [[nodiscard]] std::vector<SampleResult> readMany(const std::vector<std::pair<std::array<std::int64_t, 3>, std::array<std::int64_t, 3>>>& regions);
    // One target only, with no companions: data and validity.
    [[nodiscard]] std::pair<SampledArray, SampledArray> sampleTarget(const std::string& name, const std::array<std::int64_t, 3>& start, const std::array<std::int64_t, 3>& size);
    [[nodiscard]] bool hasTarget(const std::string& name) const;

    // Physical-offset layers over a YX patch in full-resolution surface UV.
    [[nodiscard]] SampleResult renderSurface(const Surface& surface, const std::array<std::int64_t, 2>& start, const std::array<std::int64_t, 2>& size, const std::vector<double>& offsetsUm = {0});
    // Uses a label YX patch to bound a crop, then projects its sheet across it. Geometric
    // coverage includes sheet outside the selecting patch; label validity additionally
    // requires annotation support at the surface point.
    [[nodiscard]] SampleResult projectSurface(const Surface& surface, const std::array<std::int64_t, 2>& start, const std::array<std::int64_t, 2>& size, const std::string& labelTarget, double thicknessUm = 0, double paddingUm = 0);
    // The same, as bounded tiles: read each of projection.regions(tileSize).
    [[nodiscard]] std::shared_ptr<SurfaceProjection> projection(const Surface& surface, const std::array<std::int64_t, 2>& start, const std::array<std::int64_t, 2>& size, const std::string& labelTarget, double thicknessUm = 0, double paddingUm = 0);
    // A surface one of the targets registers to, by id.
    [[nodiscard]] Surface surfaceNamed(const std::string& id) const;
};

class SurfaceProjection {
public:
    struct Impl;
    explicit SurfaceProjection(std::unique_ptr<Impl> impl);
    ~SurfaceProjection();
    [[nodiscard]] std::array<std::int64_t, 3> origin() const;
    [[nodiscard]] std::array<std::int64_t, 3> shape() const;
    [[nodiscard]] std::vector<std::pair<std::array<std::int64_t, 3>, std::array<std::int64_t, 3>>> regions(const std::array<std::int64_t, 3>& tile) const;
    [[nodiscard]] SampleResult readCrop(const std::array<std::int64_t, 3>& origin, const std::array<std::int64_t, 3>& shape);
private:
    std::unique_ptr<Impl> impl_;
};

// A flattened image/label pair: shared indexing is YX on the selected image, with independent
// layer depth. Label and mask grids are selected independently: a matching stored grid when
// available, otherwise exact nearest-neighbor sampling of categorical values.
class PairedReader : public AlignedEngine {
public:
    struct Layers {
        std::int64_t start = 0;
        std::optional<std::int64_t> count;  // default: to the end
        bool drop = false;                  // a single selected layer loses its axis
    };
    // `layers`: the layers read, given the selected image's depth.
    using LayerChoice = std::function<Layers(std::int64_t depth)>;
    PairedReader(const Source& label, const Source& image, const catalog::Pairing& pairing, std::int64_t level, const LayerChoice& layers, AlignedOptions options);
    ~PairedReader() override;
    [[nodiscard]] std::array<std::int64_t, 2> shape() const;
    [[nodiscard]] std::int64_t layerDepth() const;   // stored layers of the selected image
    [[nodiscard]] std::int64_t layerCount() const;
    [[nodiscard]] std::int64_t layerStart() const;
    [[nodiscard]] bool dropLayer() const;
    [[nodiscard]] SampleResult read(const std::array<std::int64_t, 2>& start, const std::array<std::int64_t, 2>& size);
    [[nodiscard]] std::vector<SampleResult> readMany(const std::vector<std::pair<std::array<std::int64_t, 2>, std::array<std::int64_t, 2>>>& regions);
    // One target at (layer, y, x) start and size.
    [[nodiscard]] std::pair<SampledArray, SampledArray> sampleTarget(const std::string& name, const std::array<std::int64_t, 3>& start, const std::array<std::int64_t, 3>& size);
    [[nodiscard]] bool hasTarget(const std::string& name) const;
    [[nodiscard]] AlignedStats stats() override;
private:
    std::array<std::int64_t, 2> shape_{};
    std::int64_t layerStart_ = 0, layerCount_ = 0, layerDepth_ = 0;
    bool dropLayer_ = false;
};

// Reusable geometry-only UV interpolation; no volume is sampled.
class SurfaceGeometry : public AlignedEngine {
public:
    SurfaceGeometry(const Surface& surface, AlignedOptions options);
    ~SurfaceGeometry() override;
    // Rows and columns of the stored coordinate grid.
    [[nodiscard]] std::array<std::int64_t, 2> shape() const;
    // XYZ float32 points (layers, y, x, 3) in the surface's coordinate-value frame and
    // their validity (layers, y, x); offsets are physical micrometers.
    [[nodiscard]] std::pair<Block, Block> readPoints(const std::array<std::int64_t, 2>& start, const std::array<std::int64_t, 2>& size, const std::vector<double>& offsetsUm = {0});
};

} // namespace vcr::sources
