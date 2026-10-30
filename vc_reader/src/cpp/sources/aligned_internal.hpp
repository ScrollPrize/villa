#pragma once
// What the aligned readers and surface operations share.
#include "vc_reader/aligned_reader.hpp"

#include <chrono>
#include <cmath>
#include <list>
#include <mutex>
#include <unordered_map>

#include "../array.hpp"
#include "../aligned/plan.hpp"
#include "../surface/plan.hpp"
#ifdef VC_READER_CUDA
#include "../cuda_buffer.hpp"
#include "../reader_cuda/reader.hpp"
#endif

namespace vcr::sources {

using Triple = std::array<std::int64_t, 3>;

// Least-recently-used entries within a byte limit.
template <class T>
class Bounded {
public:
    explicit Bounded(std::size_t limit = 0) : limit(limit) {}
    std::size_t limit, bytes = 0, hits = 0;
    const T* get(const std::string& key)
    {
        const auto it = index_.find(key);
        if (it == index_.end()) return nullptr;
        order_.splice(order_.end(), order_, it->second);
        ++hits;
        return &it->second->value;
    }
    void put(const std::string& key, T value, std::size_t size)
    {
        if (size > limit) return;
        if (const auto it = index_.find(key); it != index_.end()) {
            bytes -= it->second->size;
            order_.erase(it->second);
            index_.erase(it);
        }
        while (!order_.empty() && bytes + size > limit) {
            bytes -= order_.front().size;
            index_.erase(order_.front().key);
            order_.pop_front();
        }
        order_.push_back({key, std::move(value), size});
        index_[key] = std::prev(order_.end());
        bytes += size;
    }
    void clear()
    {
        order_.clear();
        index_.clear();
        bytes = 0;
    }
private:
    struct Item {
        std::string key;
        T value;
        std::size_t size;
    };
    std::list<Item> order_;
    std::unordered_map<std::string, typename std::list<Item>::iterator> index_;
};

struct FieldSpec {
    std::array<float, 9> matrix{};
    std::array<float, 3> translation{};
    std::array<int, 3> components{};
    bool normalize = false, normal = false;
};

// One aligned array: how a reader's values reach the shared grid.
struct Target {
    Source source;
    std::shared_ptr<SourceReader> reader;
    VolumeTransform mapping;
    Triple shape{1, 1, 1};
    std::int64_t channels = 1;
    std::string dtype, nativeDtype;  // typestrs
    bool plane = false;
    std::string storageKey;
    int order = 1;  // 0: nearest, for categorical values
    bool direct = false;
    std::optional<FieldSpec> field;
    aligned::Field fieldState{};
    nlohmann::json metadata;
    std::vector<std::pair<std::string, Target>> companions;
    std::shared_ptr<void> cuda;  // reader_cuda::Reader of its array, once a render needs it
};

// A tile on the host or device.
struct Tile {
    Block host;
#ifdef VC_READER_CUDA
    device::Storage device;
#endif
};

struct Geometry;

struct AlignedEngine::Impl {
    explicit Impl(AlignedOptions o);
    AlignedOptions options;
    std::string backend;
    int device = 0;
    std::recursive_mutex mutex;
    bool closed = false;
    std::map<std::string, std::shared_ptr<SourceReader>> readers;
    std::vector<std::shared_ptr<SourceReader>> opened;
    std::vector<std::pair<std::string, Target>> targets, surfaceTargets, masks;
    Bounded<std::shared_ptr<void>> tiles, plans;
    std::size_t workBytes, uploads = 0, tileReads = 0, peakWork = 0;
    std::size_t geometryReadBytes = 0, candidateUploadBytes = 0, largestGeometryCells = 0;
    double preparationSeconds = 0, candidateIndexSeconds = 0, rasterCallSeconds = 0, outputAllocationSeconds = 0;
    bool usedSurface = false;
    std::map<std::string, std::shared_ptr<Geometry>> geometries;

    // Reference state of a SampleReader.
    std::optional<catalog::Catalog> catalog;
    Source reference;
    std::shared_ptr<SourceReader> referenceReader;
    VolumeTransform referenceMap;
    std::array<std::string, 2> referenceFrame;
    nlohmann::json metadata;

    std::shared_ptr<SourceReader> open(const Source& source);
    Target samplingTarget(const Source& source, std::shared_ptr<SourceReader> reader, const VolumeTransform& mapping, bool plane = false, bool planarGrid = false,
                          std::optional<FieldSpec> field = std::nullopt, nlohmann::json metadata = nullptr);
    Target* find(const std::string& name);
    VolumeTransform valuePlacement(const ValueFrame& value);
    VolumeTransform between(const Source& source, const std::array<std::string, 2>& frame);
    Target prepare(const Source& source, std::shared_ptr<SourceReader> reader);

    std::pair<std::shared_ptr<aligned::Plan>, Triple> plan(const Target& target, const Triple& start, const Triple& size, const Triple& offset, const Triple& block, bool tiles = true);
    std::optional<std::vector<std::shared_ptr<Tile>>> tileInputs(const Target& target, const aligned::Plan& plan, const Triple& tile);
    void execute(const Target& target, const Triple& start, const Triple& size, const Triple& offset, const Triple& block, SampledArray& output, SampledArray& valid);
    std::vector<std::pair<SampledArray, SampledArray>> sampleMany(const Target& target, const std::vector<std::pair<Triple, Triple>>& regions);
    std::pair<SampledArray, SampledArray> samplePoints(const Target& target, const std::vector<float>& points, const Triple& base);
    SampledArray empty(std::vector<std::int64_t> shape, const std::string& dtype);
    SampledArray fromHost(const Block& block);
    Block canonical(const Block& block, const std::vector<std::string>& axes);
    void checkOpen() const;
    void finishTargetResults();
#ifdef VC_READER_CUDA
    void cudaBlocks(const Target& target, const Triple& start, const Triple& size, SampledArray& output, SampledArray& valid);
#endif
private:
    void pointBlock(const Target& target, const float* points, std::size_t n, const Triple& base, SampledArray& out, SampledArray& valid, std::size_t& cursor);
};

// A surface's coordinate grid: metadata and bounded tiles through shared source readers.
struct GeometryTile {
    std::int64_t rows = 0, cols = 0;
    std::vector<float> xyz;      // rows x cols x 3
    std::vector<std::uint8_t> valid;
    std::vector<float> normals;  // empty, or rows x cols x 3
};
struct Geometry {
    Geometry(AlignedEngine::Impl& owner, const Surface& surface);
    AlignedEngine::Impl& owner;
    Surface surface;
    std::vector<std::shared_ptr<SourceReader>> coordinateReaders;
    std::shared_ptr<SourceReader> mask;
    std::array<std::int64_t, 2> shape{};
    VolumeTransform toUv, fromUv;
    int orientation = 1;
    std::optional<Target> explicitMask, normals;
    std::array<float, 9> normalTransform{};
    std::string key;
    [[nodiscard]] std::shared_ptr<const GeometryTile> tile(const std::array<std::int64_t, 2>& start, const std::array<std::int64_t, 2>& size);
    std::optional<Target> auxiliary(const std::optional<Shared<Source>>& source);
};
std::shared_ptr<Geometry> geometryOf(AlignedEngine::Impl& impl, const Surface& surface);
Target prepareSurfaceTarget(AlignedEngine::Impl& impl, const Source& source, std::shared_ptr<SourceReader> opened);

// Conversion and checks shared by the readers.
Block convertBlock(const Block& block, const std::string& dtype);
std::string keyOf(const Triple& value);
} // namespace vcr::sources
