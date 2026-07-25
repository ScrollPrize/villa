// Drives SurfaceCache against the sampling path it memoizes: QuadSurface::gen()
// plus ChunkedPlaneSampler. The cache is only correct if, for the same view
// geometry, it reproduces what the legacy flattened render path produced -- so
// that comparison is the spine of this file.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/render/SurfaceCache.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

using vc::render::ChunkedPlaneSampler;
using vc::render::ChunkKey;
using vc::render::ChunkKeyHash;
using vc::render::ChunkResult;
using vc::render::ChunkStatus;
using vc::render::SurfaceCache;

namespace {

constexpr int kChunk = 32;
// Level-0 extent on every axis. Large enough that the fixture surface below,
// plus its +-16 voxel band, lies wholly inside the volume -- otherwise
// out-of-bounds reads return the fill value and every comparison passes
// vacuously on all-zero frames.
constexpr int kShape0 = 512;

// Smooth synthetic intensity so a sub-voxel coordinate difference between the
// two paths cannot show up as a large value difference.
uint8_t syntheticValue(double x, double y, double z)
{
    const double v = 128.0 + 60.0 * std::sin(x * 0.05) + 40.0 * std::sin(y * 0.07) +
                     30.0 * std::sin(z * 0.11);
    return static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L));
}

// Synthetic pyramid: level L has extent kShape0 >> L and maps level-0 XYZ by
// 2^-L, matching a real zarr pyramid's LevelTransform. Every chunk resolves
// immediately, so prefetchChunks() is a no-op and resident-only reads always
// succeed.
class SyntheticArray : public vc::render::IChunkedArray {
public:
    explicit SyntheticArray(int levels = 3) : levels_(levels) {}

    int numLevels() const override { return levels_; }

    std::array<int, 3> shape(int level) const override
    {
        const int extent = std::max(1, kShape0 >> level);
        return {extent, extent, extent};
    }

    std::array<int, 3> chunkShape(int) const override { return {kChunk, kChunk, kChunk}; }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }

    LevelTransform levelTransform(int level) const override
    {
        LevelTransform t;
        const double s = std::ldexp(1.0, -level);
        t.scaleFromLevel0 = {s, s, s};
        return t;
    }

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        return build(level, iz, iy, ix);
    }

    ChunkResult getChunkIfCached(int level, int iz, int iy, int ix) override
    {
        return build(level, iz, iy, ix);
    }

    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return build(level, iz, iy, ix);
    }

    void prefetchChunks(const std::vector<ChunkKey>&, bool, int) override
    {
        ++prefetchRequests_;
    }
    void beginViewRequest(bool discardPending) override
    {
        ++viewRequests_;
        if (discardPending)
            ++discardRequests_;
    }
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}

    int viewRequests() const { return viewRequests_.load(); }
    int discardRequests() const { return discardRequests_.load(); }
    int prefetchRequests() const { return prefetchRequests_.load(); }
private:
    ChunkResult build(int level, int iz, int iy, int ix)
    {
        ChunkResult result;
        result.dtype = vc::render::ChunkDtype::UInt8;
        result.shape = chunkShape(level);
        if (level < 0 || level >= levels_) {
            result.status = ChunkStatus::Missing;
            return result;
        }
        const auto extent = shape(level);
        if (iz < 0 || iy < 0 || ix < 0 || iz * kChunk >= extent[0] ||
            iy * kChunk >= extent[1] || ix * kChunk >= extent[2]) {
            result.status = ChunkStatus::Missing;
            return result;
        }

        const ChunkKey key{level, iz, iy, ix};
        std::lock_guard lock(mutex_);
        auto it = chunks_.find(key);
        if (it == chunks_.end()) {
            auto bytes = std::make_shared<std::vector<std::byte>>(
                std::size_t(kChunk) * kChunk * kChunk);
            const double scale = std::ldexp(1.0, level);
            for (int z = 0; z < kChunk; ++z) {
                for (int y = 0; y < kChunk; ++y) {
                    for (int x = 0; x < kChunk; ++x) {
                        // Level L voxel (x,y,z) stands for level-0 position
                        // (x,y,z) * 2^L, so all levels agree where they overlap.
                        const uint8_t value = syntheticValue(double(ix * kChunk + x) * scale,
                                                             double(iy * kChunk + y) * scale,
                                                             double(iz * kChunk + z) * scale);
                        (*bytes)[(std::size_t(z) * kChunk + y) * kChunk + x] =
                            std::byte(value);
                    }
                }
            }
            it = chunks_.emplace(key, std::move(bytes)).first;
        }
        result.status = ChunkStatus::Data;
        result.bytes = it->second;
        return result;
    }

    int levels_;
    std::mutex mutex_;
    std::unordered_map<ChunkKey, std::shared_ptr<const std::vector<std::byte>>, ChunkKeyHash>
        chunks_;
    std::atomic<int> viewRequests_{0};
    std::atomic<int> discardRequests_{0};
    std::atomic<int> prefetchRequests_{0};
};

// No decoded storage: used to exercise dependency discovery over a huge
// logical volume without making the test's memory proportional to that volume.
class SparseAllFillArray : public vc::render::IChunkedArray {
public:
    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override
    {
        return {1'000'000, 1'000'000, 1'000'000};
    }
    std::array<int, 3> chunkShape(int) const override { return {32, 32, 32}; }
    vc::render::ChunkDtype dtype() const override
    {
        return vc::render::ChunkDtype::UInt8;
    }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int) const override { return {}; }

    ChunkResult tryGetChunk(int, int, int, int) override { return allFill(); }
    ChunkResult getChunkIfCached(int, int, int, int) override { return allFill(); }
    ChunkResult getChunkBlocking(int, int, int, int) override { return allFill(); }

    void prefetchChunks(const std::vector<ChunkKey>& keys, bool, int) override
    {
        ++prefetchRequests_;
        auto peak = maxPrefetchKeys_.load();
        while (peak < keys.size() &&
               !maxPrefetchKeys_.compare_exchange_weak(peak, keys.size())) {
        }
    }
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override
    {
        return 0;
    }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}

    std::size_t maxPrefetchKeys() const { return maxPrefetchKeys_.load(); }
    int prefetchRequests() const { return prefetchRequests_.load(); }

private:
    static ChunkResult allFill()
    {
        ChunkResult result;
        result.status = ChunkStatus::AllFill;
        result.dtype = vc::render::ChunkDtype::UInt8;
        result.shape = {32, 32, 32};
        return result;
    }

    std::atomic_size_t maxPrefetchKeys_{0};
    std::atomic<int> prefetchRequests_{0};
};

// A gently curved sheet whose nominal spacing is one voxel, so SurfaceCache's
// (u, v) and the viewer's surface pointer share one unit. `gridStep` voxels
// between grid vertices gives scale = 1/gridStep, and the surface's nominal
// extent is gridCols * gridStep -- here 384 units, spanning several tiles
// while staying inside the volume.
std::shared_ptr<QuadSurface> makeSurface(int gridCols = 48,
                                         int gridRows = 48,
                                         float gridStep = 8.0f)
{
    cv::Mat_<cv::Vec3f> points(gridRows, gridCols);
    for (int r = 0; r < gridRows; ++r) {
        for (int c = 0; c < gridCols; ++c) {
            const float x = 40.0f + float(c) * gridStep;
            const float y = 40.0f + float(r) * gridStep;
            // Curvature keeps normals off-axis so the fill exercises real
            // trilinear taps rather than an axis-aligned special case.
            const float z = 120.0f + 6.0f * std::sin(float(c) * 0.08f) +
                            4.0f * std::cos(float(r) * 0.06f);
            points(r, c) = {x, y, z};
        }
    }
    return std::make_shared<QuadSurface>(points, cv::Vec2f(1.0f / gridStep, 1.0f / gridStep));
}

std::shared_ptr<QuadSurface> makeStretchedSurface()
{
    constexpr int gridSize = 48;
    constexpr float nominalStep = 8.0f;
    constexpr float volumeStep = 20'000.0f;
    cv::Mat_<cv::Vec3f> points(gridSize, gridSize);
    for (int r = 0; r < gridSize; ++r) {
        for (int c = 0; c < gridSize; ++c) {
            points(r, c) = {40.0f + float(c) * volumeStep,
                            40.0f + float(r) * volumeStep, 500'000.0f};
        }
    }
    return std::make_shared<QuadSurface>(
        points, cv::Vec2f(1.0f / nominalStep, 1.0f / nominalStep));
}

struct LegacyFrame {
    cv::Mat_<uint8_t> values;
    cv::Mat_<uint8_t> coverage;
};

void applyNormalOffset(cv::Mat_<cv::Vec3f>& coords,
                       const cv::Mat_<cv::Vec3f>& normals,
                       float zOff)
{
    if (zOff == 0.0f || coords.empty() || normals.empty())
        return;
    for (int y = 0; y < coords.rows; ++y) {
        auto* coordRow = coords.ptr<cv::Vec3f>(y);
        const auto* normalRow = normals.ptr<cv::Vec3f>(y);
        for (int x = 0; x < coords.cols; ++x) {
            const cv::Vec3f& n = normalRow[x];
            if (!std::isfinite(coordRow[x][0]) || !std::isfinite(n[0]) ||
                !std::isfinite(n[1]) || !std::isfinite(n[2]))
                continue;
            coordRow[x] += n * zOff;
        }
    }
}

// The pre-cache render path for the flattened view: gen() at the frame's
// geometry, per-pixel normal offset, then one level of coord sampling.
LegacyFrame renderLegacy(vc::render::IChunkedArray& array,
                         QuadSurface& surface,
                         int level,
                         double uMin,
                         double vMin,
                         double scale,
                         int fbW,
                         int fbH,
                         double zOff,
                         vc::Sampling sampling)
{
    LegacyFrame frame;
    frame.values = cv::Mat_<uint8_t>(fbH, fbW, uint8_t(0));
    frame.coverage = cv::Mat_<uint8_t>(fbH, fbW, uint8_t(0));

    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
    const cv::Vec3f offset(float(uMin * scale), float(vMin * scale), 0.0f);
    surface.gen(&coords, &normals, cv::Size(fbW, fbH), {0, 0, 0}, float(scale), offset);
    applyNormalOffset(coords, normals, float(zOff));

    ChunkedPlaneSampler::Options options(sampling, 32);
    ChunkedPlaneSampler::sampleCoordsLevel(array, level, coords, frame.values, frame.coverage,
                                           options);
    return frame;
}

// Legacy composite: one full coord-sampling pass per layer, reduced per pixel.
LegacyFrame renderLegacyComposite(vc::render::IChunkedArray& array,
                                  QuadSurface& surface,
                                  int level,
                                  double uMin,
                                  double vMin,
                                  double scale,
                                  int fbW,
                                  int fbH,
                                  double zOff,
                                  const CompositeRenderSettings& composite,
                                  vc::Sampling sampling)
{
    LegacyFrame frame;
    frame.values = cv::Mat_<uint8_t>(fbH, fbW, uint8_t(0));
    frame.coverage = cv::Mat_<uint8_t>(fbH, fbW, uint8_t(0));

    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
    const cv::Vec3f offset(float(uMin * scale), float(vMin * scale), 0.0f);
    surface.gen(&coords, &normals, cv::Size(fbW, fbH), {0, 0, 0}, float(scale), offset);
    applyNormalOffset(coords, normals, float(zOff));

    const int front = std::max(0, composite.layersFront);
    const int behind = std::max(0, composite.layersBehind);
    const int numLayers = front + behind + 1;
    const float zStep = composite.reverseDirection ? -1.0f : 1.0f;
    ChunkedPlaneSampler::Options options(sampling, 32);

    std::vector<cv::Mat_<uint8_t>> layerValues;
    std::vector<cv::Mat_<uint8_t>> layerCoverage;
    cv::Mat_<cv::Vec3f> layerCoords(coords.rows, coords.cols);
    for (int i = 0; i < numLayers; ++i) {
        const float layerOffset = float(-behind + i) * zStep;
        for (int y = 0; y < coords.rows; ++y) {
            const auto* src = coords.ptr<cv::Vec3f>(y);
            const auto* nrow = normals.ptr<cv::Vec3f>(y);
            auto* dst = layerCoords.ptr<cv::Vec3f>(y);
            for (int x = 0; x < coords.cols; ++x) {
                if (!std::isfinite(src[x][0]) || src[x][0] == -1.0f)
                    dst[x] = src[x];
                else
                    dst[x] = src[x] + nrow[x] * layerOffset;
            }
        }
        layerValues.emplace_back(fbH, fbW, uint8_t(0));
        layerCoverage.emplace_back(fbH, fbW, uint8_t(0));
        ChunkedPlaneSampler::sampleCoordsLevel(array, level, layerCoords, layerValues.back(),
                                               layerCoverage.back(), options);
    }

    LayerStack stack;
    stack.values.resize(std::size_t(numLayers));
    for (int y = 0; y < fbH; ++y) {
        auto* outRow = frame.values.ptr<uint8_t>(y);
        auto* coverageRow = frame.coverage.ptr<uint8_t>(y);
        for (int x = 0; x < fbW; ++x) {
            stack.validCount = 0;
            for (int i = 0; i < numLayers; ++i) {
                if (!layerCoverage[std::size_t(i)](y, x))
                    continue;
                const float value = float(layerValues[std::size_t(i)](y, x));
                if (value < float(composite.params.isoCutoff))
                    continue;
                stack.values[std::size_t(stack.validCount++)] = value;
            }
            if (stack.validCount > 0) {
                outRow[x] = static_cast<uint8_t>(
                    std::clamp(compositeLayerStack(stack, composite.params), 0.0f, 255.0f));
                coverageRow[x] = 1;
            }
        }
    }
    return frame;
}

struct Diff {
    int comparedPixels = 0;
    int legacyOnly = 0;   // covered by the legacy path but not by the cache
    int cacheOnly = 0;
    int maxAbs = 0;
    double meanAbs = 0.0;
    // Spread of the legacy values actually compared. A zero spread means the
    // frame carries no signal (e.g. the surface fell outside the volume and
    // every read returned the fill value), which would make an equality
    // assertion vacuous.
    int valueSpread = 0;
};

Diff compare(const LegacyFrame& legacy, const cv::Mat_<uint8_t>& values,
             const cv::Mat_<uint8_t>& coverage)
{
    Diff diff;
    double total = 0.0;
    int lowest = 255;
    int highest = 0;
    for (int y = 0; y < legacy.values.rows; ++y) {
        for (int x = 0; x < legacy.values.cols; ++x) {
            const bool a = legacy.coverage(y, x) != 0;
            const bool b = coverage(y, x) != 0;
            if (a && !b) {
                ++diff.legacyOnly;
                continue;
            }
            if (b && !a) {
                ++diff.cacheOnly;
                continue;
            }
            if (!a)
                continue;
            const int reference = int(legacy.values(y, x));
            const int d = std::abs(reference - int(values(y, x)));
            diff.maxAbs = std::max(diff.maxAbs, d);
            lowest = std::min(lowest, reference);
            highest = std::max(highest, reference);
            total += d;
            ++diff.comparedPixels;
        }
    }
    if (diff.comparedPixels > 0) {
        diff.meanAbs = total / diff.comparedPixels;
        diff.valueSpread = highest - lowest;
    }
    return diff;
}

// requestView admits a bounded page per call, so drive it until the view's
// tiles are resident (or the deadline passes).
bool fillView(SurfaceCache& cache, int level, double uMin, double vMin, double scale, int fbW,
              int fbH, std::chrono::milliseconds timeout = std::chrono::seconds(20))
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::uint64_t generation = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        cache.requestView(level, uMin, vMin, scale, fbW, fbH, ++generation);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const auto stats = cache.stats();
        if (stats.tilesInFlight != 0)
            continue;
        // A second identical request that admits nothing means every tile the
        // view needs is already complete.
        const auto before = cache.stats().tiles;
        cache.requestView(level, uMin, vMin, scale, fbW, fbH, ++generation);
        if (cache.stats().tilesInFlight == 0 && cache.stats().tiles == before)
            return true;
    }
    return false;
}

SurfaceCache::Options defaultOptions()
{
    SurfaceCache::Options options;
    options.byteCapacity = 256ULL << 20;
    options.sampling = vc::Sampling::Trilinear;
    return options;
}

} // namespace

TEST_CASE("SurfaceCache bandCovers covers exactly the stored band")
{
    auto array = std::make_shared<SyntheticArray>();
    auto surface = makeSurface();
    SurfaceCache cache(array, surface, defaultOptions());

    CHECK(cache.wMin() == -16);
    CHECK(cache.wCount() == 32);
    CHECK(cache.bandCovers(0.0, 0.0));
    CHECK(cache.bandCovers(-16.0, 15.0));
    CHECK(cache.bandCovers(-16.0, -16.0));
    CHECK(cache.bandCovers(15.0, 15.0));
    CHECK(cache.bandCovers(-0.5, 8.25));
    CHECK_FALSE(cache.bandCovers(-16.5, 0.0));
    CHECK_FALSE(cache.bandCovers(0.0, 15.5));
    CHECK_FALSE(cache.bandCovers(0.0, 40.0));
    CHECK_FALSE(cache.bandCovers(std::nan(""), 0.0));
}

TEST_CASE("SurfaceCache supersedes chunk dependencies only when its view changes")
{
    auto array = std::make_shared<SyntheticArray>();
    auto options = defaultOptions();
    options.supersedeChunkRequests = true;
    SurfaceCache cache(array, makeSurface(), options);

    cache.requestView(0, 0.0, 0.0, 1.0, 128, 128, 1);
    CHECK(array->viewRequests() == 1);
    CHECK(array->discardRequests() == 1);

    // Progressive rerenders of the same viewport must not cancel their own
    // fills merely because the viewer's presentation generation advanced.
    cache.requestView(0, 0.0, 0.0, 1.0, 128, 128, 2);
    CHECK(array->viewRequests() == 1);

    cache.requestView(0, 256.0, 0.0, 1.0, 128, 128, 3);
    CHECK(array->viewRequests() == 2);
    CHECK(array->discardRequests() == 2);
}

TEST_CASE("SurfaceCache prefetches each tile as one parallel dependency batch")
{
    auto array = std::make_shared<SyntheticArray>();
    SurfaceCache cache(array, makeSurface(), defaultOptions());

    REQUIRE(fillView(cache, 0, 0.0, 0.0, 1.0,
                     SurfaceCache::kTileSize, SurfaceCache::kTileSize));
    const auto stats = cache.stats();
    REQUIRE(stats.tiles > 0);
    CHECK(array->prefetchRequests() > 0);
    // The slow workaround issued up to 256 blocking prefetches per tile.
    // Invalid edge tiles may need no source chunks, hence the upper bound.
    CHECK(std::size_t(array->prefetchRequests()) <= stats.tiles);
    CHECK(stats.dependencyRegionSplits == 0);
}

TEST_CASE("SurfaceCache bounds dependency metadata for a pathological tile AABB")
{
    auto array = std::make_shared<SparseAllFillArray>();
    SurfaceCache cache(array, makeStretchedSurface(), defaultOptions());

    // The old 8x8-block AABB collector attempted to enumerate billions of
    // chunks here. Adaptive discovery should split only this pathological
    // region while keeping every individual prefetch batch bounded.
    REQUIRE(fillView(cache, 0, 0.0, 0.0, 1.0,
                     SurfaceCache::kTileSize, SurfaceCache::kTileSize));
    const auto stats = cache.stats();
    CHECK(stats.dependencyRegionSplits > 0);
    CHECK(stats.peakDependencyKeys <= 8192);
    CHECK(array->prefetchRequests() > 1);
    CHECK(array->maxPrefetchKeys() <= 8192);
}

TEST_CASE("SurfaceCache reproduces the legacy plain frame")
{
    auto array = std::make_shared<SyntheticArray>();
    auto surface = makeSurface();

    SUBCASE("tile-aligned view at level 0 matches sample for sample")
    {
        // uMin/vMin on a tile boundary with scale == 2^-level makes every
        // bilinear weight 0 or 1, so the cache must return exactly the samples
        // it stored -- and gen() sees identical arguments in both paths.
        constexpr int fb = SurfaceCache::kTileSize;
        SurfaceCache cache(array, surface, defaultOptions());
        REQUIRE(fillView(cache, 0, 0.0, 0.0, 1.0, fb, fb));

        cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
        cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
        CompositeRenderSettings composite;
        const auto stats = cache.sampleView(0, 0.0, 0.0, 1.0, 0.0, composite, values, coverage);
        CHECK(stats.coveredPixels > fb * fb / 2);

        const LegacyFrame legacy = renderLegacy(*array, *surface, 0, 0.0, 0.0, 1.0, fb, fb, 0.0,
                                                vc::Sampling::Trilinear);
        const Diff diff = compare(legacy, values, coverage);
        CHECK(diff.comparedPixels > fb * fb / 2);
        CHECK(diff.valueSpread > 20);
        CHECK(diff.legacyOnly == 0);
        CHECK(diff.maxAbs == 0);
    }

    SUBCASE("unaligned view and coarser level stay within resampling tolerance")
    {
        constexpr int fbW = 200;
        constexpr int fbH = 150;
        const double uMin = -37.25;
        const double vMin = 11.5;
        const double scale = 0.5;  // level 1 is the matching pyramid level
        SurfaceCache cache(array, surface, defaultOptions());
        REQUIRE(fillView(cache, 1, uMin, vMin, scale, fbW, fbH));

        cv::Mat_<uint8_t> values(fbH, fbW, uint8_t(0));
        cv::Mat_<uint8_t> coverage(fbH, fbW, uint8_t(0));
        CompositeRenderSettings composite;
        cache.sampleView(1, uMin, vMin, scale, 0.0, composite, values, coverage);

        const LegacyFrame legacy = renderLegacy(*array, *surface, 1, uMin, vMin, scale, fbW, fbH,
                                                0.0, vc::Sampling::Trilinear);
        const Diff diff = compare(legacy, values, coverage);
        CHECK(diff.comparedPixels > 1000);
        CHECK(diff.valueSpread > 20);
        CHECK(diff.meanAbs < 2.0);
        CHECK(diff.maxAbs <= 8);
    }

    SUBCASE("integer normal offset inside the band")
    {
        constexpr int fb = SurfaceCache::kTileSize;
        SurfaceCache cache(array, surface, defaultOptions());
        REQUIRE(fillView(cache, 0, 0.0, 0.0, 1.0, fb, fb));

        cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
        cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
        CompositeRenderSettings composite;
        cache.sampleView(0, 0.0, 0.0, 1.0, -7.0, composite, values, coverage);

        const LegacyFrame legacy = renderLegacy(*array, *surface, 0, 0.0, 0.0, 1.0, fb, fb, -7.0,
                                                vc::Sampling::Trilinear);
        const Diff diff = compare(legacy, values, coverage);
        CHECK(diff.comparedPixels > fb * fb / 2);
        CHECK(diff.valueSpread > 20);
        CHECK(diff.maxAbs == 0);
    }

    SUBCASE("fractional normal offset interpolates between band slices")
    {
        // The cache blends two stored slices; the legacy path samples the
        // volume at the exact position. For smooth data the two agree closely.
        constexpr int fb = SurfaceCache::kTileSize;
        SurfaceCache cache(array, surface, defaultOptions());
        REQUIRE(fillView(cache, 0, 0.0, 0.0, 1.0, fb, fb));

        cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
        cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
        CompositeRenderSettings composite;
        cache.sampleView(0, 0.0, 0.0, 1.0, 3.4, composite, values, coverage);

        const LegacyFrame legacy = renderLegacy(*array, *surface, 0, 0.0, 0.0, 1.0, fb, fb, 3.4,
                                                vc::Sampling::Trilinear);
        const Diff diff = compare(legacy, values, coverage);
        CHECK(diff.comparedPixels > fb * fb / 2);
        CHECK(diff.valueSpread > 20);
        CHECK(diff.legacyOnly == 0);
        CHECK(diff.meanAbs < 1.0);
        CHECK(diff.maxAbs <= 4);
    }
}

TEST_CASE("SurfaceCache composite reduction matches the per-layer legacy pass")
{
    auto array = std::make_shared<SyntheticArray>();
    auto surface = makeSurface();
    constexpr int fb = SurfaceCache::kTileSize;

    // The legacy composite samples layers with Nearest, so compare like for
    // like by filling the cache the same way.
    auto options = defaultOptions();
    options.sampling = vc::Sampling::Nearest;
    SurfaceCache cache(array, surface, options);
    REQUIRE(fillView(cache, 0, 0.0, 0.0, 1.0, fb, fb));

    for (const char* method : {"mean", "max", "min", "alpha"}) {
        CAPTURE(method);
        CompositeRenderSettings composite;
        composite.enabled = true;
        composite.layersFront = 6;
        composite.layersBehind = 3;
        composite.params.method = method;

        double wLow = 0.0, wHigh = 0.0;
        SurfaceCache::compositeWRange(composite, 0.0, wLow, wHigh);
        CHECK(wLow == -3.0);
        CHECK(wHigh == 6.0);
        REQUIRE(cache.bandCovers(wLow, wHigh));

        cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
        cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
        cache.sampleView(0, 0.0, 0.0, 1.0, 0.0, composite, values, coverage);

        const LegacyFrame legacy = renderLegacyComposite(
            *array, *surface, 0, 0.0, 0.0, 1.0, fb, fb, 0.0, composite, vc::Sampling::Nearest);
        const Diff diff = compare(legacy, values, coverage);
        CHECK(diff.comparedPixels > fb * fb / 2);
        CHECK(diff.valueSpread > 20);
        CHECK(diff.legacyOnly == 0);
        CHECK(diff.maxAbs <= 1);
    }

    SUBCASE("reverseDirection flips the band the reduction walks")
    {
        CompositeRenderSettings composite;
        composite.enabled = true;
        composite.layersFront = 8;
        composite.layersBehind = 0;
        composite.reverseDirection = true;
        composite.params.method = "max";

        double wLow = 0.0, wHigh = 0.0;
        SurfaceCache::compositeWRange(composite, 0.0, wLow, wHigh);
        CHECK(wLow == -8.0);
        CHECK(wHigh == 0.0);

        cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
        cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
        cache.sampleView(0, 0.0, 0.0, 1.0, 0.0, composite, values, coverage);
        const LegacyFrame legacy = renderLegacyComposite(
            *array, *surface, 0, 0.0, 0.0, 1.0, fb, fb, 0.0, composite, vc::Sampling::Nearest);
        const Diff diff = compare(legacy, values, coverage);
        CHECK(diff.comparedPixels > fb * fb / 2);
        CHECK(diff.valueSpread > 20);
        CHECK(diff.maxAbs <= 1);
    }

    SUBCASE("iso cutoff drops layers below the threshold")
    {
        // Derive the cutoff from the data so it actually splits the stack
        // instead of accepting or rejecting every layer.
        cv::Mat_<uint8_t> plainValues(fb, fb, uint8_t(0));
        cv::Mat_<uint8_t> plainCoverage(fb, fb, uint8_t(0));
        CompositeRenderSettings plain;
        cache.sampleView(0, 0.0, 0.0, 1.0, 0.0, plain, plainValues, plainCoverage);
        std::vector<uint8_t> observed;
        for (int y = 0; y < fb; ++y) {
            for (int x = 0; x < fb; ++x) {
                if (plainCoverage(y, x))
                    observed.push_back(plainValues(y, x));
            }
        }
        REQUIRE(observed.size() > 100);
        std::sort(observed.begin(), observed.end());
        const uint8_t median = observed[observed.size() / 2];
        REQUIRE(median > observed.front());

        CompositeRenderSettings composite;
        composite.enabled = true;
        composite.layersFront = 6;
        composite.layersBehind = 3;
        composite.params.method = "mean";
        composite.params.isoCutoff = median;

        cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
        cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
        cache.sampleView(0, 0.0, 0.0, 1.0, 0.0, composite, values, coverage);
        const LegacyFrame legacy = renderLegacyComposite(
            *array, *surface, 0, 0.0, 0.0, 1.0, fb, fb, 0.0, composite, vc::Sampling::Nearest);
        const Diff diff = compare(legacy, values, coverage);
        CHECK(diff.comparedPixels > 0);
        CHECK(diff.legacyOnly == 0);
        CHECK(diff.cacheOnly == 0);
        CHECK(diff.maxAbs <= 1);
    }
}

TEST_CASE("SurfaceCache overlay reduction honours method and direction")
{
    auto array = std::make_shared<SyntheticArray>();
    auto surface = makeSurface();
    constexpr int fb = SurfaceCache::kTileSize;
    auto options = defaultOptions();
    options.sampling = vc::Sampling::Nearest;
    SurfaceCache cache(array, surface, options);
    REQUIRE(fillView(cache, 0, 0.0, 0.0, 1.0, fb, fb));

    OverlayCompositeSettings overlay;
    overlay.enabled = true;
    overlay.method = "max";
    overlay.layersFront = 5;
    overlay.layersBehind = 2;

    double wLow = 0.0, wHigh = 0.0;
    SurfaceCache::overlayCompositeWRange(overlay, 0.0, 1.0, wLow, wHigh);
    CHECK(wLow == -2.0);
    CHECK(wHigh == 5.0);
    SurfaceCache::overlayCompositeWRange(overlay, 0.0, -1.0, wLow, wHigh);
    CHECK(wLow == -5.0);
    CHECK(wHigh == 2.0);

    cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
    cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
    cache.sampleViewOverlay(0, 0.0, 0.0, 1.0, 0.0, overlay, 1.0, values, coverage);

    CompositeRenderSettings equivalent;
    equivalent.enabled = true;
    equivalent.layersFront = overlay.layersFront;
    equivalent.layersBehind = overlay.layersBehind;
    equivalent.params.method = overlay.method;
    const LegacyFrame legacy = renderLegacyComposite(*array, *surface, 0, 0.0, 0.0, 1.0, fb, fb,
                                                     0.0, equivalent, vc::Sampling::Nearest);
    const Diff diff = compare(legacy, values, coverage);
    CHECK(diff.comparedPixels > fb * fb / 2);
    CHECK(diff.valueSpread > 20);
    CHECK(diff.maxAbs <= 1);

    SUBCASE("a zero-thickness overlay stack degenerates to a plain read")
    {
        OverlayCompositeSettings flat;
        flat.enabled = true;
        flat.method = "max";
        flat.layersFront = 0;
        flat.layersBehind = 0;
        double lo = 1.0, hi = 1.0;
        SurfaceCache::overlayCompositeWRange(flat, 2.0, 1.0, lo, hi);
        CHECK(lo == 2.0);
        CHECK(hi == 2.0);
    }
}

TEST_CASE("SurfaceCache bounds its residency by the byte capacity")
{
    auto array = std::make_shared<SyntheticArray>();
    auto surface = makeSurface();
    auto options = defaultOptions();
    // Four tiles is the enforced floor, so ask for six.
    SurfaceCache probe(array, surface, options);
    options.byteCapacity = 6 * probe.tileBytes();
    SurfaceCache cache(array, surface, options);

    // Sweep the surface so far more than six tiles get filled.
    for (int step = 0; step < 6; ++step) {
        const double uMin = -256.0 + double(step) * 128.0;
        cache.requestView(0, uMin, 0.0, 1.0, 128, 128, std::uint64_t(step + 1));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (cache.stats().tilesInFlight != 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    auto stats = cache.stats();
    CHECK(stats.capacity == 6 * probe.tileBytes());
    CHECK(stats.tiles <= 6);
    CHECK(stats.bytes <= stats.capacity);

    SUBCASE("lowering the capacity evicts in place")
    {
        cache.setByteCapacity(probe.tileBytes());
        stats = cache.stats();
        // The floor keeps a workable minimum rather than emptying the cache.
        CHECK(stats.capacity == 4 * probe.tileBytes());
        CHECK(stats.tiles <= 4);
    }
}

TEST_CASE("SurfaceCache invalidation")
{
    auto array = std::make_shared<SyntheticArray>();
    auto surface = makeSurface();
    constexpr int fb = SurfaceCache::kTileSize;

    SUBCASE("invalidateAll drops every tile")
    {
        SurfaceCache cache(array, surface, defaultOptions());
        REQUIRE(fillView(cache, 0, 0.0, 0.0, 1.0, fb, fb));
        REQUIRE(cache.stats().tiles > 0);
        cache.invalidateAll();
        CHECK(cache.stats().tiles == 0);
        CHECK(cache.stats().bytes == 0);
    }

    SUBCASE("invalidateSurfaceRegion drops exactly the overlapping tiles")
    {
        SurfaceCache cache(array, surface, defaultOptions());
        // Nominal (u, v) is centred on the surface, so derive both view
        // origins from the grid rather than assuming where zero lands.
        const cv::Vec2d nearGridOrigin = surface->gridToSurface({0.0, 0.0});
        const cv::Vec2d farInU = surface->gridToSurface({36.0, 0.0});
        REQUIRE(fillView(cache, 0, nearGridOrigin[0], nearGridOrigin[1], 1.0, fb, fb));
        REQUIRE(fillView(cache, 0, farInU[0], nearGridOrigin[1], 1.0, fb, fb));
        const std::size_t before = cache.stats().tiles;
        REQUIRE(before >= 2);

        // Grid cells at the origin only overlap the first view's tiles.
        cache.invalidateSurfaceRegion(cv::Rect(0, 0, 2, 2));
        const std::size_t after = cache.stats().tiles;
        CHECK(after < before);
        CHECK(after > 0);

        // The far view is still resident and still samples.
        cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
        cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
        CompositeRenderSettings composite;
        const auto stats = cache.sampleView(0, farInU[0], nearGridOrigin[1], 1.0, 0.0,
                                            composite, values, coverage);
        CHECK(stats.missingTiles == 0);
        CHECK(stats.coveredPixels > 0);
    }

    SUBCASE("an empty region falls back to dropping everything")
    {
        SurfaceCache cache(array, surface, defaultOptions());
        REQUIRE(fillView(cache, 0, 0.0, 0.0, 1.0, fb, fb));
        cache.invalidateSurfaceRegion(cv::Rect());
        CHECK(cache.stats().tiles == 0);
    }
}

TEST_CASE("SurfaceCache falls back to coarser cached levels")
{
    auto array = std::make_shared<SyntheticArray>();
    auto surface = makeSurface();
    constexpr int fb = SurfaceCache::kTileSize;
    SurfaceCache cache(array, surface, defaultOptions());

    // Fill level 1 only, then ask for a level-0 frame over the same nominal
    // area. Nothing is resident at level 0, so every covered pixel came from
    // the coarser level.
    REQUIRE(fillView(cache, 1, 0.0, 0.0, 0.5, fb, fb));

    cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
    cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
    CompositeRenderSettings composite;
    const auto stats = cache.sampleView(0, 0.0, 0.0, 1.0, 0.0, composite, values, coverage);
    CHECK(stats.missingTiles > 0);
    CHECK(stats.coveredPixels > 0);
    CHECK(stats.usedCoarseFallback);
}

TEST_CASE("SurfaceCache serves zoom beyond its coarsest level")
{
    auto array = std::make_shared<SyntheticArray>();
    auto surface = makeSurface();
    auto options = defaultOptions();
    options.levels = 1;  // coarsest cached level is 0
    SurfaceCache cache(array, surface, options);
    CHECK(cache.levels() == 1);

    constexpr int fb = 64;
    // Zoomed further out than level 0: the request clamps and sampleView
    // resamples the coarsest tiles rather than falling through to the volume.
    REQUIRE(fillView(cache, 3, 0.0, 0.0, 0.25, fb, fb));
    cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
    cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
    CompositeRenderSettings composite;
    const auto stats = cache.sampleView(3, 0.0, 0.0, 0.25, 0.0, composite, values, coverage);
    CHECK(stats.coveredPixels > 0);
}

TEST_CASE("SurfaceCache tiles survive eviction while a render reads them")
{
    // Eviction only drops the map's reference; sampleView holds its own, so a
    // capacity change concurrent with a resample must not tear.
    auto array = std::make_shared<SyntheticArray>();
    auto surface = makeSurface();
    auto options = defaultOptions();
    SurfaceCache cache(array, surface, options);
    constexpr int fb = 192;
    REQUIRE(fillView(cache, 0, 0.0, 0.0, 1.0, fb, fb));
    const std::size_t tileBytes = cache.tileBytes();

    std::atomic<bool> stop{false};
    std::thread churn([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            cache.setByteCapacity(tileBytes);
            cache.setByteCapacity(256ULL << 20);
        }
    });

    CompositeRenderSettings composite;
    composite.enabled = true;
    composite.layersFront = 8;
    for (int i = 0; i < 40; ++i) {
        cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
        cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
        cache.sampleView(0, 0.0, 0.0, 1.0, 0.0, composite, values, coverage);
    }
    stop.store(true, std::memory_order_relaxed);
    churn.join();
    CHECK(cache.stats().bytes <= cache.stats().capacity);
}

// An array whose chunks are permanently Missing beyond a boundary, so tiles
// covering that region can never be completed.
class PartlyMissingArray : public SyntheticArray {
public:
    using SyntheticArray::SyntheticArray;

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        return gate(level, iz, iy, ix, [&] { return SyntheticArray::tryGetChunk(level, iz, iy, ix); });
    }
    ChunkResult getChunkIfCached(int level, int iz, int iy, int ix) override
    {
        return gate(level, iz, iy, ix,
                    [&] { return SyntheticArray::getChunkIfCached(level, iz, iy, ix); });
    }
    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return gate(level, iz, iy, ix,
                    [&] { return SyntheticArray::getChunkBlocking(level, iz, iy, ix); });
    }

    int missingReads() const { return missingReads_.load(); }

private:
    template <typename Fn>
    ChunkResult gate(int level, int, int, int ix, Fn&& fn)
    {
        // Everything from the third chunk column on is unavailable.
        if (ix >= 2) {
            ++missingReads_;
            ChunkResult result;
            result.dtype = vc::render::ChunkDtype::UInt8;
            result.shape = chunkShape(level);
            result.status = ChunkStatus::Missing;
            return result;
        }
        return fn();
    }

    std::atomic<int> missingReads_{0};
};

TEST_CASE("SurfaceCache stops refilling a tile it can never complete")
{
    // Publishing a tile wakes the render loop, which calls requestView again.
    // Without a bound on refills of an incomplete tile that is an unbounded
    // spin, so assert the retries actually stop.
    auto array = std::make_shared<PartlyMissingArray>();
    auto surface = makeSurface();
    SurfaceCache cache(array, surface, defaultOptions());

    constexpr int fb = 256;
    const double uMin = 0.0;
    const double vMin = 0.0;
    std::uint64_t generation = 0;
    // Drive the loop the way the viewer does: request, let fills land, repeat.
    for (int round = 0; round < 24; ++round) {
        cache.requestView(0, uMin, vMin, 1.0, fb, fb, ++generation);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (cache.stats().tilesInFlight != 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    const auto stats = cache.stats();
    REQUIRE(stats.tilesIncomplete > 0);   // the fixture really did block chunks
    const int readsAfterSettling = array->missingReads();

    // Further identical requests must not queue any more work.
    for (int round = 0; round < 8; ++round)
        cache.requestView(0, uMin, vMin, 1.0, fb, fb, ++generation);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(cache.stats().tilesInFlight == 0);
    CHECK(array->missingReads() == readsAfterSettling);

    // The incomplete tiles are still stored and still usable -- computed work is
    // never discarded, it is only not recomputed.
    cv::Mat_<uint8_t> values(fb, fb, uint8_t(0));
    cv::Mat_<uint8_t> coverage(fb, fb, uint8_t(0));
    CompositeRenderSettings composite;
    const auto sampled = cache.sampleView(0, uMin, vMin, 1.0, 0.0, composite, values, coverage);
    CHECK(sampled.coveredPixels > 0);
}

TEST_CASE("SurfaceGeometryTileCache computes each tile once and is shareable")
{
    auto surface = makeSurface();
    auto geometry = std::make_shared<vc::render::SurfaceGeometryTileCache>(surface);
    auto first = geometry->get(0, 0, 0);
    REQUIRE(first);
    CHECK(first->coords.size() == cv::Size(SurfaceCache::kTileSize, SurfaceCache::kTileSize));
    CHECK(geometry->size() == 1);
    // Same key -> same object, so a base and overlay cache share one gen().
    CHECK(geometry->get(0, 0, 0).get() == first.get());
    CHECK(geometry->size() == 1);

    auto array = std::make_shared<SyntheticArray>();
    auto options = defaultOptions();
    options.geometry = geometry;
    SurfaceCache base(array, surface, options);
    SurfaceCache overlay(array, surface, options);
    CHECK(base.geometryTiles().get() == geometry.get());
    CHECK(overlay.geometryTiles().get() == geometry.get());
}
