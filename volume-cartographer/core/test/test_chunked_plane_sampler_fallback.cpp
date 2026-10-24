#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "utils/thread_pool.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/render/PinnedChunkRegion.hpp"
#include "vc/core/render/LocalChunkCache.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <optional>
#include <tuple>
#include <stdexcept>
#include <vector>

namespace {

class PyramidChunkedArray final : public vc::render::IChunkedArray {
public:
    PyramidChunkedArray(vc::render::ChunkStatus level0Status,
                        uint8_t level0Value,
                        vc::render::ChunkStatus level1Status,
                        uint8_t level1Value,
                        std::array<int, 3> level0Shape = {4, 4, 4},
                        std::array<int, 3> level1Shape = {2, 2, 2},
                        std::array<int, 3> level0ChunkShape = {0, 0, 0},
                        std::array<int, 3> level1ChunkShape = {0, 0, 0},
                        std::array<double, 3> level1Scale = {0.5, 0.5, 0.5})
        : statuses_{level0Status, level1Status}
        , values_{level0Value, level1Value}
        , shapes_{level0Shape, level1Shape}
        , chunkShapes_{
              level0ChunkShape[0] > 0 ? level0ChunkShape : level0Shape,
              level1ChunkShape[0] > 0 ? level1ChunkShape : level1Shape}
        , level1Scale_(level1Scale)
    {
    }

    int numLevels() const override { return 2; }

    std::array<int, 3> shape(int level) const override
    {
        return shapes_[level];
    }

    std::array<int, 3> chunkShape(int level) const override
    {
        return chunkShapes_[level];
    }

    vc::render::ChunkDtype dtype() const override
    {
        return vc::render::ChunkDtype::UInt8;
    }

    double fillValue() const override { return 0.0; }

    LevelTransform levelTransform(int level) const override
    {
        LevelTransform transform;
        if (level == 1)
            transform.scaleFromLevel0 = level1Scale_;
        return transform;
    }

    vc::render::ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        if (level >= 0 && level < numLevels())
            ++queuedLookups[level];
        return chunkResult(level, iz, iy, ix);
    }

    vc::render::ChunkResult getChunkIfCached(
        int level, int iz, int iy, int ix) override
    {
        if (level >= 0 && level < numLevels())
            ++cachedLookups[level];
        return chunkResult(level, iz, iy, ix);
    }

    std::array<int, 2> queuedLookups{};
    std::array<int, 2> cachedLookups{};
    int blockingLookups=0,prefetchCalls=0;

private:
    vc::render::ChunkResult chunkResult(int level, int iz, int iy, int ix)
    {
        vc::render::ChunkResult result;
        result.dtype = vc::render::ChunkDtype::UInt8;
        if (level < 0 || level >= numLevels() || iz != 0 || iy != 0 || ix != 0) {
            result.status = vc::render::ChunkStatus::MissQueued;
            return result;
        }

        result.status = statuses_[level];
        result.shape = shape(level);
        if (result.status == vc::render::ChunkStatus::Data) {
            const auto dims = chunkShape(level);
            auto bytes = std::make_shared<std::vector<std::byte>>(
                std::size_t(dims[0]) * std::size_t(dims[1]) * std::size_t(dims[2]),
                std::byte{values_[level]});
            result.bytes = std::move(bytes);
        }
        return result;
    }

public:
    vc::render::ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        ++blockingLookups;
        return chunkResult(level, iz, iy, ix);
    }

    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override {++prefetchCalls;}

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override
    {
        return 0;
    }

    void removeChunkReadyListener(ChunkReadyCallbackId) override {}

private:
    std::array<vc::render::ChunkStatus, 2> statuses_;
    std::array<uint8_t, 2> values_;
    std::array<std::array<int, 3>, 2> shapes_;
    std::array<std::array<int, 3>, 2> chunkShapes_;
    std::array<double, 3> level1Scale_;
};

class ProceduralChunkedArray final : public vc::render::IChunkedArray {
public:
    using MissingKey = std::tuple<int, int, int>;

    explicit ProceduralChunkedArray(
        std::array<int, 3> shape = {11, 13, 17},
        std::array<int, 3> chunkShape = {3, 5, 7},
        std::optional<MissingKey> missing = std::nullopt,
        bool transformed = false)
        : shape_(shape), chunkShape_(chunkShape), missing_(missing), transformed_(transformed)
    {
    }

    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return shape_; }
    std::array<int, 3> chunkShape(int) const override { return chunkShape_; }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 19.0; }
    LevelTransform levelTransform(int) const override
    {
        LevelTransform transform;
        if (transformed_) {
            transform.scaleFromLevel0 = {0.75, 0.5, 1.25};
            transform.offsetFromLevel0 = {0.125, 0.25, 0.375};
        }
        return transform;
    }
    vc::render::ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        return chunkResult(level, iz, iy, ix);
    }
    vc::render::ChunkResult getChunkIfCached(int level, int iz, int iy, int ix) override
    {
        return chunkResult(level, iz, iy, ix);
    }
    vc::render::ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return chunkResult(level, iz, iy, ix);
    }
    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}

private:
    vc::render::ChunkResult chunkResult(int level, int cz, int cy, int cx) const
    {
        vc::render::ChunkResult result;
        result.dtype = vc::render::ChunkDtype::UInt8;
        result.shape = chunkShape_;
        if (level != 0 || cz < 0 || cy < 0 || cx < 0 ||
            cz * chunkShape_[0] >= shape_[0] ||
            cy * chunkShape_[1] >= shape_[1] ||
            cx * chunkShape_[2] >= shape_[2]) {
            result.status = vc::render::ChunkStatus::Missing;
            return result;
        }
        if (missing_ && *missing_ == MissingKey{cz, cy, cx}) {
            result.status = vc::render::ChunkStatus::Missing;
            return result;
        }
        result.status = vc::render::ChunkStatus::Data;
        auto bytes = std::make_shared<std::vector<std::byte>>(
            std::size_t(chunkShape_[0]) * std::size_t(chunkShape_[1]) *
                std::size_t(chunkShape_[2]));
        for (int z = 0; z < chunkShape_[0]; ++z) {
            for (int y = 0; y < chunkShape_[1]; ++y) {
                for (int x = 0; x < chunkShape_[2]; ++x) {
                    const int gz = cz * chunkShape_[0] + z;
                    const int gy = cy * chunkShape_[1] + y;
                    const int gx = cx * chunkShape_[2] + x;
                    const auto offset = (std::size_t(z) * chunkShape_[1] + y) *
                        chunkShape_[2] + x;
                    (*bytes)[offset] = std::byte(
                        static_cast<uint8_t>((gz * 37 + gy * 11 + gx * 3) & 255));
                }
            }
        }
        result.bytes = std::move(bytes);
        return result;
    }

    std::array<int, 3> shape_;
    std::array<int, 3> chunkShape_;
    std::optional<MissingKey> missing_;
    bool transformed_ = false;
};

cv::Mat_<cv::Vec3f> singleCoord(const cv::Vec3f& coord)
{
    cv::Mat_<cv::Vec3f> coords(1, 1);
    coords(0, 0) = coord;
    return coords;
}

std::vector<vc::render::ChunkKey> sortedKeys(std::vector<vc::render::ChunkKey> keys)
{
    std::sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) {
        return std::tie(a.level, a.iz, a.iy, a.ix) <
               std::tie(b.level, b.iz, b.iy, b.ix);
    });
    return keys;
}

} // namespace

TEST_CASE("Pinned sampling owns payloads and never queues an unexpected dependency")
{
    using namespace vc::render;
    auto array=std::make_shared<PyramidChunkedArray>(ChunkStatus::Data,17,ChunkStatus::Data,9);
    PinnedChunkRegion region;
    region.pin(array,{{0,0,0,0}});
    region.seal();
    CHECK(region.bytes()==64);
    const auto* expected=&region.get(array.get(),0,0,0,0);
    {
        PinnedChunkLookup local(region, array.get());
        CHECK(&local.get({0,0,0,0})==expected);
        CHECK(&local.get({0,0,0,0})==expected); // last-chunk fast path
        CHECK(array->queuedLookups[0]==0);
        CHECK(array->cachedLookups[0]==0);
        CHECK_THROWS_AS(local.get({0,0,0,1}),PinnedRegionMiss);
        CHECK_THROWS_AS(region.pin(array,{{0,0,0,0}}),std::logic_error);
    }
    CHECK(std::to_integer<int>((*expected->bytes)[0])==17);
    CHECK_THROWS_AS(region.pin(array,{{0,0,0,0}}),std::logic_error);
    {
        int visits=0;
        const auto visitor=+[](void* context,size_t,const cv::Vec3f&,bool valid,
            std::span<const std::array<uint8_t,8>> corners) {
            ++*static_cast<int*>(context);
            CHECK(valid);
            REQUIRE(corners.size()==1);
            for(auto value:corners[0]) CHECK(value==17);
        };
        (void)ChunkedPlaneSampler::visitTrilinearCornersLevelBlockingRequestedLevel(
            {array.get()},0,{{1.25f,1.5f,1.75f}},&visits,visitor,1,true,{},&region);
        CHECK(visits==1);
        CHECK(array->blockingLookups==1); // preparation only
        CHECK(array->prefetchCalls==1); // preparation only
    }
}

TEST_CASE("Local acquisition variants reuse consecutive and revisited chunks")
{
    using namespace vc::render;
    auto array = std::make_shared<PyramidChunkedArray>(
        ChunkStatus::Data, 17, ChunkStatus::Data, 9);
    const ChunkKey a{0, 0, 0, 0}, b{1, 0, 0, 0};
    int requested = 0, errors = 0;
    const auto exercise = [&](auto& cache) {
        const auto first = cache.get(a, requested, errors).bytes;
        CHECK(cache.get(a, requested, errors).bytes == first);
        CHECK(std::to_integer<int>((*cache.get(b, requested, errors).bytes)[0]) == 9);
        CHECK(cache.get(a, requested, errors).bytes == first);
    };
    LocalChunkCache queued(*array);
    exercise(queued);
    CHECK(array->queuedLookups[0] == 1);
    CHECK(array->queuedLookups[1] == 1);
    LocalChunkCache resident(*array, 8, false);
    exercise(resident);
    CHECK(array->cachedLookups[0] == 1);
    CHECK(array->cachedLookups[1] == 1);
    BlockingLocalChunkCache blocking(*array);
    exercise(blocking);
    CHECK(array->blockingLookups == 2);

    PinnedChunkRegion region;
    region.pin(array, {a, b});
    CHECK_THROWS_AS(PinnedChunkLookup(region, array.get()), std::logic_error);
    region.seal();
    CHECK_THROWS_AS(PinnedChunkLookup(region, nullptr), std::logic_error);
    PinnedChunkLookup pinned(region, array.get());
    const auto* first = &pinned.get(a);
    const auto ownershipCount = first->bytes.use_count();
    CHECK(&pinned.get(a) == first);
    CHECK(std::to_integer<int>((*pinned.get(b).bytes)[0]) == 9);
    CHECK(&pinned.get(a) == first);
    CHECK(first->bytes.use_count() == ownershipCount); // borrows, never copies
    CHECK(array->blockingLookups == 4); // only region preparation
    CHECK(array->queuedLookups[0] == 1);
    CHECK(array->cachedLookups[0] == 1);
    // Explicit pins must never intercept ordinary cache access.
    LocalChunkCache independent(*array);
    CHECK(independent.get(a, requested, errors).bytes != first->bytes);
    CHECK(array->queuedLookups[0] == 2);
}

TEST_CASE("Explicit pinned corner batches match ordinary batches across chunks and workers")
{
    using namespace vc::render;
    auto array = std::make_shared<ProceduralChunkedArray>();
    auto other = std::make_shared<ProceduralChunkedArray>();
    PinnedChunkRegion region;
    std::vector<ChunkKey> keys;
    const auto shape = array->shape(0), chunk = array->chunkShape(0);
    for (int z = 0; z < shape[0]; z += chunk[0])
        for (int y = 0; y < shape[1]; y += chunk[1])
            for (int x = 0; x < shape[2]; x += chunk[2])
                keys.push_back({0, z / chunk[0], y / chunk[1], x / chunk[2]});
    region.pin(array, keys);
    region.pin(other, keys);
    region.seal();
    std::vector<cv::Vec3f> points;
    for (int z = 0; z < shape[0]; ++z)
        for (int y = 0; y < shape[1]; ++y)
            for (int x = 0; x < shape[2]; ++x)
                points.emplace_back(x + .25f, y + .5f, z + .75f);
    using Corners = std::vector<std::array<uint8_t, 8>>;
    struct Output {
        std::vector<Corners> corners;
        std::vector<cv::Vec3f> fractions;
        std::vector<uint8_t> valid;
        explicit Output(size_t n) : corners(n), fractions(n), valid(n) {}
    };
    const auto visitor = +[](void* context, size_t i, const cv::Vec3f& fraction,
                             bool valid, std::span<const std::array<uint8_t, 8>> corners) {
        auto& output = *static_cast<Output*>(context);
        output.corners[i].assign(corners.begin(), corners.end());
        output.fractions[i] = fraction;
        output.valid[i] = valid;
    };
    Output expected(points.size());
    (void)ChunkedPlaneSampler::visitTrilinearCornersLevelBlockingRequestedLevel(
        {array.get(), other.get()}, 0, points, &expected, visitor, 1);
    for (int workers : {1, 4}) {
        Output actual(points.size());
        (void)ChunkedPlaneSampler::visitTrilinearCornersLevelBlockingRequestedLevel(
            {array.get(), other.get()}, 0, points, &actual, visitor, workers, true, {}, &region);
        CHECK(actual.corners == expected.corners);
        CHECK(actual.fractions == expected.fractions);
        CHECK(actual.valid == expected.valid);
    }
}

TEST_CASE("ChunkedPlaneSampler selects one analytic source level from camera scale")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Data, 1,
                              vc::render::ChunkStatus::Data, 2);
    CHECK(vc::render::ChunkedPlaneSampler::maximumBaseVoxelExtent(array, 0) ==
          doctest::Approx(1.0));
    CHECK(vc::render::ChunkedPlaneSampler::maximumBaseVoxelExtent(array, 1) ==
          doctest::Approx(2.0));
    CHECK(vc::render::ChunkedPlaneSampler::sourceLevelForView(array, 2.0f) == 0);
    CHECK(vc::render::ChunkedPlaneSampler::sourceLevelForView(array, 1.0f) == 1);
    CHECK(vc::render::ChunkedPlaneSampler::sourceLevelForView(array, 0.25f) == 1);

    PyramidChunkedArray anisotropic(
        vc::render::ChunkStatus::Data, 1,
        vc::render::ChunkStatus::Data, 2,
        {4, 4, 4}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0},
        {0.5, 0.25, 0.5});
    CHECK(vc::render::ChunkedPlaneSampler::maximumBaseVoxelExtent(anisotropic, 1) ==
          doctest::Approx(4.0));
    CHECK(vc::render::ChunkedPlaneSampler::sourceLevelForView(anisotropic, 1.0f) == 0);
    CHECK(vc::render::ChunkedPlaneSampler::sourceLevelForView(anisotropic, 0.5f) == 1);
}

TEST_CASE("ThreadPool indexed batch visits every index once")
{
    utils::ThreadPool pool(4);
    std::array<std::atomic<int>, 17> visits{};

    pool.run_indexed_batch(visits.size(), [&](size_t index) {
        visits[index].fetch_add(1, std::memory_order_relaxed);
    });

    for (const auto& count : visits)
        CHECK(count.load(std::memory_order_relaxed) == 1);
}

TEST_CASE("ThreadPool indexed batch propagates worker exceptions")
{
    utils::ThreadPool pool(4);

    CHECK_THROWS_WITH_AS(
        pool.run_indexed_batch(8, [](size_t index) {
            if (index == 3)
                throw std::runtime_error("indexed batch failure");
        }),
        doctest::Contains("indexed batch failure"),
        std::runtime_error);
}

TEST_CASE("ChunkedPlaneSampler fine-to-coarse fills missing high-res from coarse level")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Missing, 0,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 42);
}

TEST_CASE("ChunkedPlaneSampler fine-to-coarse fills queued high-res from coarse level")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::MissQueued, 0,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 42);
}

TEST_CASE("ChunkedPlaneSampler fine-to-coarse keeps high-res value when present")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Data, 7,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 7);
}

TEST_CASE("ChunkedPlaneSampler fine-to-coarse skips empty high-res scale")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::AllFill, 0,
                              vc::render::ChunkStatus::Data, 42,
                              {0, 0, 0}, {2, 2, 2});
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 42);
}

TEST_CASE("ChunkedPlaneSampler coarse-to-fine lets ready high-res overwrite coarse preview")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Data, 7,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsCoarseToFine(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 7);
}

TEST_CASE("ChunkedPlaneSampler coarse-to-fine keeps coarse preview when high-res scale is empty")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::AllFill, 0,
                              vc::render::ChunkStatus::Data, 42,
                              {0, 0, 0}, {2, 2, 2});
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsCoarseToFine(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 42);
}

TEST_CASE("ChunkedPlaneSampler fallback leaves sentinel surface coords uncovered")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Data, 7,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({0.0f, 0.0f, 0.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 0);
    CHECK(out(0, 0) == 0);
}

TEST_CASE("ChunkedPlaneSampler base and overlay buffers fall back independently")
{
    PyramidChunkedArray baseArray(vc::render::ChunkStatus::Missing, 0,
                                  vc::render::ChunkStatus::Data, 11);
    PyramidChunkedArray overlayArray(vc::render::ChunkStatus::Data, 99,
                                     vc::render::ChunkStatus::Data, 22);
    cv::Mat_<uint8_t> baseOut(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> baseCoverage(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> overlayOut(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> overlayCoverage(1, 1, uint8_t(0));
    const cv::Mat_<cv::Vec3f> coords = singleCoord({1.0f, 1.0f, 1.0f});

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        baseArray, 0, coords, baseOut, baseCoverage, {vc::Sampling::Nearest, 1});
    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        overlayArray, 0, coords, overlayOut, overlayCoverage, {vc::Sampling::Nearest, 1});

    CHECK(baseCoverage(0, 0) == 1);
    CHECK(baseOut(0, 0) == 11);
    CHECK(overlayCoverage(0, 0) == 1);
    CHECK(overlayOut(0, 0) == 99);
}

TEST_CASE("requested-level locality path matches scalar reference on transformed non-power-of-two chunks")
{
    ProceduralChunkedArray array({11, 13, 17}, {3, 5, 7}, std::nullopt, true);
    cv::Mat_<cv::Vec3f> coords(8, 16);
    const std::array<cv::Vec3f, 24> points{{
        {1.20f, 1.30f, 1.40f}, {1.80f, 1.30f, 1.40f},
        {2.20f, 1.30f, 1.40f}, {2.80f, 1.30f, 1.40f},
        {6.20f, 4.20f, 2.20f}, {6.80f, 4.80f, 2.80f},
        {7.20f, 5.20f, 3.20f}, {7.80f, 5.80f, 3.80f},
        {15.75f, 11.25f, 9.50f}, {16.00f, 12.00f, 10.00f},
        {-0.25f, 2.0f, 2.0f}, {17.25f, 2.0f, 2.0f},
        {0.0f, 0.0f, 0.0f},
        {std::numeric_limits<float>::quiet_NaN(), 2.0f, 2.0f},
        {std::numeric_limits<float>::infinity(), 2.0f, 2.0f},
        {3.10f, 9.90f, 5.90f}, {3.20f, 9.80f, 5.80f},
        {3.30f, 9.70f, 5.70f}, {13.95f, 4.95f, 8.95f},
        {14.05f, 5.05f, 9.05f}, {8.5f, 7.5f, 4.5f},
        {8.6f, 7.6f, 4.6f}, {8.7f, 7.7f, 4.7f}, {8.8f, 7.8f, 4.8f}}};
    uint32_t state = 0x6d2b79f5U;
    const auto randomUnit = [&] {
        state = state * 1664525U + 1013904223U;
        return float(state >> 8U) / float(uint32_t{1} << 24U);
    };
    for (auto& point : coords) {
        point = {randomUnit() * 21.0f - 2.0f,
                 randomUnit() * 17.0f - 2.0f,
                 randomUnit() * 15.0f - 2.0f};
    }
    std::copy(points.begin(), points.end(), coords.begin());
    cv::Mat_<uint8_t> initialCoverage(coords.rows, coords.cols, uint8_t{0});
    initialCoverage(0, 3) = 1;

    for (const vc::Sampling sampling : {vc::Sampling::Nearest, vc::Sampling::Trilinear}) {
        vc::render::ChunkedPlaneSampler::Options reference(sampling, 4);
        reference.parallelThreads = 1;
        reference.chunkLocalityFastPath = false;
        reference.collectLocalityStats = true;
        auto optimized = reference;
        optimized.chunkLocalityFastPath = true;

        CHECK(sortedKeys(vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
                  array, 0, coords, initialCoverage, reference)) ==
              sortedKeys(vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
                  array, 0, coords, initialCoverage, optimized)));

        cv::Mat_<uint8_t> referenceOut(coords.rows, coords.cols, uint8_t{73});
        cv::Mat_<uint8_t> optimizedOut = referenceOut.clone();
        cv::Mat_<uint8_t> referenceCoverage = initialCoverage.clone();
        cv::Mat_<uint8_t> optimizedCoverage = initialCoverage.clone();
        const auto referenceStats =
            vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
                array, 0, coords, referenceOut, referenceCoverage, reference);
        const auto optimizedStats =
            vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
                array, 0, coords, optimizedOut, optimizedCoverage, optimized);
        CHECK(cv::countNonZero(referenceOut != optimizedOut) == 0);
        CHECK(cv::countNonZero(referenceCoverage != optimizedCoverage) == 0);
        CHECK(referenceStats.requestedChunks == optimizedStats.requestedChunks);
        CHECK(referenceStats.missingChunks == optimizedStats.missingChunks);
        CHECK(referenceStats.errorChunks == optimizedStats.errorChunks);
        if (sampling == vc::Sampling::Trilinear) {
            CHECK(optimizedStats.sampleDirectGatherPoints > 0);
            CHECK(optimizedStats.samplePinnedMapLookups < referenceStats.samplePinnedMapLookups);
            CHECK(optimizedStats.dependencyKeyLookups < referenceStats.dependencyKeyLookups);
        }
    }
}

TEST_CASE("requested-level locality path preserves missing-chunk scalar policy")
{
    ProceduralChunkedArray array({11, 13, 17}, {3, 5, 7},
                                 ProceduralChunkedArray::MissingKey{1, 1, 1});
    cv::Mat_<cv::Vec3f> coords(1, 3);
    coords(0, 0) = {7.2f, 5.2f, 3.2f};
    coords(0, 1) = {7.8f, 5.8f, 3.8f};
    coords(0, 2) = {1.2f, 1.2f, 1.2f};
    vc::render::ChunkedPlaneSampler::Options reference(vc::Sampling::Trilinear, 2);
    reference.parallelThreads = 1;
    reference.chunkLocalityFastPath = false;
    auto optimized = reference;
    optimized.chunkLocalityFastPath = true;
    cv::Mat_<uint8_t> referenceOut(1, 3, uint8_t{99});
    cv::Mat_<uint8_t> optimizedOut = referenceOut.clone();
    cv::Mat_<uint8_t> referenceCoverage(1, 3, uint8_t{0});
    cv::Mat_<uint8_t> optimizedCoverage = referenceCoverage.clone();
    const auto referenceStats =
        vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
            array, 0, coords, referenceOut, referenceCoverage, reference);
    const auto optimizedStats =
        vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
            array, 0, coords, optimizedOut, optimizedCoverage, optimized);
    CHECK(cv::countNonZero(referenceOut != optimizedOut) == 0);
    CHECK(cv::countNonZero(referenceCoverage != optimizedCoverage) == 0);
    CHECK(referenceStats.missingChunks == optimizedStats.missingChunks);
    CHECK(referenceOut(0, 0) == 0);
    CHECK(referenceOut(0, 1) == 0);
}

TEST_CASE("ChunkedPlaneSampler blocking requested-level does not fall back to coarse level")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Missing, 0,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(123));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    const auto stats = vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 0);
    CHECK(stats.requestedLevelOnly);
    CHECK(stats.fallbackLevels == 0);
    CHECK(stats.requestedChunks == 1);
    CHECK(stats.missingChunks == 1);
}

TEST_CASE("ChunkedPlaneSampler blocking requested-level keeps requested-level data")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Data, 7,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    const auto stats = vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 7);
    CHECK(stats.requestedLevelOnly);
    CHECK(stats.fallbackLevels == 0);
    CHECK(stats.missingChunks == 0);
}

TEST_CASE("ChunkedPlaneSampler coordinate worker limit preserves blocking samples")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Data, 7,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<cv::Vec3f> coords(128, 128, cv::Vec3f{1.25f, 1.25f, 1.25f});
    cv::Mat_<uint8_t> serialOut(128, 128, uint8_t(0));
    cv::Mat_<uint8_t> serialCoverage(128, 128, uint8_t(0));
    cv::Mat_<uint8_t> boundedOut(128, 128, uint8_t(0));
    cv::Mat_<uint8_t> boundedCoverage(128, 128, uint8_t(0));

    vc::render::ChunkedPlaneSampler::Options serialOptions(
        vc::Sampling::Trilinear, 32);
    serialOptions.parallelThreads = 1;
    const auto serialStats =
        vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
            array, 0, coords, serialOut, serialCoverage, serialOptions);

    vc::render::ChunkedPlaneSampler::Options boundedOptions(
        vc::Sampling::Trilinear, 32);
    boundedOptions.parallelThreads = 3;
    const auto boundedStats =
        vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
            array, 0, coords, boundedOut, boundedCoverage, boundedOptions);

    CHECK(serialStats.sampleWorkers == 1);
    CHECK(boundedStats.sampleWorkers >= 1);
    CHECK(boundedStats.sampleWorkers <= 3);
    CHECK(serialStats.requestedLevelOnly);
    CHECK(boundedStats.requestedLevelOnly);
    CHECK(serialStats.dependencySeconds >= 0.0);
    CHECK(serialStats.chunkResolveSeconds >= 0.0);
    CHECK(serialStats.sampleSeconds > 0.0);
    CHECK(cv::countNonZero(serialOut != boundedOut) == 0);
    CHECK(cv::countNonZero(serialCoverage != boundedCoverage) == 0);
}

TEST_CASE("ChunkedPlaneSampler blocking requested-level rejects unresolved chunks")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::MissQueued, 0,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    CHECK_THROWS_AS(
        vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
            array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
            {vc::Sampling::Nearest, 1}),
        std::runtime_error);
}

TEST_CASE("ChunkedPlaneSampler grouped corner batch shares geometry across arrays")
{
    PyramidChunkedArray first(vc::render::ChunkStatus::Data, 17,
                              vc::render::ChunkStatus::Data, 0);
    PyramidChunkedArray second(vc::render::ChunkStatus::Data, 93,
                               vc::render::ChunkStatus::Data, 0);
    std::vector<vc::render::IChunkedArray*> arrays{&first, &second};
    std::vector<std::vector<std::array<uint8_t, 8>>> values;
    std::vector<cv::Vec3f> fractions;
    std::vector<uint8_t> valid;

    const auto stats =
        vc::render::ChunkedPlaneSampler::sampleTrilinearCornersLevelBlockingRequestedLevel(
            arrays, 0, {{1.25f, 1.5f, 1.75f}}, values, fractions, valid);

    REQUIRE(values.size() == 2);
    REQUIRE(values[0].size() == 1);
    CHECK(std::all_of(values[0][0].begin(), values[0][0].end(),
                      [](uint8_t value) { return value == 17; }));
    CHECK(std::all_of(values[1][0].begin(), values[1][0].end(),
                      [](uint8_t value) { return value == 93; }));
    REQUIRE(fractions.size() == 1);
    CHECK(fractions[0][0] == doctest::Approx(0.25f));
    CHECK(fractions[0][1] == doctest::Approx(0.5f));
    CHECK(fractions[0][2] == doctest::Approx(0.75f));
    CHECK(valid == std::vector<uint8_t>{1});
    CHECK(stats.requestedChunks == 2);
}

TEST_CASE("ChunkedPlaneSampler grouped corner batch supports mixed chunk grids")
{
    PyramidChunkedArray coarseChunks(
        vc::render::ChunkStatus::Data, 17,
        vc::render::ChunkStatus::Data, 0,
        {4, 4, 4}, {2, 2, 2}, {4, 4, 4});
    PyramidChunkedArray fineChunks(
        vc::render::ChunkStatus::Data, 93,
        vc::render::ChunkStatus::Data, 0,
        {4, 4, 4}, {2, 2, 2}, {2, 2, 2});
    std::vector<vc::render::IChunkedArray*> arrays{&coarseChunks, &fineChunks};
    std::vector<std::vector<std::array<uint8_t, 8>>> values;
    std::vector<cv::Vec3f> fractions;
    std::vector<uint8_t> valid;

    const auto stats =
        vc::render::ChunkedPlaneSampler::sampleTrilinearCornersLevelBlockingRequestedLevel(
            arrays, 0, {{0.25f, 0.5f, 0.75f}}, values, fractions, valid);

    CHECK(valid == std::vector<uint8_t>{1});
    CHECK(std::all_of(values[0][0].begin(), values[0][0].end(),
                      [](uint8_t value) { return value == 17; }));
    CHECK(std::all_of(values[1][0].begin(), values[1][0].end(),
                      [](uint8_t value) { return value == 93; }));
    CHECK(stats.requestedChunks == 2);
}

TEST_CASE("ChunkedPlaneSampler corner visitor preserves grouped corner semantics")
{
    PyramidChunkedArray coarseChunks(
        vc::render::ChunkStatus::Data, 17,
        vc::render::ChunkStatus::Data, 0,
        {4, 4, 4}, {2, 2, 2}, {4, 4, 4});
    PyramidChunkedArray fineChunks(
        vc::render::ChunkStatus::Data, 93,
        vc::render::ChunkStatus::Data, 0,
        {4, 4, 4}, {2, 2, 2}, {2, 2, 2});
    std::vector<vc::render::IChunkedArray*> arrays{&coarseChunks, &fineChunks};
    const std::vector<cv::Vec3f> points{
        {0.25f, 0.5f, 0.75f},
        {0.75f, 0.25f, 0.5f},
        {-1.0f, 0.0f, 0.0f},
    };
    struct VisitorOutput {
        std::vector<std::vector<std::array<uint8_t, 8>>> values;
        std::vector<cv::Vec3f> fractions;
        std::vector<uint8_t> valid;
        std::vector<int> visits;
        std::vector<size_t> cornerVolumes;
    } output{
        std::vector<std::vector<std::array<uint8_t, 8>>>(
            arrays.size(),
            std::vector<std::array<uint8_t, 8>>(points.size())),
        std::vector<cv::Vec3f>(points.size()),
        std::vector<uint8_t>(points.size()),
        std::vector<int>(points.size()),
        std::vector<size_t>(points.size()),
    };
    const auto visitor = +[](
        void* rawOutput,
        size_t pointIndex,
        const cv::Vec3f& fraction,
        bool valid,
        std::span<const std::array<uint8_t, 8>> volumeCorners) {
        auto& out = *static_cast<VisitorOutput*>(rawOutput);
        ++out.visits[pointIndex];
        out.fractions[pointIndex] = fraction;
        out.valid[pointIndex] = valid ? uint8_t{1} : uint8_t{0};
        out.cornerVolumes[pointIndex] = volumeCorners.size();
        if (!valid)
            return;
        for (size_t volumeIndex = 0; volumeIndex < volumeCorners.size(); ++volumeIndex)
            out.values[volumeIndex][pointIndex] = volumeCorners[volumeIndex];
    };

    const auto stats =
        vc::render::ChunkedPlaneSampler::visitTrilinearCornersLevelBlockingRequestedLevel(
            arrays, 0, points, &output, visitor, 2, true);

    CHECK(output.visits == std::vector<int>{1, 1, 1});
    CHECK(output.valid == std::vector<uint8_t>{1, 1, 0});
    CHECK(output.cornerVolumes == std::vector<size_t>{2, 2, 0});
    CHECK(output.fractions[0][0] == doctest::Approx(0.25f));
    CHECK(output.fractions[0][1] == doctest::Approx(0.5f));
    CHECK(output.fractions[0][2] == doctest::Approx(0.75f));
    CHECK(std::all_of(
        output.values[0][0].begin(),
        output.values[0][0].end(),
        [](uint8_t value) { return value == 17; }));
    CHECK(std::all_of(
        output.values[1][0].begin(),
        output.values[1][0].end(),
        [](uint8_t value) { return value == 93; }));
    CHECK(stats.requestedChunks == 2);
    CHECK(stats.cornerPointCount == 2);
    CHECK(stats.cornerUniqueVoxelCubes == 1);
    CHECK(stats.cornerMaxCandidatesPerCube == 2);
    CHECK(stats.cornerCubeOccupancyHistogram[2] == 1);
    CHECK(stats.cornerWorkerTasks > 0);
    CHECK(!stats.cornerDependencyIds.empty());
}

TEST_CASE("ChunkedPlaneSampler can use fallback without queueing or promoting it")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::MissQueued, 0,
                              vc::render::ChunkStatus::MissQueued, 0);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));
    vc::render::ChunkedPlaneSampler::Options options(vc::Sampling::Nearest, 1);
    options.queuedFallbackLevels = 0;

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage, options);

    CHECK(array.queuedLookups[0] > 0);
    CHECK(array.queuedLookups[1] == 0);
    CHECK(array.cachedLookups[1] > 0);
}

TEST_CASE("ChunkedPlaneSampler can bound transition work to one fallback level")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::MissQueued, 0,
                              vc::render::ChunkStatus::MissQueued, 0);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));
    vc::render::ChunkedPlaneSampler::Options options(vc::Sampling::Nearest, 1);
    options.queuedFallbackLevels = 1;

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage, options);

    CHECK(array.queuedLookups[0] > 0);
    CHECK(array.queuedLookups[1] > 0);
    CHECK(array.cachedLookups[1] == 0);
}
