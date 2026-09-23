// SurfaceCache tile fills must stay bounded when a surface's geometry streaks
// across the volume, and must not report such tiles as incomplete (which makes
// requestView refill them on every chunk arrival).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/render/SurfaceCache.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using vc::render::ChunkDtype;
using vc::render::ChunkKey;
using vc::render::ChunkResult;
using vc::render::ChunkStatus;
using vc::render::SurfaceCache;

namespace {

constexpr int kVolumeEdge = 8192;
constexpr int kChunkEdge = 128;  // 2 MiB uint8 chunks, as in the s1 volumes

// Every chunk is resident and uniform, so a fill can only be incomplete for
// reasons other than missing data. Records how much each fill prefetches.
class UniformArray : public vc::render::IChunkedArray {
public:
    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override
    {
        return {kVolumeEdge, kVolumeEdge, kVolumeEdge};
    }
    std::array<int, 3> chunkShape(int) const override
    {
        return {kChunkEdge, kChunkEdge, kChunkEdge};
    }
    ChunkDtype dtype() const override { return ChunkDtype::UInt8; }
    double fillValue() const override { return 7.0; }
    LevelTransform levelTransform(int) const override { return {}; }

    ChunkResult tryGetChunk(int, int, int, int) override
    {
        ChunkResult r;
        r.dtype = ChunkDtype::UInt8;
        r.status = ChunkStatus::AllFill;
        r.shape = chunkShape(0);
        return r;
    }
    ChunkResult getChunkIfCached(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }
    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }
    void prefetchChunks(const std::vector<ChunkKey>& keys, bool, int) override
    {
        prefetchedKeys += keys.size();
    }
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 1; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}

    std::atomic<std::size_t> prefetchedKeys{0};
};

// Valid vertices scattered over the whole volume: every interpolated pixel's
// coordinate and normal band lands somewhere unrelated to its neighbours.
cv::Mat_<cv::Vec3f> scatteredGrid(int rows, int cols)
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(200.0f, float(kVolumeEdge - 200));
    cv::Mat_<cv::Vec3f> points(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            points(r, c) = cv::Vec3f(u(rng), u(rng), u(rng));
    return points;
}

// One level-0 tile, (0, 0), covers surface pixels [0, 128) x [0, 128).
void fillOneTile(SurfaceCache& cache)
{
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false;
    const auto id = cache.addTileReadyListener([&] {
        std::lock_guard lock(mutex);
        ready = true;
        cv.notify_all();
    });
    cache.requestView(0, 1.0, 1.0, 1.0, 64, 64);
    {
        std::unique_lock lock(mutex);
        REQUIRE(cv.wait_for(lock, std::chrono::seconds(60), [&] { return ready; }));
    }
    cache.removeTileReadyListener(id);
    for (int i = 0; i < 600 && cache.stats().tilesInFlight > 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

}  // namespace

TEST_CASE("a tile whose geometry streaks across the volume fills with bounded prefetch")
{
    auto array = std::make_shared<UniformArray>();
    auto surface = std::make_shared<QuadSurface>(scatteredGrid(300, 300), cv::Vec2f(1.0f, 1.0f));

    SurfaceCache::Options options;
    options.byteCapacity = 256ULL << 20;
    SurfaceCache cache(array, surface, options);
    fillOneTile(cache);

    const auto stats = cache.stats();
    CHECK(stats.tiles >= 1);
    CHECK(stats.tilesInFlight == 0);
    // Pixels beyond the dependency bounds are left uncovered, which refilling
    // cannot change, so the tile is complete and never re-queued.
    CHECK(stats.tilesIncomplete == 0);

    // 512 MiB of 2 MiB chunks per tile fill; unbounded, this geometry asks
    // for tens of thousands.
    const std::size_t perTileLimit = (512ULL << 20) / (std::size_t(kChunkEdge) * kChunkEdge * kChunkEdge);
    CHECK(array->prefetchedKeys.load() <= stats.tiles * perTileLimit);

    // A complete tile is not refilled by a repeated request.
    const std::size_t before = array->prefetchedKeys.load();
    cache.requestView(0, 1.0, 1.0, 1.0, 64, 64);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    CHECK(array->prefetchedKeys.load() == before);
    cache.shutdown();
}

TEST_CASE("a well-formed tile still fills every pixel")
{
    auto array = std::make_shared<UniformArray>();
    cv::Mat_<cv::Vec3f> points(300, 300);
    for (int r = 0; r < points.rows; ++r)
        for (int c = 0; c < points.cols; ++c)
            points(r, c) = cv::Vec3f(1000.0f + float(c), 1000.0f + float(r), 2000.0f);
    auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f(1.0f, 1.0f));

    SurfaceCache::Options options;
    options.byteCapacity = 256ULL << 20;
    SurfaceCache cache(array, surface, options);
    fillOneTile(cache);

    CHECK(cache.stats().tilesIncomplete == 0);
    cv::Mat_<uint8_t> out(64, 64, uint8_t{0});
    cv::Mat_<uint8_t> coverage(64, 64, uint8_t{0});
    const auto sampled = cache.sampleView(0, 1.0, 1.0, 1.0, 0.0, {}, out, coverage);
    CHECK(sampled.coveredPixels == 64 * 64);
    CHECK(out(10, 10) == 7);
    cache.shutdown();
}
