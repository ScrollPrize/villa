// An edit tool (push/pull, drag) moves vertices of the displayed surface and
// invalidates the edited grid region on every tick. The flattened view must
// keep drawing the region while it refills -- never a blank square -- and the
// refills must land while the edit is still running. A full invalidation, as
// an edit without a known region ends in, must refill without the view moving.

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
#include <thread>
#include <vector>

using vc::render::ChunkDtype;
using vc::render::ChunkKey;
using vc::render::ChunkResult;
using vc::render::ChunkStatus;
using vc::render::SurfaceCache;

namespace {

constexpr int kVolumeEdge = 4096;
constexpr int kChunkEdge = 128;
constexpr int kGridEdge = 300;

// Every voxel of chunk row iz holds 10 + iz, so a sample tells which z the
// surface was at when its tile was filled. Prefetches can be slowed down to
// keep fills in flight while edits arrive.
class LayeredArray : public vc::render::IChunkedArray {
public:
    LayeredArray()
    {
        const std::size_t voxels = std::size_t(kChunkEdge) * kChunkEdge * kChunkEdge;
        for (int iz = 0; iz < kVolumeEdge / kChunkEdge; ++iz) {
            _layers.push_back(std::make_shared<const std::vector<std::byte>>(
                voxels, std::byte(10 + iz)));
        }
    }

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
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int) const override { return {}; }

    ChunkResult tryGetChunk(int, int iz, int, int) override
    {
        ChunkResult r;
        r.dtype = ChunkDtype::UInt8;
        r.status = ChunkStatus::Data;
        r.shape = chunkShape(0);
        r.bytes = _layers.at(std::size_t(iz));
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
    void prefetchChunks(const std::vector<ChunkKey>&, bool, int) override
    {
        if (const int ms = prefetchDelayMs.load(); ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 1; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}

    std::atomic<int> prefetchDelayMs{0};

private:
    std::vector<std::shared_ptr<const std::vector<std::byte>>> _layers;
};

// A flat sheet at height z, one grid cell per nominal unit.
cv::Mat_<cv::Vec3f> sheetAt(float z)
{
    cv::Mat_<cv::Vec3f> points(kGridEdge, kGridEdge);
    for (int r = 0; r < kGridEdge; ++r)
        for (int c = 0; c < kGridEdge; ++c)
            points(r, c) = cv::Vec3f(1000.0f + float(c), 1000.0f + float(r), z);
    return points;
}

void moveSheet(QuadSurface& surface, float z)
{
    cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    for (int r = 0; r < points->rows; ++r)
        for (int c = 0; c < points->cols; ++c)
            (*points)(r, c)[2] = z;
    surface.invalidateCache();
}

constexpr double kViewMin = 1.0;
constexpr int kViewEdge = 64;  // inside level-0 tile (0, 0)

struct Frame {
    int covered = 0;
    uint8_t value = 0;
};

// What the renderer does each frame: request the view's tiles, then draw from
// whatever is resident.
Frame renderFrame(SurfaceCache& cache)
{
    cache.requestView(0, kViewMin, kViewMin, 1.0, kViewEdge, kViewEdge);
    cv::Mat_<uint8_t> out(kViewEdge, kViewEdge, uint8_t{0});
    cv::Mat_<uint8_t> coverage(kViewEdge, kViewEdge, uint8_t{0});
    const auto stats =
        cache.sampleView(0, kViewMin, kViewMin, 1.0, 0.0, {}, out, coverage);
    return {stats.coveredPixels, out(kViewEdge / 2, kViewEdge / 2)};
}

template <typename Done>
bool renderUntil(SurfaceCache& cache, std::chrono::milliseconds timeout, Done done)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (done(renderFrame(cache)))
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}

const cv::Rect kWholeGrid(0, 0, kGridEdge, kGridEdge);
constexpr uint8_t kAtZ2000 = 10 + 2000 / kChunkEdge;
constexpr uint8_t kAtZ3000 = 10 + 3000 / kChunkEdge;

}  // namespace

TEST_CASE("an edited region keeps drawing its tiles until the refill replaces them")
{
    auto array = std::make_shared<LayeredArray>();
    auto surface = std::make_shared<QuadSurface>(sheetAt(2000.0f), cv::Vec2f(1.0f, 1.0f));
    SurfaceCache::Options options;
    options.byteCapacity = 256ULL << 20;
    SurfaceCache cache(array, surface, options);

    REQUIRE(renderUntil(cache, std::chrono::seconds(30), [](const Frame& f) {
        return f.covered == kViewEdge * kViewEdge && f.value == kAtZ2000;
    }));

    // Hold the refill so the frame right after the edit has to draw from what
    // was resident before it.
    array->prefetchDelayMs = 200;
    moveSheet(*surface, 3000.0f);
    cache.invalidateSurfaceRegion(kWholeGrid);

    const Frame afterEdit = renderFrame(cache);
    CHECK(afterEdit.covered == kViewEdge * kViewEdge);
    CHECK(afterEdit.value == kAtZ2000);

    array->prefetchDelayMs = 0;
    CHECK(renderUntil(cache, std::chrono::seconds(30), [](const Frame& f) {
        return f.covered == kViewEdge * kViewEdge && f.value == kAtZ3000;
    }));
    cache.shutdown();
}

TEST_CASE("refills land while an edit keeps invalidating the region")
{
    auto array = std::make_shared<LayeredArray>();
    auto surface = std::make_shared<QuadSurface>(sheetAt(2000.0f), cv::Vec2f(1.0f, 1.0f));
    SurfaceCache::Options options;
    options.byteCapacity = 256ULL << 20;
    SurfaceCache cache(array, surface, options);

    REQUIRE(renderUntil(cache, std::chrono::seconds(30), [](const Frame& f) {
        return f.covered == kViewEdge * kViewEdge && f.value == kAtZ2000;
    }));

    // A fill now takes longer than the edit tick, as on a real volume. The
    // edit moves the sheet once, then keeps re-invalidating, as push/pull does
    // every tick while its key is held.
    array->prefetchDelayMs = 40;
    moveSheet(*surface, 3000.0f);
    cache.invalidateSurfaceRegion(kWholeGrid);

    bool sawBlank = false;
    bool sawMoved = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!sawMoved && std::chrono::steady_clock::now() < deadline) {
        cache.invalidateSurfaceRegion(kWholeGrid);
        const Frame frame = renderFrame(cache);
        sawBlank |= frame.covered != kViewEdge * kViewEdge;
        sawMoved = frame.value == kAtZ3000;
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    CHECK_FALSE(sawBlank);
    CHECK(sawMoved);

    array->prefetchDelayMs = 0;
    cache.shutdown();
}

TEST_CASE("a full invalidation during a fill refills without the view moving")
{
    auto array = std::make_shared<LayeredArray>();
    auto surface = std::make_shared<QuadSurface>(sheetAt(2000.0f), cv::Vec2f(1.0f, 1.0f));
    SurfaceCache::Options options;
    options.byteCapacity = 256ULL << 20;
    SurfaceCache cache(array, surface, options);

    // The viewer redraws only when a tile is published (or the view moves,
    // which it does not here).
    std::mutex mutex;
    std::condition_variable cv;
    int notifications = 0;
    const auto id = cache.addTileReadyListener([&] {
        std::lock_guard lock(mutex);
        ++notifications;
        cv.notify_all();
    });

    array->prefetchDelayMs = 200;
    renderFrame(cache);  // admits the fill, which is now waiting on chunks
    moveSheet(*surface, 3000.0f);
    cache.invalidateAll();
    renderFrame(cache);  // the redraw the edit's surface update submits
    array->prefetchDelayMs = 0;

    bool moved = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!moved) {
        std::unique_lock lock(mutex);
        const int seen = notifications;
        if (!cv.wait_until(lock, deadline, [&] { return notifications != seen; }))
            break;
        lock.unlock();
        const Frame frame = renderFrame(cache);
        moved = frame.covered == kViewEdge * kViewEdge && frame.value == kAtZ3000;
    }
    CHECK(moved);

    cache.removeTileReadyListener(id);
    cache.shutdown();
}
