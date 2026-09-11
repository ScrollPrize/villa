#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../../apps/src/RenderPrefetch.hpp"
#include "vc/core/util/Compositing.hpp"

#include <limits>
#include <mutex>

namespace {
using Keys = std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash>;

// Record actual blocking reads made by Slicing.cpp, independently of its
// opportunistic prefetch requests. Constant chunks avoid large fixture payloads.
class RecordingArray : public vc::render::IChunkedArray {
public:
    std::array<int, 3> dimensions{256, 256, 256};
    std::array<int, 3> chunks{128, 128, 128};
    Keys reads;
    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return dimensions; }
    std::array<int, 3> chunkShape(int) const override { return chunks; }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0; }
    LevelTransform levelTransform(int) const override { return {}; }
    vc::render::ChunkResult tryGetChunk(int level, int z, int y, int x) override
    {
        std::lock_guard lock(mutex_);
        reads.insert({level, z, y, x});
        vc::render::ChunkResult result;
        result.status = vc::render::ChunkStatus::AllFill;
        result.shape = chunks;
        return result;
    }
    vc::render::ChunkResult getChunkBlocking(int level, int z, int y, int x) override
    { return tryGetChunk(level, z, y, x); }
    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
private:
    std::mutex mutex_;
};

Keys plan(RecordingArray& array, const cv::Mat_<cv::Vec3f>& points,
          const cv::Mat_<cv::Vec3f>& dirs, const std::vector<float>& offsets, bool composite)
{
    Keys keys;
    vc::render::prefetch::insertExactChunksForSamples(
        points, dirs, offsets, &array, 0,
        vc::render::prefetch::samplingForRender(composite), keys);
    return keys;
}
}

TEST_CASE("ordinary prefetch covers trilinear neighbours across a chunk boundary")
{
    RecordingArray array;
    cv::Mat_<cv::Vec3f> points(1, 1, cv::Vec3f{127.25f, 64.25f, 64.25f});
    cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0, 0, 0});
    const auto keys = plan(array, points, dirs, {0}, false);
    std::vector<cv::Mat_<uint8_t>> output;
    readMultiSlice(output, &array, 0, points, dirs, {0});
    CHECK(keys.size() == 2);
    CHECK(keys == array.reads);
    array.reads.clear();
    sampleTileSlices(output, &array, 0, points, dirs, {0});
    CHECK(keys == array.reads);
}

TEST_CASE("planner and ordinary renderer round displaced positions in float")
{
    RecordingArray array;
    cv::Mat_<cv::Vec3f> points(1, 1, cv::Vec3f{127, 64, 64});
    cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0.999999f, 0, 0});
    const auto keys = plan(array, points, dirs, {1}, false);
    std::vector<cv::Mat_<uint8_t>> output;
    readMultiSlice(output, &array, 0, points, dirs, {1});
    CHECK(keys.size() == 1);
    CHECK(keys == array.reads);
}

TEST_CASE("composite prefetch matches nearest rounding and layer offsets")
{
    RecordingArray array;
    array.dimensions = {4, 4, 4};
    array.chunks = {1, 1, 1};
    cv::Mat_<cv::Vec3f> points(1, 1, cv::Vec3f{std::nextafter(0.5f, 0.f), 1, 1});
    cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0, 0, 1});
    const auto keys = plan(array, points, dirs, {-1, 0, 1}, true);
    cv::Mat_<uint8_t> output(1, 1, uint8_t{0});
    CompositeParams params;
    params.method = "mean";
    // Default composite sampling is deliberately used as an independent oracle.
    readCompositeFast(output, &array, 0, points, dirs, 1, -1, 1, params);
    CHECK(keys == array.reads);
    CHECK(keys.size() == 3);
}

TEST_CASE("ordinary planner matches reads around all chunk and volume faces")
{
    const float coords[] = {0, 0.25f, 127, 127.25f, 127.75f, 128,
                            std::nextafter(128.f, 0.f), 255.75f};
    RecordingArray array;
    for (int axis = 0; axis < 3; ++axis) {
        for (float value : coords) {
            cv::Vec3f point{127.25f, 127.25f, 127.25f};
            point[axis] = value;
            cv::Mat_<cv::Vec3f> points(1, 1, point);
            cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0.25f, -0.5f, 1});
            const auto keys = plan(array, points, dirs, {-1, 0, 1}, false);
            array.reads.clear();
            std::vector<cv::Mat_<uint8_t>> output;
            readMultiSlice(output, &array, 0, points, dirs, {-1, 0, 1});
            CHECK(keys == array.reads);
        }
    }
}

TEST_CASE("nonfinite and outside samples produce no prefetch requests")
{
    RecordingArray array;
    for (float value : {-1.f, 256.f, std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        cv::Mat_<cv::Vec3f> points(1, 1, cv::Vec3f{value, 64, 64});
        cv::Mat_<cv::Vec3f> dirs(1, 1, cv::Vec3f{0, 0, 0});
        CHECK(plan(array, points, dirs, {0}, false).empty());
    }
}

TEST_CASE("sparse samples in huge volumes do not allocate a volume sized bitmap")
{
    RecordingArray array;
    array.dimensions = {1000000000, 1000000000, 1000000000};
    array.chunks = {1, 1, 1};
    cv::Mat_<cv::Vec3f> points(1, 2);
    points(0, 0) = {1, 1, 1};
    points(0, 1) = {999999936.f, 999999936.f, 999999936.f};
    cv::Mat_<cv::Vec3f> dirs(1, 2, cv::Vec3f{0, 0, 0});
    const auto keys = plan(array, points, dirs, {0}, false);
    CHECK(keys.size() == 16);
    CHECK(keys.contains({0, 1, 1, 1}));
    CHECK(keys.contains({0, 999999937, 999999937, 999999937}));
}
