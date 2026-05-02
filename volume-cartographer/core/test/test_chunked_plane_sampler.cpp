#include "test.hpp"

#include "vc/core/render/ChunkedPlaneSampler.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

class FakeChunkedArray final : public vc::render::IChunkedArray {
public:
    std::array<int, 3> levelShape{8, 8, 8};
    std::array<int, 3> levelChunkShape{4, 4, 4};
    LevelTransform transform;
    int tryGetCount = 0;
    std::vector<vc::render::ChunkKey> requested;

    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return levelShape; }
    std::array<int, 3> chunkShape(int) const override { return levelChunkShape; }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int) const override { return transform; }

    vc::render::ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        ++tryGetCount;
        requested.push_back({level, iz, iy, ix});
        return {vc::render::ChunkStatus::MissQueued,
                vc::render::ChunkDtype::UInt8,
                levelChunkShape,
                {},
                {}};
    }

    vc::render::ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }

    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool) override {}

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 1; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
};

} // namespace

TEST(ChunkedPlaneSampler, SurfaceNegativeOneSentinelDoesNotRequestChunk)
{
    FakeChunkedArray array;
    array.transform.offsetFromLevel0 = {1.0, 1.0, 1.0};

    cv::Mat_<cv::Vec3f> coords(1, 1, cv::Vec3f(-1.0f, -1.0f, -1.0f));
    cv::Mat_<uint8_t> values(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    const auto stats = vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
        array, 0, coords, values, coverage,
        vc::render::ChunkedPlaneSampler::Options(vc::Sampling::Nearest, 1));

    EXPECT_EQ(array.tryGetCount, 0);
    EXPECT_EQ(stats.requestedChunks, 0);
    EXPECT_EQ(stats.coveredPixels, 0);
    EXPECT_EQ(coverage(0, 0), uint8_t(0));
}

TEST(ChunkedPlaneSampler, PlaneOutOfBoundsDoesNotRequestChunk)
{
    FakeChunkedArray array;

    cv::Mat_<uint8_t> values(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    const auto stats = vc::render::ChunkedPlaneSampler::samplePlaneLevel(
        array, 0,
        cv::Vec3f(-2.0f, 2.0f, 2.0f),
        cv::Vec3f(0.0f, 0.0f, 0.0f),
        cv::Vec3f(0.0f, 0.0f, 0.0f),
        values, coverage,
        vc::render::ChunkedPlaneSampler::Options(vc::Sampling::Nearest, 1));

    EXPECT_EQ(array.tryGetCount, 0);
    EXPECT_EQ(stats.requestedChunks, 0);
    EXPECT_EQ(stats.coveredPixels, 1);
    EXPECT_EQ(coverage(0, 0), uint8_t(1));
}
