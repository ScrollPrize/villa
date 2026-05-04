#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/VcDataset.hpp"
#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kVolDim = 32;
constexpr int kChunkDim = 16;

uint8_t voxelValue(int z, int y, int x) noexcept
{
    return static_cast<uint8_t>((z + y + x) % 256);
}

struct TmpDir {
    fs::path path;
    TmpDir()
    {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        path = fs::temp_directory_path() /
               ("vc_slicing_test_" + std::to_string(rng()));
        fs::create_directories(path);
    }
    ~TmpDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

fs::path makeFixture(const fs::path& root)
{
    auto ds = vc::createZarrDataset(
        root, "0",
        {kVolDim, kVolDim, kVolDim},
        {kChunkDim, kChunkDim, kChunkDim},
        vc::VcDtype::uint8, "blosc", ".", 0);

    std::vector<uint8_t> chunkBuf(kChunkDim * kChunkDim * kChunkDim);
    for (int cz = 0; cz < kVolDim / kChunkDim; ++cz)
        for (int cy = 0; cy < kVolDim / kChunkDim; ++cy)
            for (int cx = 0; cx < kVolDim / kChunkDim; ++cx) {
                for (int z = 0; z < kChunkDim; ++z)
                    for (int y = 0; y < kChunkDim; ++y)
                        for (int x = 0; x < kChunkDim; ++x)
                            chunkBuf[(z * kChunkDim + y) * kChunkDim + x] =
                                voxelValue(cz * kChunkDim + z,
                                           cy * kChunkDim + y,
                                           cx * kChunkDim + x);
                ds->writeChunk(cz, cy, cx, chunkBuf.data(), chunkBuf.size());
            }
    return root / "0";
}

std::shared_ptr<vc::render::ChunkCache>
makeCacheWithAllChunks(const fs::path& root)
{
    auto opened = vc::render::openLocalZarrPyramid(root);
    auto cache = vc::render::createChunkCache(std::move(opened),
                                               64ULL * 1024 * 1024, 4);
    std::vector<vc::render::ChunkKey> all;
    for (int cz = 0; cz < kVolDim / kChunkDim; ++cz)
        for (int cy = 0; cy < kVolDim / kChunkDim; ++cy)
            for (int cx = 0; cx < kVolDim / kChunkDim; ++cx)
                all.push_back({0, cz, cy, cx});
    cache->prefetchChunks(all, true);
    return std::shared_ptr<vc::render::ChunkCache>(cache.release());
}

}

TEST_CASE("Slicing::readInterpolated3D: nearest sampling matches voxel grid")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    cv::Mat_<cv::Vec3f> coords(8, 8);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            coords(y, x) = cv::Vec3f(float(x), float(y), 7.0f);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(0));
    readInterpolated3D(out, cache.get(), 0, coords, /*nearest_neighbor=*/true);

    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            CHECK(out(y, x) == voxelValue(7, y, x));
}

TEST_CASE("Slicing::readInterpolated3D: trilinear at integer coords matches nearest")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    cv::Mat_<cv::Vec3f> coords(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            coords(y, x) = cv::Vec3f(float(x + 5), float(y + 5), 7.0f);

    cv::Mat_<uint8_t> outNear(4, 4, uint8_t(0));
    cv::Mat_<uint8_t> outTri(4, 4, uint8_t(0));
    readInterpolated3D(outNear, cache.get(), 0, coords, vc::Sampling::Nearest);
    readInterpolated3D(outTri,  cache.get(), 0, coords, vc::Sampling::Trilinear);

    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            const int diff = std::abs(int(outNear(y, x)) - int(outTri(y, x)));
            CHECK(diff <= 1);
        }
}

TEST_CASE("Slicing::readInterpolated3D: tricubic produces values near reference")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    cv::Mat_<cv::Vec3f> coords(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            coords(y, x) = cv::Vec3f(float(x + 5) + 0.5f, float(y + 5) + 0.5f, 7.5f);

    cv::Mat_<uint8_t> out(4, 4, uint8_t(0));
    readInterpolated3D(out, cache.get(), 0, coords, vc::Sampling::Tricubic);

    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            const int expectedNeighborhood = (8 + y) + (8 + x);
            CHECK(std::abs(int(out(y, x)) - expectedNeighborhood) <= 8);
        }
}

TEST_CASE("Slicing::readInterpolated3D: uint16 path produces same gradient")
{
    TmpDir tmp;
    auto ds = vc::createZarrDataset(
        tmp.path, "0", {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint16, "blosc", ".", 0);
    std::vector<uint16_t> chunk(16 * 16 * 16);
    for (int z = 0; z < 16; ++z)
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x)
                chunk[(z * 16 + y) * 16 + x] = uint16_t((z + y + x) * 100);
    REQUIRE(ds->writeChunk(0, 0, 0, chunk.data(), chunk.size() * sizeof(uint16_t)));

    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    auto cache = vc::render::createChunkCache(std::move(opened),
                                               64ULL * 1024 * 1024, 4);
    cache->prefetchChunks({{0, 0, 0, 0}}, true);

    cv::Mat_<cv::Vec3f> coords(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            coords(y, x) = cv::Vec3f(float(x), float(y), 3.0f);

    cv::Mat_<uint16_t> out(4, 4, uint16_t(0));
    readInterpolated3D(out, cache.get(), 0, coords, true);

    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            CHECK(out(y, x) == uint16_t((3 + y + x) * 100));
}

TEST_CASE("Slicing::readArea3D: contiguous voxel block matches gradient")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    Array3D<uint8_t> region({4, 4, 4});
    readArea3D(region, cv::Vec3i(2, 5, 5), cache.get(), 0);

    for (size_t z = 0; z < 4; ++z)
        for (size_t y = 0; y < 4; ++y)
            for (size_t x = 0; x < 4; ++x)
                CHECK(region(z, y, x) == voxelValue(int(2 + z), int(5 + y), int(5 + x)));
}

TEST_CASE("Slicing::readArea3D: region spanning chunk boundary")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    Array3D<uint8_t> region({4, 4, 4});
    readArea3D(region, cv::Vec3i(14, 14, 14), cache.get(), 0);

    for (size_t z = 0; z < 4; ++z)
        for (size_t y = 0; y < 4; ++y)
            for (size_t x = 0; x < 4; ++x)
                CHECK(region(z, y, x) == voxelValue(int(14 + z), int(14 + y), int(14 + x)));
}

TEST_CASE("Slicing::samplePlane: fused path matches readInterpolated3D")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    const cv::Vec3f origin(2.0f, 3.0f, 7.0f);
    const cv::Vec3f vxStep(1.0f, 0.0f, 0.0f);
    const cv::Vec3f vyStep(0.0f, 1.0f, 0.0f);
    constexpr int W = 8, H = 8;

    cv::Mat_<uint8_t> outFused(H, W, uint8_t(0));
    samplePlane(outFused, cache.get(), 0, origin, vxStep, vyStep, W, H, vc::Sampling::Nearest);

    cv::Mat_<cv::Vec3f> coords(H, W);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            coords(y, x) = origin + vxStep * float(x) + vyStep * float(y);
    cv::Mat_<uint8_t> outRef(H, W, uint8_t(0));
    readInterpolated3D(outRef, cache.get(), 0, coords, vc::Sampling::Nearest);

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            CHECK(outFused(y, x) == outRef(y, x));
}

TEST_CASE("Slicing::readMultiSlice: produces one output per offset")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    cv::Mat_<cv::Vec3f> base(4, 4);
    cv::Mat_<cv::Vec3f> step(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            base(y, x) = cv::Vec3f(float(x + 5), float(y + 5), 5.0f);
            step(y, x) = cv::Vec3f(0.0f, 0.0f, 1.0f);
        }

    std::vector<cv::Mat_<uint8_t>> outputs;
    std::vector<float> offsets = {0.0f, 1.0f, 2.0f};
    readMultiSlice(outputs, cache.get(), 0, base, step, offsets);

    REQUIRE(outputs.size() == offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) {
        CHECK(outputs[i].rows == 4);
        CHECK(outputs[i].cols == 4);
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x) {
                const int expected = int(5 + offsets[i] + 0.5f) + (5 + y) + (5 + x);
                CHECK(std::abs(int(outputs[i](y, x)) - (expected % 256)) <= 1);
            }
    }
}

TEST_CASE("Slicing::readArea3D: out-of-bounds offset clamps without crashing")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    Array3D<uint8_t> region({4, 4, 4}, 99);
    readArea3D(region, cv::Vec3i(100, 100, 100), cache.get(), 0);
    CHECK(true);
}

TEST_CASE("Slicing::readInterpolated3D: NaN coords skip gracefully")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    cv::Mat_<cv::Vec3f> coords(2, 2);
    coords(0, 0) = cv::Vec3f(5, 5, 5);
    coords(0, 1) = cv::Vec3f(std::numeric_limits<float>::quiet_NaN(), 0, 0);
    coords(1, 0) = cv::Vec3f(0, std::numeric_limits<float>::quiet_NaN(), 0);
    coords(1, 1) = cv::Vec3f(0, 0, std::numeric_limits<float>::quiet_NaN());

    cv::Mat_<uint8_t> out(2, 2, uint8_t(255));
    readInterpolated3D(out, cache.get(), 0, coords, true);

    CHECK(out(0, 0) == voxelValue(5, 5, 5));
}

TEST_CASE("Slicing::samplePlane: zero-size output is a no-op")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    cv::Mat_<uint8_t> out(1, 1, uint8_t(7));
    samplePlane(out, cache.get(), 0, cv::Vec3f(5, 5, 5),
                cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0), 0, 0, vc::Sampling::Nearest);
    CHECK(true);
}

TEST_CASE("Slicing::readArea3D: zero-size Array3D does not crash")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    Array3D<uint8_t> region({0, 0, 0});
    readArea3D(region, cv::Vec3i(0, 0, 0), cache.get(), 0);
    CHECK(region.size() == 0);
}

namespace {
fs::path makeFixtureUint16(const fs::path& root)
{
    auto ds = vc::createZarrDataset(
        root, "0",
        {kVolDim, kVolDim, kVolDim},
        {kChunkDim, kChunkDim, kChunkDim},
        vc::VcDtype::uint16, "blosc", ".", 0);
    std::vector<uint16_t> chunkBuf(kChunkDim * kChunkDim * kChunkDim);
    for (int cz = 0; cz < kVolDim / kChunkDim; ++cz)
        for (int cy = 0; cy < kVolDim / kChunkDim; ++cy)
            for (int cx = 0; cx < kVolDim / kChunkDim; ++cx) {
                for (int z = 0; z < kChunkDim; ++z)
                    for (int y = 0; y < kChunkDim; ++y)
                        for (int x = 0; x < kChunkDim; ++x)
                            chunkBuf[(z * kChunkDim + y) * kChunkDim + x] =
                                uint16_t(((cz * kChunkDim + z) +
                                          (cy * kChunkDim + y) +
                                          (cx * kChunkDim + x)) * 100);
                ds->writeChunk(cz, cy, cx, chunkBuf.data(),
                               chunkBuf.size() * sizeof(uint16_t));
            }
    return root / "0";
}
}

TEST_CASE("Slicing::readMultiSlice: uint16 multi-offset multi-pixel grid")
{
    TmpDir tmp;
    makeFixtureUint16(tmp.path);
    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    auto cache = vc::render::createChunkCache(std::move(opened),
                                               64ULL * 1024 * 1024, 4);
    std::vector<vc::render::ChunkKey> keys;
    for (int cz = 0; cz < 2; ++cz)
        for (int cy = 0; cy < 2; ++cy)
            for (int cx = 0; cx < 2; ++cx)
                keys.push_back({0, cz, cy, cx});
    cache->prefetchChunks(keys, true);

    cv::Mat_<cv::Vec3f> base(4, 4);
    cv::Mat_<cv::Vec3f> step(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            base(y, x) = cv::Vec3f(float(x + 5), float(y + 5), 5.0f);
            step(y, x) = cv::Vec3f(0, 0, 1);
        }
    std::vector<cv::Mat_<uint16_t>> outs;
    std::vector<float> offsets = {0.0f, 2.0f};
    readMultiSlice(outs, cache.get(), 0, base, step, offsets);
    REQUIRE(outs.size() == 2);
    CHECK(outs[0].rows == 4);
    CHECK(outs[1].cols == 4);

    int nonzero = 0;
    for (auto& m : outs)
        for (int y = 0; y < m.rows; ++y)
            for (int x = 0; x < m.cols; ++x)
                if (m(y, x) != 0) ++nonzero;
    CHECK(nonzero > 0);
}

TEST_CASE("Slicing::sampleTileSlices: uint8 + uint16 each produce per-offset matrices")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    cv::Mat_<cv::Vec3f> base(4, 4);
    cv::Mat_<cv::Vec3f> step(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            base(y, x) = cv::Vec3f(float(x + 5), float(y + 5), 5.0f);
            step(y, x) = cv::Vec3f(0, 0, 1);
        }
    std::vector<float> offsets = {0.0f, 1.0f, 2.0f};

    std::vector<cv::Mat_<uint8_t>> outU8;
    sampleTileSlices(outU8, cache.get(), 0, base, step, offsets);
    REQUIRE(outU8.size() == 3);
    CHECK(outU8[0].rows == 4);
    CHECK(outU8[2].cols == 4);
}

TEST_CASE("Slicing::sampleTileSlices uint16")
{
    TmpDir tmp;
    makeFixtureUint16(tmp.path);
    auto opened = vc::render::openLocalZarrPyramid(tmp.path);
    auto cache = vc::render::createChunkCache(std::move(opened),
                                               64ULL * 1024 * 1024, 4);
    std::vector<vc::render::ChunkKey> keys;
    for (int cz = 0; cz < 2; ++cz)
        for (int cy = 0; cy < 2; ++cy)
            for (int cx = 0; cx < 2; ++cx)
                keys.push_back({0, cz, cy, cx});
    cache->prefetchChunks(keys, true);

    cv::Mat_<cv::Vec3f> base(4, 4);
    cv::Mat_<cv::Vec3f> step(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            base(y, x) = cv::Vec3f(float(x + 5), float(y + 5), 5.0f);
            step(y, x) = cv::Vec3f(0, 0, 1);
        }
    std::vector<float> offsets = {0.0f, 1.0f};
    std::vector<cv::Mat_<uint16_t>> out;
    sampleTileSlices(out, cache.get(), 0, base, step, offsets);
    REQUIRE(out.size() == 2);
}

TEST_CASE("Slicing::readCompositeFast: mean / max / min methods")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    cv::Mat_<cv::Vec3f> base(4, 4);
    cv::Mat_<cv::Vec3f> normals(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            base(y, x) = cv::Vec3f(float(x + 5), float(y + 5), 10.0f);
            normals(y, x) = cv::Vec3f(0, 0, 1);
        }

    for (const std::string& method : {"mean", "max", "min", "median", "minabs", "alpha"}) {
        cv::Mat_<uint8_t> out(4, 4, uint8_t(0));
        CompositeParams params;
        params.method = method;
        readCompositeFast(out, cache.get(), 0, base, normals, /*zStep=*/1.0f,
                          /*zStart=*/-2, /*zEnd=*/2, params, vc::Sampling::Trilinear);
        CHECK(out.rows == 4);
    }
}

TEST_CASE("Slicing::readCompositeFast: all sampling modes")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    cv::Mat_<cv::Vec3f> base(4, 4);
    cv::Mat_<cv::Vec3f> normals(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            base(y, x) = cv::Vec3f(float(x + 5), float(y + 5), 10.0f);
            normals(y, x) = cv::Vec3f(0, 0, 1);
        }
    CompositeParams params;
    params.method = "mean";
    for (auto m : {vc::Sampling::Nearest, vc::Sampling::Trilinear, vc::Sampling::Tricubic}) {
        cv::Mat_<uint8_t> out(4, 4, uint8_t(0));
        readCompositeFast(out, cache.get(), 0, base, normals, 1.0f, -1, 1, params, m);
    }
    CHECK(true);
}

TEST_CASE("Slicing::samplePlane: tricubic dispatch")
{
    TmpDir tmp;
    makeFixture(tmp.path);
    auto cache = makeCacheWithAllChunks(tmp.path);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(0));
    samplePlane(out, cache.get(), 0,
                cv::Vec3f(5.5f, 5.5f, 5.5f),
                cv::Vec3f(1.0f, 0.0f, 0.0f),
                cv::Vec3f(0.0f, 1.0f, 0.0f),
                8, 8, vc::Sampling::Tricubic);
    int nonzero = 0;
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            if (out(y, x) != 0) ++nonzero;
    CHECK(nonzero > 0);
}
