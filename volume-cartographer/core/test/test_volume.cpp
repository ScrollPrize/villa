#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/types/SampleParams.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/VcDataset.hpp"
#include "vc/core/types/Volume.hpp"
#include "utils/Json.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct TmpDir {
    fs::path path;
    TmpDir()
    {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        path = fs::temp_directory_path() /
               ("vc_volume_test_" + std::to_string(rng()));
        fs::create_directories(path);
    }
    ~TmpDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

constexpr int kVolDim = 32;
constexpr int kChunkDim = 16;

uint8_t voxelValue(int z, int y, int x) noexcept
{
    return static_cast<uint8_t>((z + y + x) % 256);
}

void writeMetaJson(const fs::path& volumePath, const std::string& uuid = "test_vol")
{
    utils::Json meta;
    meta["uuid"] = uuid;
    meta["name"] = uuid;
    meta["type"] = "vol";
    meta["format"] = "zarr";
    meta["width"] = kVolDim;
    meta["height"] = kVolDim;
    meta["slices"] = kVolDim;
    meta["voxelsize"] = 7.91;
    meta["min"] = 0.0;
    meta["max"] = 255.0;
    std::ofstream f(volumePath / "meta.json");
    f << meta.dump(2);
}

void writeFixtureLevel(const fs::path& volumePath, const std::string& level = "0")
{
    auto ds = vc::createZarrDataset(
        volumePath, level,
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
}

}

TEST_CASE("Volume: open from meta.json + level-0 zarr")
{
    TmpDir tmp;
    writeMetaJson(tmp.path);
    writeFixtureLevel(tmp.path);

    Volume vol(tmp.path);
    CHECK(vol.id() == "test_vol");
    CHECK(vol.name() == "test_vol");
    CHECK(vol.sliceWidth() == kVolDim);
    CHECK(vol.sliceHeight() == kVolDim);
    CHECK(vol.numSlices() == kVolDim);
    CHECK(vol.shape() == std::array<int, 3>{kVolDim, kVolDim, kVolDim});
    CHECK(vol.voxelSize() == doctest::Approx(7.91));
    CHECK(vol.numScales() == 1);
    CHECK_FALSE(vol.isRemote());
    CHECK(vol.path() == tmp.path);
}

TEST_CASE("Volume::checkDir: recognizes valid + invalid layouts")
{
    TmpDir tmp;
    CHECK_FALSE(Volume::checkDir(tmp.path));

    writeMetaJson(tmp.path);
    CHECK(Volume::checkDir(tmp.path));

    fs::create_directories(tmp.path / "../empty");
    CHECK_FALSE(Volume::checkDir(tmp.path / "non_existent"));
}

TEST_CASE("Volume::chunkedCache: returns a valid IChunkedArray with the right shape")
{
    TmpDir tmp;
    writeMetaJson(tmp.path);
    writeFixtureLevel(tmp.path);

    Volume vol(tmp.path);
    auto* cache = vol.chunkedCache();
    REQUIRE(cache != nullptr);
    CHECK(cache->numLevels() == 1);
    CHECK(cache->shape(0) == std::array<int, 3>{kVolDim, kVolDim, kVolDim});
    CHECK(cache->dtype() == vc::render::ChunkDtype::UInt8);
}

TEST_CASE("Volume::setCacheBudget: invalidates cached IChunkedArray")
{
    TmpDir tmp;
    writeMetaJson(tmp.path);
    writeFixtureLevel(tmp.path);

    Volume vol(tmp.path);
    auto* a = vol.chunkedCache();
    REQUIRE(a != nullptr);
    vol.setCacheBudget(2ULL << 20);
    auto* b = vol.chunkedCache();
    REQUIRE(b != nullptr);
}

TEST_CASE("Volume::setIOThreads: invalidates cached IChunkedArray")
{
    TmpDir tmp;
    writeMetaJson(tmp.path);
    writeFixtureLevel(tmp.path);

    Volume vol(tmp.path);
    (void)vol.chunkedCache();
    vol.setIOThreads(2);
    auto* b = vol.chunkedCache();
    REQUIRE(b != nullptr);
}

TEST_CASE("Volume::sample: uint8 writes voxel-grid values to output")
{
    TmpDir tmp;
    writeMetaJson(tmp.path);
    writeFixtureLevel(tmp.path);

    Volume vol(tmp.path);
    auto* cache = vol.chunkedCache();
    REQUIRE(cache != nullptr);

    std::vector<vc::render::ChunkKey> keys;
    for (int cz = 0; cz < 2; ++cz)
        for (int cy = 0; cy < 2; ++cy)
            for (int cx = 0; cx < 2; ++cx)
                keys.push_back({0, cz, cy, cx});
    cache->prefetchChunks(keys, /*wait=*/true);

    cv::Mat_<cv::Vec3f> coords(8, 8);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            coords(y, x) = cv::Vec3f(float(x), float(y), 7.0f);

    cv::Mat_<uint8_t> out(8, 8, uint8_t(0));
    vc::SampleParams params;
    params.level = 0;
    params.method = vc::Sampling::Nearest;
    vol.sample(out, coords, params);

    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            CHECK(out(y, x) == voxelValue(7, y, x));
}

TEST_CASE("Volume: missing meta.json + missing zarr levels throws on construction")
{
    TmpDir tmp;
    CHECK_THROWS(([&] { Volume vol(tmp.path); })());
}

TEST_CASE("Volume: New() factory")
{
    TmpDir tmp;
    writeMetaJson(tmp.path);
    writeFixtureLevel(tmp.path);

    auto vol = Volume::New(tmp.path);
    REQUIRE(vol);
    CHECK(vol->sliceWidth() == kVolDim);
}

namespace {
void writeFixtureLevel16(const fs::path& volumePath, const std::string& level)
{
    auto ds = vc::createZarrDataset(
        volumePath, level,
        {kVolDim / 2, kVolDim / 2, kVolDim / 2},
        {kChunkDim / 2, kChunkDim / 2, kChunkDim / 2},
        vc::VcDtype::uint8, "blosc", ".", 0);
    std::vector<uint8_t> chunkBuf(8 * 8 * 8, 0);
    for (int cz = 0; cz < 2; ++cz)
        for (int cy = 0; cy < 2; ++cy)
            for (int cx = 0; cx < 2; ++cx)
                ds->writeChunk(cz, cy, cx, chunkBuf.data(), chunkBuf.size());
}

void writeFixtureLevelUint16(const fs::path& volumePath, const std::string& level = "0")
{
    auto ds = vc::createZarrDataset(
        volumePath, level,
        {kVolDim, kVolDim, kVolDim},
        {kChunkDim, kChunkDim, kChunkDim},
        vc::VcDtype::uint16, "blosc", ".", 0);
    std::vector<uint16_t> chunkBuf(kChunkDim * kChunkDim * kChunkDim, 0);
    for (int cz = 0; cz < 2; ++cz)
        for (int cy = 0; cy < 2; ++cy)
            for (int cx = 0; cx < 2; ++cx) {
                for (size_t i = 0; i < chunkBuf.size(); ++i)
                    chunkBuf[i] = uint16_t(i * 7);
                ds->writeChunk(cz, cy, cx, chunkBuf.data(),
                               chunkBuf.size() * sizeof(uint16_t));
            }
}
}

TEST_CASE("Volume: multi-level pyramid loads with consistent shapes")
{
    TmpDir tmp;
    writeMetaJson(tmp.path);
    writeFixtureLevel(tmp.path, "0");
    writeFixtureLevel16(tmp.path, "1");

    Volume vol(tmp.path);
    CHECK(vol.numScales() == 2);
    CHECK(vol.sliceWidth() == kVolDim);
}

TEST_CASE("Volume: auto-generates metadata when meta.json is missing but zarr exists")
{
    TmpDir tmp;
    writeFixtureLevel(tmp.path);

    Volume vol(tmp.path);
    CHECK(vol.sliceWidth() == kVolDim);
    CHECK(vol.sliceHeight() == kVolDim);
    CHECK(vol.numSlices() == kVolDim);
}

TEST_CASE("Volume: metadata.json (alt) with 'scan' wrapper is accepted")
{
    TmpDir tmp;
    writeFixtureLevel(tmp.path);

    utils::Json full;
    utils::Json scan;
    scan["uuid"] = "alt_vol";
    scan["name"] = "alt_vol";
    scan["type"] = "vol";
    scan["width"] = kVolDim;
    scan["height"] = kVolDim;
    scan["slices"] = kVolDim;
    scan["voxelsize"] = 1.0;
    scan["min"] = 0.0;
    scan["max"] = 255.0;
    full["scan"] = scan;
    {
        std::ofstream f(tmp.path / "metadata.json");
        f << full.dump();
    }

    Volume vol(tmp.path);
    CHECK(vol.id() == "alt_vol");
}

TEST_CASE("Volume: metadata.json without 'scan' key is rejected")
{
    TmpDir tmp;
    writeFixtureLevel(tmp.path);

    utils::Json bad;
    bad["uuid"] = "x";
    {
        std::ofstream f(tmp.path / "metadata.json");
        f << bad.dump();
    }

    CHECK_THROWS([&] { Volume vol(tmp.path); }());
}

TEST_CASE("Volume::sample uint16 path")
{
    TmpDir tmp;
    writeMetaJson(tmp.path);
    writeFixtureLevelUint16(tmp.path);

    Volume vol(tmp.path);
    auto* cache = vol.chunkedCache();
    REQUIRE(cache != nullptr);

    std::vector<vc::render::ChunkKey> keys;
    for (int cz = 0; cz < 2; ++cz)
        for (int cy = 0; cy < 2; ++cy)
            for (int cx = 0; cx < 2; ++cx)
                keys.push_back({0, cz, cy, cx});
    cache->prefetchChunks(keys, true);

    cv::Mat_<cv::Vec3f> coords(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            coords(y, x) = cv::Vec3f(float(x), float(y), 7.0f);

    cv::Mat_<uint16_t> out(4, 4, uint16_t(0));
    vc::SampleParams params;
    params.level = 0;
    params.method = vc::Sampling::Nearest;
    vol.sample(out, coords, params);
    bool nonzero = false;
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            if (out(y, x) != 0) nonzero = true;
    CHECK(nonzero);
}

TEST_CASE("Volume: RemoteConstructTag direct construction")
{
    Volume vol(fs::path("/some/remote/path"), Volume::RemoteConstructTag{});
    CHECK(vol.path() == fs::path("/some/remote/path"));
}
