#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/SeedScanCheck.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kChunkEdge = 32;

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_seed_scan_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

std::shared_ptr<Volume> makeScan(const fs::path& dir)
{
    Volume::ZarrCreateOptions opts;
    opts.shapeZYX = {64, 64, 64};
    opts.chunkShapeZYX = {kChunkEdge, kChunkEdge, kChunkEdge};
    opts.numLevels = 1;
    opts.fillValue = 0.0;
    opts.voxelSize = 1.0;
    opts.uuid = "test-scan-uuid";
    opts.name = "test-scan";
    opts.overwriteExisting = true;
    opts.compressor = "none";

    auto scan = Volume::New(dir, opts);
    REQUIRE(scan);

    std::vector<std::byte> chunk(scan->chunkByteSize(0), std::byte{200});
    chunk[(1 * kChunkEdge + 2) * kChunkEdge + 3] = std::byte{0};
    scan->writeChunk(0, {0, 0, 0}, chunk);

    vc::render::processChunkCacheService()->configureDecodedByteCapacity(4 * 1024 * 1024);
    return scan;
}

}  // namespace

TEST_CASE("a seed on scan data draws no warning under either policy")
{
    auto d = tmpDir("data");
    auto scan = makeScan(d);
    const cv::Vec3d seed(5.0, 6.0, 7.0);

    const auto sample = vc::util::sampleSeedInScan(*scan->chunkedCache(), seed);
    CHECK(sample.status == vc::util::SeedScanStatus::Data);
    CHECK(sample.value == 200.0);
    CHECK(sample.chunk == std::array<int, 3>{0, 0, 0});

    const auto warned = vc::util::evaluateSeedScan(sample, seed, vc::util::SeedScanPolicy::Warn);
    CHECK(warned.message.empty());
    CHECK_FALSE(warned.abort);

    const auto required = vc::util::evaluateSeedScan(sample, seed, vc::util::SeedScanPolicy::Require);
    CHECK(required.message.empty());
    CHECK_FALSE(required.abort);

    fs::remove_all(d);
}

TEST_CASE("a seed on a scan voxel of 0 warns, and is refused when required")
{
    auto d = tmpDir("zero");
    auto scan = makeScan(d);
    const cv::Vec3d seed(3.0, 2.0, 1.0);

    const auto sample = vc::util::sampleSeedInScan(*scan->chunkedCache(), seed);
    CHECK(sample.status == vc::util::SeedScanStatus::ZeroVoxel);
    CHECK(sample.value == 0.0);

    const auto warned = vc::util::evaluateSeedScan(sample, seed, vc::util::SeedScanPolicy::Warn);
    CHECK_FALSE(warned.abort);
    CHECK(warned.message.find("scan voxel 0") != std::string::npos);

    const auto required = vc::util::evaluateSeedScan(sample, seed, vc::util::SeedScanPolicy::Require);
    CHECK(required.abort);
    CHECK(required.message == warned.message);

    fs::remove_all(d);
}

TEST_CASE("a seed in a scan chunk the store does not hold warns, and is refused when required")
{
    auto d = tmpDir("absent");
    auto scan = makeScan(d);
    const cv::Vec3d seed(40.0, 41.0, 42.0);

    const auto sample = vc::util::sampleSeedInScan(*scan->chunkedCache(), seed);
    CHECK(sample.status == vc::util::SeedScanStatus::MissingChunk);
    CHECK(sample.value == 0.0);
    CHECK(sample.chunk == std::array<int, 3>{1, 1, 1});

    const auto warned = vc::util::evaluateSeedScan(sample, seed, vc::util::SeedScanPolicy::Warn);
    CHECK_FALSE(warned.abort);
    CHECK(warned.message.find("scan chunk 1/1/1") != std::string::npos);

    const auto required = vc::util::evaluateSeedScan(sample, seed, vc::util::SeedScanPolicy::Require);
    CHECK(required.abort);

    fs::remove_all(d);
}

TEST_CASE("the policy defaults to warning, so an existing command still runs")
{
    auto d = tmpDir("default");
    auto scan = makeScan(d);
    const cv::Vec3d seed(40.0, 41.0, 42.0);

    const auto sample = vc::util::sampleSeedInScan(*scan->chunkedCache(), seed);
    const auto decision = vc::util::evaluateSeedScan(sample, seed);
    CHECK_FALSE(decision.abort);
    CHECK_FALSE(decision.message.empty());

    fs::remove_all(d);
}

TEST_CASE("a seed outside the scan is reported rather than read as data")
{
    auto d = tmpDir("outside");
    auto scan = makeScan(d);
    const cv::Vec3d seed(64.0, 10.0, 10.0);

    const auto sample = vc::util::sampleSeedInScan(*scan->chunkedCache(), seed);
    CHECK(sample.status == vc::util::SeedScanStatus::OutsideVolume);

    const auto required = vc::util::evaluateSeedScan(sample, seed, vc::util::SeedScanPolicy::Require);
    CHECK(required.abort);
    CHECK(required.message.find("outside the scan volume") != std::string::npos);

    fs::remove_all(d);
}
