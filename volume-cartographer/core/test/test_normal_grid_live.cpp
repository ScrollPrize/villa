// Pull PHerc 0172 level 5 from the public S3 open-data bucket, stage it as a
// minimal local zarr volume, and exercise the NormalGridGenerate API on it.
//
// Soft-skips on network failure (set VC_TEST_REQUIRE_NETWORK=1 to make it
// hard). Only level 5 (~58 MB total) is downloaded.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/NormalGridGenerate.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/types/Volume.hpp"

#include "utils/Json.hpp"
#include <opencv2/core.hpp>

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using vc::core::util::BinarySliceTarget;
using vc::core::util::NormalGridSliceDirection;
using vc::core::util::fillBinarySliceBatchFromVolume;
using vc::core::util::normalGridSliceSize;

namespace {

constexpr const char* kZarrRoot =
    "s3://vesuvius-challenge-open-data/PHerc0172/volumes/"
    "20241024131838-7.910um-53keV-masked.zarr";

// PHerc 0172 level 5 pin (observed 2026-05): 651 z, 210 y, 285 x; 128^3 chunks.
constexpr int kSliceCount = 651;
constexpr int kHeight = 210;
constexpr int kWidth = 285;
constexpr int kChunk = 128;

bool requireNetwork()
{
    const char* env = std::getenv("VC_TEST_REQUIRE_NETWORK");
    return env && env[0] && env[0] != '0';
}

// Returns nullopt on network failure (after soft-skipping via MESSAGE).
std::optional<std::string> fetch(const std::string& url)
{
    try {
        auto body = vc::httpGetString(url);
        if (body.empty()) return std::nullopt;
        return body;
    } catch (const std::exception& e) {
        if (requireNetwork()) FAIL("network fetch failed: " << e.what());
        MESSAGE("Skipping (network?): " << e.what());
        return std::nullopt;
    }
}

void writeBytes(const fs::path& p, const std::string& bytes)
{
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_ng_live_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

// Stage a minimal local volume directory by downloading the level 5 zarr
// chunks and adding the wrapping meta.json + .zgroup needed by Volume::New.
// Returns the local path on success, empty on network failure.
fs::path stagePHerc172Level5(const fs::path& dst)
{
    const std::string base = std::string(kZarrRoot);
    auto httpRoot = vc::resolveRemoteUrl(base).httpsUrl;

    // 1. Pull root .zattrs (for OME multiscales).
    auto zattrs = fetch(httpRoot + "/.zattrs");
    if (!zattrs) return {};

    // 2. Pull the level 5 .zarray.
    auto zarray = fetch(httpRoot + "/5/.zarray");
    if (!zarray) return {};

    // Stage as the SOLE level "0" in dst (so Volume sees a 1-level pyramid).
    fs::create_directories(dst);
    {
        std::ofstream zg(dst / ".zgroup");
        zg << R"({"zarr_format":2})";
    }
    // Write a minimal but valid OME .zattrs with one dataset at path "0".
    {
        std::ofstream zat(dst / ".zattrs");
        zat << R"({"multiscales":[{)"
            << R"("axes":[{"name":"z","type":"space"},)"
            << R"({"name":"y","type":"space"},)"
            << R"({"name":"x","type":"space"}],)"
            << R"("datasets":[{)"
            << R"("path":"0",)"
            << R"("coordinateTransformations":[{"scale":[1.0,1.0,1.0],"type":"scale"}])"
            << R"(}])"
            << R"(}]})";
    }
    writeBytes(dst / "0" / ".zarray", *zarray);

    // 3. Pull each level-5 chunk file and write it under dst/0/<iz>/<iy>/<ix>.
    // PHerc 0172 chunks use '/' as dimension separator.
    const int cz = (kSliceCount + kChunk - 1) / kChunk;  // 6
    const int cy = (kHeight + kChunk - 1) / kChunk;      // 2
    const int cx = (kWidth + kChunk - 1) / kChunk;       // 3
    int downloaded = 0;
    int missing = 0;
    for (int iz = 0; iz < cz; ++iz) {
        for (int iy = 0; iy < cy; ++iy) {
            for (int ix = 0; ix < cx; ++ix) {
                const std::string chunkUrl = httpRoot + "/5/" +
                    std::to_string(iz) + "/" + std::to_string(iy) + "/" +
                    std::to_string(ix);
                try {
                    auto body = vc::httpGetString(chunkUrl);
                    if (body.empty()) { ++missing; continue; }
                    writeBytes(dst / "0" / std::to_string(iz) /
                               std::to_string(iy) /
                               std::to_string(ix),
                               body);
                    ++downloaded;
                } catch (const std::exception& e) {
                    if (requireNetwork()) FAIL("chunk fetch: " << e.what());
                    return {};
                }
            }
        }
    }
    MESSAGE("Staged PHerc 0172 level 5: " << downloaded << " chunks ("
            << missing << " missing/skipped)");

    // 4. Write a meta.json that Volume::New expects.
    {
        std::ofstream m(dst / "meta.json");
        m << R"({"type":"vol","uuid":"pherc0172-level5","name":"PHerc0172-L5",)"
          << R"("width":)" << kWidth << R"(,"height":)" << kHeight
          << R"(,"slices":)" << kSliceCount
          << R"(,"format":"zarr","voxelsize":7.91})";
    }

    return dst;
}

} // namespace

TEST_CASE("Stage PHerc 0172 level 5 and run fillBinarySliceBatchFromVolume")
{
    auto d = tmpDir("pherc172_l5");
    auto staged = stagePHerc172Level5(d);
    if (staged.empty()) {
        // Soft-skip when network not available.
        fs::remove_all(d);
        return;
    }

    std::shared_ptr<Volume> v;
    try {
        v = Volume::New(staged);
    } catch (const std::exception& e) {
        if (requireNetwork()) FAIL("Volume::New on staged dir: " << e.what());
        MESSAGE("Skipping: Volume::New failed: " << e.what());
        fs::remove_all(d);
        return;
    }
    REQUIRE(v);
    REQUIRE_FALSE(v->isRemote());

    auto sh = v->shape(0);
    CHECK(sh[0] == kSliceCount);
    CHECK(sh[1] == kHeight);
    CHECK(sh[2] == kWidth);
    CHECK(v->dtypeSize() == 1);

    // Run normal-grid generation on the first chunk-along-Z (sliceAxisChunkIndex=0)
    // for the XY direction. The chunk depth is 128, so we sample localSlices
    // 0, 8, 16, 24 — four binary slices per chunk-walk.
    const std::array<int, 3> shape{kSliceCount, kHeight, kWidth};
    const std::array<int, 3> chunks{kChunk, kChunk, kChunk};
    auto sliceSize = normalGridSliceSize(
        {size_t(kSliceCount), size_t(kHeight), size_t(kWidth)},
        NormalGridSliceDirection::XY);

    std::vector<cv::Mat> slices(4);
    std::vector<BinarySliceTarget> targets(4);
    for (int i = 0; i < 4; ++i) {
        slices[i] = cv::Mat::zeros(sliceSize, CV_8UC1);
        targets[i].binarySlice = &slices[i];
        targets[i].localSliceIndex = i * 8;
    }

    fillBinarySliceBatchFromVolume(
        *v, /*level=*/0, NormalGridSliceDirection::XY,
        /*sliceAxisChunkIndex=*/0,
        shape, chunks,
        std::span<BinarySliceTarget>(targets),
        /*ioThreads=*/1);

    // PHerc 0172 is a masked scroll volume — most early slices should have
    // non-zero voxels (mask + scan data). Pin "at least one" since masking
    // may zero out boundary regions.
    int withData = 0;
    for (auto& t : targets) if (t.anyNonZero) ++withData;
    CHECK(withData >= 1);

    fs::remove_all(d);
}

TEST_CASE("XZ direction also produces some non-empty slices on PHerc 0172 L5")
{
    auto d = tmpDir("pherc172_l5_xz");
    auto staged = stagePHerc172Level5(d);
    if (staged.empty()) { fs::remove_all(d); return; }

    std::shared_ptr<Volume> v;
    try {
        v = Volume::New(staged);
    } catch (const std::exception&) {
        fs::remove_all(d);
        return;
    }
    REQUIRE(v);

    const std::array<int, 3> shape{kSliceCount, kHeight, kWidth};
    const std::array<int, 3> chunks{kChunk, kChunk, kChunk};
    auto sliceSize = normalGridSliceSize(
        {size_t(kSliceCount), size_t(kHeight), size_t(kWidth)},
        NormalGridSliceDirection::XZ);

    cv::Mat slice0 = cv::Mat::zeros(sliceSize, CV_8UC1);
    BinarySliceTarget t;
    t.binarySlice = &slice0;
    t.localSliceIndex = 0;
    std::vector<BinarySliceTarget> ts = {t};

    fillBinarySliceBatchFromVolume(
        *v, 0, NormalGridSliceDirection::XZ,
        /*sliceAxisChunkIndex=*/0, shape, chunks,
        std::span<BinarySliceTarget>(ts), 1);

    // Y axis chunkIndex=0 spans rows 0..127 — should hit data.
    CHECK(ts[0].anyNonZero);
    fs::remove_all(d);
}
