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

// PHerc 0172 level 5 pin (observed 2026-06): 657 z, 210 y, 285 x; 128^3 chunks.
constexpr int kSliceCount = 657;
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

// --threshold (vc_gen_normalgrids): a cut above 1 on the same staged scan.
// The expected slices come from a separate per-chunk read (readChunkOrFill),
// not from fillBinarySliceBatchFromVolume, clipped to the volume extent.
#include <map>
#include <optional>

namespace {

using ChunkKey = std::array<size_t, 3>;

struct ThrRun {
    std::vector<cv::Mat> slices;
    std::vector<bool> any;
};

ThrRun runFill(Volume& v, NormalGridSliceDirection dir, int sliceChunk,
               const std::vector<int>& locals, int threshold)
{
    // threshold < 0: call without the threshold argument (the old call).
    const std::array<int, 3> shape{kSliceCount, kHeight, kWidth};
    const std::array<int, 3> chunks{kChunk, kChunk, kChunk};
    const auto size = normalGridSliceSize(
        {size_t(kSliceCount), size_t(kHeight), size_t(kWidth)}, dir);
    ThrRun r;
    r.slices.resize(locals.size());
    std::vector<BinarySliceTarget> targets(locals.size());
    for (size_t i = 0; i < locals.size(); ++i) {
        r.slices[i] = cv::Mat::zeros(size, CV_8UC1);
        targets[i].binarySlice = &r.slices[i];
        targets[i].localSliceIndex = locals[i];
    }
    if (threshold < 0) {
        fillBinarySliceBatchFromVolume(
            v, 0, dir, sliceChunk, shape, chunks,
            std::span<BinarySliceTarget>(targets), 1);
    } else {
        fillBinarySliceBatchFromVolume(
            v, 0, dir, sliceChunk, shape, chunks,
            std::span<BinarySliceTarget>(targets), 1,
            static_cast<uint8_t>(threshold));
    }
    for (auto& t : targets) r.any.push_back(t.anyNonZero);
    return r;
}

cv::Mat expectedSlice(Volume& v, std::map<ChunkKey, std::vector<std::byte>>& cache,
                      NormalGridSliceDirection dir, int sliceChunk, int local,
                      int threshold)
{
    const auto size = normalGridSliceSize(
        {size_t(kSliceCount), size_t(kHeight), size_t(kWidth)}, dir);
    cv::Mat out = cv::Mat::zeros(size, CV_8UC1);
    const int s = sliceChunk * kChunk + local;
    for (int r = 0; r < out.rows; ++r) {
        for (int c = 0; c < out.cols; ++c) {
            int z = 0, y = 0, x = 0;
            if (dir == NormalGridSliceDirection::XY) { z = s; y = r; x = c; }
            else if (dir == NormalGridSliceDirection::XZ) { z = r; y = s; x = c; }
            else { z = r; y = c; x = s; }
            if (z >= kSliceCount || y >= kHeight || x >= kWidth) continue;
            const ChunkKey key{size_t(z / kChunk), size_t(y / kChunk), size_t(x / kChunk)};
            auto it = cache.find(key);
            if (it == cache.end()) it = cache.emplace(key, v.readChunkOrFill(0, key)).first;
            const size_t off = (size_t(z % kChunk) * kChunk + size_t(y % kChunk)) * kChunk + size_t(x % kChunk);
            const auto val = std::to_integer<uint8_t>(it->second.at(off));
            out.ptr<uint8_t>(r)[c] = val >= threshold ? 255 : 0;
        }
    }
    return out;
}

int differingPixels(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat d;
    cv::compare(a, b, d, cv::CMP_NE);
    return cv::countNonZero(d);
}

} // namespace

TEST_CASE("threshold above 1 keeps exactly the voxels >= threshold on PHerc 0172 L5")
{
    auto d = tmpDir("pherc172_l5_thr");
    auto staged = stagePHerc172Level5(d);
    if (staged.empty()) { fs::remove_all(d); return; }

    std::shared_ptr<Volume> v;
    try {
        v = Volume::New(staged);
    } catch (const std::exception& e) {
        if (requireNetwork()) FAIL("Volume::New on staged dir: " << e.what());
        fs::remove_all(d);
        return;
    }
    REQUIRE(v);
    std::map<ChunkKey, std::vector<std::byte>> cache;

    struct Case { NormalGridSliceDirection dir; int sliceChunk; };
    // XY: z 256..383 (chunk 2); XZ: y 128..209 (chunk 1, clipped at 210).
    for (const Case cs : {Case{NormalGridSliceDirection::XY, 2},
                          Case{NormalGridSliceDirection::XZ, 1}}) {
        const std::vector<int> locals{0, 16, 40, 64};
        const auto noArg = runFill(*v, cs.dir, cs.sliceChunk, locals, -1);
        const auto one = runFill(*v, cs.dir, cs.sliceChunk, locals, 1);
        int cutDiffers = 0;
        for (size_t i = 0; i < locals.size(); ++i) {
            // (a) threshold 1 = the old call, pixel for pixel, and = voxel != 0.
            CHECK(differingPixels(one.slices[i], noArg.slices[i]) == 0);
            CHECK(one.any[i] == noArg.any[i]);
            CHECK(differingPixels(one.slices[i],
                expectedSlice(*v, cache, cs.dir, cs.sliceChunk, locals[i], 1)) == 0);
        }
        for (const int thr : {51, 128}) {
            const auto cut = runFill(*v, cs.dir, cs.sliceChunk, locals, thr);
            for (size_t i = 0; i < locals.size(); ++i) {
                // (b) threshold N = 255 * (voxel >= N), from the separate read.
                const cv::Mat want = expectedSlice(*v, cache, cs.dir, cs.sliceChunk, locals[i], thr);
                CHECK(differingPixels(cut.slices[i], want) == 0);
                CHECK(cut.any[i] == (cv::countNonZero(want) > 0));
                cutDiffers += differingPixels(cut.slices[i], one.slices[i]);
            }
        }
        // (c) the cut changes the slices (the test is not trivially equal).
        CHECK(cutDiffers > 0);
        MESSAGE("direction " << static_cast<int>(cs.dir) << ": " << cutDiffers
                << " pixels differ between threshold 1 and the cuts");
    }

    // (d) threshold 0 is refused (with a non-empty target list).
    CHECK_THROWS_AS(runFill(*v, NormalGridSliceDirection::XY, 2, {0}, 0),
                    std::runtime_error);

    fs::remove_all(d);
}
