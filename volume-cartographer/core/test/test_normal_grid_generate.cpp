#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/core/util/NormalGridGenerate.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <fstream>
#include <random>
#include <opencv2/core.hpp>
#include "vc_reader/zarr_write.hpp"

TEST_CASE("Native normal-grid assembly covers sparse edges and parallel read errors")
{
    namespace fs = std::filesystem;
    using namespace vc::core::util;
    const auto root = fs::temp_directory_path() / ("vc-ng-" + std::to_string(std::random_device{}()));
    struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
    for (int version : {2, 3}) {
        vcr::zarr::CreateOptions options;
        options.version = version;
        options.shape = {3, 5, 7};
        options.chunks = {2, 3, 4};
        options.dtype = "|u1";
        const auto path = root / std::to_string(version);
        auto writer = vcr::zarr::Writer::create(path.string(), options);
        std::vector<std::uint8_t> bytes(24, 1);
        writer.writeChunk({1, 1, 1}, bytes.data(), bytes.size());
        auto source = vcr::zarr::openArray(path.string(), {});
        for (auto direction : {NormalGridSliceDirection::XY, NormalGridSliceDirection::XZ, NormalGridSliceDirection::YZ}) {
            cv::Mat image = cv::Mat::zeros(normalGridSliceSize({3,5,7}, direction), CV_8U);
            BinarySliceTarget target{&image, 0, false};
            fillBinarySliceBatchFromArray(source, direction, 1, {3,5,7}, {2,3,4}, {&target, 1}, 2);
            CHECK(target.anyNonZero);
            const int expected = direction == NormalGridSliceDirection::XY ? 6 :
                                 direction == NormalGridSliceDirection::XZ ? 3 : 2;
            CHECK(cv::countNonZero(image) == expected);
        }
        {
            std::ofstream broken(writer.chunkPath({1,1,1}), std::ios::binary | std::ios::trunc);
            broken << "invalid compressed chunk";
        }
        cv::Mat brokenImage = cv::Mat::zeros(5,7,CV_8U);
        BinarySliceTarget brokenTarget{&brokenImage,0,false};
        // A corrupt chunk is noted and skipped: it reads as fill.
        auto reopened = vcr::zarr::openArray(path.string(), {});
        CHECK_NOTHROW(fillBinarySliceBatchFromArray(reopened, NormalGridSliceDirection::XY, 1,
                                                    {3,5,7}, {2,3,4}, {&brokenTarget,1}, 2));
        CHECK_FALSE(brokenTarget.anyNonZero);
        CHECK(reopened.array->stats().corrupt == 1);
        writer.setFillValue(1);
        auto nonzero = vcr::zarr::openArray(path.string(), {});
        cv::Mat image = cv::Mat::zeros(5,7,CV_8U);
        BinarySliceTarget target{&image,0,false};
        CHECK_THROWS(fillBinarySliceBatchFromArray(nonzero, NormalGridSliceDirection::XY, 0,
                                                  {3,5,7}, {2,3,4}, {&target,1}, 2));
    }
}

namespace {

// The read_many formulation of a slab: one region per target slice, written straight into
// its Mat, then nonzero voxels become 255 and anyNonZero is set.
void fillBinarySliceBatchReadMany(vcr::zarr::OpenedArray& source, vc::core::util::NormalGridSliceDirection direction,
                                  int sliceAxisChunkIndex, const std::array<int, 3>& volumeShape,
                                  const std::array<int, 3>& chunkShape, std::span<vc::core::util::BinarySliceTarget> targets)
{
    using vc::core::util::NormalGridSliceDirection;
    const int axis = int(vc::core::util::normalGridSliceAxis(direction));
    std::vector<vcr::Index> starts, sizes;
    std::vector<std::uint8_t*> outs;
    for (auto& target : targets) {
        const std::int64_t slice = std::int64_t(sliceAxisChunkIndex) * chunkShape[axis] + target.localSliceIndex;
        if (slice >= volumeShape[axis]) continue;
        vcr::Index start{0, 0, 0}, size{volumeShape[0], volumeShape[1], volumeShape[2]};
        start[axis] = slice;
        size[axis] = 1;
        starts.push_back(start);
        sizes.push_back(size);
        outs.push_back(target.binarySlice->ptr<std::uint8_t>());
    }
    source.array->read_many(starts, sizes, outs);
    for (auto& target : targets) {
        cv::Mat& m = *target.binarySlice;
        bool any = false;
        for (int r = 0; r < m.rows; ++r) {
            auto* row = m.ptr<std::uint8_t>(r);
            for (int c = 0; c < m.cols; ++c) {
                any |= row[c] != 0;
                row[c] = row[c] ? 255 : 0;
            }
        }
        target.anyNonZero = any;
    }
}

}  // namespace

// VC_BENCH_NORMAL_GRID_LEVEL=<zarr level dir> [VC_BENCH_NORMAL_GRID_CHUNK=<slab chunk index>]:
// times the chunk walk against the read_many formulation on one slab per direction (5.1).
TEST_CASE("normal-grid slab reads: chunk walk against read_many on a real level")
{
    const char* level = std::getenv("VC_BENCH_NORMAL_GRID_LEVEL");
    if (!level) {
        MESSAGE("set VC_BENCH_NORMAL_GRID_LEVEL to benchmark");
        return;
    }
    using namespace vc::core::util;
    const int chunkIndex = std::getenv("VC_BENCH_NORMAL_GRID_CHUNK") ? std::atoi(std::getenv("VC_BENCH_NORMAL_GRID_CHUNK")) : 0;
    for (const int threads : {0}) {
        auto source = vcr::zarr::openArray(level, {});
        const std::array<int, 3> shape{int(source.meta.shape[0]), int(source.meta.shape[1]), int(source.meta.shape[2])};
        const std::array<int, 3> chunks{int(source.meta.chunks[0]), int(source.meta.chunks[1]), int(source.meta.chunks[2])};
        for (auto direction : {NormalGridSliceDirection::XY, NormalGridSliceDirection::XZ, NormalGridSliceDirection::YZ}) {
            const int axis = int(normalGridSliceAxis(direction));
            const int depth = std::min(chunks[axis], shape[axis] - chunkIndex * chunks[axis]);
            const auto size = normalGridSliceSize({size_t(shape[0]), size_t(shape[1]), size_t(shape[2])}, direction);
            std::vector<cv::Mat> a(depth), b(depth);
            std::vector<BinarySliceTarget> ta, tb;
            for (int i = 0; i < depth; ++i) {
                a[i] = cv::Mat::zeros(size, CV_8U); b[i] = cv::Mat::zeros(size, CV_8U);
                ta.push_back({&a[i], i, false}); tb.push_back({&b[i], i, false});
            }
            double walkMs = 0, manyMs = 0;
            for (int rep = 0; rep < 3; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                fillBinarySliceBatchFromArray(source, direction, chunkIndex, shape, chunks, ta, threads);
                walkMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                t0 = std::chrono::steady_clock::now();
                fillBinarySliceBatchReadMany(source, direction, chunkIndex, shape, chunks, tb);
                manyMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
            bool same = true;
            long nonzero = 0;
            for (int i = 0; i < depth && same; ++i) {
                same = cv::countNonZero(a[i] != b[i]) == 0 && ta[i].anyNonZero == tb[i].anyNonZero;
                nonzero += cv::countNonZero(a[i]);
            }
            CHECK(same);
            MESSAGE("direction " << axis << ": " << depth << " slices of " << size.width << "x" << size.height << ", " << nonzero
                                 << " nonzero: chunk walk " << walkMs / 3 << " ms, read_many " << manyMs / 3 << " ms (mean of 3)");
        }
    }
}
