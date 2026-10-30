#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/core/util/NormalGridGenerate.hpp"
#include <filesystem>
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
