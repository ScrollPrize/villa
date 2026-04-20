#include "test.hpp"

#include <vector>

#include "vc/core/util/NormalGridGenerate.hpp"

TEST(GenNormalGridsHelpers, BatchPlanRespectsBudgetForRealShape)
{
    const std::vector<size_t> shape = {21000, 6700, 9100};
    const auto xy = vc::core::util::planNormalGridBatch(
        shape, vc::core::util::NormalGridSliceDirection::XY, 32, 4, 512, 1);
    const auto xz = vc::core::util::planNormalGridBatch(
        shape, vc::core::util::NormalGridSliceDirection::XZ, 32, 4, 512, 1);
    const auto yz = vc::core::util::planNormalGridBatch(
        shape, vc::core::util::NormalGridSliceDirection::YZ, 32, 4, 512, 1);

    EXPECT_EQ(xy.chunkSizeTarget, static_cast<size_t>(8));
    EXPECT_EQ(xz.chunkSizeTarget, static_cast<size_t>(2));
    EXPECT_EQ(yz.chunkSizeTarget, static_cast<size_t>(3));
    EXPECT_LE(xy.estimatedBatchBytes, 512ull * 1024ull * 1024ull);
    EXPECT_LE(xz.estimatedBatchBytes, 512ull * 1024ull * 1024ull);
    EXPECT_LE(yz.estimatedBatchBytes, 512ull * 1024ull * 1024ull);
}

TEST(GenNormalGridsHelpers, BatchPlanFallsBackToOneSliceWithZeroBudget)
{
    const std::vector<size_t> shape = {128, 128, 128};
    const auto plan = vc::core::util::planNormalGridBatch(
        shape, vc::core::util::NormalGridSliceDirection::XY, 16, 4, 0, 1);
    EXPECT_EQ(plan.chunkSizeTarget, static_cast<size_t>(1));
}

TEST(GenNormalGridsHelpers, SampledChunkPlanGroupsSlicesBySourceChunk)
{
    const std::vector<size_t> shape = {10, 6, 8};
    const std::vector<size_t> chunk_shape = {4, 3, 5};
    const auto plans = vc::core::util::planNormalGridSampledChunks(
        shape,
        chunk_shape,
        vc::core::util::NormalGridSliceDirection::XY,
        3);

    ASSERT_EQ(plans.size(), static_cast<size_t>(3));
    EXPECT_EQ(plans[0].sourceChunkIndex, static_cast<size_t>(0));
    EXPECT_EQ(plans[0].sourceSliceBegin, static_cast<size_t>(0));
    ASSERT_EQ(plans[0].sampledSlices.size(), static_cast<size_t>(2));
    EXPECT_EQ(plans[0].sampledSlices[0].sliceIndex, static_cast<size_t>(0));
    EXPECT_EQ(plans[0].sampledSlices[0].localSliceIndex, static_cast<size_t>(0));
    EXPECT_EQ(plans[0].sampledSlices[1].sliceIndex, static_cast<size_t>(3));
    EXPECT_EQ(plans[0].sampledSlices[1].localSliceIndex, static_cast<size_t>(3));

    EXPECT_EQ(plans[1].sourceChunkIndex, static_cast<size_t>(1));
    ASSERT_EQ(plans[1].sampledSlices.size(), static_cast<size_t>(1));
    EXPECT_EQ(plans[1].sampledSlices[0].sliceIndex, static_cast<size_t>(6));
    EXPECT_EQ(plans[1].sampledSlices[0].localSliceIndex, static_cast<size_t>(2));

    EXPECT_EQ(plans[2].sourceChunkIndex, static_cast<size_t>(2));
    ASSERT_EQ(plans[2].sampledSlices.size(), static_cast<size_t>(1));
    EXPECT_EQ(plans[2].sampledSlices[0].sliceIndex, static_cast<size_t>(9));
    EXPECT_EQ(plans[2].sampledSlices[0].localSliceIndex, static_cast<size_t>(1));
}

TEST(GenNormalGridsHelpers, ExtractBinarySliceMatchesExpectedXYXZYZ)
{
    Array3D<uint8_t> chunk({2, 3, 4});
    chunk(1, 2, 3) = 5;
    chunk(0, 1, 2) = 9;
    chunk(1, 0, 1) = 7;

    cv::Mat binary;

    EXPECT_TRUE(vc::core::util::extractBinarySliceFromChunk(
        chunk, vc::core::util::NormalGridSliceDirection::XY, 1, binary));
    EXPECT_EQ(binary.rows, 3);
    EXPECT_EQ(binary.cols, 4);
    EXPECT_EQ(binary.at<uint8_t>(2, 3), 255);
    EXPECT_EQ(binary.at<uint8_t>(0, 0), 0);

    EXPECT_TRUE(vc::core::util::extractBinarySliceFromChunk(
        chunk, vc::core::util::NormalGridSliceDirection::XZ, 1, binary));
    EXPECT_EQ(binary.rows, 2);
    EXPECT_EQ(binary.cols, 4);
    EXPECT_EQ(binary.at<uint8_t>(0, 2), 255);
    EXPECT_EQ(binary.at<uint8_t>(1, 3), 0);

    EXPECT_TRUE(vc::core::util::extractBinarySliceFromChunk(
        chunk, vc::core::util::NormalGridSliceDirection::YZ, 2, binary));
    EXPECT_EQ(binary.rows, 2);
    EXPECT_EQ(binary.cols, 3);
    EXPECT_EQ(binary.at<uint8_t>(0, 1), 255);
    EXPECT_EQ(binary.at<uint8_t>(1, 2), 0);
}

TEST(GenNormalGridsHelpers, CopyBinarySliceRegionBlitsSubrect)
{
    std::array<int, 3> chunk_shape = {3, 4, 5};
    std::vector<uint8_t> chunk_data(static_cast<size_t>(chunk_shape[0] * chunk_shape[1] * chunk_shape[2]), 0);
    auto index_of = [&](int z, int y, int x) {
        return static_cast<size_t>((z * chunk_shape[1] + y) * chunk_shape[2] + x);
    };
    chunk_data[index_of(1, 0, 0)] = 5;
    chunk_data[index_of(1, 1, 2)] = 7;

    cv::Mat binary = cv::Mat::zeros(6, 7, CV_8U);
    bool any_nonzero = false;
    vc::core::util::copyBinarySliceRegionFromChunk(
        chunk_data.data(),
        chunk_shape,
        vc::core::util::NormalGridSliceDirection::XY,
        1,
        2,
        3,
        2,
        4,
        binary,
        any_nonzero);

    EXPECT_TRUE(any_nonzero);
    EXPECT_EQ(binary.at<uint8_t>(2, 4), 255);
    EXPECT_EQ(binary.at<uint8_t>(3, 6), 255);
    EXPECT_EQ(binary.at<uint8_t>(0, 0), 0);
}
