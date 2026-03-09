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

TEST(GenNormalGridsHelpers, ExtractBinarySliceMatchesExpectedXYXZYZ)
{
    xt::xtensor<uint8_t, 3, xt::layout_type::column_major> chunk =
        xt::xtensor<uint8_t, 3, xt::layout_type::column_major>::from_shape({2, 3, 4});
    chunk.fill(0);
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
