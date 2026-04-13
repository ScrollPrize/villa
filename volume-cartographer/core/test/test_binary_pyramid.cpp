#include "test.hpp"

#include <array>
#include <vector>

#include "vc/core/util/BinaryPyramid.hpp"

namespace {

using Shape3 = vc::core::util::Shape3;

std::size_t idx(const Shape3& shape, std::size_t z, std::size_t y, std::size_t x)
{
    return vc::core::util::linearIndex(shape, z, y, x);
}

}  // namespace

TEST(BinaryPyramid, DownsampleBinaryOrPreservesAnyOccupiedVoxel)
{
    const Shape3 srcShape = {2, 2, 2};
    const Shape3 dstShape = {1, 1, 1};
    std::vector<uint8_t> src(8, 0);
    std::vector<uint8_t> dst(1, 0);

    src[idx(srcShape, 1, 1, 1)] = 255;
    vc::core::util::downsampleBinaryOr(src.data(), srcShape, dst.data(), dstShape);

    EXPECT_EQ(dst[0], 255);
}

TEST(BinaryPyramid, DownsampleBinaryOrKeepsEmptyBlocksEmpty)
{
    const Shape3 srcShape = {4, 4, 4};
    const Shape3 dstShape = {2, 2, 2};
    std::vector<uint8_t> src(4 * 4 * 4, 0);
    std::vector<uint8_t> dst(2 * 2 * 2, 17);

    vc::core::util::downsampleBinaryOr(src.data(), srcShape, dst.data(), dstShape);

    for (const uint8_t v : dst) {
        EXPECT_EQ(v, 0);
    }
}

TEST(BinaryPyramid, DownsampleBinaryOrHandlesPartialEdgeChunks)
{
    const Shape3 srcShape = {3, 3, 3};
    const Shape3 dstShape = {2, 2, 2};
    std::vector<uint8_t> src(3 * 3 * 3, 0);
    std::vector<uint8_t> dst(2 * 2 * 2, 99);

    src[idx(srcShape, 2, 2, 2)] = 255;
    src[idx(srcShape, 0, 1, 0)] = 255;

    vc::core::util::downsampleBinaryOr(src.data(), srcShape, dst.data(), dstShape);

    EXPECT_EQ(dst[idx(dstShape, 0, 0, 0)], 255);
    EXPECT_EQ(dst[idx(dstShape, 1, 1, 1)], 255);
    EXPECT_EQ(dst[idx(dstShape, 0, 0, 1)], 0);
    EXPECT_EQ(dst[idx(dstShape, 0, 1, 0)], 0);
    EXPECT_EQ(dst[idx(dstShape, 0, 1, 1)], 0);
    EXPECT_EQ(dst[idx(dstShape, 1, 0, 0)], 0);
    EXPECT_EQ(dst[idx(dstShape, 1, 0, 1)], 0);
    EXPECT_EQ(dst[idx(dstShape, 1, 1, 0)], 0);
}

TEST(BinaryPyramid, DownsampleLabelPriorityPrefersForegroundOverIgnore)
{
    const Shape3 srcShape = {2, 2, 2};
    const Shape3 dstShape = {1, 1, 1};
    std::vector<uint8_t> src(8, 0);
    std::vector<uint8_t> dst(1, 0);

    src[idx(srcShape, 0, 0, 0)] = 127;
    src[idx(srcShape, 1, 1, 1)] = 255;

    vc::core::util::downsampleLabelPriority(src.data(), srcShape, dst.data(), dstShape, 127);

    EXPECT_EQ(dst[0], 255);
}

TEST(BinaryPyramid, DownsampleLabelPriorityFallsBackToIgnore)
{
    const Shape3 srcShape = {2, 2, 2};
    const Shape3 dstShape = {1, 1, 1};
    std::vector<uint8_t> src(8, 0);
    std::vector<uint8_t> dst(1, 0);

    src[idx(srcShape, 0, 1, 0)] = 127;

    vc::core::util::downsampleLabelPriority(src.data(), srcShape, dst.data(), dstShape, 127);

    EXPECT_EQ(dst[0], 127);
}

TEST(BinaryPyramid, DownsampleLabelPriorityPreservesForegroundLabelDeterministically)
{
    const Shape3 srcShape = {2, 2, 2};
    const Shape3 dstShape = {1, 1, 1};
    std::vector<uint8_t> src(8, 0);
    std::vector<uint8_t> dst(1, 0);

    src[idx(srcShape, 0, 0, 1)] = 11;
    src[idx(srcShape, 1, 1, 1)] = 255;

    vc::core::util::downsampleLabelPriority(src.data(), srcShape, dst.data(), dstShape, 127);

    EXPECT_EQ(dst[0], 11);
}
