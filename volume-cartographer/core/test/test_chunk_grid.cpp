#include "test.hpp"

#include "vc/core/util/ChunkGrid.hpp"

TEST(ChunkGrid, PowerOfTwoIndexingMatchesBitMath)
{
    constexpr int chunkSize = 64;
    constexpr int shift = 6;
    constexpr int mask = chunkSize - 1;

    EXPECT_TRUE(vc::core::detail::isPowerOfTwo(chunkSize));
    EXPECT_EQ(vc::core::detail::chunkIndex(0, chunkSize, true, shift), 0);
    EXPECT_EQ(vc::core::detail::chunkIndex(63, chunkSize, true, shift), 0);
    EXPECT_EQ(vc::core::detail::chunkIndex(64, chunkSize, true, shift), 1);
    EXPECT_EQ(vc::core::detail::localOffset(0, chunkSize, true, mask), 0);
    EXPECT_EQ(vc::core::detail::localOffset(63, chunkSize, true, mask), 63);
    EXPECT_EQ(vc::core::detail::localOffset(64, chunkSize, true, mask), 0);
}

TEST(ChunkGrid, NonPowerOfTwoIndexingUsesDivisionAndModulo)
{
    constexpr int chunkSize = 96;
    constexpr int shift = 0;
    constexpr int mask = chunkSize - 1;

    EXPECT_FALSE(vc::core::detail::isPowerOfTwo(chunkSize));
    EXPECT_EQ(vc::core::detail::chunkIndex(0, chunkSize, false, shift), 0);
    EXPECT_EQ(vc::core::detail::chunkIndex(95, chunkSize, false, shift), 0);
    EXPECT_EQ(vc::core::detail::chunkIndex(96, chunkSize, false, shift), 1);
    EXPECT_EQ(vc::core::detail::chunkIndex(191, chunkSize, false, shift), 1);
    EXPECT_EQ(vc::core::detail::chunkIndex(192, chunkSize, false, shift), 2);

    EXPECT_EQ(vc::core::detail::localOffset(0, chunkSize, false, mask), 0);
    EXPECT_EQ(vc::core::detail::localOffset(95, chunkSize, false, mask), 95);
    EXPECT_EQ(vc::core::detail::localOffset(96, chunkSize, false, mask), 0);
    EXPECT_EQ(vc::core::detail::localOffset(191, chunkSize, false, mask), 95);
    EXPECT_EQ(vc::core::detail::localOffset(192, chunkSize, false, mask), 0);
}
