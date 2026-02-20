#include "test.hpp"

#include <opencv2/core.hpp>

#include "vc/core/util/Geometry.hpp"

// --- tdist -------------------------------------------------------------------

TEST(Tdist, ZeroWhenDistanceMatchesTarget)
{
    cv::Vec3f a{0, 0, 0};
    cv::Vec3f b{3, 4, 0};
    // actual distance = 5
    EXPECT_FLOAT_EQ(tdist(a, b, 5.0f), 0.0f);
}

TEST(Tdist, ReturnsAbsDifference)
{
    cv::Vec3f a{0, 0, 0};
    cv::Vec3f b{3, 4, 0};
    // actual distance = 5, target = 3 => |5-3| = 2
    EXPECT_FLOAT_EQ(tdist(a, b, 3.0f), 2.0f);
    // target = 7 => |5-7| = 2
    EXPECT_FLOAT_EQ(tdist(a, b, 7.0f), 2.0f);
}

TEST(Tdist, IdenticalPointsReturnTarget)
{
    cv::Vec3f a{1, 2, 3};
    // distance = 0 => |0 - target| = target
    EXPECT_FLOAT_EQ(tdist(a, a, 4.5f), 4.5f);
}

// --- tdist_sum ---------------------------------------------------------------

TEST(TdistSum, EmptyTargetsReturnsZero)
{
    cv::Vec3f v{0, 0, 0};
    EXPECT_FLOAT_EQ(tdist_sum(v, {}, {}), 0.0f);
}

TEST(TdistSum, SumsSquaredErrors)
{
    cv::Vec3f v{0, 0, 0};
    std::vector<cv::Vec3f> tgts = {{3, 4, 0}, {0, 0, 5}};
    std::vector<float> tds = {5.0f, 5.0f};
    // Both distances match targets exactly => sum of squared errors = 0
    EXPECT_FLOAT_EQ(tdist_sum(v, tgts, tds), 0.0f);
}

TEST(TdistSum, AccumulatesNonZeroErrors)
{
    cv::Vec3f v{0, 0, 0};
    std::vector<cv::Vec3f> tgts = {{3, 4, 0}};  // dist = 5
    std::vector<float> tds = {3.0f};             // error = 2, squared = 4
    EXPECT_FLOAT_EQ(tdist_sum(v, tgts, tds), 4.0f);
}

// --- loc_valid ---------------------------------------------------------------

TEST(LocValid, ValidInteriorLocation)
{
    cv::Mat_<cv::Vec3f> grid(4, 4, cv::Vec3f(1, 1, 1));
    // l is [y, x] — (0.5, 0.5) is a valid interior point
    EXPECT_TRUE(loc_valid(grid, {0.5, 0.5}));
}

TEST(LocValid, OutOfBoundsReturnsFalse)
{
    cv::Mat_<cv::Vec3f> grid(4, 4, cv::Vec3f(1, 1, 1));
    EXPECT_FALSE(loc_valid(grid, {10.0, 10.0}));
    EXPECT_FALSE(loc_valid(grid, {-1.0, 0.0}));
}

TEST(LocValid, InvalidMarkerReturnsFalse)
{
    cv::Mat_<cv::Vec3f> grid(4, 4, cv::Vec3f(1, 1, 1));
    grid(1, 1) = cv::Vec3f(-1, -1, -1);
    // (1.0, 1.0) needs corners at (1,1),(1,2),(2,1),(2,2) — (1,1) is invalid
    EXPECT_FALSE(loc_valid(grid, {1.0, 1.0}));
}

TEST(LocValid, NegativeOneFirstComponentReturnsFalse)
{
    cv::Mat_<cv::Vec3f> grid(4, 4, cv::Vec3f(1, 1, 1));
    // The implementation checks l[0] == -1 as a fast-path
    EXPECT_FALSE(loc_valid(grid, {-1.0, 0.5}));
}

// --- at_int (bilinear interpolation) -----------------------------------------

TEST(AtInt, ExactIntegerCoordinates)
{
    cv::Mat_<cv::Vec3f> m(2, 2);
    m(0, 0) = {1, 2, 3};
    m(0, 1) = {4, 5, 6};
    m(1, 0) = {7, 8, 9};
    m(1, 1) = {10, 11, 12};

    auto v = at_int(m, {0.0f, 0.0f});
    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
    EXPECT_FLOAT_EQ(v[2], 3.0f);
}

TEST(AtInt, MidpointInterpolation)
{
    cv::Mat_<cv::Vec3f> m(2, 2);
    m(0, 0) = {0, 0, 0};
    m(0, 1) = {2, 0, 0};
    m(1, 0) = {0, 2, 0};
    m(1, 1) = {2, 2, 0};

    // Center of 2x2 cell
    auto v = at_int(m, {0.5f, 0.5f});
    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 1.0f);
    EXPECT_FLOAT_EQ(v[2], 0.0f);
}
