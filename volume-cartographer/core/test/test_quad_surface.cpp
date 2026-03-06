#include "test.hpp"

#include <opencv2/core.hpp>

#include "vc/core/util/QuadSurface.hpp"

TEST(QuadSurfaceResample, SupportsAnisotropicFactors)
{
    cv::Mat_<cv::Vec3f> points(4, 4);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            points(row, col) = cv::Vec3f(static_cast<float>(col),
                                         static_cast<float>(row),
                                         1.0f);
        }
    }

    QuadSurface surf(points, cv::Vec2f(1.0f, 1.0f));
    surf.resample(2.0f, 0.5f, 1);

    EXPECT_EQ(surf.rawPointsPtr()->cols, 8);
    EXPECT_EQ(surf.rawPointsPtr()->rows, 2);
    EXPECT_FLOAT_EQ(surf._scale[0], 0.5f);
    EXPECT_FLOAT_EQ(surf._scale[1], 2.0f);
}

TEST(QuadSurfaceResample, UniformOverloadDelegatesToAxisWisePath)
{
    cv::Mat_<cv::Vec3f> points(3, 3, cv::Vec3f(1.0f, 2.0f, 3.0f));
    QuadSurface surf(points, cv::Vec2f(2.0f, 4.0f));

    surf.resample(2.0f, 1);

    EXPECT_EQ(surf.rawPointsPtr()->cols, 6);
    EXPECT_EQ(surf.rawPointsPtr()->rows, 6);
    EXPECT_FLOAT_EQ(surf._scale[0], 1.0f);
    EXPECT_FLOAT_EQ(surf._scale[1], 2.0f);
}
