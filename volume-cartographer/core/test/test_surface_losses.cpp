#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/core/util/QuadSurface.hpp"
#include "vc/tracer/CostFunctions.hpp"

TEST_CASE("surface residuals use bicubic positions and analytic tangents")
{
    cv::Mat_<cv::Vec3f> points(8, 9);
    for (int r = 0; r < points.rows; ++r)
        for (int c = 0; c < points.cols; ++c)
            points(r, c) = {float(c), float(r), float(c*c + 2*r*r)};
    const auto surface = QuadSurface(points, {1, 1}).pointSurface();
    double xyz[]{2.25, 3.5, 29.5625}, location[]{3.5, 2.25};
    const double* parameters[]{xyz, location};
    double residual[3], pointJacobian[9], locationJacobian[6];
    double* jacobians[]{pointJacobian, locationJacobian};
    SurfaceLossD loss(surface, 2);
    CHECK(loss.Evaluate(parameters, residual, jacobians));
    for (double value : residual) CHECK(value == doctest::Approx(0));
    const double expected[]{0, 2, 2, 0, 28, 9};
    for (int i = 0; i < 6; ++i) CHECK(locationJacobian[i] == doctest::Approx(expected[i]));
    for (int i = 0; i < 9; ++i) CHECK(pointJacobian[i] == doctest::Approx(i % 4 ? 0 : -2));
    ZLocationLoss zloss(surface, float(xyz[2]), 2);
    double zResidual, zJacobian[2];
    double* zJacobians[]{zJacobian};
    CHECK(zloss.Evaluate(parameters + 1, &zResidual, zJacobians));
    CHECK(zResidual == doctest::Approx(0));
    CHECK(zJacobian[0] == doctest::Approx(28));
    CHECK(zJacobian[1] == doctest::Approx(9));
    location[0] = -1;
    CHECK(loss.Evaluate(parameters, residual, jacobians));
    for (double value : residual) CHECK(value == 0);
    for (double value : locationJacobian) CHECK(value == 0);
}
