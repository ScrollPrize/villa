#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/AffineTransform.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace vc::core::util;

namespace {

cv::Matx44d makeScaleTranslate(double s, double tx, double ty, double tz)
{
    cv::Matx44d m = cv::Matx44d::eye();
    m(0, 0) = s; m(1, 1) = s; m(2, 2) = s;
    m(0, 3) = tx; m(1, 3) = ty; m(2, 3) = tz;
    return m;
}

bool matxApproxEq(const cv::Matx44d& a, const cv::Matx44d& b, double eps = 1e-9)
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            if (std::abs(a(r, c) - b(r, c)) > eps) return false;
    return true;
}

} // namespace

TEST_CASE("parseAffineTransformMatrix accepts a 3x4 row-major matrix")
{
    auto j = utils::Json::parse(R"({
        "transformation_matrix": [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 2.0, 30.0]
        ]
    })");
    auto m = parseAffineTransformMatrix(j);
    CHECK(m(0, 0) == doctest::Approx(2.0));
    CHECK(m(1, 3) == doctest::Approx(20.0));
    // bottom row defaulted to [0,0,0,1]
    CHECK(m(3, 0) == doctest::Approx(0.0));
    CHECK(m(3, 3) == doctest::Approx(1.0));
}

TEST_CASE("parseAffineTransformMatrix accepts a 4x4 with valid bottom row")
{
    auto j = utils::Json::parse(R"({
        "transformation_matrix": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    })");
    auto m = parseAffineTransformMatrix(j);
    CHECK(matxApproxEq(m, cv::Matx44d::eye()));
}

TEST_CASE("parseAffineTransformMatrix rejects missing key")
{
    auto j = utils::Json::parse(R"({"other": 1})");
    CHECK_THROWS_AS(parseAffineTransformMatrix(j), std::runtime_error);
}

TEST_CASE("parseAffineTransformMatrix rejects non-3/4 row count")
{
    auto j = utils::Json::parse(R"({"transformation_matrix": [[1,0,0,0],[0,1,0,0]]})");
    CHECK_THROWS_AS(parseAffineTransformMatrix(j), std::runtime_error);
}

TEST_CASE("parseAffineTransformMatrix rejects bad row width")
{
    auto j = utils::Json::parse(R"({"transformation_matrix": [[1,0,0],[0,1,0],[0,0,1]]})");
    CHECK_THROWS_AS(parseAffineTransformMatrix(j), std::runtime_error);
}

TEST_CASE("parseAffineTransformMatrix rejects bad bottom row in 4x4")
{
    auto j = utils::Json::parse(R"({
        "transformation_matrix": [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,2]
        ]
    })");
    CHECK_THROWS_AS(parseAffineTransformMatrix(j), std::runtime_error);
}

TEST_CASE("loadAffineTransformMatrix throws on empty path")
{
    CHECK_THROWS_AS(loadAffineTransformMatrix(""), std::runtime_error);
}

TEST_CASE("loadAffineTransformMatrix throws on missing file")
{
    CHECK_THROWS_AS(loadAffineTransformMatrix("/nonexistent/__no__/transform.json"),
                    std::runtime_error);
}

TEST_CASE("loadAffineTransformMatrix reads a real file")
{
    auto dir = std::filesystem::temp_directory_path() / "vc_test_affine";
    std::filesystem::create_directories(dir);
    auto p = dir / "transform.json";
    {
        std::ofstream f(p);
        f << R"({"transformation_matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0]]})";
    }
    auto m = loadAffineTransformMatrix(p);
    CHECK(matxApproxEq(m, cv::Matx44d::eye()));
    std::filesystem::remove_all(dir);
}

TEST_CASE("loadAffineTransformMatrixFromString")
{
    auto m = loadAffineTransformMatrixFromString(
        R"({"transformation_matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0]]})");
    CHECK(matxApproxEq(m, cv::Matx44d::eye()));
    CHECK_THROWS_AS(loadAffineTransformMatrixFromString(""), std::runtime_error);
}

TEST_CASE("composeAffineTransform yields second*first")
{
    auto a = makeScaleTranslate(2.0, 0, 0, 0);
    auto b = makeScaleTranslate(1.0, 1, 2, 3);
    auto c = composeAffineTransform(a, b); // = b * a
    // Apply to origin: a maps origin->origin, then b maps origin->(1,2,3)
    cv::Vec4d v(0, 0, 0, 1);
    cv::Vec4d out = c * v;
    CHECK(out[0] == doctest::Approx(1.0));
    CHECK(out[1] == doctest::Approx(2.0));
    CHECK(out[2] == doctest::Approx(3.0));
}

TEST_CASE("tryInvertAffineTransformMatrix succeeds on invertible affine")
{
    auto m = makeScaleTranslate(2.0, 5, 6, 7);
    auto inv = tryInvertAffineTransformMatrix(m);
    REQUIRE(inv.has_value());
    auto prod = composeAffineTransform(m, *inv); // *inv * m
    CHECK(matxApproxEq(prod, cv::Matx44d::eye(), 1e-9));
}

TEST_CASE("tryInvertAffineTransformMatrix returns nullopt for singular")
{
    cv::Matx44d m = cv::Matx44d::eye();
    m(0, 0) = 0.0; m(1, 1) = 0.0; m(2, 2) = 0.0;
    auto inv = tryInvertAffineTransformMatrix(m);
    CHECK_FALSE(inv.has_value());
}

TEST_CASE("invertAffineTransformMatrix throws on singular")
{
    cv::Matx44d m = cv::Matx44d::eye();
    m(0, 0) = 0.0; m(1, 1) = 0.0; m(2, 2) = 0.0;
    CHECK_THROWS_AS(invertAffineTransformMatrix(m), std::runtime_error);
}

TEST_CASE("invertAffineTransformMatrix returns the value for invertible")
{
    auto m = makeScaleTranslate(3.0, 1, 2, 3);
    auto inv = invertAffineTransformMatrix(m);
    auto prod = composeAffineTransform(m, inv);
    CHECK(matxApproxEq(prod, cv::Matx44d::eye(), 1e-9));
}

TEST_CASE("applyAffineTransform Vec3d happy path")
{
    auto m = makeScaleTranslate(2.0, 1, 2, 3);
    cv::Vec3d out;
    CHECK(applyAffineTransform(cv::Vec3d(1, 1, 1), m, out));
    CHECK(out[0] == doctest::Approx(3.0));
    CHECK(out[1] == doctest::Approx(4.0));
    CHECK(out[2] == doctest::Approx(5.0));
}

TEST_CASE("applyAffineTransform Vec3d rejects NaN input")
{
    cv::Vec3d out;
    CHECK_FALSE(applyAffineTransform(
        cv::Vec3d(std::nan(""), 0, 0), cv::Matx44d::eye(), out));
}

TEST_CASE("applyAffineTransform Vec3d rejects non-finite result")
{
    // Build a matrix that explodes a finite input into NaN: NaN in matrix.
    cv::Matx44d m = cv::Matx44d::eye();
    m(0, 0) = std::nan("");
    cv::Vec3d out;
    CHECK_FALSE(applyAffineTransform(cv::Vec3d(1, 1, 1), m, out));
}

TEST_CASE("applyAffineTransform Vec3f passes through sentinel")
{
    auto m = makeScaleTranslate(2.0, 1, 2, 3);
    auto out = applyAffineTransform(cv::Vec3f(-1.f, -1.f, -1.f), m);
    CHECK(out[0] == -1.0f);
}

TEST_CASE("applyAffineTransform Vec3f computes transformed")
{
    auto m = makeScaleTranslate(2.0, 1, 2, 3);
    auto out = applyAffineTransform(cv::Vec3f(1, 1, 1), m);
    CHECK(out[0] == doctest::Approx(3.0f));
    CHECK(out[1] == doctest::Approx(4.0f));
    CHECK(out[2] == doctest::Approx(5.0f));
}

TEST_CASE("applyAffineTransform Vec3f returns invalid sentinel on overflow")
{
    cv::Matx44d m = cv::Matx44d::eye();
    m(0, 0) = 1e40; // > float max
    auto out = applyAffineTransform(cv::Vec3f(1, 0, 0), m);
    CHECK(out[0] == -1.0f);
}

TEST_CASE("transformNormal: identity preserves and normalizes")
{
    cv::Vec3f n(2.f, 0.f, 0.f); // not unit; transformNormal renormalizes
    auto out = transformNormal(n, cv::Matx44d::eye());
    CHECK(out[0] == doctest::Approx(1.0f));
    CHECK(out[1] == doctest::Approx(0.0f));
    CHECK(out[2] == doctest::Approx(0.0f));
}

TEST_CASE("transformNormal passes through non-finite input")
{
    cv::Vec3f n(std::nanf(""), 0, 0);
    auto out = transformNormal(n, cv::Matx44d::eye());
    CHECK(std::isnan(out[0]));
}

TEST_CASE("transformNormal returns input when matrix is singular")
{
    cv::Vec3f n(1.f, 0.f, 0.f);
    cv::Matx44d m = cv::Matx44d::eye();
    m(0, 0) = m(1, 1) = m(2, 2) = 0.0;
    auto out = transformNormal(n, m);
    CHECK(out[0] == doctest::Approx(1.0f));
}

TEST_CASE("affineUniformScaleFactor: uniform scale")
{
    auto m = makeScaleTranslate(2.5, 0, 0, 0);
    auto s = affineUniformScaleFactor(m);
    REQUIRE(s.has_value());
    CHECK(*s == doctest::Approx(2.5));
}

TEST_CASE("affineUniformScaleFactor: non-uniform returns nullopt")
{
    cv::Matx44d m = cv::Matx44d::eye();
    m(0, 0) = 1.0; m(1, 1) = 2.0; m(2, 2) = 3.0;
    auto s = affineUniformScaleFactor(m);
    CHECK_FALSE(s.has_value());
}

TEST_CASE("affineUniformScaleFactor: zero scale returns nullopt")
{
    cv::Matx44d m = cv::Matx44d::eye();
    m(0, 0) = m(1, 1) = m(2, 2) = 0.0;
    auto s = affineUniformScaleFactor(m);
    CHECK_FALSE(s.has_value());
}

TEST_CASE("applyPreAffineScale: scale 1 is identity")
{
    cv::Vec3f p(1.f, 2.f, 3.f);
    auto out = applyPreAffineScale(p, 1);
    CHECK(out[0] == doctest::Approx(1.0f));
}

TEST_CASE("applyPreAffineScale: sentinel passes through")
{
    cv::Vec3f p(-1.f, -1.f, -1.f);
    auto out = applyPreAffineScale(p, 4);
    CHECK(out[0] == -1.0f);
}

TEST_CASE("applyPreAffineScale: multiplies by scale")
{
    cv::Vec3f p(1.f, 2.f, 3.f);
    auto out = applyPreAffineScale(p, 4);
    CHECK(out[0] == doctest::Approx(4.0f));
    CHECK(out[1] == doctest::Approx(8.0f));
    CHECK(out[2] == doctest::Approx(12.0f));
}

TEST_CASE("transformSurfacePoints null surface is a no-op")
{
    transformSurfacePoints(nullptr, 1, std::nullopt);
    transformSurfacePoints(nullptr, 1.0, std::nullopt, 1.0);
    CHECK(true);
}

TEST_CASE("refreshTransformedSurfaceState null surface is a no-op")
{
    refreshTransformedSurfaceState(nullptr);
    CHECK(true);
}

TEST_CASE("cloneSurfaceForTransform null returns null")
{
    auto out = cloneSurfaceForTransform(nullptr);
    CHECK(out == nullptr);
}
