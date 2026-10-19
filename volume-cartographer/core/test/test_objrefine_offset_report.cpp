// Fixtures for the summary behind vc_objrefine's opt-in offset report
// (apps/src/vc_objrefine_offset_report.hpp). The tool's end-to-end contract
// (maps on the segment grid, refined surface unchanged by the flag) is in
// test_objrefine_offset_report_cli.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "../../apps/src/vc_objrefine_offset_report.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using vc_objrefine::percentile;
using vc_objrefine::summarizeOffsets;

namespace {
const float NaN = std::numeric_limits<float>::quiet_NaN();

cv::Mat_<std::uint8_t> allValid(int rows, int cols)
{
    return cv::Mat_<std::uint8_t>(rows, cols, std::uint8_t(255));
}
}

TEST_CASE("percentile interpolates linearly between order statistics")
{
    const std::vector<float> v{0.f, 1.f, 2.f, 3.f, 4.f};
    CHECK(percentile(v, 0.0) == doctest::Approx(0.0));
    CHECK(percentile(v, 0.5) == doctest::Approx(2.0));
    CHECK(percentile(v, 0.9) == doctest::Approx(3.6));
    CHECK(percentile(v, 1.0) == doctest::Approx(4.0));
    const std::vector<float> one{7.f};
    CHECK(percentile(one, 0.95) == doctest::Approx(7.0));
}

TEST_CASE("unmeasured cells are skipped; no-sheet cells are counted but not summarized")
{
    // One cell never measured (NaN), one that found no papyrus (opacity 0.1,
    // offset 0, which must not pull the statistics toward "on the sheet"),
    // and six measured offsets: -3 -1 0 2 6 10. Opacity exactly at the limit
    // (0.5) counts as found. The unmeasured cell is still a valid point, as
    // on a segment's last row and column, so it counts in valid_vertices.
    cv::Mat_<float> off = (cv::Mat_<float>(2, 4) <<
        NaN, 0.f, -3.f, -1.f,
        0.f, 2.f, 6.f, 10.f);
    cv::Mat_<float> alpha = (cv::Mat_<float>(2, 4) <<
        NaN, 0.1f, 1.f, 1.f,
        0.9f, 0.6f, 1.f, 0.5f);
    utils::Json s = summarizeOffsets(off, alpha, allValid(2, 4), 5.f, 0.5f);

    CHECK(s["rows"].get_int() == 2);
    CHECK(s["cols"].get_int() == 4);
    CHECK(s["valid_vertices"].get_int() == 8);
    CHECK(s["measured_vertices"].get_int() == 7);
    CHECK(s["no_sheet_vertices"].get_int() == 1);
    CHECK(s["summarized_vertices"].get_int() == 6);
    CHECK(s["no_sheet_fraction"].get_double() == doctest::Approx(1.0 / 7.0));

    CHECK(s["beyond_threshold_vertices"].get_int() == 2);   // 6 and 10
    CHECK(s["beyond_threshold_fraction"].get_double() == doctest::Approx(2.0 / 6.0));

    CHECK(s["offset"]["mean"].get_double() == doctest::Approx(14.0 / 6.0));
    CHECK(s["offset"]["p05"].get_double() == doctest::Approx(-2.5));
    CHECK(s["offset"]["p50"].get_double() == doctest::Approx(1.0));
    CHECK(s["offset"]["p95"].get_double() == doctest::Approx(9.0));

    // |offset| sorted: 0 1 2 3 6 10
    CHECK(s["abs_offset"]["p50"].get_double() == doctest::Approx(2.5));
    CHECK(s["abs_offset"]["p90"].get_double() == doctest::Approx(8.0));
    CHECK(s["abs_offset"]["p95"].get_double() == doctest::Approx(9.0));
    CHECK(s["abs_offset"]["p99"].get_double() == doctest::Approx(9.8));
    CHECK(s["abs_offset"]["max"].get_double() == doctest::Approx(10.0));
}

TEST_CASE("the threshold is strict and applies to the magnitude")
{
    cv::Mat_<float> off = (cv::Mat_<float>(1, 4) << -5.f, 5.f, -5.5f, 4.9f);
    cv::Mat_<float> alpha(1, 4, 1.f);
    utils::Json s = summarizeOffsets(off, alpha, allValid(1, 4), 5.f, 0.5f);
    CHECK(s["beyond_threshold_vertices"].get_int() == 1);
    CHECK(s["beyond_threshold_fraction"].get_double() == doctest::Approx(0.25));
}

TEST_CASE("nothing summarized gives null statistics, not zeros")
{
    cv::Mat_<float> off(3, 3, NaN);
    cv::Mat_<float> alpha(3, 3, NaN);
    off(1, 1) = 0.f;
    alpha(1, 1) = 0.f;
    utils::Json s = summarizeOffsets(off, alpha, allValid(3, 3), 5.f, 0.5f);
    CHECK(s["valid_vertices"].get_int() == 9);
    CHECK(s["measured_vertices"].get_int() == 1);
    CHECK(s["no_sheet_fraction"].get_double() == doctest::Approx(1.0));
    CHECK(s["summarized_vertices"].get_int() == 0);
    CHECK(s["beyond_threshold_fraction"].is_null());
    CHECK(s["offset"].size() == 0);
    CHECK(s["abs_offset"].size() == 0);

    cv::Mat_<float> none(2, 2, NaN);
    utils::Json e = summarizeOffsets(none, none, cv::Mat_<std::uint8_t>(2, 2, std::uint8_t(0)), 5.f, 0.5f);
    CHECK(e["valid_vertices"].get_int() == 0);
    CHECK(e["measured_vertices"].get_int() == 0);
    CHECK(e["no_sheet_fraction"].is_null());
}

TEST_CASE("maps of different sizes are rejected")
{
    cv::Mat_<float> a(2, 2, 0.f);
    cv::Mat_<float> b(2, 3, 0.f);
    CHECK_THROWS_AS(summarizeOffsets(a, b, allValid(2, 2), 5.f, 0.5f), std::invalid_argument);
    CHECK_THROWS_AS(summarizeOffsets(a, a, allValid(2, 3), 5.f, 0.5f), std::invalid_argument);
}
