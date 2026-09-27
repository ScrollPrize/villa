// Coverage for vc::surface::onPredictionSupport (see #1675): the fraction of
// valid surface vertices landing on nonzero prediction voxels. A grown surface
// that follows its prediction scores ~1.0; one that cut across windings scores
// near the volume background rate (~0.06-0.10 measured on PHerc1203).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/SurfaceSupport.hpp"

#include <limits>
#include <random>
#include <vector>

using vc::surface::onPredictionSupport;

// 8x8x8 toy prediction: a single lit "sheet" at z == 3, everything else 0.
struct ToyPrediction {
    std::vector<uint8_t> v = std::vector<uint8_t>(8 * 8 * 8, 0);
    ToyPrediction()
    {
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x) v[(3 * 8 + y) * 8 + x] = 255;
    }
    uint8_t operator()(int z, int y, int x) const
    {
        if (z < 0 || z >= 8 || y < 0 || y >= 8 || x < 0 || x >= 8)
            return 0;
        return v[(z * 8 + y) * 8 + x];
    }
};

static cv::Mat_<cv::Vec3f> make_points(int rows, int cols, cv::Vec3f fill)
{
    cv::Mat_<cv::Vec3f> pts(rows, cols);
    pts.setTo(fill);
    return pts;
}

TEST_CASE("onPredictionSupport: vertices on the sheet score 1.0")
{
    ToyPrediction pred;
    auto pts = make_points(4, 5, cv::Vec3f(2.f, 2.f, 3.f));  // x, y, z
    CHECK(onPredictionSupport(pts, pred).fraction == doctest::Approx(1.0));
}

TEST_CASE("onPredictionSupport: vertices off the sheet score 0.0")
{
    ToyPrediction pred;
    auto pts = make_points(4, 5, cv::Vec3f(2.f, 2.f, 6.f));
    CHECK(onPredictionSupport(pts, pred).fraction == doctest::Approx(0.0));
}

TEST_CASE("onPredictionSupport: mixed vertices give the exact fraction")
{
    ToyPrediction pred;
    auto pts = make_points(2, 4, cv::Vec3f(1.f, 1.f, 3.f));
    pts(0, 0) = cv::Vec3f(1.f, 1.f, 0.f);
    pts(1, 3) = cv::Vec3f(1.f, 1.f, 7.f);
    CHECK(onPredictionSupport(pts, pred).fraction == doctest::Approx(0.75));
}

TEST_CASE("onPredictionSupport: invalid vertices are skipped")
{
    ToyPrediction pred;
    auto pts = make_points(2, 2, cv::Vec3f(2.f, 2.f, 3.f));
    pts(0, 0) = cv::Vec3f(-1.f, -1.f, -1.f);  // invalid: not counted
    pts(0, 1) = cv::Vec3f(2.f, 2.f, 0.f);     // valid, off sheet
    // 2 of 3 valid vertices on the sheet
    const auto res = onPredictionSupport(pts, pred);
    CHECK(res.fraction == doctest::Approx(2.0 / 3.0));
    CHECK(res.total == 3);
    CHECK(res.on == 2);
}

TEST_CASE("onPredictionSupport: no valid vertices returns 1.0 (nothing to judge)")
{
    ToyPrediction pred;
    auto pts = make_points(3, 3, cv::Vec3f(-1.f, -1.f, -1.f));
    CHECK(onPredictionSupport(pts, pred).fraction == doctest::Approx(1.0));
}

TEST_CASE("onPredictionSupport: samples the nearest voxel")
{
    ToyPrediction pred;
    auto pts = make_points(1, 2, cv::Vec3f(2.f, 2.f, 3.4f));  // rounds to z=3
    pts(0, 1) = cv::Vec3f(2.f, 2.f, 3.6f);                   // rounds to z=4
    CHECK(onPredictionSupport(pts, pred).fraction == doctest::Approx(0.5));
}

TEST_CASE("onPredictionSupport: non-finite vertices are skipped, not rounded")
{
    ToyPrediction pred;
    auto pts = make_points(2, 2, cv::Vec3f(2.f, 2.f, 3.f));  // on the sheet
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    pts(0, 0) = cv::Vec3f(nan, 2.f, 3.f);  // NaN: invalid, must not reach lround
    pts(0, 1) = cv::Vec3f(2.f, inf, 3.f);   // Inf: invalid, must not reach lround
    pts(1, 0) = cv::Vec3f(2.f, 2.f, nan);  // NaN in z: invalid
    // only pts(1, 1) is valid and on the sheet
    const auto res = onPredictionSupport(pts, pred);
    CHECK(res.total == 1);
    CHECK(res.on == 1);
    CHECK(res.fraction == doctest::Approx(1.0));
}

TEST_CASE("validSupportThreshold: accepts only finite fractions in [0, 1]")
{
    using vc::surface::validSupportThreshold;
    CHECK(validSupportThreshold(0.0));
    CHECK(validSupportThreshold(0.5));
    CHECK(validSupportThreshold(1.0));
    CHECK(!validSupportThreshold(-0.1));   // would silently disable the check
    CHECK(!validSupportThreshold(1.1));    // would reject even a perfect surface
    CHECK(!validSupportThreshold(std::numeric_limits<double>::quiet_NaN()));
    CHECK(!validSupportThreshold(std::numeric_limits<double>::infinity()));
}

TEST_CASE("noBetterThanChance: flags surfaces at or below the background rate")
{
    using vc::surface::noBetterThanChance;
    using vc::surface::OnPredictionSupport;
    auto make = [](double fraction) {
        OnPredictionSupport s;
        s.fraction = fraction;
        s.total = 1000;  // a measured background: total != 0
        return s;
    };
    CHECK(noBetterThanChance(make(0.08), make(0.08)));   // equal: no better
    CHECK(noBetterThanChance(make(0.05), make(0.60)));   // below: no better
    CHECK(!noBetterThanChance(make(0.65), make(0.60)));  // above background: tracking
    CHECK(!noBetterThanChance(make(1.0), make(0.05)));   // clearly tracking
}

TEST_CASE("onPredictionSupport: background rate matches prediction density")
{
    // Uniform random points over the toy volume should land on the sheet at
    // roughly the sheet's share of voxels (1/8 of the 8x8x8 volume).
    ToyPrediction pred;
    cv::Mat_<cv::Vec3f> bg(2000, 1);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 8.0f);
    for (int i = 0; i < bg.rows; ++i) {
        bg(i, 0) = cv::Vec3f(dist(rng), dist(rng), dist(rng));
    }
    const auto res = onPredictionSupport(bg, pred);
    CHECK(res.total == 2000);
    CHECK(res.fraction == doctest::Approx(1.0 / 8.0).epsilon(0.05));
}

TEST_CASE("noBetterThanChance: unmeasured background never warns")
{
    using vc::surface::noBetterThanChance;
    using vc::surface::OnPredictionSupport;
    // An empty surface leaves the background at its default (total 0,
    // fraction 1.0); comparing two default 1.0 fractions must not warn.
    const OnPredictionSupport empty;
    CHECK(!noBetterThanChance(empty, empty));
    CHECK(!noBetterThanChance(empty, OnPredictionSupport{0.03, 60, 2000}));
    // Sanity: a measured background still behaves as before.
    CHECK(noBetterThanChance(OnPredictionSupport{0.05, 50, 1000},
                             OnPredictionSupport{0.06, 120, 2000}));
    CHECK(!noBetterThanChance(OnPredictionSupport{0.7, 700, 1000},
                              OnPredictionSupport{0.03, 60, 2000}));
}
TEST_CASE("isValidVertex: rejects sentinels and non-finite coordinates")
{
    using vc::surface::isValidVertex;
    CHECK(isValidVertex(cv::Vec3f(1.f, 2.f, 3.f)));
    CHECK(!isValidVertex(cv::Vec3f(-1.f, 2.f, 3.f)));
    CHECK(!isValidVertex(cv::Vec3f(1.f, -1.f, 3.f)));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    CHECK(!isValidVertex(cv::Vec3f(nan, 2.f, 3.f)));
    CHECK(!isValidVertex(cv::Vec3f(1.f, inf, 3.f)));
}

TEST_CASE("validVertexBounds: bounds of valid vertices, false when empty")
{
    using vc::surface::validVertexBounds;
    cv::Mat_<cv::Vec3f> pts(4, 1);
    pts(0, 0) = cv::Vec3f(1.f, 5.f, 3.f);
    pts(1, 0) = cv::Vec3f(4.f, 2.f, 7.f);
    pts(2, 0) = cv::Vec3f(-1.f, -1.f, -1.f);  // invalid sentinel
    pts(3, 0) = cv::Vec3f(std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f);
    cv::Vec3f lo, hi;
    CHECK(validVertexBounds(pts, lo, hi));
    CHECK(lo[0] == 1.f);
    CHECK(lo[1] == 2.f);
    CHECK(lo[2] == 3.f);
    CHECK(hi[0] == 4.f);
    CHECK(hi[1] == 5.f);
    CHECK(hi[2] == 7.f);

    cv::Mat_<cv::Vec3f> empty(2, 1);
    empty(0, 0) = cv::Vec3f(-1.f, -1.f, -1.f);
    empty(1, 0) = cv::Vec3f(-1.f, -1.f, -1.f);
    CHECK(!validVertexBounds(empty, lo, hi));
}

TEST_CASE("supportVerdict: rejected checks warn by default, discard in strict mode")
{
    using vc::surface::supportVerdict;
    using vc::surface::SupportVerdict;
    // A passing check always saves the surface.
    CHECK(supportVerdict(false, false) == SupportVerdict::Accept);
    CHECK(supportVerdict(false, true) == SupportVerdict::Accept);
    // A rejected check (below threshold or no better than chance) is advisory
    // by default: warn and keep the surface.
    CHECK(supportVerdict(true, false) == SupportVerdict::Warn);
    // Strict mode turns a rejected check into a discard.
    CHECK(supportVerdict(true, true) == SupportVerdict::Reject);
}

TEST_CASE("samplingFailureVerdict: unverified surface discarded only in strict mode")
{
    using vc::surface::samplingFailureVerdict;
    using vc::surface::SupportVerdict;
    // Default mode: warn that the check could not run, but keep the surface.
    CHECK(samplingFailureVerdict(false) == SupportVerdict::Warn);
    // Strict mode: the surface is unverified, discard it like a rejection.
    CHECK(samplingFailureVerdict(true) == SupportVerdict::Reject);
}

TEST_CASE("strictCleanupMayDeleteSegDir: --segment-name protects the shared target dir")
{
    using vc::surface::strictCleanupMayDeleteSegDir;
    // Default layout: the tool created a fresh per-run subfolder, so strict
    // cleanup may remove it.
    CHECK(strictCleanupMayDeleteSegDir(""));
    // With --segment-name, seg_dir IS the shared target directory (not created
    // by this run, may hold pre-existing segments): strict cleanup must leave
    // it in place; the rejected surface is simply not saved.
    CHECK(!strictCleanupMayDeleteSegDir("seg01"));
}

TEST_CASE("backgroundSampleHiBound: rounded samples always land inside the volume")
{
    using vc::surface::backgroundSampleHiBound;
    // 8-voxel dimension with a neighborhood far beyond the volume: the old
    // bound (8.0) let coordinates in (7.5, 8) round to 8, outside the volume,
    // where they were counted as off-prediction.
    CHECK(backgroundSampleHiBound(8, 72.0f) == doctest::Approx(7.5f));
    // A one-voxel dimension: roughly half the samples would previously round
    // to 1 and be miscounted.
    CHECK(backgroundSampleHiBound(1, 65.0f) == doctest::Approx(0.5f));
    // The neighborhood still wins when it is the tighter bound.
    CHECK(backgroundSampleHiBound(1000, 100.0f) == doctest::Approx(100.0f));
    // Empty dimension: hi lands below any lo >= 0, so the existing lo >= hi
    // guard reports "nothing to sample" instead of building a reversed
    // distribution.
    CHECK(backgroundSampleHiBound(0, 64.0f) < 0.0f);

    // End-to-end on the formula: 5000 uniform samples in [0, hi) never round
    // out of bounds.
    std::mt19937 rng(7);
    const float hi = backgroundSampleHiBound(8, 72.0f);
    std::uniform_real_distribution<float> dist(0.0f, hi);
    for (int i = 0; i < 5000; ++i) {
        const int r = static_cast<int>(std::lround(dist(rng)));
        CHECK(r >= 0);
        CHECK(r < 8);
    }
}
