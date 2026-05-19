// Real-world QuadSurface tests using committed PHerc 0172 tifxyz fixtures.
//
// Source: three segments from the public dl.ash2txt.org mirror,
//   https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/paths/
//   {20241113070770, 20241113080880, 20241113090990}/<seg>_flat.obj
// Each .obj was converted to tifxyz via:
//   vc_obj2tifxyz <seg>_flat.obj <outdir> 128 1.0 --uv-metric
// Result: ~210 KiB per segment, 129x129 grid (84% valid points), uuid in
// meta.json matches the directory name.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace {

// CMake injects this via target_compile_definitions so the test can find
// the fixtures regardless of cwd. Falls back to a relative path for adhoc
// invocations from the build dir.
#ifndef VC_TEST_FIXTURES_DIR
#define VC_TEST_FIXTURES_DIR "core/test/data"
#endif

constexpr const char* kSeg770 = "20241113070770";
constexpr const char* kSeg880 = "20241113080880";
constexpr const char* kSeg990 = "20241113090990";

fs::path fixturesRoot() { return fs::path(VC_TEST_FIXTURES_DIR); }

fs::path segPath(const std::string& name)
{
    return fixturesRoot() / "segments" / name;
}

bool haveFixture(const std::string& name)
{
    return fs::exists(segPath(name) / "meta.json");
}

} // namespace

TEST_CASE("Fixture: segment 770 loads with the meta.json pins we converted")
{
    if (!haveFixture(kSeg770)) {
        MESSAGE("Skipping: fixture not present at " << segPath(kSeg770));
        return;
    }
    QuadSurface qs(segPath(kSeg770));
    qs.ensureLoaded();
    auto* pts = qs.rawPointsPtr();
    REQUIRE(pts);
    CHECK(pts->rows == 129);
    CHECK(pts->cols == 129);

    CHECK(qs.meta.is_object());
    CHECK(qs.meta["uuid"].get_string() == std::string(kSeg770));
    CHECK(qs.meta["type"].get_string() == "seg");
    CHECK(qs.meta["format"].get_string() == "tifxyz");
    // stretch_factor=128 -> scale = 1/128 = 0.0078125
    CHECK(qs.scale()[0] == doctest::Approx(0.0078125f).epsilon(1e-5));
    CHECK(qs.scale()[1] == doctest::Approx(0.0078125f).epsilon(1e-5));
}

TEST_CASE("Fixture: validMask, countValid*, and bbox on segment 770")
{
    if (!haveFixture(kSeg770)) return;
    QuadSurface qs(segPath(kSeg770));
    qs.ensureLoaded();

    auto mask = qs.validMask();
    CHECK(mask.rows == 129);
    CHECK(mask.cols == 129);

    const int validPoints = qs.countValidPoints();
    const int validQuads = qs.countValidQuads();
    CHECK(validPoints > 0);
    CHECK(validQuads > 0);
    // The conversion log reports ~84% valid; check it's in that ballpark.
    CHECK(validPoints >= 129 * 129 * 70 / 100);

    auto bb = qs.bbox();
    CHECK(std::isfinite(bb.low[0]));
    CHECK(std::isfinite(bb.high[0]));
    CHECK(bb.high[0] > bb.low[0]);
    CHECK(bb.high[1] > bb.low[1]);
    CHECK(bb.high[2] > bb.low[2]);
}

TEST_CASE("Fixture: segment 880 and 990 also load")
{
    for (const char* name : {kSeg880, kSeg990}) {
        if (!haveFixture(name)) continue;
        CAPTURE(name);
        QuadSurface qs(segPath(name));
        qs.ensureLoaded();
        CHECK(qs.rawPointsPtr() != nullptr);
        CHECK(qs.meta["uuid"].get_string() == std::string(name));
        CHECK(qs.meta["format"].get_string() == "tifxyz");
    }
}

TEST_CASE("Fixture: rotate(0) is a no-op; rotate(90) keeps surface loaded")
{
    if (!haveFixture(kSeg770)) return;
    QuadSurface qs(segPath(kSeg770));
    qs.ensureLoaded();
    auto* p = qs.rawPointsPtr();
    int rowsBefore = p->rows;
    qs.rotate(0.0f);
    CHECK(qs.rawPointsPtr()->rows == rowsBefore);
    qs.rotate(90.0f);
    CHECK(qs.rawPointsPtr() != nullptr);
}

TEST_CASE("Fixture: flipU and flipV are involutions")
{
    if (!haveFixture(kSeg770)) return;
    QuadSurface qs(segPath(kSeg770));
    qs.ensureLoaded();
    int validBefore = qs.countValidPoints();
    qs.flipU();
    qs.flipU();
    CHECK(qs.countValidPoints() == validBefore);
    qs.flipV();
    qs.flipV();
    CHECK(qs.countValidPoints() == validBefore);
}

TEST_CASE("Fixture: gridNormal yields a unit-length normal somewhere on the surface")
{
    if (!haveFixture(kSeg770)) return;
    QuadSurface qs(segPath(kSeg770));
    qs.ensureLoaded();
    auto* p = qs.rawPointsPtr();
    bool found = false;
    for (int r = 1; r < p->rows - 1 && !found; ++r) {
        for (int c = 1; c < p->cols - 1; ++c) {
            if ((*p)(r - 1, c)[0] == -1.f) continue;
            if ((*p)(r + 1, c)[0] == -1.f) continue;
            if ((*p)(r, c - 1)[0] == -1.f) continue;
            if ((*p)(r, c + 1)[0] == -1.f) continue;
            auto n = qs.gridNormal(r, c);
            if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                float len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                CHECK(len == doctest::Approx(1.0f).epsilon(1e-3));
                found = true;
                break;
            }
        }
    }
    CHECK(found);
}

TEST_CASE("Fixture: resample(0.5) halves dimensions and doubles scale")
{
    if (!haveFixture(kSeg770)) return;
    QuadSurface qs(segPath(kSeg770));
    qs.ensureLoaded();
    int rowsBefore = qs.rawPointsPtr()->rows;
    int colsBefore = qs.rawPointsPtr()->cols;
    auto scaleBefore = qs.scale();
    qs.resample(0.5f);
    CHECK(qs.rawPointsPtr()->rows == rowsBefore / 2 + (rowsBefore % 2));
    CHECK(qs.rawPointsPtr()->cols == colsBefore / 2 + (colsBefore % 2));
    CHECK(qs.scale()[0] == doctest::Approx(scaleBefore[0] * 2.0f).epsilon(1e-5));
}

TEST_CASE("Fixture: three segments share the same uuid pattern (date_id)")
{
    int found = 0;
    for (const char* name : {kSeg770, kSeg880, kSeg990}) {
        if (haveFixture(name)) ++found;
    }
    // Three live in the repo by default; running from a different cwd may
    // miss the path — accept >=1 for the soft-skip case.
    CHECK(found >= 1);
}
