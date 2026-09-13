// Coverage for the physical-area helpers in SurfaceArea.hpp. The point of
// these is that an unknown voxel size must not turn into a 0 cm^2 area: a
// volume whose metadata carries no resolution reports voxelSize() == 0, and a
// fabricated 0 is indistinguishable from a genuinely tiny surface. See #1603.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/SurfaceArea.hpp"

#include <cmath>
#include <limits>

using vc::surface::areaCm2FromVox2;
using vc::surface::storeAreaMeta;

TEST_CASE("areaCm2FromVox2: converts with a known voxel size")
{
    // Measured on a real PHercParis4 trace at 2.4 um/voxel.
    const auto cm2 = areaCm2FromVox2(3625.31668921149, 2.4);
    REQUIRE(cm2.has_value());
    CHECK(*cm2 == doctest::Approx(0.000208818241).epsilon(1e-9));

    // 1e8 um^2 per cm^2, so a 1 um voxel makes the two numbers differ by 1e8.
    const auto unit = areaCm2FromVox2(1e8, 1.0);
    REQUIRE(unit.has_value());
    CHECK(*unit == doctest::Approx(1.0));
}

TEST_CASE("areaCm2FromVox2: refuses an unusable voxel size")
{
    CHECK_FALSE(areaCm2FromVox2(1000.0, 0.0).has_value());
    CHECK_FALSE(areaCm2FromVox2(1000.0, -2.4).has_value());
    CHECK_FALSE(areaCm2FromVox2(1000.0, std::numeric_limits<double>::quiet_NaN()).has_value());
    CHECK_FALSE(areaCm2FromVox2(1000.0, std::numeric_limits<double>::infinity()).has_value());
}

TEST_CASE("areaCm2FromVox2: refuses a non-finite area")
{
    CHECK_FALSE(areaCm2FromVox2(std::numeric_limits<double>::quiet_NaN(), 2.4).has_value());
    CHECK_FALSE(areaCm2FromVox2(std::numeric_limits<double>::infinity(), 2.4).has_value());
}

TEST_CASE("storeAreaMeta: records both areas when the voxel size is known")
{
    auto meta = utils::Json::object();
    storeAreaMeta(meta, 3625.31668921149, 2.4);

    REQUIRE(meta.contains("area_vx2"));
    REQUIRE(meta.contains("area_cm2"));
    CHECK(meta["area_vx2"].get_double() == doctest::Approx(3625.31668921149));
    CHECK(meta["area_cm2"].get_double() == doctest::Approx(0.000208818241).epsilon(1e-9));
}

TEST_CASE("storeAreaMeta: omits area_cm2 when the voxel size is unknown")
{
    auto meta = utils::Json::object();
    storeAreaMeta(meta, 15627277.310839027, 0.0);

    REQUIRE(meta.contains("area_vx2"));
    CHECK(meta["area_vx2"].get_double() == doctest::Approx(15627277.310839027));
    CHECK_FALSE(meta.contains("area_cm2"));
}

TEST_CASE("storeAreaMeta: clears a stale area_cm2 rather than leaving it")
{
    // The tracers seed meta from caller-supplied params, so the key can
    // already be present when the voxel size turns out to be unusable.
    auto meta = utils::Json::object();
    meta["area_cm2"] = 12.5;
    storeAreaMeta(meta, 4096.0, 0.0);

    CHECK_FALSE(meta.contains("area_cm2"));
    CHECK(meta["area_vx2"].get_double() == doctest::Approx(4096.0));
}
