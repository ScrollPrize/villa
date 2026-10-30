// VC3D tools share one opened array per volume level: repeated requests return the
// same array, levels sample the stored voxels, and a coarser level's indices follow
// its placement (scale-only levels are block-reduced, half a voxel off-centre).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../../apps/VC3D/OpenedVolumes.hpp"
#include "support/TestZarrVolume.hpp"
#include "vc/core/types/Volume.hpp"

#include <vc_reader/zarr.hpp>

#include <chrono>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

TEST_CASE("OpenedVolumes shares opened levels and maps coordinates through placements")
{
    const auto dir = fs::temp_directory_path() /
                     ("vc-opened-volumes-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    vc::test::ZarrVolumeSpec spec;
    spec.shapeZYX = {16, 16, 16};
    spec.chunkShapeZYX = {8, 8, 8};
    spec.numLevels = 2;
    auto volume = vc::test::createZarrVolume(dir, spec);
    REQUIRE(volume);
    std::vector<std::uint8_t> level0(16 * 16 * 16), level1(8 * 8 * 8);
    for (std::size_t i = 0; i < level0.size(); ++i)
        level0[i] = std::uint8_t(i % 251);
    for (std::size_t i = 0; i < level1.size(); ++i)
        level1[i] = std::uint8_t(7 * i % 251);
    vcr::zarr::Writer::open(dir / "0").writeRegion({0, 0, 0}, {16, 16, 16}, level0.data());
    vcr::zarr::Writer::open(dir / "1").writeRegion({0, 0, 0}, {8, 8, 8}, level1.data());

    OpenedVolumes opened(64 << 20);
    const auto& pyramid = opened.pyramid(*volume);
    CHECK(pyramid.levels == std::vector<int>{0, 1});
    CHECK(pyramid.count() == 2);
    CHECK(pyramid.has(1));
    CHECK_FALSE(pyramid.has(2));
    CHECK(pyramid.dtype == "|u1");
    const auto a = opened.level(*volume, 0);
    const auto b = opened.level(*volume, 0);
    CHECK(a.array == b.array);
    const double zyx[3] = {3, 5, 7};
    float value = 0;
    a.array->array->sample_linear(zyx, 1, a.array->meta.type, false, &value);
    CHECK(value == float(level0[(3 * 16 + 5) * 16 + 7]));

    const auto coarse = opened.level(*volume, 1);
    CHECK(coarse.array != a.array);
    CHECK(coarse.working.scale[0] == doctest::Approx(1.0));
    // Level-0 voxel 4.5 is the centre of level-1 voxel 2.
    CHECK(coarse.array->placement.fromLevel0(0, coarse.working.toLevel0(0, 4.5)) == doctest::Approx(2.0));
    double centre[3];
    for (int axis = 0; axis < 3; ++axis)
        centre[axis] = coarse.array->placement.fromLevel0(axis, coarse.working.toLevel0(axis, 4.5));
    coarse.array->array->sample_linear(centre, 1, coarse.array->meta.type, false, &value);
    CHECK(value == float(level1[(2 * 8 + 2) * 8 + 2]));
    fs::remove_all(dir);
}
