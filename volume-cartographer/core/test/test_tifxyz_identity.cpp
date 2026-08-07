#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/TifxyzIdentity.hpp"

TEST_CASE("explicit TIFXYZ UUID overrides a generic output directory")
{
    CHECK(vc::util::resolveTifxyzUuid(
              "/tmp/output_tifxyz", "/tmp/mesh.obj", "PHerc1447-segment") ==
          "PHerc1447-segment");
}

TEST_CASE("TIFXYZ UUID keeps the historical output-directory default")
{
    CHECK(vc::util::resolveTifxyzUuid(
              "/tmp/output_tifxyz", "/tmp/mesh.obj") == "output_tifxyz");
}

TEST_CASE("TIFXYZ UUID falls back to the input stem for a trailing output path")
{
    CHECK(vc::util::resolveTifxyzUuid(
              std::filesystem::path("/tmp/output/"), "/tmp/source-mesh.obj") ==
          "source-mesh");
}

TEST_CASE("TIFXYZ UUID rejects completely empty identity inputs")
{
    CHECK_THROWS_AS(
        vc::util::resolveTifxyzUuid(std::filesystem::path{}, std::filesystem::path{}),
        std::invalid_argument);
}
