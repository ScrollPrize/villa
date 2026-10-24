#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/TifxyzIdentity.hpp"

TEST_CASE("explicit TIFXYZ UUID is kept as given")
{
    CHECK(vc::util::resolveTifxyzUuid("PHerc1447-segment") == "PHerc1447-segment");
}

TEST_CASE("TIFXYZ UUID defaults to a fresh random UUID, not a path name")
{
    const std::string first = vc::util::resolveTifxyzUuid();
    const std::string second = vc::util::resolveTifxyzUuid();
    CHECK(vc::util::isUuid(first));
    CHECK(vc::util::isUuid(second));
    CHECK(first != second);
}
