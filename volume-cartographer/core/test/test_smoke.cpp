#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("smoke: arithmetic")
{
    CHECK(1 + 1 == 2);
}

TEST_CASE("smoke: floating point")
{
    CHECK(2.0 * 3.0 == doctest::Approx(6.0));
}
