// vc_grow_seg_from_seed picks mode "explicit_seed" from the --seed
// coordinates through is_explicit_seed(). A seed with one coordinate at zero
// is a seed the user gave and must be kept; only 0 0 0, the value origin
// starts with, reads as no seed.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../../apps/src/GrowSeedMode.hpp"

using vc::grow_seed::is_explicit_seed;

TEST_CASE("a seed with every coordinate nonzero is explicit")
{
    CHECK(is_explicit_seed({2464.0, 2411.0, 12410.0}));
}

TEST_CASE("a seed with one or two coordinates at zero is explicit")
{
    CHECK(is_explicit_seed({0.0, 3936.0, 830.0}));
    CHECK(is_explicit_seed({3936.0, 0.0, 830.0}));
    CHECK(is_explicit_seed({3936.0, 830.0, 0.0}));
    CHECK(is_explicit_seed({0.0, 0.0, 830.0}));
}

TEST_CASE("0 0 0 reads as no seed")
{
    CHECK_FALSE(is_explicit_seed({0.0, 0.0, 0.0}));
    CHECK_FALSE(is_explicit_seed(cv::Vec3d{}));
}
