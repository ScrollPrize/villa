#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/tracer/Tracer.hpp"

#include "utils/Json.hpp"

namespace {

utils::Json growth_params(int generations)
{
    utils::Json params = utils::Json::object();
    params["generations"] = generations;
    return params;
}

} // namespace

TEST_CASE("resume_stop_generation: without resume_generations the JSON limit stands")
{
    utils::Json params = growth_params(8);
    CHECK(resume_stop_generation(params, 7, 8) == 8);
    CHECK(resume_stop_generation(params, 7, 16) == 16);
}

TEST_CASE("resume_stop_generation: resume_generations adds that many generations")
{
    utils::Json params = growth_params(8);
    params["resume_generations"] = 8;
    CHECK(resume_stop_generation(params, 7, 8) == 16);
}

TEST_CASE("resume_stop_generation: the JSON limit no longer caps the resumed run")
{
    utils::Json params = growth_params(8);
    params["resume_generations"] = 1;
    CHECK(resume_stop_generation(params, 7, 8) == 9);

    params["generations"] = 100;
    params["resume_generations"] = 2;
    CHECK(resume_stop_generation(params, 7, 100) == 10);
}

TEST_CASE("resume_stop_generation: counts from the rewind generation")
{
    utils::Json params = growth_params(8);
    params["resume_generations"] = 4;
    CHECK(resume_stop_generation(params, 3, 8) == 8);
}

TEST_CASE("resume_stop_generation: non-positive values grow nothing")
{
    utils::Json params = growth_params(8);
    params["resume_generations"] = 0;
    CHECK(resume_stop_generation(params, 7, 8) == 8);

    params["resume_generations"] = -5;
    CHECK(resume_stop_generation(params, 7, 8) == 8);
}
