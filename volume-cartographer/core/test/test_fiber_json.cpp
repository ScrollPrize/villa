#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberJson.hpp"

TEST_CASE("Fiber JSON parses version-4 transfer documents and rejects other versions")
{
    nlohmann::json input = {{"type", "vc3d_fiber"}, {"version", 4},
        {"uuid", "12345678-1234-1234-1234-123456789abc"},
        {"line_points", {{1, 2, 3}, {4, 5, 6}}},
        {"control_points", {{{"position", {1, 2, 3}}}}}};
    auto fiber = vc::fiber_tracer::parseVc3dFiberJson(input, "test");
    REQUIRE(fiber.linePoints.size() == 2);
    CHECK(fiber.linePoints[1][2] == 6);
    CHECK(fiber.optimizationMode == "lasagna");
    input["version"] = 1;
    CHECK_THROWS_AS(vc::fiber_tracer::parseVc3dFiberJson(input, "test"), std::runtime_error);
}
