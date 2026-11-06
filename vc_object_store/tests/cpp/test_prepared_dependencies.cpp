#include "TestSupport.hpp"
#include "internal/PreparedDependencies.hpp"
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;

namespace {
bool accepts(const auto& geometry, const Anchor& anchor)
{
    try { validateDependencies(anchor, geometry); return true; }
    catch (const std::exception&) { return false; }
}

void parity(GeometryData data, const std::vector<PointReference>& references)
{
    const auto prepared = prepareDependencies(data);
    const auto native = dependencyGeometry(data);
    std::vector<std::pair<Anchor,bool>> cases;
    Anchor whole;
    whole.wholeGeometry = true;
    cases.emplace_back(whole, accepts(native, whole));
    cases.emplace_back(Anchor{}, accepts(native, Anchor{}));
    for (const auto& point : references) {
        Anchor anchor;
        anchor.points = {point};
        cases.emplace_back(anchor, accepts(native, anchor));
        anchor.wholeGeometry = true;
        cases.emplace_back(anchor, accepts(native, anchor));
    }
    // The descriptor must not keep callbacks, spans or pointers into the source.
    data.arrays.clear();
    data.attributes.clear();
    for (const auto& [anchor, expected] : cases)
        require(accepts(prepared, anchor) == expected, "prepared/native dependency validation differs");
}
}

int main()
{
    try {
        vc::geometry::Fiber fiber;
        fiber.controlPoints = {{0,1,2},{3,4,5}};
        fiber.controlPointIds = {17,3};
        fiber.linePoints = {{6,7,8},{9,10,11},{12,13,14}};
        parity(vc::geometry::encode(fiber), {
            {"control_point_ids",{17}}, {"control_point_ids",{3}}, {"control_point_ids",{0}},
            {"control_point_ids",{-1}}, {"control_point_ids",{17,0}},
            {"control_points",{0}}, {"control_points",{2}}, {"line_points",{2}},
            {"line_points",{3}}, {"line_points",{-1}}, {"line_points",{1,0}},
            {"points",{0}}, {"point_ids",{3}}, {"missing",{0}}
        });
        parity(vc::geometry::encode(vc::geometry::PointCollection{{9,2},{{1,2,3},{4,5,6}}}), {
            {"point_ids",{9}}, {"point_ids",{2}}, {"point_ids",{1}},
            {"points",{0}}, {"control_point_ids",{9}}
        });
        parity(vc::geometry::encode(vc::geometry::Track{{{1,2,3},{4,5,6}}}), {
            {"points",{0}}, {"points",{1}}, {"points",{2}}, {"points",{-1}},
            {"point_ids",{0}}, {"points",{0,1}}
        });
        parity(vc::geometry::encode(vc::geometry::Surface{2,2,{1,1},
            {{1,2,3},{-1,-1,-1},{4,5,6},{7,8,9}}}), {
            {"points",{0,0}}, {"points",{0,1}}, {"points",{1,1}},
            {"points",{2,0}}, {"points",{0}}, {"points",{-1,0}}, {"point_ids",{0}}
        });
        std::cout << "prepared dependency parity tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
