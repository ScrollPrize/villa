#pragma once

#include "vc/fiber_graph/ScaffoldConstraints.hpp"
#include "vc/fiber_graph/ScaffoldNormals.hpp"
#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"

#include <filesystem>
#include <string>

// The default scaffold solve, in one place: what VC3D's solver dock runs and
// what `vc_scaffold_solve` starts from before its configuration overrides
// anything. A default changed here changes both; neither states a default of
// its own.
//
// The option structs' member defaults are the profile. This adds the choices
// that belong to no option struct (whether the crossings are measured, where
// the strip normals come from) and the ones the scaffold run sets differently
// from a struct's general default (the flow backend).
namespace vc::fiber_graph {

struct ScaffoldSolveProfile {
    // Measure the scaffold's own crossings beside the manual links: strip
    // normals, their signs, and the relations the crossings assert.
    bool measureCrossings = true;
    // Where the strip normals come from: "geometry" (the fibers' own),
    // "fused" (pulled towards the fiber-inference prediction's sheet axis) or
    // "neighbours" (pulled towards, and faced by, the signed normals of the
    // resolved fiber volume around the scaffold; scaffoldNeighbourNormalsPath).
    std::string normalSource = "neighbours";
    // Manual links (approved and pending) at weight 100, continuation links.
    ScaffoldSolveOptions solve;
    ScaffoldNormalOptions normals;
    ScaffoldEventOptions events;
    // Split separations, perpendicular 1, sign 3, no parallel or enclosure.
    ScaffoldMeasuredOptions measured;
    ScaffoldSignOptions sign;
};

[[nodiscard]] inline ScaffoldSolveProfile scaffoldSolveProfile()
{
    ScaffoldSolveProfile profile;
    profile.solve.flow.backend = geometry::WindingFlowBackend::CostScaling;
    // The links that anchor a normal are the links that constrain a winding.
    profile.normals.linkApprovalStates = profile.solve.links.approvalStates;
    profile.normals.includeAutoLinks = profile.solve.links.includeAuto;
    return profile;
}

// The resolved fiber volume's signed strip normals a `neighbours` run reads:
// the file beside the scaffold store, under this name.
[[nodiscard]] inline std::filesystem::path scaffoldNeighbourNormalsPath(const std::filesystem::path& fiberStore)
{
    return fiberStore.string() + ".neighbour_normals.vcstrip";
}

} // namespace vc::fiber_graph
