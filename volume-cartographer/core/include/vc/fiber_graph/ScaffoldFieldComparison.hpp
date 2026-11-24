#pragma once
// A scaffold solve against the winding solve of a resolved fiber volume.
//
// The two are solved separately and share no fiber. What relates them are the
// crossings between a scaffold fiber and a fiber of the volume: each states,
// exactly as a crossing inside either solve does, how many windings the two
// fibers are apart. This header extracts those crossings between the two
// stores - no union store, no second extraction of the volume - turns them
// into constraints with the shared assembly (ScaffoldConstraints.hpp, which
// calls vc_object_store's windingEventConstraints), and reads both solutions
// against them.
//
// The scaffold's normals and signs are the run's own; with the neighbours
// normal source they are transferred from the same volume (ScaffoldNormals),
// which is what makes the signed order of a cross crossing meaningful. The
// volume's fibers carry their signed strip normals.
#include "vc/fiber_graph/ScaffoldConstraints.hpp"
#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"
#include "vc/fiber_tracer/FiberNeighbourNormals.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vc::fiber_graph {

// The resolved volume: its fibers as signed strip samples, and per fiber the
// family and the winding its own solve gave it.
struct ScaffoldFieldVolume {
    std::vector<std::string> uuids;
    std::vector<std::vector<fiber_tracer::FiberStripSample>> samples;
    // 0 horizontal, 1 vertical; anything else leaves the fiber out.
    std::vector<int> family;
    // The volume's own solution; `solved` is 0 for a fiber it did not place.
    std::vector<std::int64_t> winding, component;
    std::vector<char> solved;
};

struct ScaffoldFieldComparisonOptions {
    ScaffoldEventOptions events;
    // How a cross crossing is turned into constraints: the scaffold run's own
    // options. Separations are rounded here whatever they say, so that every
    // crossing is one statement to score.
    ScaffoldMeasuredOptions measured;
};

// One constraint between a scaffold fiber and a fiber of the volume, with both
// solutions read against it.
struct ScaffoldFieldConstraint {
    ScaffoldConstraintKind kind = ScaffoldConstraintKind::Perpendicular;
    std::int64_t scaffold = 0, field = 0; // indices into the graph and the volume
    double arcScaffold = 0, arcField = 0;
    // The step from the scaffold fiber to the volume's fiber the crossing states.
    std::int64_t target = 0;
    double weight = 0, windingDistance = 0;
    // What the two solutions make of it: field winding - scaffold winding,
    // minus the offset of this pair of components. `placed` is false when
    // either solution did not place its fiber or the pair of components has
    // no offset.
    bool placed = false;
    std::int64_t step = 0;
    // The cost per unit weight: |step - target| for an equation, the part on
    // the wrong side for a sign constraint.
    double residual = 0;
};

// The two solutions have independent gauges, one per component each. A pair
// (scaffold component, volume component) joined by cross equations gets the
// integer offset that satisfies the largest weight of them.
struct ScaffoldFieldOffset {
    std::int64_t scaffoldComponent = 0, fieldComponent = 0, offset = 0;
    std::int64_t equations = 0;
    double weight = 0, agreeingWeight = 0;
};

struct ScaffoldFieldComparison {
    ScaffoldEventReport events;
    ScaffoldMeasuredReport measured;
    // Volume fibers taken into the search (a usable family and at least two samples).
    std::int64_t fieldFibers = 0;
    std::vector<ScaffoldFieldConstraint> constraints;
    std::vector<ScaffoldFieldOffset> offsets;
};

// `sign`, `winding` and `component` are the scaffold run's per-fiber results.
// `distances` measures the winding separation of a crossing; required when the
// event options ask for it.
[[nodiscard]] ScaffoldFieldComparison compareScaffoldWithField(
    const ScaffoldGraph& graph, const ScaffoldNormals& normals, std::span<const std::int64_t> sign,
    std::span<const std::int64_t> winding, std::span<const std::int64_t> component, const ScaffoldFieldVolume& field,
    const ScaffoldFieldComparisonOptions& options, const ScaffoldWindingDistances& distances);

} // namespace vc::fiber_graph
