#include "vc/fiber_graph/ScaffoldFieldComparison.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace vc::fiber_graph {

ScaffoldFieldComparison compareScaffoldWithField(const ScaffoldGraph& graph, const ScaffoldNormals& normals,
                                                 const std::span<const std::int64_t> sign,
                                                 const std::span<const std::int64_t> winding,
                                                 const std::span<const std::int64_t> component,
                                                 const ScaffoldFieldVolume& field,
                                                 const ScaffoldFieldComparisonOptions& options,
                                                 const ScaffoldWindingDistances& distances)
{
    const auto scaffoldFibers = graph.fibers.size();
    const auto fieldFibers = field.uuids.size();
    if (sign.size() != scaffoldFibers || winding.size() != scaffoldFibers || component.size() != scaffoldFibers)
        throw std::invalid_argument("scaffold field comparison: the scaffold results cover a different fiber set");
    if (field.samples.size() != fieldFibers || field.family.size() != fieldFibers || field.winding.size() != fieldFibers ||
        field.component.size() != fieldFibers || field.solved.size() != fieldFibers)
        throw std::invalid_argument("scaffold field comparison: the volume's columns differ in length");

    // One graph for the search: the scaffold's fibers, then the volume's as
    // the polylines of their strip samples. Nothing of it is stored.
    ScaffoldGraph combined = graph;
    combined.links.clear();
    std::vector<std::int64_t> fieldOf;                  // combined index - scaffoldFibers -> volume fiber
    std::vector<std::vector<double>> arcs;              // per taken volume fiber: arc at each sample
    for (std::size_t f = 0; f < fieldFibers; ++f) {
        const auto& samples = field.samples[f];
        if ((field.family[f] != 0 && field.family[f] != 1) || samples.size() < 2) continue;
        ScaffoldFiber fiber;
        fiber.uuid = field.uuids[f];
        fiber.family = field.family[f] == 0 ? ScaffoldFamily::Horizontal : ScaffoldFamily::Vertical;
        fiber.pointFirst = std::int64_t(combined.linePoints.size() / 3);
        fiber.pointCount = std::int64_t(samples.size());
        std::vector<double> arc(samples.size(), 0.);
        for (std::size_t i = 0; i < samples.size(); ++i) {
            for (int k = 0; k < 3; ++k) combined.linePoints.push_back(samples[i].point[k]);
            if (i > 0) arc[i] = arc[i - 1] + cv::norm(samples[i].point - samples[i - 1].point);
        }
        combined.index[fiber.uuid] = combined.fibers.size();
        combined.fibers.push_back(std::move(fiber));
        combined.offsets.push_back(std::int64_t(combined.linePoints.size() / 3));
        fieldOf.push_back(std::int64_t(f));
        arcs.push_back(std::move(arc));
    }
    ScaffoldFieldComparison out;
    out.fieldFibers = std::int64_t(fieldOf.size());
    if (fieldOf.empty()) throw std::invalid_argument("scaffold field comparison: the volume holds no usable fiber");
    const ScaffoldGeometry geometry(combined, std::max<std::size_t>(1, options.events.workers));

    auto eventOptions = options.events;
    eventOptions.cache = nullptr;
    eventOptions.sink = nullptr;
    eventOptions.crossQueryPrefix = scaffoldFibers;
    // The scaffold's normals are the run's own; the volume's are its signed strip normals.
    eventOptions.normalAt = [&](std::size_t fiber, double arc) -> std::optional<geometry::Point> {
        if (fiber < scaffoldFibers) return normals.at(fiber, arc);
        const auto local = fiber - scaffoldFibers;
        const auto& samples = field.samples[std::size_t(fieldOf[local])];
        const auto& grid = arcs[local];
        const auto upper = std::size_t(std::upper_bound(grid.begin(), grid.end(), arc) - grid.begin());
        const auto hi = std::clamp<std::size_t>(upper, 1, grid.size() - 1), lo = hi - 1;
        const double span = grid[hi] - grid[lo], t = span > 0 ? std::clamp((arc - grid[lo]) / span, 0., 1.) : 0.;
        const cv::Vec3d normal = samples[lo].normal * (1. - t) + samples[hi].normal * t;
        const double length = cv::norm(normal);
        if (!(length > 0)) return std::nullopt;
        return geometry::Point{normal[0] / length, normal[1] / length, normal[2] / length};
    };
    const auto events = scaffoldEvents(combined, geometry, normals, eventOptions, distances, out.events);

    // The scaffold's solved signs; the volume's normals are signed already.
    std::vector<std::int64_t> signs(combined.fibers.size(), 1);
    std::copy(sign.begin(), sign.end(), signs.begin());
    auto measuredOptions = options.measured;
    measuredOptions.separations = ScaffoldSeparationMode::Rounded;
    const auto constraints = scaffoldMeasuredConstraints(combined, geometry, events, signs, measuredOptions, out.measured);

    out.constraints.reserve(constraints.size());
    for (const auto& constraint : constraints) {
        // Node order puts the scaffold fiber first: its indices are the smaller ones.
        if (std::size_t(constraint.a) >= scaffoldFibers || std::size_t(constraint.b) < scaffoldFibers)
            throw std::logic_error("scaffold field comparison: a constraint does not join the two sets");
        ScaffoldFieldConstraint row;
        row.kind = constraint.kind;
        row.scaffold = constraint.a;
        row.field = fieldOf[std::size_t(constraint.b) - scaffoldFibers];
        row.arcScaffold = constraint.arcA;
        row.arcField = constraint.arcB;
        row.target = constraint.target;
        row.weight = constraint.weight;
        row.windingDistance = constraint.windingDistance;
        out.constraints.push_back(row);
    }

    // Per pair of components: the offset most of the equation weight agrees on.
    const auto side = [](ScaffoldConstraintKind kind) {
        return kind == ScaffoldConstraintKind::SignAtLeast ? 1 : kind == ScaffoldConstraintKind::SignAtMost ? -1 : 0;
    };
    std::map<std::pair<std::int64_t, std::int64_t>, std::map<std::int64_t, double>> votes;
    std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> equations;
    for (const auto& row : out.constraints) {
        if (side(row.kind) != 0 || !field.solved[std::size_t(row.field)]) continue;
        const std::pair key{component[std::size_t(row.scaffold)], field.component[std::size_t(row.field)]};
        votes[key][field.winding[std::size_t(row.field)] - winding[std::size_t(row.scaffold)] - row.target] += row.weight;
        ++equations[key];
    }
    std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> offsetOf;
    for (const auto& [key, candidates] : votes) {
        ScaffoldFieldOffset offset;
        offset.scaffoldComponent = key.first;
        offset.fieldComponent = key.second;
        offset.equations = equations[key];
        for (const auto& [value, weight] : candidates) {
            offset.weight += weight;
            if (weight > offset.agreeingWeight) { offset.agreeingWeight = weight; offset.offset = value; }
        }
        offsetOf[key] = offset.offset;
        out.offsets.push_back(offset);
    }
    for (auto& row : out.constraints) {
        if (!field.solved[std::size_t(row.field)]) continue;
        const auto found = offsetOf.find({component[std::size_t(row.scaffold)], field.component[std::size_t(row.field)]});
        if (found == offsetOf.end()) continue;
        row.placed = true;
        row.step = field.winding[std::size_t(row.field)] - winding[std::size_t(row.scaffold)] - found->second;
        const double difference = double(row.step - row.target);
        const int s = side(row.kind);
        row.residual = s == 0 ? std::abs(difference) : std::max(-double(s) * difference, 0.);
    }
    return out;
}

} // namespace vc::fiber_graph
