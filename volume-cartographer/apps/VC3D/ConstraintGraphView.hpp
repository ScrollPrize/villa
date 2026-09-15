#pragma once
#include "vc/constraints/ConstraintGraph.hpp"
#include <limits>

namespace vc3d::constraint_graph {
struct InitialView {
    cv::Vec2d center{0,0};
    double scale = .5;
};

// Open at an inspectable scale (ten pixels per grid interval), at a junction
// on the seed fiber. Distant graph extents must not shrink this initial view.
inline InitialView initialView(const vc::constraints::Graph& graph)
{
    InitialView out;
    out.scale = 10.0 / graph.settings.spacing;
    const auto seed = graph.nodes.find(graph.seed);
    if (seed == graph.nodes.end() || seed->second.uv.empty()) return out;
    const auto& placement = seed->second;
    const auto& node = *placement.source;
    const size_t middle = node.fiber ? (node.rows/2)*node.cols + node.cols/2 : placement.uv.size()/2;
    out.center = placement.uv.at(middle);
    const auto center = out.center;
    double nearest = std::numeric_limits<double>::infinity();
    for (const auto& link : graph.links) {
        const auto* location = link.a.node == graph.seed ? &link.a : link.b.node == graph.seed ? &link.b : nullptr;
        if (!location) continue;
        const auto uv = vc::constraints::placedLocation(graph,*location);
        const auto delta = uv-center;
        if (delta.dot(delta) < nearest) { nearest = delta.dot(delta); out.center = uv; }
    }
    return out;
}
} // namespace vc3d::constraint_graph
