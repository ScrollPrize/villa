#include "vc/fiber_graph/ScaffoldNormals.hpp"

#include "vc/fiber_tracer/FiberDisplay.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace vc::fiber_graph {
namespace {

cv::Vec3d vectorOf(const geometry::Point& point) { return {point[0], point[1], point[2]}; }
geometry::Point pointOf(const cv::Vec3d& value) { return {value[0], value[1], value[2]}; }

// The spans approved for one aspect, as a membership test per span.
std::vector<char> approvedSpanFlags(const ScaffoldFiber& fiber, objects::FiberPieceAspect aspect)
{
    const auto controls = fiber.content.controlIds.size();
    std::vector<char> flags(controls == 0 ? 0 : controls - 1, 0);
    for (const auto span : fiber.approvals.approvedSpans[std::size_t(aspect)])
        if (span < flags.size()) flags[span] = 1;
    return flags;
}

// The H and V fiber of an adjacent link, or nullopt when the pair is not one
// of each (an annotation error that fixes no normal).
std::optional<std::pair<std::int64_t, std::int64_t>> adjacentPair(const ScaffoldGraph& graph, const ScaffoldLink& link)
{
    if (link.a < 0 || link.b < 0 || link.a == link.b) return std::nullopt;
    const auto a = graph.fibers[std::size_t(link.a)].family, b = graph.fibers[std::size_t(link.b)].family;
    if (a == ScaffoldFamily::Horizontal && b == ScaffoldFamily::Vertical) return std::pair{link.a, link.b};
    if (a == ScaffoldFamily::Vertical && b == ScaffoldFamily::Horizontal) return std::pair{link.b, link.a};
    return std::nullopt;
}

} // namespace

const char* scaffoldSignSourceName(ScaffoldSignSource source)
{
    switch (source) {
        case ScaffoldSignSource::None: return "none";
        case ScaffoldSignSource::Manual: return "manual";
        case ScaffoldSignSource::ApprovedPiece: return "approved_piece";
        case ScaffoldSignSource::AdjacentLink: return "adjacent_link";
        case ScaffoldSignSource::Crossings: return "crossings";
    }
    throw std::invalid_argument("scaffold normals: unknown sign source");
}

ScaffoldNormals::ScaffoldNormals(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                const ScaffoldNormalOptions& options)
    : graph_(&graph)
    , geometry_(&geometry)
{
    if (!std::isfinite(options.anchorWeight) || options.anchorWeight <= 0)
        throw std::invalid_argument("scaffold normals: the anchor weight must be finite and positive");
    const auto started = std::chrono::steady_clock::now();
    const auto fibers = graph.fibers.size();
    report_.fibers = std::int64_t(fibers);
    signSources_.assign(fibers, ScaffoldSignSource::None);
    radialAgreement_.assign(fibers, 0.);
    fixedSigns_.assign(fibers, 0);
    solved_.assign(fibers, 0);
    normals_.assign(graph.linePoints.size(), 0.);

    // The annotated anchors, in precedence order: a manual display normal
    // beats an approved span's, which beats an adjacent link's.
    if (options.manualDisplayNormals || options.approvedNormalPieces) {
        for (std::size_t i = 0; i < fibers; ++i) {
            const auto& fiber = graph.fibers[i];
            const auto approved = approvedSpanFlags(fiber, objects::FiberPieceAspect::Normals);
            for (std::size_t control = 0; control < fiber.content.annotations.size(); ++control) {
                const auto& annotation = fiber.content.annotations[control];
                const auto normal = fiber_tracer::displayNormalFromJson(annotation);
                if (!normal) continue;
                const auto source = fiber_tracer::displayNormalSourceFromJson(annotation);
                ScaffoldSignSource kind = ScaffoldSignSource::None;
                if (options.manualDisplayNormals && source == "manual") kind = ScaffoldSignSource::Manual;
                else if (options.approvedNormalPieces &&
                         ((control > 0 && control - 1 < approved.size() && approved[control - 1] != 0) ||
                          (control < approved.size() && approved[control] != 0)))
                    kind = ScaffoldSignSource::ApprovedPiece;
                if (kind == ScaffoldSignSource::None) continue;
                ScaffoldNormalAnchor anchor;
                anchor.fiber = std::int64_t(i);
                anchor.arc = geometry.controlArc(i, control);
                // The format's display normals point outward.
                anchor.inward = pointOf(-*normal);
                anchor.source = kind;
                anchor.control = std::int64_t(control);
                anchor.weight = options.anchorWeight;
                anchors_.push_back(anchor);
                if (kind == ScaffoldSignSource::Manual) ++report_.manualAnchors;
                else ++report_.approvedPieceAnchors;
            }
        }
    }
    if (options.adjacentLinkAnchors) {
        for (std::size_t index = 0; index < graph.links.size(); ++index) {
            const auto& link = graph.links[index];
            if (!link.adjacent) continue;
            if (std::find(options.linkApprovalStates.begin(), options.linkApprovalStates.end(), link.approvalState) ==
                options.linkApprovalStates.end()) {
                ++report_.skippedLinks;
                continue;
            }
            if (link.autoLink && !options.includeAutoLinks) { ++report_.skippedLinks; continue; }
            if (!link.anchorA || !link.anchorB) { ++report_.skippedLinks; continue; }
            const auto pair = adjacentPair(graph, link);
            if (!pair) { ++report_.skippedLinks; continue; }
            const auto [horizontal, vertical] = *pair;
            const auto& anchorH = horizontal == link.a ? *link.anchorA : *link.anchorB;
            const auto& anchorV = vertical == link.a ? *link.anchorA : *link.anchorB;
            const auto arcH = geometry.project(std::size_t(horizontal), anchorH).arc;
            const auto arcV = geometry.project(std::size_t(vertical), anchorV).arc;
            const auto pointH = geometry.packed().sample(horizontal, arcH).point;
            const auto pointV = geometry.packed().sample(vertical, arcV).point;
            // The V fiber lies one winding inside the H fiber: both inward
            // normals point from the H anchor towards the V anchor.
            const auto inward = fiber_tracer::displayUnit(vectorOf(pointV) - vectorOf(pointH));
            if (!inward) { ++report_.degenerateAnchors; continue; }
            for (const auto [fiber, arc] : {std::pair{horizontal, arcH}, std::pair{vertical, arcV}}) {
                ScaffoldNormalAnchor anchor;
                anchor.fiber = fiber;
                anchor.arc = arc;
                anchor.inward = pointOf(*inward);
                anchor.source = ScaffoldSignSource::AdjacentLink;
                anchor.link = std::int64_t(index);
                anchor.weight = options.anchorWeight;
                anchors_.push_back(anchor);
                ++report_.adjacentLinkAnchors;
            }
        }
    }

    // The strip solve, per fiber, with the anchors as fixed targets on the
    // chain point nearest each anchor arc. A chain point already claimed by a
    // stronger anchor keeps it.
    std::vector<std::vector<cv::Vec3d>> lines(fibers);
    std::vector<std::vector<fiber_tracer::FiberStripAxisTarget>> targets(fibers);
    std::vector<std::vector<int>> claimed(fibers);
    for (std::size_t i = 0; i < fibers; ++i) {
        const auto& fiber = graph.fibers[i];
        lines[i].resize(std::size_t(fiber.pointCount));
        for (std::int64_t j = 0; j < fiber.pointCount; ++j) {
            const auto at = std::size_t(3 * (fiber.pointFirst + j));
            lines[i][std::size_t(j)] = {graph.linePoints[at], graph.linePoints[at + 1], graph.linePoints[at + 2]};
        }
    }
    std::vector<std::vector<double>> chainArcs(fibers);
    for (std::size_t i = 0; i < fibers; ++i) {
        const auto chain = fiber_tracer::fiberStripChain(lines[i], options.strip);
        chainArcs[i] = chain.arc;
        targets[i].assign(chain.points.size(), fiber_tracer::FiberStripAxisTarget{});
        claimed[i].assign(chain.points.size(), 0);
    }
    std::stable_sort(anchors_.begin(), anchors_.end(), [](const ScaffoldNormalAnchor& x, const ScaffoldNormalAnchor& y) {
        return std::tie(x.fiber, x.source, x.arc) < std::tie(y.fiber, y.source, y.arc);
    });
    for (const auto& anchor : anchors_) {
        const auto fiber = std::size_t(anchor.fiber);
        const auto& arcs = chainArcs[fiber];
        if (arcs.empty()) continue;
        const auto upper = std::lower_bound(arcs.begin(), arcs.end(), anchor.arc);
        std::size_t at = std::size_t(upper - arcs.begin());
        if (at >= arcs.size()) at = arcs.size() - 1;
        else if (at > 0 && anchor.arc - arcs[at - 1] < arcs[at] - anchor.arc) at = at - 1;
        if (claimed[fiber][at]) continue;
        claimed[fiber][at] = 1;
        targets[fiber][at] = {vectorOf(anchor.inward), anchor.weight, true};
        if (signSources_[fiber] == ScaffoldSignSource::None ||
            int(anchor.source) < int(signSources_[fiber]))
            signSources_[fiber] = anchor.source;
    }
    for (std::size_t i = 0; i < fibers; ++i)
        if (signSources_[i] != ScaffoldSignSource::None) fixedSigns_[i] = 1;

    const auto results = fiber_tracer::solveFiberStripNormals(lines, options.strip,
                                                              int(std::max<std::size_t>(1, options.workers)), &targets);
    if (results.size() != fibers) throw std::runtime_error("scaffold normals: strip solve result count differs");
    for (std::size_t i = 0; i < fibers; ++i) {
        const auto& fiber = graph.fibers[i];
        const auto& result = results[i];
        if (result.normals.size() != std::size_t(fiber.pointCount)) {
            ++report_.degenerate;
            // A fiber whose geometry carries no normal keeps zeros and no
            // sign: it witnesses nothing.
            signSources_[i] = ScaffoldSignSource::None;
            fixedSigns_[i] = 0;
            continue;
        }
        ++report_.solved;
        if (result.converged) ++report_.converged;
        solved_[i] = 1;
        double weightedDot = 0, totalWeight = 0;
        const auto arcs = geometry.arcs();
        for (std::int64_t j = 0; j < fiber.pointCount; ++j) {
            const auto at = std::size_t(3 * (fiber.pointFirst + j));
            const auto& normal = result.normals[std::size_t(j)];
            for (int k = 0; k < 3; ++k) normals_[at + std::size_t(k)] = normal[k];
            // The arc each line point stands for (half of its neighbouring
            // segments), so an unevenly sampled fiber is not skewed.
            const auto first = std::size_t(fiber.pointFirst);
            const double previous = j > 0 ? arcs[first + std::size_t(j)] - arcs[first + std::size_t(j - 1)] : 0.;
            const double next = j + 1 < fiber.pointCount ? arcs[first + std::size_t(j + 1)] - arcs[first + std::size_t(j)] : 0.;
            const double weight = (previous + next) / 2.;
            const auto relative = geometry.seam().relative(
                geometry::Point{graph.linePoints[at], graph.linePoints[at + 1], graph.linePoints[at + 2]});
            const double radius = std::sqrt(relative[0] * relative[0] + relative[1] * relative[1]);
            if (!(radius > 0) || !(weight > 0)) continue;
            weightedDot += weight * (normal[0] * relative[0] + normal[1] * relative[1]) / radius;
            totalWeight += weight;
        }
        radialAgreement_[i] = totalWeight > 0 ? std::clamp(weightedDot / totalWeight, -1., 1.) : 0.;
    }
    // An anchored fiber's gauge is the inward one by construction; the
    // others are decided by the sign solve.
    for (std::size_t i = 0; i < fibers; ++i) {
        if (signSources_[i] == ScaffoldSignSource::None) ++report_.unanchoredFibers;
        else ++report_.anchoredFibers;
    }
    report_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

std::optional<geometry::Point> ScaffoldNormals::at(std::size_t fiber, double arc) const
{
    if (!solved(fiber)) return std::nullopt;
    const auto& entry = graph_->fibers.at(fiber);
    const auto arcs = geometry_->arcs();
    const auto first = std::size_t(entry.pointFirst), count = std::size_t(entry.pointCount);
    const auto begin = arcs.begin() + std::ptrdiff_t(first), end = begin + std::ptrdiff_t(count);
    auto upper = std::lower_bound(begin, end, arc);
    if (upper == end) upper = end - 1;
    std::size_t second = std::size_t(upper - begin);
    if (second == 0) second = 1;
    const std::size_t low = second - 1;
    const double span = arcs[first + second] - arcs[first + low];
    const double t = span > 0 ? std::clamp((arc - arcs[first + low]) / span, 0., 1.) : 0.;
    geometry::Point value{};
    double length = 0;
    for (std::size_t k = 0; k < 3; ++k) {
        value[k] = normals_[3 * (first + low) + k] * (1 - t) + normals_[3 * (first + second) + k] * t;
        length += value[k] * value[k];
    }
    length = std::sqrt(length);
    if (!(length > 1e-9)) return std::nullopt;
    for (auto& component : value) component /= length;
    return value;
}

} // namespace vc::fiber_graph
