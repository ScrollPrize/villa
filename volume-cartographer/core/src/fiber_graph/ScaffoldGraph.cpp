#include "vc/fiber_graph/ScaffoldGraph.hpp"

#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/VolumeUmbilicus.hpp"
#include "vc_object_store/store/GeometryViews.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace vc::fiber_graph {
namespace {
using Json = nlohmann::json;

Json metaOf(const std::string& text)
{
    if (text.empty()) return Json::object();
    auto parsed = Json::parse(text, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object())
        throw std::invalid_argument("scaffold graph: record metadata is not a JSON object");
    return parsed;
}

ScaffoldFamily familyOf(const std::string& label)
{
    if (label == "H") return ScaffoldFamily::Horizontal;
    if (label == "V") return ScaffoldFamily::Vertical;
    return ScaffoldFamily::Untagged;
}

// The frame the fibers are annotated in, from their own coordinate
// declarations. Fibers declaring different volumes are an annotation error:
// one solve cannot span two frames.
std::string declaredVolume(const std::vector<objects::ObjectRecord>& records)
{
    std::string volume;
    for (const auto& record : records) {
        const auto coordinates = objects::geometryCoordinates(metaOf(record.meta).value("coordinates", Json::object()));
        if (coordinates["volume_uuid"].is_null()) continue;
        const auto declared = coordinates["volume_uuid"].get<std::string>();
        if (volume.empty()) volume = declared;
        else if (volume != declared)
            throw std::invalid_argument("scaffold graph: fibers declare different annotation volumes (" + volume +
                                        " and " + declared + ")");
    }
    if (volume.empty())
        throw std::invalid_argument("scaffold graph: no fiber declares its annotation volume; pass the volume explicitly");
    return volume;
}

// The handedness the store's umbilicus objects state, or nullopt.
std::optional<int> statedHandedness(const objects::StoreReader& reader, const std::string& volume)
{
    std::optional<int> stated;
    for (const auto& record : objects::volumeUmbilici(reader, volume)) {
        const auto handedness = objects::umbilicusWindingHandedness(record);
        if (!handedness) continue;
        const auto value = *handedness;
        if (stated && *stated != value)
            throw std::invalid_argument("scaffold graph: umbilicus objects state different winding handedness values");
        stated = value;
    }
    return stated;
}

std::optional<geometry::Point> anchorPosition(const objects::Anchor& anchor)
{
    if (!anchor.position) return std::nullopt;
    const auto& value = *anchor.position;
    for (const auto component : value)
        if (!std::isfinite(component)) throw std::invalid_argument("scaffold graph: nonfinite link anchor");
    return geometry::Point{value[0], value[1], value[2]};
}
} // namespace

std::optional<std::size_t> ScaffoldGraph::indexOf(const std::string& fiberUuid) const
{
    const auto found = index.find(fiberUuid);
    if (found == index.end()) return std::nullopt;
    return found->second;
}

std::int64_t scaffoldAnnotationCommit(const objects::StoreReader& reader)
{
    std::int64_t last = 0;
    for (const char* kind : {"fiber", "umbilicus"}) {
        objects::ObjectSelection selection;
        selection.kind = kind;
        for (const auto& record : reader.selectObjects(selection)) last = std::max(last, record.commitSequence);
    }
    objects::EdgeSelection links;
    links.kind = "fiber_branch";
    for (const auto& edge : reader.selectEdges(links)) last = std::max(last, edge.commitSequence);
    return last;
}

ScaffoldGraph loadScaffoldGraph(const objects::StoreReader& reader, const ScaffoldLoadOptions& options)
{
    if (!std::isfinite(options.scale) || options.scale <= 0)
        throw std::invalid_argument("scaffold graph: target scale must be finite and positive");
    for (const auto& value : {options.handedness, options.fallbackHandedness})
        if (value && *value != 1 && *value != -1)
            throw std::invalid_argument("scaffold graph: winding handedness must be +1 or -1");

    ScaffoldGraph graph;
    graph.seamAngle = options.seamAngle;
    graph.scale = options.scale;
    graph.source = reader.status();

    objects::ObjectSelection selection;
    selection.kind = "fiber";
    const auto records = reader.selectObjects(selection);
    graph.report.fiberRecords = std::int64_t(records.size());
    graph.volume = options.volume.empty() ? declaredVolume(records) : options.volume;
    if (options.handedness) {
        graph.handedness = *options.handedness;
        graph.report.handednessSource = ScaffoldHandednessSource::Option;
    } else if (const auto stated = statedHandedness(reader, graph.volume)) {
        graph.handedness = *stated;
        graph.report.handednessSource = ScaffoldHandednessSource::Store;
    } else if (options.fallbackHandedness) {
        graph.handedness = *options.fallbackHandedness;
        graph.report.handednessSource = ScaffoldHandednessSource::Fallback;
    } else {
        throw std::invalid_argument("scaffold graph: the store states no winding handedness; tag the umbilicus object "
                                    "with winding_handedness, or pass the handedness explicitly");
    }

    // Accept the fibers whose stored family the caller asked for, in the
    // store's UUID order: that order is the solve's node order.
    struct Accepted { objects::ObjectRecord record; ScaffoldFamily family; bool manual; };
    std::vector<Accepted> accepted;
    // A restricted run: the subset is a set of UUIDs, and a UUID the store
    // does not hold is simply not found (the caller's selection may be older
    // than the store).
    const std::set<std::string> subset(options.fiberSubset.begin(), options.fiberSubset.end());
    for (const auto& record : records) {
        if (!subset.empty() && subset.count(record.uuid) == 0) { ++graph.report.filteredSubset; continue; }
        if (!record.fiber) { ++graph.report.untagged; continue; }
        const auto family = familyOf(record.fiber->family);
        const bool manual = record.fiber->manualFamily;
        if (family == ScaffoldFamily::Untagged) {
            ++graph.report.untagged;
            if (!options.includeUntagged) continue;
        } else {
            if ((options.familySource == ScaffoldFamilySource::Manual && !manual) ||
                (options.familySource == ScaffoldFamilySource::Automatic && manual)) {
                ++graph.report.filteredFamily;
                continue;
            }
            if (manual) ++graph.report.manualFamily; else ++graph.report.automaticFamily;
        }
        accepted.push_back({record, family, manual});
    }

    std::vector<objects::ObjectVersion> references;
    references.reserve(accepted.size());
    for (const auto& entry : accepted) references.push_back(entry.record.reference());
    auto revisions = reader.objectsAtVersions(references);
    if (revisions.size() != accepted.size())
        throw std::runtime_error("scaffold graph: fiber revision batch size differs");
    std::vector<std::string> geometryRevisions;
    for (const auto& revision : revisions)
        if (revision && revision->geometryRevision) geometryRevisions.push_back(*revision->geometryRevision);
    reader.prefetchGeometry(geometryRevisions, {"line_points", "control_points"}, {"control_points"});

    objects::GeometrySnapshot snapshot(reader);
    graph.offsets.push_back(0);
    // Per fiber: the affine its geometry was read through, so a link's
    // anchors can be mapped the same way. A run is one frame; an anchor left
    // in the stored frame lands a whole volume away from its polyline.
    std::map<std::string, vcr::catalog::Affine> mappingOf;
    if (options.monitor) options.monitor->enter("reading the fibers", std::int64_t(accepted.size()));
    for (std::size_t i = 0; i < accepted.size(); ++i) {
        if (options.monitor) {
            options.monitor->checkpoint();
            options.monitor->report(std::int64_t(i));
        }
        const auto& entry = accepted[i];
        if (!revisions[i]) throw std::runtime_error("scaffold graph: fiber revision disappeared: " + entry.record.uuid);
        const auto& revision = *revisions[i];
        const auto view = snapshot.viewTransformedAt(revision, graph.volume, graph.scale);
        const auto controls = view.points("control_points", 0, entry.record.fiber->controlPointCount);
        // Only the annotated line: a manual fiber's extrapolated tails past its
        // end controls are not fiber, so arc 0 is its first control and no
        // normal, crossing, anchor or enclosure ever sees a tail.
        const auto raw = view.points("line_points", 0, entry.record.fiber->linePointCount);
        const auto annotated = geometry::annotatedLine(controls, raw);
        const std::vector<geometry::Point> lines(annotated.begin(), annotated.end());
        // Seam transport needs a segment: a fiber of one point carries no
        // winding relation of its own.
        if (lines.size() < 2) { ++graph.report.shortFibers; continue; }

        ScaffoldFiber fiber;
        fiber.uuid = entry.record.uuid;
        fiber.name = entry.record.name;
        fiber.family = entry.family;
        fiber.manualFamily = entry.manual;
        fiber.version = revision.version;
        fiber.pointFirst = graph.offsets.back();
        fiber.pointCount = std::int64_t(lines.size());
        fiber.content.controlPoints = controls;
        fiber.content.controlIds =
            std::get<objects::FiberView>(view.canonical()).controlPointIds(0, entry.record.fiber->controlPointCount);
        fiber.content.linePoints = lines;
        fiber.content.annotations.assign(fiber.content.controlPoints.size(), Json::object());
        if (revision.geometryRevision) {
            const auto annotations = reader.geometryPointAnnotations(*revision.geometryRevision, "control_points");
            if (annotations.is_array())
                for (std::size_t k = 0; k < std::min(annotations.size(), fiber.content.annotations.size()); ++k)
                    if (annotations[k].is_object()) fiber.content.annotations[k] = annotations[k];
        }
        fiber.content.resolveLineIndices();
        mappingOf.insert_or_assign(fiber.uuid, view.mapping());
        if (options.readPieceApprovals) {
            const auto rows = objects::readFiberPieceApprovals(reader, fiber.uuid);
            graph.report.pieceApprovalRows += std::int64_t(rows.size());
            fiber.approvals = objects::fiberPieceApprovalStates(fiber.content, rows);
            fiber.approvedLineNormals = objects::approvedFiberLineNormals(fiber.content, fiber.approvals);
        }
        // The polylines live once, packed for seam transport; the per-fiber
        // copy would double a scaffold's largest allocation.
        for (const auto& point : fiber.content.linePoints)
            graph.linePoints.insert(graph.linePoints.end(), point.begin(), point.end());
        fiber.content.linePoints.clear();
        fiber.content.linePoints.shrink_to_fit();
        graph.offsets.push_back(fiber.pointFirst + fiber.pointCount);
        graph.index.emplace(fiber.uuid, graph.fibers.size());
        graph.fibers.push_back(std::move(fiber));
    }
    graph.report.fibers = std::int64_t(graph.fibers.size());

    // Each anchor through the mapping of the fiber it sits on.
    const auto mappedAnchor = [&](const objects::Anchor& anchor,
                                  const std::string& fiberUuid) -> std::optional<geometry::Point> {
        const auto position = anchorPosition(anchor);
        if (!position) return std::nullopt;
        const auto found = mappingOf.find(fiberUuid);
        if (found == mappingOf.end()) return position;  // a fiber this run did not load
        const auto mapped = found->second.apply({(*position)[0], (*position)[1], (*position)[2]});
        return geometry::Point{mapped[0], mapped[1], mapped[2]};
    };
    objects::EdgeSelection links;
    links.kind = "fiber_branch";
    for (const auto& edge : reader.selectEdges(links)) {
        ++graph.report.linkRecords;
        ScaffoldLink link;
        link.uuid = edge.uuid;
        link.version = edge.version;
        link.aUuid = edge.aUuid;
        link.bUuid = edge.bUuid;
        link.approvalState = edge.approvalState;
        const auto meta = metaOf(edge.meta);
        link.adjacent = meta.value("adjacent", false);
        link.autoLink = meta.value("auto", false);
        link.anchorA = mappedAnchor(edge.aAnchor, edge.aUuid);
        link.anchorB = mappedAnchor(edge.bAnchor, edge.bUuid);
        if (const auto a = graph.indexOf(edge.aUuid)) link.a = std::int64_t(*a);
        if (const auto b = graph.indexOf(edge.bUuid)) link.b = std::int64_t(*b);
        graph.links.push_back(std::move(link));
    }
    graph.report.links = std::int64_t(graph.links.size());

    const auto umbilicus = objects::volumeUmbilicus(reader, objects::VolumeVoxels{graph.volume, graph.scale});
    if (!umbilicus)
        throw std::invalid_argument("scaffold graph: volume " + graph.volume + " has no umbilicus; seam transport is undefined");
    graph.umbilicus = umbilicus->points;
    return graph;
}

objects::FiberPieceContent scaffoldFiberContent(const ScaffoldGraph& graph, std::size_t fiber)
{
    const auto& entry = graph.fibers.at(fiber);
    auto content = entry.content;
    content.linePoints.resize(std::size_t(entry.pointCount));
    for (std::int64_t j = 0; j < entry.pointCount; ++j) {
        const auto at = std::size_t(3 * (entry.pointFirst + j));
        content.linePoints[std::size_t(j)] = {graph.linePoints[at], graph.linePoints[at + 1],
                                              graph.linePoints[at + 2]};
    }
    return content;
}

ScaffoldGeometry::ScaffoldGeometry(const ScaffoldGraph& graph, std::size_t workers)
    : graph_(&graph)
    , columns_(geometry::packWindingGeometry(graph.offsets, graph.linePoints,
          geometry::WindingSeam(graph.umbilicus, graph.seamAngle, graph.handedness), 0, std::max<std::size_t>(1, workers)))
    , packed_(graph.offsets, columns_.arc.data(), columns_.arc.size(), 1, graph.linePoints, columns_.turns,
          geometry::WindingSeam(graph.umbilicus, graph.seamAngle, graph.handedness))
{
    // Gap spans, from the control annotations: the span's own `gap` tag, or
    // both of its controls tagged `break` (which is what the tag records).
    const auto tagged = [](const nlohmann::json& tags, const char* tag) {
        if (!tags.is_array()) return false;
        for (const auto& entry : tags)
            if (entry.is_string() && entry.get<std::string>() == tag) return true;
        return false;
    };
    gaps_.resize(graph.fibers.size());
    for (std::size_t fiber = 0; fiber < graph.fibers.size(); ++fiber) {
        const auto& annotations = graph.fibers[fiber].content.annotations;
        const auto controls = graph.fibers[fiber].content.controlIds.size();
        for (std::size_t control = 0; control + 1 < controls && control + 1 < annotations.size(); ++control) {
            const auto& here = annotations[control];
            const auto& next = annotations[control + 1];
            if (!here.is_object() || !next.is_object()) continue;
            const auto span = here.find("segment_to_next");
            const bool gapTag = span != here.end() && span->is_object() && span->contains("tags") &&
                                tagged((*span)["tags"], "gap");
            const bool breaks = here.contains("tags") && next.contains("tags") && tagged(here["tags"], "break") &&
                                tagged(next["tags"], "break");
            if (!gapTag && !breaks) continue;
            const double from = controlArc(fiber, control), to = controlArc(fiber, control + 1);
            gaps_[fiber].emplace_back(std::min(from, to), std::max(from, to));
        }
    }
}

bool ScaffoldGeometry::nearGap(std::size_t fiber, double arc, double margin) const
{
    for (const auto& [from, to] : gaps_.at(fiber))
        if (arc >= from - margin && arc <= to + margin) return true;
    return false;
}

ScaffoldGeometry::Projection ScaffoldGeometry::project(std::size_t fiber, const geometry::Point& point) const
{
    const auto [first, last] = packed_.pointRange(std::int64_t(fiber));
    Projection best{0, std::numeric_limits<double>::infinity()};
    for (std::int64_t j = first; j + 1 < last; ++j) {
        const auto a = packed_.point(j), b = packed_.point(j + 1);
        const double dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
        const double length = dx * dx + dy * dy + dz * dz;
        double t = length > 0 ? ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy + (point[2] - a[2]) * dz) /
                                    std::max(length, 1e-300)
                              : 0.;
        t = std::clamp(t, 0., 1.);
        const double ex = a[0] + t * dx - point[0], ey = a[1] + t * dy - point[1], ez = a[2] + t * dz - point[2];
        const double distance = std::sqrt(ex * ex + ey * ey + ez * ez);
        if (distance < best.distance) {
            best.distance = distance;
            best.arc = columns_.arc[std::size_t(j)] + t * (columns_.arc[std::size_t(j) + 1] - columns_.arc[std::size_t(j)]);
        }
    }
    return best;
}

std::int64_t ScaffoldGeometry::lineIndexAt(std::size_t fiber, double arc) const
{
    const auto& entry = graph_->fibers.at(fiber);
    const auto first = std::size_t(entry.pointFirst), count = std::size_t(entry.pointCount);
    if (count == 0) return -1;
    const auto begin = columns_.arc.begin() + std::ptrdiff_t(first);
    const auto end = begin + std::ptrdiff_t(count);
    auto upper = std::lower_bound(begin, end, arc);
    if (upper == end) return std::int64_t(count) - 1;
    if (upper == begin) return 0;
    const auto index = std::size_t(upper - begin);
    return std::int64_t(arc - *(upper - 1) <= *upper - arc ? index - 1 : index);
}

double ScaffoldGeometry::controlArc(std::size_t fiber, std::size_t control) const
{
    const auto& entry = graph_->fibers.at(fiber);
    const auto index = entry.content.controlLineIndex.at(control);
    if (index < 0 || index >= entry.pointCount) return project(fiber, entry.content.controlPoints.at(control)).arc;
    return columns_.arc[std::size_t(entry.pointFirst + index)];
}

} // namespace vc::fiber_graph
