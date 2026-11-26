#include "vc/fiber_graph/ScaffoldConstraints.hpp"

#include "vc/fiber_graph/ScaffoldCache.hpp"

#include "vc_object_store/geometry/FiberWindingGraph.hpp"

#include <atomic>
#include <thread>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace vc::fiber_graph {
namespace {

geometry::Point pointOf(const cv::Vec3d& value) { return {value[0], value[1], value[2]}; }

double length(const geometry::Point& value)
{
    return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
}

std::int64_t familyIndex(ScaffoldFamily family) { return std::int64_t(family); }

} // namespace

namespace {
// Each perpendicular crossing followed through its two fibers' normal
// surfaces. Computed on every run: the findings depend on the run's normals,
// not only on the geometry the cache keys the crossings by.
void crossingFindings(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                      const std::function<std::optional<geometry::Point>(std::size_t, double)>& normalOf,
                      const ScaffoldEventOptions& options, std::vector<ScaffoldEvent>& events,
                      ScaffoldEventReport& report)
{
    if (!options.crossingFindings) return;
    // The strip normals of every fiber at a regular arc spacing, then each
    // perpendicular crossing through its two fibers' surfaces. Computed on
    // every run: the findings depend on the run's normals, not only on the
    // geometry the cache keys the crossings by.
    if (!(options.crossingSampleSpacing > 0))
        throw std::invalid_argument("scaffold constraints: the crossing sample spacing must be positive");
    std::vector<std::vector<fiber_tracer::FiberStripSample>> samples(graph.fibers.size());
    std::vector<char> needed(graph.fibers.size(), 0);
    for (const auto& event : events)
        if (!event.parallel && event.validA && event.validB && event.a != event.b)
            needed[std::size_t(event.a)] = needed[std::size_t(event.b)] = 1;
    const auto threads = std::max<std::size_t>(1, options.workers ? options.workers : std::thread::hardware_concurrency());
    const auto parallel = [&](std::size_t count, const auto& body) {
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> pool;
        const auto work = [&] { for (std::size_t i; (i = next.fetch_add(1)) < count;) body(i); };
        for (std::size_t t = 1; t < std::min(threads, count); ++t) pool.emplace_back(work);
        work();
        for (auto& thread : pool) thread.join();
    };
    parallel(graph.fibers.size(), [&](std::size_t fiber) {
        if (!needed[fiber]) return;
        const double length = geometry.arcLength(fiber);
        for (double arc = 0; arc <= length; arc += options.crossingSampleSpacing) {
            const auto normal = normalOf(fiber, arc);
            if (!normal) continue;
            const auto point = geometry.packed().sample(std::int64_t(fiber), arc).point;
            samples[fiber].push_back({arc, {point[0], point[1], point[2]}, {(*normal)[0], (*normal)[1], (*normal)[2]}, {0, 0, 0}});
        }
    });
    std::atomic<std::int64_t> asked{0}, found{0};
    parallel(events.size(), [&](std::size_t index) {
        auto& event = events[index];
        if (event.parallel || !event.validA || !event.validB || event.a == event.b) return;
        const auto findings = fiber_tracer::fiberCrossingFindings(
            samples[std::size_t(event.a)], samples[std::size_t(event.b)], event.arcA, event.arcB,
            options.crossingSurface, options.crossingFilter, options.crossingEncounterArcGap);
        event.crossingFindings = findings.findings;
        event.crossingAlternative = findings.alternative;
        ++asked;
        found += (findings.findings & ~geometry::crossing_finding::Matched) != 0;
    });
    report.crossingsAsked = asked.load();
    report.crossingsWithFindings = found.load();
}
} // namespace

std::vector<ScaffoldEvent> scaffoldEvents(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                          const ScaffoldNormals& normals, const ScaffoldEventOptions& options,
                                          const ScaffoldWindingDistances& distances, ScaffoldEventReport& report)
{
    if (!(options.resampleSpacing > 0) || !(options.maximumDistance > 0) || !(options.tangentWindow > 0) ||
        !(options.windingIntegrationStep > 0) || !std::isfinite(options.maximumWindingDistance))
        throw std::invalid_argument("scaffold constraints: invalid extraction geometry settings");
    if (options.measureWindingDistances && !distances)
        throw std::invalid_argument("scaffold constraints: measuring winding separations needs a winding sampler; "
                                    "bind a Lasagna normal dataset or turn the measurement off");
    report = ScaffoldEventReport{};
    const auto started = std::chrono::steady_clock::now();
    if (options.normalAt && (options.cache || options.sink))
        throw std::invalid_argument("scaffold constraints: events with their own normal source are not cached");
    if (options.crossQueryPrefix >= graph.fibers.size() && options.crossQueryPrefix)
        throw std::invalid_argument("scaffold constraints: the cross query prefix leaves no fibers to search against");
    const auto normalOf = [&](std::size_t fiber, double arc) {
        return options.normalAt ? options.normalAt(fiber, arc) : normals.at(fiber, arc);
    };

    std::vector<fiber_tracer::FiberletCropTraceLine> lines(graph.fibers.size());
    for (std::size_t i = 0; i < graph.fibers.size(); ++i) {
        const auto& fiber = graph.fibers[i];
        auto& line = lines[i];
        line.pointsBaseXYZ.reserve(std::size_t(fiber.pointCount));
        for (std::int64_t j = 0; j < fiber.pointCount; ++j) {
            const auto at = std::size_t(3 * (fiber.pointFirst + j));
            line.pointsBaseXYZ.emplace_back(graph.linePoints[at], graph.linePoints[at + 1], graph.linePoints[at + 2]);
        }
        if (line.pointsBaseXYZ.size() < 2) ++report.shortLines;
    }
    report.lines = std::int64_t(lines.size());

    fiber_tracer::FiberTraceConstraintConfig config;
    // The annotation's own polylines are the pieces, never split. The two
    // ways of saying so differ in endpoint certification and the extractor
    // refuses them together: `localKeepWholeFibers` certifies a crossing
    // against the trace's endpoint support (the automatic extraction's own
    // mode), `preserveInputLinesAsPieces` relaxes that, which a hand-drawn
    // fiber may need because it ends where the annotator stopped.
    config.localEvents = true;
    if (options.relaxEndpointCertification) config.preserveInputLinesAsPieces = true;
    else config.localKeepWholeFibers = true;
    // The automatic extraction's own choice: geometry decides a relation,
    // the measured separation never certifies it.
    config.certifyWinding = false;
    config.resampleSpacingBaseVoxels = options.resampleSpacing;
    config.maximumDistanceBaseVoxels = options.maximumDistance;
    config.tangentWindowBaseVoxels = options.tangentWindow;
    config.measureWindingDistances = options.measureWindingDistances;
    config.windingIntegrationStepBaseVoxels = options.windingIntegrationStep;
    config.maximumWindingDistance = options.maximumWindingDistance;
    config.enforceMaximumWindingDistance = options.enforceMaximumWindingDistance;
    config.parallelThreads = options.workers;
    config.crossQueryPrefix = options.crossQueryPrefix;

    // The extraction reports its own phases (prepare, spatial_index, search,
    // candidate_order, score, deduplicate); each becomes a stage of the run,
    // so the dock shows where the minutes go, and the same monitor is what
    // the extraction asks whether to stop - which it does within a block
    // rather than at the end of the phase.
    fiber_tracer::FiberTraceConstraintProgressCallback progress;
    if (options.monitor) {
        auto* monitor = options.monitor;
        config.cancelled = [monitor]() { return monitor->cancelled(); };
        progress = [monitor, phase = std::make_shared<std::string>()](
                       const fiber_tracer::FiberTraceConstraintProgress& reported) {
            // A stage per phase: entering resets the count, so a phase is
            // entered when its name changes. Calls arrive from workers; the
            // monitor guards its own state.
            std::string name = "crossings: ";
            name += reported.phase;
            if (*phase != name) {
                *phase = name;
                monitor->enter(name, std::int64_t(reported.total));
            }
            monitor->report(std::int64_t(reported.completed), std::int64_t(reported.total));
        };
        options.monitor->enter("crossings");
        options.monitor->checkpoint();
    }
    // Reuse: every pair whose two fibers are unchanged and whose parameters
    // match is read from the cache, and the search is told to skip it. The
    // filter answers from the pair alone, so the search keeps its workers.
    const auto& normalRows = normals.cacheRows();
    const auto pairInputs = scaffoldCrossingInputs(options);
    // A pair is cached under the normals rows it was extracted with; a fiber
    // without one (never cached) makes its pairs uncacheable.
    const auto normalsRowOf = [&](std::size_t fiber) -> const std::string& {
        static const std::string none;
        return fiber < normalRows.size() ? normalRows[fiber] : none;
    };
    const auto versionOf = [&](std::size_t fiber) -> std::int64_t {
        const auto& version = graph.fibers[fiber].version;
        return version ? *version : 0;
    };
    std::map<std::pair<std::size_t, std::size_t>, std::vector<objects::FiberScaffoldCrossingRow>> reused;
    if (options.cache) {
        for (std::size_t a = 0; a < graph.fibers.size(); ++a) {
            if (versionOf(a) == 0) continue;
            for (std::size_t b = a + 1; b < graph.fibers.size(); ++b) {
                if (versionOf(b) == 0) continue;
                if (normalsRowOf(a).empty() || normalsRowOf(b).empty()) continue;
                const auto rows = options.cache->crossings(graph.fibers[a].uuid, versionOf(a), normalsRowOf(a),
                                                           graph.fibers[b].uuid, versionOf(b), normalsRowOf(b),
                                                           pairInputs);
                if (!rows) continue;
                reused.emplace(std::pair{a, b}, std::vector<objects::FiberScaffoldCrossingRow>(rows->begin(), rows->end()));
            }
        }
    }
    if (!reused.empty()) {
        config.parallelPairFilter = true;
        report.reusedPairs = std::int64_t(reused.size());
    }
    const fiber_tracer::FiberTraceConstraintTracePairFilter filter =
        reused.empty() ? fiber_tracer::FiberTraceConstraintTracePairFilter{}
                       : [&reused](std::size_t a, std::size_t b) {
                             return reused.find(a < b ? std::pair{a, b} : std::pair{b, a}) == reused.end();
                         };
    const auto extracted = [&] {
        try {
            return extractFiberTraceConstraints(lines, config, {}, distances, filter, nullptr, progress);
        } catch (const fiber_tracer::FiberTraceCancelled&) {
            throw ScaffoldCancelled();
        }
    }();
    if (options.monitor) options.monitor->checkpoint();
    report.resampledPoints = std::int64_t(extracted.resampledPoints);
    report.rejectedTangents = std::int64_t(extracted.rejectedTangents);
    report.rejectedWinding = std::int64_t(extracted.rejectedWinding);
    report.rejectedWindingCutoff = std::int64_t(extracted.rejectedWindingCutoff);

    // The events' arcs are the extractor's own arc grid; the normals and the
    // control points are looked up in the packed one. They must be the same
    // measure, or every lookup lands somewhere else on the fiber.
    for (const auto& piece : extracted.pieces) {
        if (piece.traceIndex >= graph.fibers.size()) throw std::runtime_error("scaffold constraints: unknown trace");
        const double packed = geometry.arcLength(piece.traceIndex);
        if (std::abs(piece.endArcBaseVoxels - packed) > 1e-6 * std::max(1., packed))
            throw std::runtime_error("scaffold constraints: the extracted arc grid of fiber " +
                                     graph.fibers[piece.traceIndex].uuid + " differs from the packed one (" +
                                     std::to_string(piece.endArcBaseVoxels) + " against " + std::to_string(packed) +
                                     " voxels)");
    }

    std::vector<ScaffoldEvent> events;
    events.reserve(extracted.constraints.size());
    // Per pair, in the pair's UUID order: what this run writes back.
    std::map<std::pair<std::size_t, std::size_t>, std::vector<ScaffoldEvent>> fresh;
    for (const auto& constraint : extracted.constraints) {
        if (constraint.pieceA >= extracted.pieces.size() || constraint.pieceB >= extracted.pieces.size())
            throw std::runtime_error("scaffold constraints: extracted event names an unknown piece");
        ScaffoldEvent event;
        event.a = std::int64_t(extracted.pieces[constraint.pieceA].traceIndex);
        event.b = std::int64_t(extracted.pieces[constraint.pieceB].traceIndex);
        event.arcA = constraint.arcABaseVoxels;
        event.arcB = constraint.arcBBaseVoxels;
        event.pointA = pointOf(constraint.pointABaseXYZ);
        event.pointB = pointOf(constraint.pointBBaseXYZ);
        event.parallelScore = constraint.parallelScore;
        event.perpendicularScore = constraint.perpendicularScore;
        event.closestDistance = constraint.closestDistanceBaseVoxels;
        // The pipeline's own verdict and its matching separation.
        event.parallel = constraint.parallelScore > constraint.perpendicularScore;
        event.distance = event.parallel ? constraint.parallelWindingDistance : constraint.windingDistance;
        const auto normalA = normalOf(std::size_t(event.a), event.arcA);
        const auto normalB = normalOf(std::size_t(event.b), event.arcB);
        event.validA = normalA.has_value();
        event.validB = normalB.has_value();
        if (normalA) event.normalA = *normalA;
        if (normalB) event.normalB = *normalB;
        const geometry::Point connector{event.pointB[0] - event.pointA[0], event.pointB[1] - event.pointA[1],
                                        event.pointB[2] - event.pointA[2]};
        event.validConnector = length(connector) > 0;
        if (!event.validA || !event.validB) ++report.withoutNormals;
        if (!std::isnan(event.distance)) ++report.withDistance;
        if (event.parallel) ++report.parallel; else ++report.perpendicular;
        // Written back per pair below, so the next run skips this pair.
        if (options.sink) {
            // Keyed by fiber index, smaller first: the order the pairs are
            // walked in when they are written.
            fresh[std::minmax(std::size_t(event.a), std::size_t(event.b))].push_back(event);
        }
        events.push_back(event);
    }
    // The cached pairs' events, filled in with the fibers' current normals
    // exactly as the fresh ones are: the pair digest covers those normals, so
    // a cached event is about the same geometry it was extracted from.
    for (const auto& [pair, rows] : reused) {
        for (const auto& row : rows) {
            const bool forward = graph.fibers[pair.first].uuid == row.fiberA;
            // Oriented as the extraction orients its events - the fiber of
            // the smaller index first - whichever way the row (UUID order)
            // happens to state the pair.
            ScaffoldEvent event;
            event.a = std::int64_t(pair.first);
            event.b = std::int64_t(pair.second);
            event.arcA = forward ? row.arcA : row.arcB;
            event.arcB = forward ? row.arcB : row.arcA;
            const auto& rowPointA = forward ? row.pointA : row.pointB;
            const auto& rowPointB = forward ? row.pointB : row.pointA;
            event.pointA = {rowPointA[0], rowPointA[1], rowPointA[2]};
            event.pointB = {rowPointB[0], rowPointB[1], rowPointB[2]};
            event.parallelScore = row.parallelScore;
            event.perpendicularScore = row.perpendicularScore;
            event.parallel = row.parallel;
            event.distance = row.parallel ? row.parallelWindingDistance : row.windingDistance;
            const geometry::Point connector{event.pointB[0] - event.pointA[0], event.pointB[1] - event.pointA[1],
                                            event.pointB[2] - event.pointA[2]};
            event.closestDistance = length(connector);
            const auto normalA = normalOf(std::size_t(event.a), event.arcA);
            const auto normalB = normalOf(std::size_t(event.b), event.arcB);
            event.validA = normalA.has_value();
            event.validB = normalB.has_value();
            if (normalA) event.normalA = *normalA;
            if (normalB) event.normalB = *normalB;
            event.validConnector = event.closestDistance > 0;
            if (!event.validA || !event.validB) ++report.withoutNormals;
            if (!std::isnan(event.distance)) ++report.withDistance;
            if (event.parallel) ++report.parallel; else ++report.perpendicular;
            ++report.reusedEvents;
            events.push_back(event);
        }
    }
    // What this run extracted, per pair. A pair the search covered and found
    // nothing in is cached too, as the "no crossings" marker: an empty answer
    // is an answer, and re-extracting it every run is the cost this avoids.
    if (options.sink) {
        const auto rowOf = [&](std::size_t a, std::size_t b, std::int64_t index,
                               const ScaffoldEvent* event) {
            objects::FiberScaffoldCrossingRow row;
            const bool ordered = graph.fibers[a].uuid < graph.fibers[b].uuid;
            const auto first = ordered ? a : b, second = ordered ? b : a;
            row.fiberA = graph.fibers[first].uuid;
            row.fiberB = graph.fibers[second].uuid;
            row.versionA = versionOf(first);
            row.versionB = versionOf(second);
            row.inputs = pairInputs;
            row.normalsA = normalsRowOf(first);
            row.normalsB = normalsRowOf(second);
            row.eventIndex = index;
            row.author = options.sink->author;
            row.computedAt = options.sink->computedAt;
            if (event != nullptr) {
                const bool forward = std::size_t(event->a) == first;
                row.arcA = forward ? event->arcA : event->arcB;
                row.arcB = forward ? event->arcB : event->arcA;
                const auto& pointFirst = forward ? event->pointA : event->pointB;
                const auto& pointSecond = forward ? event->pointB : event->pointA;
                row.pointA = {pointFirst[0], pointFirst[1], pointFirst[2]};
                row.pointB = {pointSecond[0], pointSecond[1], pointSecond[2]};
                row.parallelScore = event->parallelScore;
                row.perpendicularScore = event->perpendicularScore;
                row.parallel = event->parallel;
                if (event->parallel) {
                    row.parallelWindingDistance = event->distance;
                    row.windingDistance = std::numeric_limits<double>::quiet_NaN();
                } else {
                    row.windingDistance = event->distance;
                    row.parallelWindingDistance = std::numeric_limits<double>::quiet_NaN();
                }
            }
            return row;
        };
        for (std::size_t a = 0; a < graph.fibers.size(); ++a) {
            if (versionOf(a) == 0) continue;
            for (std::size_t b = a + 1; b < graph.fibers.size(); ++b) {
                if (versionOf(b) == 0 || normalsRowOf(a).empty() || normalsRowOf(b).empty()) continue;
                if (reused.find({a, b}) != reused.end()) continue;  // already cached
                const auto found = fresh.find({a, b});
                if (found == fresh.end()) {
                    // Only pairs the extraction could have found something in
                    // are worth a marker: an H/V pair of this run's tags.
                    if (graph.fibers[a].family == graph.fibers[b].family) continue;
                    options.sink->crossings.push_back(rowOf(a, b, objects::kFiberScaffoldNoCrossings, nullptr));
                    ++report.cachedPairs;
                    continue;
                }
                std::int64_t index = 0;
                for (const auto& event : found->second)
                    options.sink->crossings.push_back(rowOf(a, b, index++, &event));
                ++report.cachedPairs;
            }
        }
    }
    // A gap span is a bridge, not fiber: no crossing on it, or closer to it
    // than a crossing needs fiber on both sides, is evidence. Applied after
    // the cache, which keeps what the geometry alone yields.
    std::erase_if(events, [&](const ScaffoldEvent& event) {
        const double support = event.parallel ? kScaffoldParallelSupport : kScaffoldPerpendicularSupport;
        if (!geometry.nearGap(std::size_t(event.a), event.arcA, support) &&
            !geometry.nearGap(std::size_t(event.b), event.arcB, support))
            return false;
        ++report.rejectedGap;
        return true;
    });
    crossingFindings(graph, geometry, normalOf, options, events, report);
    report.events = std::int64_t(events.size());
    report.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return events;
}

void scaffoldRefreshEventNormals(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                 const ScaffoldNormals& normals, const ScaffoldEventOptions& options,
                                 std::vector<ScaffoldEvent>& events, ScaffoldEventReport& report)
{
    const std::function<std::optional<geometry::Point>(std::size_t, double)> normalOf =
        [&](std::size_t fiber, double arc) {
            return options.normalAt ? options.normalAt(fiber, arc) : normals.at(fiber, arc);
        };
    report.withoutNormals = 0;
    for (auto& event : events) {
        const auto normalA = normalOf(std::size_t(event.a), event.arcA);
        const auto normalB = normalOf(std::size_t(event.b), event.arcB);
        event.validA = normalA.has_value();
        event.validB = normalB.has_value();
        event.normalA = normalA.value_or(geometry::Point{});
        event.normalB = normalB.value_or(geometry::Point{});
        if (!event.validA || !event.validB) ++report.withoutNormals;
        event.crossingFindings = 0;
        event.crossingAlternative = 0;
    }
    crossingFindings(graph, geometry, normalOf, options, events, report);
}

std::vector<ScaffoldApprovedFacing> scaffoldApprovedFacing(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                                           const ScaffoldNormals& normals,
                                                           std::span<const ScaffoldEvent> events,
                                                           const ScaffoldNormalOptions& normalOptions,
                                                           const ScaffoldSignOptions& signs,
                                                           ScaffoldFacingReport& report)
{
    report = ScaffoldFacingReport{};
    // The stretches approved for normals, per fiber in arc order.
    std::vector<ScaffoldApprovedFacing> stretches;
    std::vector<std::pair<std::size_t, std::size_t>> ofFiber(graph.fibers.size(), {0, 0});
    for (std::size_t i = 0; i < graph.fibers.size(); ++i) {
        ofFiber[i].first = stretches.size();
        if (normals.solved(i)) {
            std::vector<std::pair<double, double>> ranges;
            for (const auto& [from, to] : graph.fibers[i].approval.approvedNormals)
                ranges.emplace_back(geometry.arcAtLineIndex(i, from), geometry.arcAtLineIndex(i, to));
            for (const auto& [from, to] : scaffoldMergeApprovedRanges(std::move(ranges)))
                stretches.push_back({std::int64_t(i), from, to, 1});
        }
        ofFiber[i].second = stretches.size();
    }
    report.stretches = std::int64_t(stretches.size());
    if (stretches.empty()) return stretches;
    const auto stretchAt = [&](std::int64_t fiber, double arc) -> std::int64_t {
        for (auto at = ofFiber[std::size_t(fiber)].first; at < ofFiber[std::size_t(fiber)].second; ++at)
            if (arc >= stretches[at].arcFrom && arc <= stretches[at].arcTo) return std::int64_t(at);
        return -1;
    };
    // Union-find with the facing of each stretch relative to its root.
    std::vector<std::size_t> parent(stretches.size());
    std::vector<int> flip(stretches.size(), 1);
    for (std::size_t i = 0; i < parent.size(); ++i) parent[i] = i;
    const auto facingOf = [&](std::size_t x) {
        int sign = 1;
        while (parent[x] != x) {
            sign *= flip[x];
            x = parent[x];
        }
        return std::pair{x, sign};
    };
    const auto place = [&](std::int64_t fiber, double arc) {
        const auto& entry = graph.fibers[std::size_t(fiber)];
        char buffer[64];
        std::snprintf(buffer, sizeof buffer, " at %.0f vx", arc);
        return "'" + (entry.name.empty() ? entry.uuid : entry.name) + "'" + buffer;
    };
    for (const auto& event : events) {
        if (event.parallel || !event.validA || !event.validB || event.a == event.b) continue;
        const auto a = stretchAt(event.a, event.arcA), b = stretchAt(event.b, event.arcB);
        if (a < 0 || b < 0) continue;
        const double dot = event.normalA[0] * event.normalB[0] + event.normalA[1] * event.normalB[1] +
                           event.normalA[2] * event.normalB[2];
        if (dot == 0.) continue;
        if (signs.crossingFilter &&
            !geometry::fiberCrossingUse(event.crossingFindings, event.crossingAlternative, dot, 0).used)
            continue;
        ++report.crossings;
        const int relation = dot > 0 ? 1 : -1;
        const auto [rootA, signA] = facingOf(std::size_t(a));
        const auto [rootB, signB] = facingOf(std::size_t(b));
        if (rootA == rootB) {
            if (signA * signB != relation)
                throw std::runtime_error(
                    "Approved normals contradict each other: at the crossing of " + place(event.a, event.arcA) +
                    " and " + place(event.b, event.arcB) + " the two normals face " +
                    (relation > 0 ? "the same way" : "opposite ways") +
                    ", the other crossings of the approved stretches say the reverse. One normals approval on that "
                    "loop is wrong: correct it or revoke it.");
            continue;
        }
        parent[rootB] = rootA;
        flip[rootB] = signA * signB * relation;
    }
    // The sign of each object: against the signed normals around it.
    struct Score { double agree = 0, against = 0; bool neighbours = false; };
    std::map<std::size_t, Score> scores;
    const auto* field = normalOptions.neighbours.field;
    const double spacing = normalOptions.neighbours.sampleSpacing > 0 ? normalOptions.neighbours.sampleSpacing : 16.;
    for (const bool radial : {false, true}) {
        for (std::size_t at = 0; at < stretches.size(); ++at) {
            const auto [root, sign] = facingOf(at);
            auto& score = scores[root];
            if (radial ? score.neighbours : field == nullptr) continue;
            const auto fiber = std::size_t(stretches[at].fiber);
            std::vector<fiber_tracer::FiberStripSample> query;
            std::vector<geometry::Point> gauge;
            for (double arc = stretches[at].arcFrom; arc <= stretches[at].arcTo; arc += spacing) {
                const auto normal = normals.at(fiber, arc);
                if (!normal) continue;
                const auto point = geometry.packed().sample(std::int64_t(fiber), arc).point;
                fiber_tracer::FiberStripSample sample;
                sample.arc = arc;
                sample.point = {point[0], point[1], point[2]};
                query.push_back(sample);
                gauge.push_back(*normal);
            }
            if (query.empty()) continue;
            const auto vote = [&](double cosine) {
                // Cosine times its magnitude: a cos^2 weight with the side it takes.
                (cosine * double(sign) >= 0 ? score.agree : score.against) += cosine * cosine;
            };
            if (radial) {
                for (std::size_t k = 0; k < query.size(); ++k) {
                    const auto relative = geometry.seam().relative(
                        geometry::Point{query[k].point[0], query[k].point[1], query[k].point[2]});
                    const double radius = std::sqrt(relative[0] * relative[0] + relative[1] * relative[1]);
                    if (radius > 0) vote((gauge[k][0] * relative[0] + gauge[k][1] * relative[1]) / radius);
                }
                continue;
            }
            const auto estimates = field->infer(
                query, std::span<const fiber_tracer::FiberNeighbourVariant>(&normalOptions.neighbours.variant, 1));
            for (std::size_t k = 0; k < query.size(); ++k) {
                if (!(estimates[k].count > 0)) continue;
                ++report.samples;
                vote(gauge[k][0] * estimates[k].axis[0] + gauge[k][1] * estimates[k].axis[1] +
                     gauge[k][2] * estimates[k].axis[2]);
            }
        }
        if (!radial)
            for (auto& [root, score] : scores) score.neighbours = score.agree + score.against > 0;
    }
    report.objects = std::int64_t(scores.size());
    for (const auto& [root, score] : scores) {
        if (score.neighbours) ++report.facedByNeighbours;
        const double total = score.agree + score.against;
        if (total > 0) report.weakestAgreement = std::min(report.weakestAgreement, std::max(score.agree, score.against) / total);
    }
    for (std::size_t at = 0; at < stretches.size(); ++at) {
        const auto [root, sign] = facingOf(at);
        const auto& score = scores[root];
        stretches[at].sign = sign * (score.agree >= score.against ? 1 : -1);
    }
    return stretches;
}

ScaffoldFacingReport scaffoldFaceApprovedNormals(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                                 ScaffoldNormals& normals, std::vector<ScaffoldEvent>& events,
                                                 const ScaffoldNormalOptions& normalOptions,
                                                 const ScaffoldEventOptions& eventOptions,
                                                 const ScaffoldSignOptions& signs, ScaffoldEventReport& eventReport)
{
    ScaffoldFacingReport report;
    if (!normalOptions.approvedNormalPieces) return report;
    const auto facing = scaffoldApprovedFacing(graph, geometry, normals, events, normalOptions, signs, report);
    if (facing.empty()) return report;
    if (normalOptions.monitor) {
        normalOptions.monitor->enter("approved normals");
        normalOptions.monitor->checkpoint();
    }
    normals.face(facing, normalOptions);
    scaffoldRefreshEventNormals(graph, geometry, normals, eventOptions, events, eventReport);
    return report;
}

ScaffoldSignResult scaffoldNormalSigns(const ScaffoldGraph& graph, const ScaffoldNormals& normals,
                                       std::span<const ScaffoldEvent> events, const ScaffoldSignOptions& options)
{
    const auto fibers = graph.fibers.size();
    if (normals.radialAgreement().size() != fibers || normals.fixedSigns().size() != fibers)
        throw std::invalid_argument("scaffold constraints: the normals cover a different fiber set");
    ScaffoldSignResult result;
    result.sign.assign(fibers, 1);
    result.source = normals.signSources();
    for (std::size_t i = 0; i < fibers; ++i)
        if (normals.fixedSigns()[i] != 0) {
            result.sign[i] = normals.fixedSigns()[i];
            ++result.fixed;
        }

    std::vector<fiber_tracer::FiberSignEvent> signEvents;
    signEvents.reserve(events.size());
    for (const auto& event : events) {
        if (event.parallel || !event.validA || !event.validB || event.a == event.b) continue;
        fiber_tracer::FiberSignEvent entry;
        entry.a = event.a;
        entry.b = event.b;
        entry.dot = std::clamp(event.normalA[0] * event.normalB[0] + event.normalA[1] * event.normalB[1] +
                                   event.normalA[2] * event.normalB[2],
                               -1., 1.);
        // No signs yet: the shared rule can only drop what decides none.
        if (options.crossingFilter &&
            !geometry::fiberCrossingUse(event.crossingFindings, event.crossingAlternative, entry.dot, 0).used) {
            ++result.crossingFiltered;
            continue;
        }
        entry.windingDistance = event.distance;
        signEvents.push_back(entry);
    }
    result.events = std::int64_t(signEvents.size());
    if (!options.solve || signEvents.empty()) {
        for (std::size_t i = 0; i < fibers; ++i)
            if (result.source[i] == ScaffoldSignSource::None) ++result.undecided;
        return result;
    }

    // The sign solve counts its own reweighting steps, and stops at one when
    // the run was taken back.
    fiber_tracer::FiberSignStepCallback onStep;
    if (options.monitor) {
        auto* monitor = options.monitor;
        monitor->enter("normal signs", std::int64_t(options.sign.iterations));
        monitor->checkpoint();
        onStep = [monitor](int step, int steps) {
            monitor->report(std::int64_t(step), std::int64_t(steps));
            return !monitor->cancelled();
        };
    }
    result.solve = fiber_tracer::solveFiberNormalSigns(normals.radialAgreement(), signEvents, options.sign,
                                                       normals.fixedSigns(), onStep);
    if (options.monitor) options.monitor->checkpoint();
    result.solved = true;
    std::vector<char> witnessed(fibers, 0);
    for (const auto& entry : signEvents) {
        witnessed[std::size_t(entry.a)] = 1;
        witnessed[std::size_t(entry.b)] = 1;
    }
    for (std::size_t i = 0; i < fibers; ++i) {
        if (!normals.solved(i)) {
            result.sign[i] = 1;
            ++result.undecided;
            continue;
        }
        const auto solved = std::int64_t(result.solve.sign[i]);
        if (normals.fixedSigns()[i] != 0) {
            if (solved != normals.fixedSigns()[i]) ++result.fixedDisagreements;
            continue;
        }
        if (!witnessed[i]) { ++result.undecided; continue; }
        result.sign[i] = solved;
        result.source[i] = ScaffoldSignSource::Crossings;
        ++result.decided;
        if (solved < 0) ++result.flipped;
    }
    return result;
}

namespace {
ScaffoldConstraint constraintOfArc(const geometry::winding_graph_detail::Arc& arc, const ScaffoldEvent& event,
                                   std::int64_t index, std::int64_t transport, ScaffoldConstraintKind kind)
{
    ScaffoldConstraint constraint;
    constraint.kind = kind;
    constraint.a = arc.a;
    constraint.b = arc.b;
    const bool forward = arc.a == event.a;
    constraint.arcA = forward ? event.arcA : event.arcB;
    constraint.arcB = forward ? event.arcB : event.arcA;
    constraint.target = arc.target;
    constraint.step = arc.target - (forward ? transport : -transport);
    constraint.weight = arc.weight;
    constraint.event = index;
    constraint.windingDistance = event.distance;
    return constraint;
}

geometry::WindingConstraintOptions sharedConstraintOptions(const ScaffoldMeasuredOptions& options)
{
    geometry::WindingConstraintOptions shared;
    shared.windingScale = options.windingScale;
    shared.perpendicularOffset = options.perpendicularOffset;
    shared.perpendicularWeight = options.perpendicularWeight;
    shared.parallelWeight = options.parallelWeight;
    shared.signWeight = options.signWeight;
    shared.crossingFilter = options.crossingFilter;
    shared.enclosures = options.enclosures && options.enclosureWeight > 0;
    // The crossing's evidence, exactly as the sign stage weights it.
    shared.dotWeightPower = options.crossingConfidence ? options.confidence.dotWeightPower : 0.;
    shared.dampByWindingDistance =
        options.crossingConfidence && options.confidence.distanceWeight == fiber_tracer::FiberSignDistanceWeight::InverseSquare;
    return shared;
}

// Whether a crossing states a relation at all; `report` counts why not.
bool eventStatesRelation(const ScaffoldGraph& graph, const ScaffoldEvent& event, ScaffoldMeasuredReport* report)
{
    if (event.a == event.b) { if (report) ++report->skippedSelf; return false; }
    if (!event.validA || !event.validB || !event.validConnector) { if (report) ++report->skippedInvalid; return false; }
    const auto familyA = graph.fibers[std::size_t(event.a)].family;
    const auto familyB = graph.fibers[std::size_t(event.b)].family;
    if (familyA == ScaffoldFamily::Untagged || familyB == ScaffoldFamily::Untagged) {
        if (report) ++report->skippedUntagged;
        return false;
    }
    if (event.parallel != (familyA == familyB)) { if (report) ++report->skippedFamilyMismatch; return false; }
    return true;
}

// The same per-crossing constraints the traced volume's winding stage builds
// (vc_object_store geometry/FiberWindingGraph.hpp), under the two whole-fiber
// normal signs given.
geometry::WindingEventConstraints eventConstraints(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                                   const ScaffoldEvent& event, std::int64_t signA, std::int64_t signB,
                                                   std::int64_t fallingSign,
                                                   const geometry::WindingConstraintOptions& shared)
{
    geometry::WindingEvent input;
    input.a = event.a;
    input.b = event.b;
    input.familyA = familyIndex(graph.fibers[std::size_t(event.a)].family);
    input.familyB = familyIndex(graph.fibers[std::size_t(event.b)].family);
    // The relation orders two fibers by the direction their winding falls in.
    input.signA = signA * fallingSign;
    input.signB = signB * fallingSign;
    input.arcA = event.arcA;
    input.arcB = event.arcB;
    input.distance = event.distance;
    input.parallel = event.parallel;
    input.connector = {event.pointB[0] - event.pointA[0], event.pointB[1] - event.pointA[1],
                       event.pointB[2] - event.pointA[2]};
    input.normalA = event.normalA;
    input.normalB = event.normalB;
    input.crossingFindings = event.crossingFindings;
    input.crossingAlternative = event.crossingAlternative;
    return geometry::windingEventConstraints(geometry.packed(), input, shared);
}

// The constraints one crossing states on its own (its sign bound and its
// separation); an enclosure is a statement about a pair and is not among them.
void appendEventConstraints(std::vector<ScaffoldConstraint>& constraints, const geometry::WindingEventConstraints& built,
                            const ScaffoldEvent& event, std::int64_t index, const ScaffoldMeasuredOptions& options,
                            ScaffoldMeasuredReport* report)
{
    const auto& relation = built.relation;
    if (report && relation.withoutDistance) ++report->withoutDistance;
    if (built.hasSign) {
        const geometry::winding_graph_detail::Arc arc{built.sign.a, built.sign.b, built.sign.target, built.sign.weight};
        constraints.push_back(constraintOfArc(
            arc, event, index, relation.transport,
            built.sign.side > 0 ? ScaffoldConstraintKind::SignAtLeast : ScaffoldConstraintKind::SignAtMost));
        if (report) ++report->signConstraints;
    }
    if (!relation.usable) return;
    const auto kind = event.parallel ? ScaffoldConstraintKind::Parallel : ScaffoldConstraintKind::Perpendicular;
    if (report && relation.hasUpper) ++report->fractional;
    if (options.separations == ScaffoldSeparationMode::Rounded) {
        // The nearer integer with the event's whole weight.
        auto nearer = relation.hasUpper && relation.upper.weight > relation.lower.weight ? relation.upper
                                                                                        : relation.lower;
        nearer.weight = built.weight;
        constraints.push_back(constraintOfArc(nearer, event, index, relation.transport, kind));
        if (report) ++report->expanded;
    } else {
        constraints.push_back(constraintOfArc(relation.lower, event, index, relation.transport, kind));
        if (report) ++report->expanded;
        if (relation.hasUpper) {
            constraints.push_back(constraintOfArc(relation.upper, event, index, relation.transport, kind));
            if (report) ++report->expanded;
        }
    }
    if (report) {
        ++report->used;
        if (event.parallel) ++report->parallel; else ++report->perpendicular;
    }
}
} // namespace

std::int64_t scaffoldCrossingEstimateHalves(double distance, const ScaffoldMeasuredOptions& options)
{
    if (!std::isfinite(distance)) return 0;
    // The half-winding minimum plus whole windings, as the rounded
    // perpendicular constraint states it.
    const double whole = std::round(distance * options.windingScale - options.perpendicularOffset);
    return 2 * std::int64_t(std::max(0., whole)) + 1;
}

std::vector<ScaffoldConstraint> scaffoldMeasuredConstraints(const ScaffoldGraph& graph,
                                                           const ScaffoldGeometry& geometry,
                                                           std::span<const ScaffoldEvent> events,
                                                           std::span<const std::int64_t> normalSign,
                                                           const ScaffoldMeasuredOptions& options,
                                                           ScaffoldMeasuredReport& report,
                                                           std::span<const ScaffoldNeighbour> stated,
                                                           std::vector<ScaffoldCrossing>* crossings)
{
    if (!stated.empty() && stated.size() != events.size())
        throw std::invalid_argument("scaffold constraints: the reviewed crossings cover a different event set");
    if (crossings) crossings->clear();
    for (const auto weight :
         {options.perpendicularWeight, options.parallelWeight, options.enclosureWeight, options.signWeight,
          options.approvedSignFactor, options.reviewedWeight})
        if (!std::isfinite(weight) || weight < 0)
            throw std::invalid_argument("scaffold constraints: a measured weight is negative or nonfinite");
    if (!std::isfinite(options.perpendicularOffset) || !std::isfinite(options.windingScale) || options.windingScale < 0)
        throw std::invalid_argument("scaffold constraints: invalid winding scale or perpendicular offset");
    if (normalSign.size() != graph.fibers.size())
        throw std::invalid_argument("scaffold constraints: the normal signs cover a different fiber set");
    for (const auto sign : normalSign)
        if (sign != 1 && sign != -1) throw std::invalid_argument("scaffold constraints: a normal sign is not +-1");

    report = ScaffoldMeasuredReport{};
    report.events = std::int64_t(events.size());
    const auto fallingSign = std::int64_t(scaffoldFallingWindingSign(graph));
    std::vector<ScaffoldConstraint> constraints;
    // One signed bound on the H->V step of a pair, with the event that
    // witnessed it: an accepted enclosure keeps that event's anchors so the
    // residual analysis can name the stretches it runs along.
    struct Bound {
        geometry::winding_graph_detail::EnclosureBound bound;
        std::int64_t event = 0, transport = 0;
    };
    std::vector<Bound> bounds;

    const auto shared = sharedConstraintOptions(options);
    const auto normalsApproved = [&](std::int64_t fiber, double arc) {
        const auto line = geometry.lineIndexAt(std::size_t(fiber), arc);
        return line >= 0 &&
               ScaffoldApproval::touches(graph.fibers[std::size_t(fiber)].approval.approvedNormals, double(line));
    };
    for (std::size_t index = 0; index < events.size(); ++index) {
        auto event = events[index];
        if (!eventStatesRelation(graph, event, &report)) continue;
        const auto review = stated.empty() || event.parallel ? ScaffoldNeighbour::Undecided : stated[index];
        const auto estimate = scaffoldCrossingEstimateHalves(event.distance, options);
        // A direct neighbour is a separation of the perpendicular offset,
        // whatever was measured.
        if (review == ScaffoldNeighbour::Direct && options.windingScale > 0)
            event.distance = options.perpendicularOffset / options.windingScale;
        auto built = eventConstraints(graph, geometry, event, normalSign[std::size_t(event.a)],
                                      normalSign[std::size_t(event.b)], fallingSign, shared);
        if (built.filtered) { ++report.crossingFiltered; continue; }
        if (review == ScaffoldNeighbour::Distant) {
            // Farther than a direct neighbour: the bound moves one winding
            // out, and a measurement that says "direct" (or none) states no
            // separation, since how far is not known.
            if (built.hasSign) built.sign.target += built.sign.side;
            if (built.relation.hasEnclosure) built.relation.enclosure.bound += built.relation.enclosure.sign;
            if (estimate <= 1) {
                built.relation.usable = false;
                ++report.statedDistantDropped;
            } else {
                ++report.statedDistantKept;
            }
        } else if (review == ScaffoldNeighbour::Direct) {
            ++report.statedDirect;
            if (built.relation.sign == 0) ++report.statedWithoutSign;
        }
        if (built.hasSign && (normalsApproved(event.a, event.arcA) || normalsApproved(event.b, event.arcB))) {
            built.sign.weight *= options.approvedSignFactor;
            ++report.signOnApprovedNormals;
        }
        if (built.relation.hasEnclosure && shared.enclosures)
            bounds.push_back({built.relation.enclosure, std::int64_t(index), built.relation.transport});
        const auto first = constraints.size();
        appendEventConstraints(constraints, built, event, std::int64_t(index), options, &report);
        if (review != ScaffoldNeighbour::Undecided) {
            // A reviewed separation is heavy, whatever the crossing's confidence.
            for (auto at = first; at < constraints.size(); ++at)
                if (scaffoldConstraintSide(constraints[at].kind) == 0) constraints[at].weight = options.reviewedWeight;
        }
        if (crossings && !event.parallel) {
            ScaffoldCrossing crossing;
            crossing.event = std::int64_t(index);
            crossing.a = event.a;
            crossing.b = event.b;
            crossing.arcA = event.arcA;
            crossing.arcB = event.arcB;
            crossing.distance = events[index].distance;
            crossing.estimateHalves = estimate;
            crossing.signKnown = built.relation.sign != 0;
            crossing.stated = review;
            crossing.separation = built.relation.usable;
            crossings->push_back(crossing);
        }
    }

    const auto append = [&](const geometry::winding_graph_detail::Arc& arc, const ScaffoldEvent& event,
                            std::int64_t index, std::int64_t transport, ScaffoldConstraintKind kind) {
        constraints.push_back(constraintOfArc(arc, event, index, transport, kind));
    };

    // Exact enclosure relations, accepted by the shared rule. An accepted
    // pair keeps the anchors of the crossing that set its lower bound.
    const auto counts = geometry::acceptWindingEnclosures(
        bounds, [](const Bound& entry) { return entry.bound; }, [&](std::int64_t step, std::size_t witness) {
            const auto& chosen = bounds[witness];
            const auto h = chosen.bound.h, v = chosen.bound.v;
            // The accepted step is stated H -> V; the constraint in node order.
            const bool ordered = h < v;
            const geometry::winding_graph_detail::Arc arc{ordered ? h : v, ordered ? v : h, ordered ? step : -step,
                                                          options.enclosureWeight};
            append(arc, events[std::size_t(chosen.event)], chosen.event, chosen.transport, ScaffoldConstraintKind::Enclosure);
        });
    report.enclosurePairs += std::int64_t(counts.pairs);
    report.enclosureOneSided += std::int64_t(counts.oneSided);
    report.enclosureContradictory += std::int64_t(counts.contradictory);
    report.enclosureNonunique += std::int64_t(counts.nonunique);
    report.enclosureArcs += std::int64_t(counts.accepted);
    return constraints;
}

objects::FiberConstraintKind scaffoldConstraintStoreKind(ScaffoldConstraintKind kind)
{
    switch (kind) {
        case ScaffoldConstraintKind::Perpendicular: return objects::FiberConstraintKind::Perpendicular;
        case ScaffoldConstraintKind::Parallel: return objects::FiberConstraintKind::Parallel;
        case ScaffoldConstraintKind::Enclosure: return objects::FiberConstraintKind::Enclosure;
        case ScaffoldConstraintKind::SignAtLeast:
        case ScaffoldConstraintKind::SignAtMost:
            throw std::invalid_argument("scaffold constraints: a sign constraint has no approval");
        case ScaffoldConstraintKind::Continuation:
            throw std::invalid_argument("scaffold constraints: a continuation link has no approval");
    }
    throw std::invalid_argument("scaffold constraints: unknown constraint kind");
}

std::optional<ScaffoldConstraintSpans> scaffoldConstraintSpans(const ScaffoldGraph& graph,
                                                               const ScaffoldGeometry& geometry,
                                                               const ScaffoldConstraint& constraint)
{
    const auto span = [&](std::int64_t fiber, double arc) -> std::optional<std::size_t> {
        const auto content = scaffoldFiberContent(graph, std::size_t(fiber));
        const auto line = geometry.lineIndexAt(std::size_t(fiber), arc);
        if (line < 0) return std::nullopt;
        return objects::fiberSpanCoveringLineIndex(content, line);
    };
    const auto a = span(constraint.a, constraint.arcA);
    const auto b = span(constraint.b, constraint.arcB);
    if (!a || !b) return std::nullopt;
    return ScaffoldConstraintSpans{*a, *b};
}

objects::FiberConstraintIdentity scaffoldConstraintIdentity(const ScaffoldGraph& graph,
                                                            const ScaffoldConstraint& constraint)
{
    return objects::fiberConstraintIdentity(graph.fibers.at(std::size_t(constraint.a)).uuid, constraint.arcA,
                                            graph.fibers.at(std::size_t(constraint.b)).uuid, constraint.arcB,
                                            scaffoldConstraintStoreKind(constraint.kind));
}

objects::FiberConstraintApproval approveScaffoldConstraint(const ScaffoldGraph& graph,
                                                           const ScaffoldGeometry& geometry,
                                                           const ScaffoldConstraint& constraint,
                                                           objects::FiberConstraintDistance distance,
                                                           const std::string& author, const std::string& approvedAt,
                                                           const std::string& host)
{
    if (constraint.link >= 0)
        throw std::invalid_argument("scaffold constraints: a map link's approval is its edge record's state");
    const auto identity = scaffoldConstraintIdentity(graph, constraint);
    const auto spans = scaffoldConstraintSpans(graph, geometry, constraint);
    if (!spans)
        throw std::invalid_argument("scaffold constraints: the constraint's anchors lie on no control point span");
    const auto& fiberA = graph.fibers.at(std::size_t(constraint.a));
    const auto& fiberB = graph.fibers.at(std::size_t(constraint.b));
    // The identity orders the two fibers by UUID; the spans follow it.
    const bool ordered = fiberA.uuid < fiberB.uuid;
    const auto contentA = scaffoldFiberContent(graph, std::size_t(ordered ? constraint.a : constraint.b));
    const auto contentB = scaffoldFiberContent(graph, std::size_t(ordered ? constraint.b : constraint.a));
    const auto versionA = (ordered ? fiberA : fiberB).version;
    const auto versionB = (ordered ? fiberB : fiberA).version;
    if (!versionA || !versionB)
        throw std::invalid_argument("scaffold constraints: an approval records both fibers' versions");
    return objects::approveFiberConstraint(identity, contentA, ordered ? spans->spanA : spans->spanB, *versionA,
                                           contentB, ordered ? spans->spanB : spans->spanA, *versionB, distance,
                                           author, approvedAt, host);
}

std::vector<ScaffoldNeighbour> scaffoldCrossingApprovals(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                                         std::span<const ScaffoldEvent> events,
                                                         std::span<const objects::FiberConstraintApproval> rows,
                                                         ScaffoldApprovalReport& report)
{
    report = ScaffoldApprovalReport{};
    report.rows = std::int64_t(rows.size());
    std::vector<ScaffoldNeighbour> stated(events.size(), ScaffoldNeighbour::Undecided);
    const auto live = objects::liveFiberConstraintApprovals(rows);
    report.live = std::int64_t(live.size());
    if (live.empty()) return stated;
    std::map<objects::FiberConstraintIdentity, std::size_t> index;
    for (std::size_t i = 0; i < live.size(); ++i) index.emplace(live[i].identity, i);
    std::vector<char> matched(live.size(), 0), current(live.size(), 0);
    // The span fingerprints of a fiber as the store holds it; a scaffold that
    // was not read from a store states them over its own line points.
    std::map<std::size_t, std::vector<objects::FiberSpanFingerprint>> computed;
    const auto fingerprintsOf = [&](std::size_t fiber) -> const std::vector<objects::FiberSpanFingerprint>& {
        if (!graph.fibers[fiber].spanFingerprints.empty()) return graph.fibers[fiber].spanFingerprints;
        const auto found = computed.find(fiber);
        if (found != computed.end()) return found->second;
        const auto content = scaffoldFiberContent(graph, fiber);
        std::vector<objects::FiberSpanFingerprint> list;
        for (std::size_t span = 0; span + 1 < content.controlPoints.size(); ++span)
            list.push_back(objects::fiberSpanFingerprint(content, span));
        return computed.emplace(fiber, std::move(list)).first->second;
    };
    // The recorded span is the covering one or a neighbour of it: an anchor
    // beside a control point may be placed on either side of it.
    const auto spanCurrent = [&](std::size_t fiber, std::size_t span,
                                 const std::optional<objects::FiberSpanFingerprint>& recorded) {
        if (!recorded) return false;
        const auto& list = fingerprintsOf(fiber);
        for (std::size_t at = span > 0 ? span - 1 : 0; at <= span + 1 && at < list.size(); ++at)
            if (list[at] == *recorded) return true;
        return false;
    };
    for (std::size_t at = 0; at < events.size(); ++at) {
        const auto& event = events[at];
        if (event.parallel || event.a == event.b || !event.validA || !event.validB) continue;
        ScaffoldConstraint constraint;
        constraint.kind = ScaffoldConstraintKind::Perpendicular;
        constraint.a = event.a;
        constraint.b = event.b;
        constraint.arcA = event.arcA;
        constraint.arcB = event.arcB;
        const auto found = index.find(scaffoldConstraintIdentity(graph, constraint));
        if (found == index.end()) continue;
        matched[found->second] = 1;
        const auto spans = scaffoldConstraintSpans(graph, geometry, constraint);
        if (!spans) continue;
        const bool ordered = graph.fibers.at(std::size_t(constraint.a)).uuid <
                             graph.fibers.at(std::size_t(constraint.b)).uuid;
        const auto& row = live[found->second];
        if (!spanCurrent(std::size_t(constraint.a), spans->spanA, ordered ? row.spanA : row.spanB) ||
            !spanCurrent(std::size_t(constraint.b), spans->spanB, ordered ? row.spanB : row.spanA))
            continue;
        current[found->second] = 1;
        stated[at] = row.distance == objects::FiberConstraintDistance::Distant ? ScaffoldNeighbour::Distant
                                                                              : ScaffoldNeighbour::Direct;
    }
    for (std::size_t i = 0; i < live.size(); ++i) {
        if (!matched[i]) ++report.unmatched;
        else if (current[i]) ++report.approved;
        else ++report.stale;
    }
    return stated;
}

std::vector<ScaffoldStretch> scaffoldStretches(std::span<const ScaffoldStretchFiber> fibers,
                                               std::span<const ScaffoldStretchPlace> places,
                                               std::span<const ScaffoldStretchViolation> violations,
                                               const ScaffoldStretchOptions& options, ScaffoldStretchReport& report)
{
    if (!(options.joinVx >= 0) || !(options.marginVx >= 0) || !(options.minStretchVx >= 0) ||
        !(options.maxStretchVx >= options.minStretchVx) || !(options.maxStretchVx > 0))
        throw std::invalid_argument("scaffold stretches: invalid stretch lengths");
    report = ScaffoldStretchReport{};
    const auto known = [&](std::int64_t fiber) { return fiber >= 0 && std::size_t(fiber) < fibers.size(); };
    // A fiber is a row of zones, and a stretch stays in one: its parts
    // between broken geometry, and within a part every stretch that is
    // already approved as it stands. An approved stretch is never cut up or
    // offset: it is offered whole, and new stretches lie in the fiber between
    // the approved ones.
    struct Zone {
        double from = 0, to = 0;
        bool fixed = false, broken = false;
    };
    std::vector<std::vector<Zone>> zones(fibers.size());
    for (std::size_t i = 0; i < fibers.size(); ++i) {
        // Ranges that overlap are one stretch; ranges that only touch stay two.
        const auto fixed = scaffoldMergeApprovedRanges(fibers[i].approved);
        const auto broken = scaffoldMergeApprovedRanges(fibers[i].broken);
        const double length = fibers[i].length;
        // Every place the fiber's kind can change at, then one zone per run
        // of the same kind: broken, one approved range, or free fiber. A cut
        // always ends a zone.
        std::vector<double> bounds{0., length};
        for (const double cut : fibers[i].cuts) bounds.push_back(cut);
        for (const auto* ranges : {&fixed, &broken})
            for (const auto& [from, to] : *ranges) {
                bounds.push_back(from);
                bounds.push_back(to);
            }
        for (auto& bound : bounds) bound = std::clamp(bound, 0., length);
        std::sort(bounds.begin(), bounds.end());
        bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
        const auto rangeAt = [](const std::vector<std::pair<double, double>>& ranges, double arc) {
            for (std::size_t at = 0; at < ranges.size(); ++at)
                if (arc > ranges[at].first && arc < ranges[at].second) return std::int64_t(at);
            return std::int64_t(-1);
        };
        const auto isCut = [&](double arc) {
            return std::find(fibers[i].cuts.begin(), fibers[i].cuts.end(), arc) != fibers[i].cuts.end();
        };
        std::int64_t lastKind = -3;
        for (std::size_t at = 0; at + 1 < bounds.size(); ++at) {
            const double middle = (bounds[at] + bounds[at + 1]) / 2.;
            const bool isBroken = rangeAt(broken, middle) >= 0;
            const auto approvedRange = isBroken ? -1 : rangeAt(fixed, middle);
            // -2 broken, -1 free, otherwise the approved range.
            const std::int64_t kind = isBroken ? -2 : approvedRange;
            if (!zones[i].empty() && kind == lastKind && !isCut(bounds[at]))
                zones[i].back().to = bounds[at + 1];
            else
                zones[i].push_back({bounds[at], bounds[at + 1], approvedRange >= 0, isBroken});
            lastKind = kind;
        }
        if (zones[i].empty()) zones[i].push_back({0., length, false, false});
    }
    const auto zoneAt = [&](std::int64_t fiber, double arc) {
        const auto& list = zones[std::size_t(fiber)];
        // The last zone starting at or before the arc.
        std::int64_t at = 0;
        while (at + 1 < std::int64_t(list.size()) && list[std::size_t(at) + 1].from <= arc) ++at;
        return at;
    };
    using Key = std::pair<std::int64_t, std::int64_t>;  // fiber, zone
    // Every place of every zone where a constraint is anchored, and the
    // constraints anchored there.
    std::map<Key, std::vector<double>> crossings;
    std::map<Key, std::vector<std::pair<double, std::int64_t>>> anchored;
    const auto onBroken = [&](std::int64_t fiber, double arc) {
        return zones[std::size_t(fiber)][std::size_t(zoneAt(fiber, arc))].broken;
    };
    for (const auto& place : places) {
        if (!known(place.fiber) || onBroken(place.fiber, place.arc)) continue;
        const Key key{place.fiber, zoneAt(place.fiber, place.arc)};
        crossings[key].push_back(place.arc);
        anchored[key].emplace_back(place.arc, place.constraints);
    }
    std::map<Key, std::vector<const ScaffoldStretchViolation*>> violated;
    for (const auto& violation : violations) {
        if (!known(violation.fiber) || onBroken(violation.fiber, violation.arc)) continue;
        const Key key{violation.fiber, zoneAt(violation.fiber, violation.arc)};
        crossings[key].push_back(violation.arc);
        violated[key].push_back(&violation);
    }
    // Places less than a voxel apart are one place of the fiber.
    for (auto& [key, arcs] : crossings) {
        std::sort(arcs.begin(), arcs.end());
        std::vector<double> distinct;
        for (const double arc : arcs)
            if (distinct.empty() || arc - distinct.back() >= 1.) distinct.push_back(arc);
        arcs = std::move(distinct);
    }
    struct Stretch {
        std::size_t from = 0, to = 0;  // indices into the zone's places
    };
    std::map<Key, std::vector<Stretch>> stretches;
    for (const auto& violation : violations) {
        ++report.violations;
        if (!known(violation.fiber)) { ++report.withoutCrossing; continue; }
        // Nothing on a broken range is listed.
        if (onBroken(violation.fiber, violation.arc)) continue;
        const double arc = violation.arc;
        const Key key{violation.fiber, zoneAt(violation.fiber, arc)};
        const auto& arcs = crossings[key];
        const auto& zoneList = zones[std::size_t(violation.fiber)];
        const auto zone = std::size_t(key.second);
        if (zoneList[zone].fixed) {
            // An approved stretch is the stretch, whole.
            stretches[key].push_back({0, arcs.size() - 1});
            continue;
        }
        // The violation's own place among the zone's places.
        auto at = std::size_t(std::lower_bound(arcs.begin(), arcs.end(), arc) - arcs.begin());
        if (at >= arcs.size() || (at > 0 && arc - arcs[at - 1] < arcs[at] - arc)) --at;
        if (arcs.size() < 2) {
            // Alone between approved stretches, it is tied by them.
            const bool beside = (zone > 0 && zoneList[zone - 1].fixed) ||
                                (zone + 1 < zoneList.size() && zoneList[zone + 1].fixed);
            if (!beside) { ++report.withoutCrossing; continue; }
            stretches[key].push_back({at, at});
            continue;
        }
        // Another place is mandatory: the nearer neighbour, whatever its
        // distance.
        std::size_t from = at, to = at;
        if (at == 0) to = 1;
        else if (at + 1 == arcs.size()) from = at - 1;
        else if (arcs[at] - arcs[at - 1] <= arcs[at + 1] - arcs[at]) from = at - 1;
        else to = at + 1;
        // Up to the minimum length: the nearer next place on either side,
        // whatever its distance (the minimum outranks the maximum).
        while (arcs[to] - arcs[from] < options.minStretchVx) {
            const double before = from > 0 ? arcs[from] - arcs[from - 1] : -1.;
            const double after = to + 1 < arcs.size() ? arcs[to + 1] - arcs[to] : -1.;
            const bool takeBefore = before >= 0 && (after < 0 || before <= after);
            if ((takeBefore ? before : after) < 0) break;
            if (takeBefore) --from; else ++to;
        }
        // Further ones join while they are close.
        for (bool grown = true; grown;) {
            grown = false;
            if (from > 0 && arcs[from] - arcs[from - 1] <= options.joinVx &&
                arcs[to] - arcs[from - 1] <= options.maxStretchVx) { --from; grown = true; }
            if (to + 1 < arcs.size() && arcs[to + 1] - arcs[to] <= options.joinVx &&
                arcs[to + 1] - arcs[from] <= options.maxStretchVx) { ++to; grown = true; }
        }
        stretches[key].push_back({from, to});
    }
    std::vector<ScaffoldStretch> out;
    for (auto& [key, list] : stretches) {
        const auto fiber = std::size_t(key.first);
        const auto& arcs = crossings[key];
        const auto& zoneList = zones[fiber];
        const auto zone = std::size_t(key.second);
        // Overlapping stretches are one while that stays within the maximum
        // length; a stretch inside the last one is that one.
        std::stable_sort(list.begin(), list.end(), [](const Stretch& x, const Stretch& y) {
            return x.from != y.from ? x.from < y.from : x.to > y.to;
        });
        std::vector<Stretch> merged;
        for (const auto& stretch : list) {
            if (!merged.empty() && stretch.from <= merged.back().to &&
                (stretch.to <= merged.back().to ||
                 arcs[stretch.to] - arcs[merged.back().from] <= options.maxStretchVx))
                merged.back().to = std::max(merged.back().to, stretch.to);
            else
                merged.push_back(stretch);
        }
        for (const auto& stretch : merged) {
            ScaffoldStretch entry;
            entry.fiber = key.first;
            // An approved stretch stands exactly as it was approved. A new
            // one reaches past its outer places by the margin and is at
            // least the minimum length, within its zone - the margin into an
            // approved neighbour, never farther.
            double arcFrom = zoneList[zone].from, arcTo = zoneList[zone].to;
            double holdFrom = arcFrom, holdTo = arcTo;  // what counts as "on the stretch"
            if (!zoneList[zone].fixed) {
                const double lower =
                    zoneList[zone].from - (zone > 0 && zoneList[zone - 1].fixed ? options.marginVx : 0.);
                const double upper = zoneList[zone].to +
                                     (zone + 1 < zoneList.size() && zoneList[zone + 1].fixed ? options.marginVx : 0.);
                arcFrom = arcs[stretch.from] - options.marginVx;
                arcTo = arcs[stretch.to] + options.marginVx;
                if (arcTo - arcFrom < options.minStretchVx) {
                    const double pad = (options.minStretchVx - (arcTo - arcFrom)) / 2.;
                    arcFrom -= pad;
                    arcTo += pad;
                }
                // What one end cannot take, the other does.
                if (arcFrom < lower) { arcTo += lower - arcFrom; arcFrom = lower; }
                if (arcTo > upper) { arcFrom -= arcTo - upper; arcTo = upper; }
                arcFrom = std::max(arcFrom, lower);
                holdFrom = arcs[stretch.from] - 1.;
                holdTo = arcs[stretch.to] + 1.;
            }
            // The cost: every violated constraint anchored on the stretch,
            // each once; the worst of them names the row.
            double worst = -1;
            for (const auto* violation : violated[key]) {
                if (violation->arc < holdFrom || violation->arc > holdTo) continue;
                entry.cost += violation->cost;
                ++entry.violations;
                if (violation->cost > worst) {
                    worst = violation->cost;
                    entry.arc = violation->arc;
                    entry.other = violation->other;
                    entry.otherArc = violation->otherArc;
                }
            }
            for (const auto& [arc, count] : anchored[key])
                if (arc >= holdFrom && arc <= holdTo) entry.constraints += count;
            entry.constraints = std::max(entry.constraints, entry.violations);
            for (const double arc : arcs)
                if (arc >= holdFrom && arc <= holdTo) ++entry.crossings;
            entry.piece.fiber = key.first;
            entry.piece.arcFrom = arcFrom;
            entry.piece.arcTo = arcTo;
            out.push_back(entry);
        }
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const ScaffoldStretch& x, const ScaffoldStretch& y) { return x.cost > y.cost; });
    report.suspects = std::int64_t(out.size());
    return out;
}

std::vector<ScaffoldStretch> scaffoldStretches(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                               const ScaffoldSolveResult& result,
                                               const ScaffoldStretchOptions& options, ScaffoldStretchReport& report)
{
    std::vector<ScaffoldStretchFiber> fibers(graph.fibers.size());
    for (std::size_t i = 0; i < fibers.size(); ++i) {
        fibers[i].length = geometry.arcAtLineIndex(i, double(graph.fibers[i].pointCount - 1));
        fibers[i].cuts = geometry.splitArcs(i);
        fibers[i].approved = scaffoldApprovedArcRanges(graph, geometry, i);
        for (const auto* ranges : {&graph.fibers[i].approval.brokenGeometry, &graph.fibers[i].approval.brokenNormals})
            for (const auto& [from, to] : *ranges)
                fibers[i].broken.emplace_back(geometry.arcAtLineIndex(i, from), geometry.arcAtLineIndex(i, to));
    }
    std::vector<ScaffoldStretchPlace> places;
    for (const auto& constraint : result.constraints) {
        places.push_back({constraint.a, constraint.arcA, 1});
        places.push_back({constraint.b, constraint.arcB, 1});
    }
    std::vector<ScaffoldStretchViolation> violations;
    for (const auto& residual : result.residuals) {
        const auto& constraint = result.constraints.at(std::size_t(residual.constraint));
        violations.push_back({constraint.a, constraint.arcA, residual.cost, constraint.b, constraint.arcB});
        violations.push_back({constraint.b, constraint.arcB, residual.cost, constraint.a, constraint.arcA});
    }
    auto out = scaffoldStretches(fibers, places, violations, options, report);
    report.violations = std::int64_t(result.residuals.size());
    for (auto& stretch : out) scaffoldPieceControls(graph, geometry, stretch.piece);
    return out;
}

std::vector<ScaffoldNormalFlip> scaffoldLinkNormalFlips(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                                        const ScaffoldNormals& normals,
                                                        std::span<const std::int64_t> normalSign,
                                                        std::span<const ScaffoldConstraint> constraints,
                                                        ScaffoldNormalFlipReport& report)
{
    if (normalSign.size() != graph.fibers.size())
        throw std::invalid_argument("scaffold constraints: the normal signs cover a different fiber set");
    report = ScaffoldNormalFlipReport{};
    struct Check { std::size_t constraint; double arc; bool agrees; };
    std::map<std::int64_t, std::vector<Check>> checks;
    std::vector<std::size_t> disagreeing;
    for (std::size_t index = 0; index < constraints.size(); ++index) {
        const auto& constraint = constraints[index];
        if (constraint.link < 0 || constraint.kind == ScaffoldConstraintKind::Enclosure) continue;
        ++report.checked;
        // The signed normal turned the way the winding falls, which is what orders two fibers.
        const auto falling = [&](std::int64_t fiber, double arc) -> std::optional<geometry::Point> {
            const auto normal = normals.at(std::size_t(fiber), arc);
            if (!normal) return std::nullopt;
            const double sign = double(normalSign[std::size_t(fiber)]) * scaffoldFallingWindingSign(graph);
            return geometry::Point{(*normal)[0] * sign, (*normal)[1] * sign, (*normal)[2] * sign};
        };
        const auto normalA = falling(constraint.a, constraint.arcA), normalB = falling(constraint.b, constraint.arcB);
        int observed = 0, expected = 1;
        if (normalA && normalB) {
            if (constraint.kind == ScaffoldConstraintKind::Parallel) {
                const double dot = ((*normalA)[0] * (*normalB)[0] + (*normalA)[1] * (*normalB)[1]) +
                                   (*normalA)[2] * (*normalB)[2];
                observed = std::abs(dot) < .1 ? 0 : (dot > 0 ? 1 : -1);
            } else {
                const auto start = geometry.packed().sample(constraint.a, constraint.arcA).point;
                const auto finish = geometry.packed().sample(constraint.b, constraint.arcB).point;
                const geometry::Point connector{finish[0] - start[0], finish[1] - start[1], finish[2] - start[2]};
                // Stated from the H fiber: +1 the V fiber outside it, -1 inside.
                const bool forward = graph.fibers[std::size_t(constraint.a)].family == ScaffoldFamily::Horizontal;
                const int order = int(geometry::fiberSignedOrder(connector, *normalA, *normalB));
                observed = forward ? order : -order;
                expected = (forward ? constraint.step : -constraint.step) >= 0 ? 1 : -1;
                // No order at all because the two normals face each other
                // across the connector: two fibers of one sheet cannot, so one
                // of them is flipped whatever the link states.
                if (order == 0) {
                    const double length = std::sqrt((connector[0] * connector[0] + connector[1] * connector[1]) +
                                                    connector[2] * connector[2]);
                    const auto along = [&](const geometry::Point& normal) {
                        return ((connector[0] * normal[0] + connector[1] * normal[1]) + connector[2] * normal[2]) /
                               std::max(length, 1e-12);
                    };
                    const double a = along(*normalA), b = along(*normalB);
                    if (length > 0 && a * b < 0 && std::min(std::abs(a), std::abs(b)) >= .1) {
                        observed = -expected;
                        ++report.opposed;
                    }
                }
            }
        }
        if (observed == 0) { ++report.undetermined; continue; }
        const bool agrees = observed == expected;
        if (agrees) ++report.agreed; else { ++report.disagreed; disagreeing.push_back(index); }
        checks[constraint.a].push_back({index, constraint.arcA, agrees});
        checks[constraint.b].push_back({index, constraint.arcB, agrees});
    }
    std::vector<ScaffoldNormalFlip> flips;
    for (const auto index : disagreeing) {
        const auto& constraint = constraints[index];
        for (const bool first : {true, false}) {
            ScaffoldNormalFlip flip;
            flip.constraint = std::int64_t(index);
            flip.fiber = first ? constraint.a : constraint.b;
            flip.other = first ? constraint.b : constraint.a;
            flip.arc = first ? constraint.arcA : constraint.arcB;
            flip.otherArc = first ? constraint.arcB : constraint.arcA;
            double nearest = std::numeric_limits<double>::quiet_NaN();
            for (const auto& check : checks[flip.fiber]) {
                if (!check.agrees) { ++flip.disagreeing; continue; }
                ++flip.agreeing;
                if (std::isnan(nearest) || std::abs(check.arc - flip.arc) < std::abs(nearest - flip.arc))
                    nearest = check.arc;
            }
            flip.bracketed = !std::isnan(nearest);
            flip.piece.fiber = flip.fiber;
            flip.piece.arcFrom = flip.bracketed ? std::min(nearest, flip.arc) : flip.arc;
            flip.piece.arcTo = flip.bracketed ? std::max(nearest, flip.arc) : flip.arc;
            scaffoldPieceControls(graph, geometry, flip.piece);
            flips.push_back(flip);
        }
    }
    std::stable_sort(flips.begin(), flips.end(), [](const ScaffoldNormalFlip& x, const ScaffoldNormalFlip& y) {
        if (x.bracketed != y.bracketed) return x.bracketed;
        return x.piece.length() < y.piece.length();
    });
    return flips;
}

} // namespace vc::fiber_graph
