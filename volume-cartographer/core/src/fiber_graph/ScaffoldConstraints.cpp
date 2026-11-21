#include "vc/fiber_graph/ScaffoldConstraints.hpp"

#include "vc/fiber_graph/ScaffoldCache.hpp"

#include "vc_object_store/geometry/FiberWindingGraph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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
        const auto normalA = normals.at(std::size_t(event.a), event.arcA);
        const auto normalB = normals.at(std::size_t(event.b), event.arcB);
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
            const auto normalA = normals.at(std::size_t(event.a), event.arcA);
            const auto normalB = normals.at(std::size_t(event.b), event.arcB);
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
    report.events = std::int64_t(events.size());
    report.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return events;
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

std::vector<ScaffoldConstraint> scaffoldMeasuredConstraints(const ScaffoldGraph& graph,
                                                           const ScaffoldGeometry& geometry,
                                                           std::span<const ScaffoldEvent> events,
                                                           std::span<const std::int64_t> normalSign,
                                                           const ScaffoldMeasuredOptions& options,
                                                           ScaffoldMeasuredReport& report)
{
    for (const auto weight : {options.perpendicularWeight, options.parallelWeight, options.enclosureWeight})
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
    std::vector<ScaffoldConstraint> constraints;
    // One signed bound on the H->V step of a pair, with the event that
    // witnessed it: an accepted enclosure keeps that event's anchors so the
    // residual analysis can name the stretches it runs along.
    struct Bound {
        geometry::winding_graph_detail::EnclosureBound bound;
        std::int64_t event = 0, transport = 0;
    };
    std::vector<Bound> bounds;

    const auto append = [&](const geometry::winding_graph_detail::Arc& arc, const ScaffoldEvent& event,
                            std::int64_t index, std::int64_t transport, ScaffoldConstraintKind kind) {
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
        constraints.push_back(constraint);
    };

    for (std::size_t index = 0; index < events.size(); ++index) {
        const auto& event = events[index];
        if (event.a == event.b) { ++report.skippedSelf; continue; }
        if (!event.validA || !event.validB || !event.validConnector) { ++report.skippedInvalid; continue; }
        const auto familyA = graph.fibers[std::size_t(event.a)].family;
        const auto familyB = graph.fibers[std::size_t(event.b)].family;
        if (familyA == ScaffoldFamily::Untagged || familyB == ScaffoldFamily::Untagged) {
            ++report.skippedUntagged;
            continue;
        }
        if (event.parallel != (familyA == familyB)) { ++report.skippedFamilyMismatch; continue; }

        geometry::WindingEvent input;
        input.a = event.a;
        input.b = event.b;
        input.familyA = familyIndex(familyA);
        input.familyB = familyIndex(familyB);
        input.signA = normalSign[std::size_t(event.a)];
        input.signB = normalSign[std::size_t(event.b)];
        input.arcA = event.arcA;
        input.arcB = event.arcB;
        input.distance = event.distance * options.windingScale;
        input.parallel = event.parallel;
        input.connector = {event.pointB[0] - event.pointA[0], event.pointB[1] - event.pointA[1],
                           event.pointB[2] - event.pointA[2]};
        input.normalA = event.normalA;
        input.normalB = event.normalB;
        input.weight = event.parallel ? options.parallelWeight : options.perpendicularWeight;
        input.enclosure = options.enclosures && options.enclosureWeight > 0;
        input.perpendicularOffset = options.perpendicularOffset;

        const auto relation = geometry::windingEventRelation(geometry.packed(), input);
        if (relation.withoutDistance) ++report.withoutDistance;
        if (relation.hasEnclosure) bounds.push_back({relation.enclosure, std::int64_t(index), relation.transport});
        if (!relation.usable) continue;
        const auto kind = event.parallel ? ScaffoldConstraintKind::Parallel : ScaffoldConstraintKind::Perpendicular;
        if (relation.hasUpper) ++report.fractional;
        if (options.separations == ScaffoldSeparationMode::Rounded) {
            // The nearer integer with the event's whole weight.
            auto nearer = relation.hasUpper && relation.upper.weight > relation.lower.weight ? relation.upper
                                                                                            : relation.lower;
            nearer.weight = input.weight;
            append(nearer, event, std::int64_t(index), relation.transport, kind);
            ++report.expanded;
        } else {
            append(relation.lower, event, std::int64_t(index), relation.transport, kind);
            ++report.expanded;
            if (relation.hasUpper) {
                append(relation.upper, event, std::int64_t(index), relation.transport, kind);
                ++report.expanded;
            }
        }
        ++report.used;
        if (event.parallel) ++report.parallel; else ++report.perpendicular;
    }

    // Exact enclosure relations: a pair is accepted when its largest lower
    // bound equals its smallest upper bound. Rejections are counted, never
    // turned into a weaker arc.
    std::stable_sort(bounds.begin(), bounds.end(), [](const Bound& x, const Bound& y) {
        if (x.bound.h != y.bound.h) return x.bound.h < y.bound.h;
        if (x.bound.v != y.bound.v) return x.bound.v < y.bound.v;
        if (x.bound.sign != y.bound.sign) return x.bound.sign < y.bound.sign;
        return x.bound.bound < y.bound.bound;
    });
    for (std::size_t i = 0; i < bounds.size();) {
        std::size_t j = i;
        bool lowerSeen = false, upperSeen = false;
        std::int64_t lower = 0, upper = 0;
        std::size_t witness = i;
        for (; j < bounds.size() && bounds[j].bound.h == bounds[i].bound.h && bounds[j].bound.v == bounds[i].bound.v; ++j) {
            if (bounds[j].bound.sign > 0) {
                if (!lowerSeen || bounds[j].bound.bound > lower) { lower = bounds[j].bound.bound; witness = j; }
                lowerSeen = true;
            } else {
                upper = upperSeen ? std::min(upper, bounds[j].bound.bound) : bounds[j].bound.bound;
                upperSeen = true;
            }
        }
        ++report.enclosurePairs;
        if (!lowerSeen || !upperSeen) ++report.enclosureOneSided;
        else if (lower > upper) ++report.enclosureContradictory;
        else if (lower < upper) ++report.enclosureNonunique;
        else {
            const auto& chosen = bounds[witness];
            const auto& event = events[std::size_t(chosen.event)];
            const auto h = chosen.bound.h, v = chosen.bound.v;
            // The accepted step is stated H -> V; the constraint is stated in
            // node order.
            const bool ordered = h < v;
            geometry::winding_graph_detail::Arc arc{ordered ? h : v, ordered ? v : h, ordered ? lower : -lower,
                                                    options.enclosureWeight};
            append(arc, event, chosen.event, chosen.transport, ScaffoldConstraintKind::Enclosure);
            ++report.enclosureArcs;
        }
        i = j;
    }
    return constraints;
}

objects::FiberConstraintKind scaffoldConstraintStoreKind(ScaffoldConstraintKind kind)
{
    switch (kind) {
        case ScaffoldConstraintKind::Perpendicular: return objects::FiberConstraintKind::Perpendicular;
        case ScaffoldConstraintKind::Parallel: return objects::FiberConstraintKind::Parallel;
        case ScaffoldConstraintKind::Enclosure: return objects::FiberConstraintKind::Enclosure;
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

std::optional<std::size_t> scaffoldSpanCoveringArc(const objects::FiberPieceContent& content, double arc)
{
    if (content.linePoints.size() < 2 || !std::isfinite(arc)) return std::nullopt;
    double walked = 0, bestDistance = std::abs(arc);
    std::size_t best = 0;
    for (std::size_t i = 1; i < content.linePoints.size(); ++i) {
        const auto& previous = content.linePoints[i - 1];
        const auto& current = content.linePoints[i];
        const double dx = current[0] - previous[0], dy = current[1] - previous[1], dz = current[2] - previous[2];
        walked += std::sqrt((dx * dx + dy * dy) + dz * dz);
        const double distance = std::abs(arc - walked);
        if (distance < bestDistance) {
            bestDistance = distance;
            best = i;
        }
    }
    return objects::fiberSpanCoveringLineIndex(content, std::int64_t(best));
}

objects::FiberConstraintApproval approveScaffoldConstraint(const ScaffoldGraph& graph,
                                                           const ScaffoldGeometry& geometry,
                                                           const ScaffoldConstraint& constraint,
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
                                           contentB, ordered ? spans->spanB : spans->spanA, *versionB, author,
                                           approvedAt, host);
}

bool scaffoldConstraintPiecesApproved(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                      const ScaffoldConstraint& constraint)
{
    const auto spans = scaffoldConstraintSpans(graph, geometry, constraint);
    if (!spans) return false;
    const auto approved = [&](std::int64_t fiber, std::size_t span) {
        if (fiber < 0 || std::size_t(fiber) >= graph.fibers.size()) return false;
        const auto& list =
            graph.fibers[std::size_t(fiber)].approvals.approvedSpans[std::size_t(objects::FiberPieceAspect::Geometry)];
        return std::find(list.begin(), list.end(), span) != list.end();
    };
    return approved(constraint.a, spans->spanA) && approved(constraint.b, spans->spanB);
}

void applyScaffoldConstraintApprovals(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                      std::span<ScaffoldConstraint> measured,
                                      std::span<const objects::FiberConstraintApproval> rows,
                                      ScaffoldApprovalReport& report)
{
    report = ScaffoldApprovalReport{};
    report.rows = std::int64_t(rows.size());
    const auto live = objects::liveFiberConstraintApprovals(rows);
    report.live = std::int64_t(live.size());
    if (live.empty()) return;
    std::map<objects::FiberConstraintIdentity, std::size_t> index;
    for (std::size_t i = 0; i < live.size(); ++i) index.emplace(live[i].identity, i);
    std::vector<char> matched(live.size(), 0);
    // One content per fiber, materialized only for the fibers an approval
    // names: the fingerprints are taken over their line points.
    std::map<std::size_t, objects::FiberPieceContent> contents;
    const auto contentOf = [&](std::size_t fiber) -> const objects::FiberPieceContent& {
        const auto found = contents.find(fiber);
        if (found != contents.end()) return found->second;
        return contents.emplace(fiber, scaffoldFiberContent(graph, fiber)).first->second;
    };
    for (auto& constraint : measured) {
        if (constraint.link >= 0) continue; // a map link's approval is its edge record's state
        const auto found = index.find(scaffoldConstraintIdentity(graph, constraint));
        if (found == index.end()) continue;
        matched[found->second] = 1;
        const auto spans = scaffoldConstraintSpans(graph, geometry, constraint);
        if (!spans) continue;
        const auto& fiberA = graph.fibers.at(std::size_t(constraint.a));
        const auto& fiberB = graph.fibers.at(std::size_t(constraint.b));
        const bool ordered = fiberA.uuid < fiberB.uuid;
        const auto& contentA = contentOf(std::size_t(ordered ? constraint.a : constraint.b));
        const auto& contentB = contentOf(std::size_t(ordered ? constraint.b : constraint.a));
        if (!objects::fiberConstraintApprovalCurrent(live[found->second], contentA,
                                                     ordered ? spans->spanA : spans->spanB, contentB,
                                                     ordered ? spans->spanB : spans->spanA))
            continue;
        // The piece approvals can be revoked after the constraint was
        // approved; the constraint then asserts nothing rigid either.
        if (!scaffoldConstraintPiecesApproved(graph, geometry, constraint)) {
            ++report.withoutPieces;
            continue;
        }
        constraint.approved = true;
        ++report.approved;
    }
    for (std::size_t i = 0; i < live.size(); ++i)
        if (!matched[i]) ++report.unmatched;
    report.stale = report.live - report.unmatched - report.approved;
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
        const auto inward = [&](std::int64_t fiber, double arc) -> std::optional<geometry::Point> {
            const auto normal = normals.at(std::size_t(fiber), arc);
            if (!normal) return std::nullopt;
            const double sign = double(normalSign[std::size_t(fiber)]);
            return geometry::Point{(*normal)[0] * sign, (*normal)[1] * sign, (*normal)[2] * sign};
        };
        const auto normalA = inward(constraint.a, constraint.arcA), normalB = inward(constraint.b, constraint.arcB);
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
