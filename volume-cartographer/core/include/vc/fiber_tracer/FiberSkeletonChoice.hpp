#pragma once

// Discrete connection choice inside an ambiguous region.
//
// A region exposes a finite set of ports, so the ways a fiber may enter and
// leave form a finite set of matchings. Each unordered port pair is connected
// once: initialized by the minimum-initial-loss path through the region's own
// skeleton voxels and refined with the ports held fixed. Matchings are then
// enumerated and scored jointly, including a penalty for region support that
// no selected connection explains.

#include "vc/fiber_tracer/FiberSkeletonRegions.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vc::fiber_tracer
{

struct FiberSkeletonChoiceConfig {
    // A region with more items (sub-ports) than this is left unresolved.
    std::size_t maxPorts = 8;
    // See `FiberSkeletonTraceConfig::routeThroughCentre`.
    bool routeThroughCentre = true;
    // The initial geometry of a connection is the cheapest path through the
    // thresholded prediction between its two candidates, in a corridor around
    // the region's route.
    FiberSkeletonPathSearchConfig search;
};

// One place a fibre may pass through a region: a particular candidate of a
// particular port. A port with several candidates contributes several items,
// so two fibres that share a merged edge can both be routed, one per
// candidate.
struct FiberSkeletonChoiceItem {
    std::size_t slot = 0;        // index into the region's port list
    std::size_t candidate = 0;   // index into that port's candidates
};

// A connection joins two items of different ports, refined once between those
// two candidate positions. Two items of the same port are never connected.
struct FiberSkeletonConnection {
    std::size_t id = 0;
    std::array<std::size_t, 2> itemIds{
        kFiberSkeletonNoIndex, kFiberSkeletonNoIndex};
    std::array<std::size_t, 2> portIds{
        kFiberSkeletonNoIndex, kFiberSkeletonNoIndex};   // region slots
    std::array<std::size_t, 2> candidates{0, 0};
    std::vector<cv::Vec3d> initialPointsBaseXYZ;
    std::vector<cv::Vec3d> pointsBaseXYZ;
    FiberSkeletonCostBreakdown cost;
    double costTotal = 0.0;
    double lengthBase = 0.0;
    // False when the region skeleton does not connect the two ports. Such a
    // pair is not a connection at all; it is never fabricated, and no route
    // may use it.
    bool available = false;
    bool searchedInitial = false;
    bool refined = false;
    int iterations = 0;
    bool converged = false;
};

// The routing options of one region. Nothing is chosen ahead of tracing: a
// fibre arriving on any item leaves along that item's cheapest connection,
// so every sub-port has its own route and none can be a dead end while any
// connection exists.
struct FiberSkeletonRegionChoice {
    std::size_t regionId = 0;
    std::vector<FiberSkeletonChoiceItem> items;
    std::vector<FiberSkeletonConnection> connections;
    // Per item, the cheapest available connection using it, or
    // `kFiberSkeletonNoIndex` when the region has none for it.
    std::vector<std::size_t> entryConnections;
    bool portLimitExceeded = false;
    std::size_t portCount = 0;
    double solveSeconds = 0.0;
};

// Refines every candidate pair of every port pair of every region and records
// each item's cheapest connection.
[[nodiscard]] std::vector<FiberSkeletonRegionChoice> solveFiberSkeletonChoices(
    const FiberSkeletonTopology& topology, const FiberSkeletonGraph& graph,
    FiberSkeletonFieldProvider& fields,
    const FiberSkeletonCostWeights& weights,
    const FiberSkeletonSolverConfig& solver,
    const FiberSkeletonChoiceConfig& choiceConfig, double stepBase);

// Every partial matching of `portCount` labelled ports, as partner tables.
// Exposed for testing; the count is the involution number of `portCount`.
[[nodiscard]] std::vector<std::vector<std::size_t>>
enumerateFiberSkeletonMatchings(std::size_t portCount);

}  // namespace vc::fiber_tracer
