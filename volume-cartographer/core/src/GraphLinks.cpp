#include "vc/core/graph_links/GraphLinks.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace vc::graph_links {
namespace {

using graph::EdgeRecord;

std::string anchorKey(const std::string& uuid, const graph::Anchor& anchor)
{
    return uuid + "#" + (anchor.index ? std::to_string(*anchor.index) : std::string("-"));
}

// Order-independent identity of a (kind, endpoints, anchor indexes)
// triple, matching vc_graph's uniqueness scopes.
std::string candidateKey(const std::string& kind, const std::string& aUuid, const graph::Anchor& aAnchor,
                         const std::string& bUuid, const graph::Anchor& bAnchor)
{
    std::string first = anchorKey(aUuid, aAnchor);
    std::string second = anchorKey(bUuid, bAnchor);
    if (second < first) std::swap(first, second);
    return kind + "|" + first + "|" + second;
}

std::string candidateKey(const LinkCandidate& c)
{
    return candidateKey(c.kind, c.aUuid, c.aAnchor, c.bUuid, c.bAnchor);
}

std::string candidateKey(const EdgeRecord& e)
{
    return candidateKey(e.kind, e.aUuid, e.aAnchor, e.bUuid, e.bAnchor);
}

bool isDetectorsOwnEdge(const EdgeRecord& e, const std::string& detectorName)
{
    try {
        const auto meta = nlohmann::json::parse(e.meta);
        return meta.is_object() && meta.contains("detector") && meta.at("detector").is_object() &&
               meta.at("detector").value("name", "") == detectorName;
    } catch (const std::exception&) {
        return false;
    }
}

DetectorReport placeholder(graph::GraphDb& db, const char* detectorName,
                           const std::vector<std::string>& objectKinds, const std::atomic_bool* cancel)
{
    DetectorReport report;
    if (cancel && cancel->load()) {
        report.problems.push_back(std::string(detectorName) + ": cancelled");
        return report;
    }
    auto txn = db.read();
    for (const auto& kind : objectKinds)
        report.objectsConsidered += int64_t(db.objectsByKind(txn, kind).size());
    report.problems.push_back(std::string(detectorName) +
                              ": detector not implemented yet (placeholder in vc_graph_links); no edges written");
    return report;
}

}  // namespace

DetectorReport commitCandidates(graph::GraphDb& db, const std::string& detectorName, int detectorVersion,
                                const std::vector<LinkCandidate>& candidates, bool replaceStalePending)
{
    if (detectorName.empty())
        throw std::runtime_error("graph links: detector name must be non-empty");
    DetectorReport report;
    report.candidatesFound = int64_t(candidates.size());

    auto txn = db.write();

    std::set<std::string> kinds;
    for (const auto& c : candidates) kinds.insert(c.kind);

    std::unordered_map<std::string, EdgeRecord> existing;
    for (const auto& kind : kinds)
        for (auto& e : db.edgesByKind(txn, kind))
            existing[candidateKey(e)] = e;

    std::set<std::string> candidateKeys;
    for (const auto& c : candidates) candidateKeys.insert(candidateKey(c));

    if (replaceStalePending) {
        std::vector<std::string> staleKeys;
        for (const auto& [key, e] : existing) {
            if (candidateKeys.contains(key)) continue;
            if (e.approvalState != "pending") continue;  // reviewed edges are never touched
            if (!isDetectorsOwnEdge(e, detectorName)) continue;
            staleKeys.push_back(key);
        }
        for (const auto& key : staleKeys) {
            db.removeEdge(txn, existing.at(key).uuid);
            ++report.staleRemoved;
            existing.erase(key);
        }
    }

    std::set<std::string> seen;
    for (const auto& c : candidates) {
        const std::string key = candidateKey(c);
        if (!seen.insert(key).second || existing.contains(key)) {
            ++report.edgesSkippedExisting;
            continue;
        }
        auto meta = nlohmann::json::parse(c.meta.empty() ? "{}" : c.meta);
        if (!meta.is_object())
            throw std::runtime_error("graph links: candidate meta must be a JSON object");
        nlohmann::json detector{{"name", detectorName}, {"version", detectorVersion}};
        if (c.score) detector["score"] = *c.score;
        meta["detector"] = std::move(detector);

        EdgeRecord e;
        e.kind = c.kind;
        e.aUuid = c.aUuid;
        e.bUuid = c.bUuid;
        e.aAnchor = c.aAnchor;
        e.bAnchor = c.bAnchor;
        e.tags = c.tags;
        e.meta = meta.dump();
        db.addEdge(txn, e);  // starts 'pending' until reviewed
        ++report.edgesCreated;
    }

    txn.commit();
    return report;
}

DetectorReport linkFibersToPatches(graph::GraphDb& db, const FiberPatchParams& /*params*/,
                                   const std::atomic_bool* cancel)
{
    // TODO: fiber line points vs patch surfaces (distance + coverage).
    return placeholder(db, kFiberPatchDetector, {"fiber", "patch"}, cancel);
}

DetectorReport linkFibersToFibers(graph::GraphDb& db, const FiberFiberParams& /*params*/,
                                  const std::atomic_bool* cancel)
{
    // TODO: fiber/fiber proximity at control points (spatial index over
    // line points, not whole-fiber bboxes).
    return placeholder(db, kFiberFiberDetector, {"fiber"}, cancel);
}

DetectorReport linkTracksToFibers(graph::GraphDb& db, const TrackFiberParams& /*params*/,
                                  const std::atomic_bool* cancel)
{
    // TODO: external track geometry vs fiber lines (distance + coverage).
    return placeholder(db, kTrackFiberDetector, {"track", "fiber"}, cancel);
}

DetectorReport linkTracksToTracks(graph::GraphDb& db, const TrackTrackParams& /*params*/,
                                  const std::atomic_bool* cancel)
{
    // TODO: track/track crossings (shared voxels between opposite
    // families), mirroring Spiral's packed-CSR crossing sidecars.
    return placeholder(db, kTrackTrackDetector, {"track"}, cancel);
}

DetectorReport linkTracksToPatches(graph::GraphDb& db, const TrackPatchParams& /*params*/,
                                   const std::atomic_bool* cancel)
{
    // TODO: external track geometry vs patch surfaces (distance + coverage).
    return placeholder(db, kTrackPatchDetector, {"track", "patch"}, cancel);
}

}  // namespace vc::graph_links
