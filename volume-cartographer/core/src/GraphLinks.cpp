#include "vc/core/graph_links/GraphLinks.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace vc::graph_links {
namespace {

using objects::EdgeRecord;

std::string anchorKey(const std::string& uuid, const objects::Anchor& anchor)
{
    return uuid + "#" + (anchor.index ? std::to_string(*anchor.index) : std::string("-"));
}

// Order-independent identity of a (kind, endpoints, anchor indexes)
// triple, matching vc_objects's uniqueness scopes.
std::string candidateKey(const std::string& kind, const std::string& aUuid, const objects::Anchor& aAnchor,
                         const std::string& bUuid, const objects::Anchor& bAnchor)
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

std::string anchorIndexSpace(const std::string& meta)
{
    try { return nlohmann::json::parse(meta).value("anchor_index_space", ""); }
    catch (const std::exception&) { return {}; }
}

DetectorReport placeholder(objects::ObjectStore& db, const char* detectorName,
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
                              ": detector not implemented yet (placeholder in vc_objects_links); no edges written");
    return report;
}

}  // namespace

DetectorReport commitCandidates(objects::ObjectStore& db, const std::string& detectorName, int detectorVersion,
                                const std::vector<LinkCandidate>& candidates, bool replaceStalePending,
                                const StaleEdgeScope* staleScope, const util::ProgressCallback& progress)
{
    auto txn = db.write();
    auto report = commitCandidates(db, txn, detectorName, detectorVersion, candidates,
                                   replaceStalePending, staleScope, progress);
    util::reportProgress(progress, "committing links");
    txn.commit();
    return report;
}

DetectorReport commitCandidates(objects::ObjectStore& db, objects::WriteTxn& txn,
                                const std::string& detectorName, int detectorVersion,
                                const std::vector<LinkCandidate>& candidates, bool replaceStalePending,
                                const StaleEdgeScope* staleScope, const util::ProgressCallback& progress)
{
    if (detectorName.empty())
        throw std::runtime_error("graph links: detector name must be non-empty");
    DetectorReport report;
    report.candidatesFound = int64_t(candidates.size());

    std::set<std::string> kinds;
    for (const auto& c : candidates) kinds.insert(c.kind);
    std::set<std::string> scopedKinds, scopedObjects;
    if (staleScope) {
        scopedKinds.insert(staleScope->edgeKinds.begin(), staleScope->edgeKinds.end());
        scopedObjects.insert(staleScope->objectUuids.begin(), staleScope->objectUuids.end());
        kinds.insert(scopedKinds.begin(), scopedKinds.end());
    }

    util::reportProgress(progress, "reading existing links");
    std::unordered_map<std::string, EdgeRecord> existing;
    for (const auto& kind : kinds)
        for (auto& e : db.edgesByKind(txn, kind))
            existing[candidateKey(e)] = e;

    std::set<std::string> candidateKeys;
    std::unordered_map<std::string, std::string> candidateIndexSpaces;
    for (const auto& c : candidates) {
        const auto key = candidateKey(c);
        candidateKeys.insert(key);
        candidateIndexSpaces[key] = anchorIndexSpace(c.meta);
    }

    if (replaceStalePending) {
        std::vector<std::string> staleKeys;
        for (const auto& [key, e] : existing) {
            if (staleScope && (!scopedKinds.contains(e.kind) || !scopedObjects.contains(e.aUuid) ||
                               !scopedObjects.contains(e.bUuid))) continue;
            if (candidateKeys.contains(key)) {
                const auto& space = candidateIndexSpaces.at(key);
                // A pending control-point anchor must not masquerade as a
                // line-point anchor merely because their integer indices match.
                if (space.empty() || space == anchorIndexSpace(e.meta)) continue;
            }
            if (e.approvalState != "pending") continue;  // reviewed edges are never touched
            if (!isDetectorsOwnEdge(e, detectorName)) continue;
            staleKeys.push_back(key);
        }
        size_t removed = 0;
        for (const auto& key : staleKeys) {
            util::reportProgress(progress, "removing stale links", removed++, staleKeys.size());
            db.removeEdge(txn, existing.at(key).uuid);
            ++report.staleRemoved;
            existing.erase(key);
        }
    }

    std::set<std::string> seen;
    size_t processed = 0;
    util::reportProgress(progress, "writing links", 0, candidates.size());
    for (const auto& c : candidates) {
        util::reportProgress(progress, "writing links", processed++, candidates.size());
        const std::string key = candidateKey(c);
        if (!seen.insert(key).second) {
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
        if (const auto found = existing.find(key); found != existing.end()) {
            const auto& old = found->second;
            if (!replaceStalePending || old.approvalState != "pending" || !isDetectorsOwnEdge(old, detectorName)) {
                ++report.edgesSkippedExisting;
                continue;
            }
            e.uuid = old.uuid;
            auto merged = nlohmann::json::parse(old.meta);
            merged.update(meta); // preserve graph annotations outside detector-owned keys
            e.meta = merged.dump();
            e.tags.insert(e.tags.end(), old.tags.begin(), old.tags.end());
            db.updateEdge(txn, e);
            if (db.edgeByUuid(txn, e.uuid)->generation == old.generation) ++report.edgesSkippedExisting;
            else ++report.edgesUpdated;
        } else {
            db.addEdge(txn, e);  // starts 'pending' until reviewed
            ++report.edgesCreated;
        }
    }

    util::reportProgress(progress, "writing links", candidates.size(), candidates.size());
    return report;
}

DetectorReport linkFibersToFibers(objects::ObjectStore& db, const FiberFiberParams& /*params*/,
                                  const std::atomic_bool* cancel)
{
    // TODO: fiber/fiber proximity at control points (spatial index over
    // line points, not whole-fiber bboxes).
    return placeholder(db, kFiberFiberDetector, {"fiber"}, cancel);
}

DetectorReport linkTracksToFibers(objects::ObjectStore& db, const TrackFiberParams& /*params*/,
                                  const std::atomic_bool* cancel)
{
    // TODO: external track geometry vs fiber lines (distance + coverage).
    return placeholder(db, kTrackFiberDetector, {"track", "fiber"}, cancel);
}

}  // namespace vc::graph_links
