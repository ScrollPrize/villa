#include "linking/StoredGeometry.hpp"
#include "internal/Anchors.hpp"
#include "vc/core/object_links/ObjectLinks.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace vc::object_links {
namespace {

using objects::EdgeRecord;

std::string candidateKey(const LinkCandidate& c)
{
    EdgeRecord edge;
    edge.kind = c.kind; edge.aUuid = c.aUuid; edge.bUuid = c.bUuid;
    edge.aAnchor = c.aAnchor; edge.bAnchor = c.bAnchor;
    return objects::detail::edgeIdentity(std::move(edge));
}

std::string candidateKey(const EdgeRecord& e)
{
    return objects::detail::edgeIdentity(e);
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

}  // namespace

static DetectorReport commitCandidatesImpl(objects::ObjectStore& db, objects::WriteTxn& txn,
                                const std::string& detectorName, int detectorVersion,
                                const std::vector<LinkCandidate>& candidates, bool replaceStalePending,
                                const StaleEdgeScope* staleScope, const util::ProgressCallback& progress, bool validateGeometry)
{
    if (detectorName.empty())
        throw std::runtime_error("object links: detector name must be non-empty");
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
    // A scoped rerun only needs links within its stale scope plus candidate
    // endpoints. Keep unscoped stale replacement's dataset-wide semantics.
    auto relevantObjects = scopedObjects;
    for (const auto& candidate : candidates) {
        relevantObjects.insert(candidate.aUuid); relevantObjects.insert(candidate.bUuid);
    }
    for (const auto& kind : kinds) {
        objects::EdgeSelection selection; selection.kind = kind;
        if (staleScope || !replaceStalePending) {
            selection.endpointUuids = std::vector<std::string>(relevantObjects.begin(), relevantObjects.end());
            selection.bothEndpoints = true;
        }
        for (auto& e : db.selectEdges(txn, selection))
            existing[candidateKey(e)] = e;
    }

    std::set<std::string> candidateKeys;
    for (const auto& c : candidates) {
        const auto key = candidateKey(c);
        candidateKeys.insert(key);
    }

    if (replaceStalePending) {
        std::vector<std::string> staleKeys;
        for (const auto& [key, e] : existing) {
            if (staleScope && (!scopedKinds.contains(e.kind) || !scopedObjects.contains(e.aUuid) ||
                               !scopedObjects.contains(e.bUuid))) continue;
            if (candidateKeys.contains(key)) continue;
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
            throw std::runtime_error("object links: candidate meta must be a JSON object");
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
            if (validateGeometry) {
                const auto updated = *db.edgeByUuid(txn, e.uuid);
                objects::EdgeGeometryRepair repair;
                repair.edgeUuid = updated.uuid; repair.expectedGeneration = updated.generation;
                if (const auto revision = db.currentGeometry(txn, updated.aUuid)) repair.aRevision = revision->uuid;
                if (const auto revision = db.currentGeometry(txn, updated.bUuid)) repair.bRevision = revision->uuid;
                repair.aAnchor = updated.aAnchor; repair.bAnchor = updated.bAnchor;
                db.repairEdgeGeometry(txn, repair);
            }
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
    const std::string& detectorName, int detectorVersion, const std::vector<LinkCandidate>& candidates,
    bool replaceStalePending, const StaleEdgeScope* staleScope, const util::ProgressCallback& progress)
{
    return commitCandidatesImpl(db, txn, detectorName, detectorVersion, candidates,
        replaceStalePending, staleScope, progress, false);
}

DetectorReport detail::commitStoredCandidates(objects::ObjectStore& db, const StoredGeometry& inputs,
    objects::WriteTxn& txn, const std::string& detectorName, int detectorVersion,
    const std::vector<LinkCandidate>& candidates, bool replaceStalePending,
    const StaleEdgeScope* staleScope, const util::ProgressCallback& progress)
{
    inputs.validate(txn);
    return commitCandidatesImpl(db, txn, detectorName, detectorVersion, candidates,
        replaceStalePending, staleScope, progress, true);
}

}  // namespace vc::object_links
