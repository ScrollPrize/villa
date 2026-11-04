#pragma once
// Client of a store hosted by vc_object_server. Requests go to the server's
// endpoint (e.g. http://127.0.0.1:8750 through an SSH tunnel); the client never
// opens the database or its artifacts. Results are the same types the local
// ObjectStore API returns, and conflicts throw the same TransactionConflict.
//
// Transport failures, 429 and 5xx are retried. That is safe for submit()
// because a retry carries the same request UUID: the server answers an
// identical committed request with its saved receipt. When retries run out
// the outcome is unknown; recover it with receipt() or by submitting the
// identical request again, never by minting a new request UUID.

#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/objects/GeometryRegion.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/Selections.hpp"
#include "vc_object_store/objects/Transactions.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vc::objects {

// A paged listing kept seeing new commits between its pages and gave up
// rather than combine snapshots. Retry later or page a pinned selection.
class StoreBusy : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct HostedStoreOptions {
    int connections = 4;
    long connectTimeoutMs = 10000;
    long stallTimeoutS = 60;
    int retries = 5;
    long requestTimeoutMs = 0;
    // Geometry array reads larger than this are fetched as several ranges
    // (the server refuses single responses above its own limit).
    int64_t rangeBytes = 32 << 20;
};

class HostedStore {
public:
    explicit HostedStore(std::string endpoint, HostedStoreOptions = {});
    ~HostedStore();
    HostedStore(const HostedStore&) = delete;
    HostedStore& operator=(const HostedStore&) = delete;

    const std::string& endpoint() const { return endpoint_; }
    StoreStatus status();
    // The service's own state as JSON text (queues, staging, connections,
    // WAL checkpoints); see vc_object_server.
    std::string serviceReport();
    // Stages the request's payloads, then submits it. Payloads the server
    // reports missing (expired staging) are uploaded again and the identical
    // request resubmitted.
    TransactionReceipt submit(const TransactionRequest&);
    std::optional<TransactionReceipt> receipt(const std::string& requestUuid);
    ChangePage changesSince(const FeedCursor&, int64_t maxChanges = 4096);
    // Current head read in one server snapshot; `asOf` receives that
    // snapshot's status (store, generation, commit sequence).
    std::optional<ObjectRevision> currentObjectRevision(const std::string& objectUuid, StoreStatus* asOf = nullptr);
    std::optional<int64_t> currentVersion(const std::string& objectUuid);
    std::optional<ObjectRevision> objectAtVersion(const ObjectVersion&);
    // Many references in order (nullopt where not recorded), `batch` per
    // request (at most the server's query limit). Each batch is one server
    // snapshot; historical versions are immutable, so batches agree.
    std::vector<std::optional<ObjectRevision>> objectsAtVersions(const std::vector<ObjectVersion>&,
                                                                 int64_t batch = 1000);
    // Geometry payloads by their immutable revision identity (from
    // ObjectRevision::geometryRevision): safe to cache by that key. Reads
    // above rangeBytes, and reads the server refuses as too large for one
    // response, arrive as several range requests within both limits.
    std::optional<GeometryRevision> geometryRevision(const std::string& revisionUuid);
    std::vector<uint8_t> readGeometryArray(const std::string& revisionUuid, const std::string& arrayName,
                                           int64_t byteOffset = 0, const std::optional<int64_t>& byteCount = std::nullopt);

    // Queries run in one server read snapshot; `asOf` receives its status.
    // Selections carry no transforms: volume targets resolve through the
    // store's recorded transforms. One page: selection.limit is required and
    // bounded by the server.
    std::vector<ObjectRecord> selectObjects(const ObjectSelection&,
        const std::optional<std::vector<std::string>>& fields = std::nullopt, StoreStatus* asOf = nullptr);
    std::vector<EdgeRecord> selectEdges(const EdgeSelection&,
        const std::optional<std::vector<std::string>>& fields = std::nullopt, StoreStatus* asOf = nullptr);
    // Every match, read in pages of `pageSize`, exact as of one commit
    // sequence (`asOf`) without holding a server snapshot. UUID-ordered
    // listings page by cursor; commits that land meanwhile are read from the
    // change feed and their records selected again, until the feed is quiet.
    // Each record carries its numbered version, so the result pins exactly
    // what it returned. A listing restarts when a change can move membership
    // beyond the records it touched (a recorded transform for volume
    // selections, a relationship for connectedTo, an endpoint for
    // applicableOnly edges, an expired feed position), and throws StoreBusy
    // after `maxRestarts` restarts or when the feed never goes quiet
    // (sustained writes far above interactive rates). Length-ordered and
    // unbound-volume listings page by offset and restart on any commit. The
    // selection must not carry a limit; its starting cursor/offset is kept.
    std::vector<ObjectRecord> selectAllObjects(ObjectSelection,
        const std::optional<std::vector<std::string>>& fields = std::nullopt, int64_t pageSize = 1000,
        int maxRestarts = 3, StoreStatus* asOf = nullptr);
    std::vector<EdgeRecord> selectAllEdges(EdgeSelection,
        const std::optional<std::vector<std::string>>& fields = std::nullopt, int64_t pageSize = 1000,
        int maxRestarts = 3, StoreStatus* asOf = nullptr);
    // selectAllObjects as a durable SelectionRecord (objects/Selections.hpp):
    // every match at its version, exact as of one commit, plus the recorded
    // transforms and route choices in force at that commit for volume and
    // coordinate selections. Restarts (then StoreBusy) like selectAllObjects,
    // and when a transform changed while they were read.
    SelectionRecord recordSelection(const ObjectSelection&, int64_t pageSize = 1000, int maxRestarts = 3);
    // Every recorded revision of an object (empty when it never existed).
    std::vector<ObjectRevision> objectHistory(const std::string& objectUuid, StoreStatus* asOf = nullptr);
    std::vector<SelectedRelationship> relationshipsAt(const std::vector<ObjectVersion>&, StoreStatus* asOf = nullptr);
    // Budgets above the server's limits are clamped; the result's stop reason
    // reports a budget stop (complete() is false).
    RegionResult geometryRegion(const RegionSlice&, const RegionOptions& = {}, StoreStatus* asOf = nullptr);
    std::vector<SurfaceProximityHit> nearestSurfaces(const SurfaceProximityQuery&, StoreStatus* asOf = nullptr);

    // Artifact bytes (objects/Artifacts.hpp). putArtifact sends `file` unless
    // the server already holds its content and returns its SHA-256; the
    // server has made it durable when this returns, so a transaction may
    // then name it in a manifest. fetchArtifact writes one manifest file to
    // `destination` (atomically), verifying its length and hash.
    std::string putArtifact(const std::filesystem::path& file);
    std::optional<uint64_t> artifactLength(const std::string& sha256);
    std::vector<uint8_t> readArtifact(const std::string& sha256, uint64_t offset = 0,
                                      const std::optional<uint64_t>& count = std::nullopt);
    void fetchArtifact(const ArtifactFile&, const std::filesystem::path& destination);

private:
    struct Impl;
    std::string endpoint_;
    int64_t rangeBytes_;
    std::unique_ptr<Impl> impl_;
};

} // namespace vc::objects
