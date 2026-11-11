#pragma once
// Deterministic JSON encoding of typed transaction requests and receipts,
// shared by C++, Python and the hosted wire protocol.
//
// Binary operation payloads (geometry arrays, edits, rectangles) are not
// embedded: the JSON names each by its SHA-256 and length, and the bytes
// travel separately (`payloads`, keyed by that hash). The request digest is the
// SHA-256 of the canonical JSON without the request UUID, so it covers the
// command version, store/generation, author, preconditions, ordered
// operations and payload identities, but not how or when bytes were staged.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vc_object_store/objects/GeometryRegion.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/Transactions.hpp"

namespace vc::objects {

using TransactionPayloads = std::map<std::string, std::vector<uint8_t>>;

// Canonical JSON text; when `payloads` is given it receives every payload.
std::string transactionRequestToJson(const TransactionRequest&, TransactionPayloads* payloads = nullptr);
// Every payload named by the JSON must be supplied and match its hash/length.
TransactionRequest transactionRequestFromJson(const std::string& json, const TransactionPayloads& payloads);
std::string transactionRequestDigest(const TransactionRequest&);
// Hash of the payload set a request needs, without materializing the JSON's
// bytes: lets a client re-stage exactly these after an upload expires.
std::vector<std::string> transactionPayloadHashes(const TransactionRequest&);
std::string payloadHash(const std::vector<uint8_t>&);
// payloadHash over bytes that arrive in pieces (streamed uploads).
class PayloadHasher {
public:
    PayloadHasher();
    ~PayloadHasher();
    PayloadHasher(const PayloadHasher&) = delete;
    PayloadHasher& operator=(const PayloadHasher&) = delete;
    void update(const void* data, std::size_t size);
    std::string finish();  // once; equals payloadHash of the concatenated bytes
private:
    void* context_;
};

// Record encodings used by CreateObjectOp/UpdateObjectOp and AddEdgeOp/UpdateEdgeOp.
std::string objectRecordJson(const ObjectRecord&);
std::string edgeRecordJson(const EdgeRecord&);

std::string transactionReceiptToJson(const TransactionReceipt&);
RecordSelection recordSelectionFromJson(const std::string&, bool relationships = false);
std::string recordSelectionToJson(const RecordSelection&, bool relationships = false);
TransactionReceipt transactionReceiptFromJson(const std::string&);
std::string storeStatusToJson(const StoreStatus&);
std::string changePageToJson(const ChangePage&);

// Wire forms shared by the hosted service and its clients.
std::vector<std::string> transactionJsonPayloadHashes(const std::string& requestJson);
// Whether the request holds an operation that is prepared before the writer
// lock (link_objects), so a service runs its preparation on a read worker.
bool transactionJsonHasPreparedOperations(const std::string& requestJson);
std::string transactionConflictToJson(const TransactionConflict&);
TransactionConflict transactionConflictFromJson(const std::string&);
StoreStatus storeStatusFromJson(const std::string&);
ChangePage changePageFromJson(const std::string&);
std::string feedCursorQuery(const FeedCursor&); // URL query parameters
std::string objectRevisionToJson(const ObjectRevision&);
ObjectRevision objectRevisionFromJson(const std::string&);
std::string geometryRevisionToJson(const GeometryRevision&);
GeometryRevision geometryRevisionFromJson(const std::string&);

// Hosted queries. A request names its filters only: selections never carry
// transforms (encoding one throws), so the service resolves volume targets
// through the store's recorded transforms, as a local query without explicit
// transforms does. Unknown request keys are rejected. Every answer carries the
// status of the one read snapshot it came from; doubles are written in their
// shortest round-trip form.
std::string objectQueryRequestToJson(const ObjectSelection&, const std::optional<std::vector<std::string>>& fields);
ObjectSelection objectQueryRequestFromJson(const std::string&, std::optional<std::vector<std::string>>* fields);
std::string objectQueryResultToJson(const StoreStatus&, const std::vector<ObjectRecord>&,
                                    const std::optional<std::vector<std::string>>& fields);
std::vector<ObjectRecord> objectQueryResultFromJson(const std::string&, StoreStatus* asOf);

std::string edgeQueryRequestToJson(const EdgeSelection&, const std::optional<std::vector<std::string>>& fields);
EdgeSelection edgeQueryRequestFromJson(const std::string&, std::optional<std::vector<std::string>>* fields);
std::string edgeQueryResultToJson(const StoreStatus&, const std::vector<EdgeRecord>&,
                                  const std::optional<std::vector<std::string>>& fields);
std::vector<EdgeRecord> edgeQueryResultFromJson(const std::string&, StoreStatus* asOf);

// Object counts and tags (ObjectStore::objectCount, unboundObjectCount,
// objectTags): counts need a kind (with `unboundVolume`, the live objects of
// it not bound to that volume); tags take an optional kind and nothing else.
struct KindQuery {
    std::optional<std::string> kind;
    bool includeRetired = false;
    std::optional<std::string> unboundVolume;
};
std::string kindQueryRequestToJson(const KindQuery&);
KindQuery kindQueryRequestFromJson(const std::string&);
std::string objectCountResultToJson(const StoreStatus&, int64_t count);
int64_t objectCountResultFromJson(const std::string&, StoreStatus* asOf);
std::string objectTagsResultToJson(const StoreStatus&, const std::vector<std::string>& tags);
std::vector<std::string> objectTagsResultFromJson(const std::string&, StoreStatus* asOf);

// Graph traversal from one object: its component (ObjectStore::componentOf)
// or its neighbours (ObjectStore::neighbours), as object UUIDs.
struct GraphQuery {
    enum class Kind { Component, Neighbours } kind = Kind::Component;
    std::string objectUuid;
    std::vector<std::string> edgeKinds;          // component: empty means all kinds
    std::optional<std::string> neighbourKind;    // neighbours only (with at most one edge kind)
    bool includeRetired = false;
};
std::string graphQueryRequestToJson(const GraphQuery&);
GraphQuery graphQueryRequestFromJson(const std::string&);
std::string uuidListResultToJson(const StoreStatus&, const std::vector<std::string>& uuids);
std::vector<std::string> uuidListResultFromJson(const std::string&, StoreStatus* asOf);

std::string objectHistoryToJson(const StoreStatus&, const std::vector<ObjectRevision>&);
std::vector<ObjectRevision> objectHistoryFromJson(const std::string&, StoreStatus* asOf);

// Lists of numbered object references: relationship queries and batched
// reads at versions send them in this form.
std::string objectVersionToJson(const ObjectVersion&);
ObjectVersion objectVersionFromJson(const std::string&);
std::string relationshipVersionToJson(const RelationshipVersion&);
RelationshipVersion relationshipVersionFromJson(const std::string&);
std::string objectVersionsRequestToJson(const std::vector<ObjectVersion>&);
std::vector<ObjectVersion> objectVersionsRequestFromJson(const std::string&);
// Batched reads at versions: one entry per reference, null where not recorded.
std::string objectsAtVersionsResultToJson(const StoreStatus&, const std::vector<std::optional<ObjectRevision>>&);
std::vector<std::optional<ObjectRevision>> objectsAtVersionsResultFromJson(const std::string&, StoreStatus* asOf);
std::string relationshipQueryResultToJson(const StoreStatus&, const std::vector<SelectedRelationship>&);
std::vector<SelectedRelationship> relationshipQueryResultFromJson(const std::string&, StoreStatus* asOf);
std::string selectedInputsToJson(const StoreStatus&, const SelectedInputs&);
SelectedInputs selectedInputsFromJson(const std::string&);
std::string selectedInputsRequestToJson(const std::vector<ObjectVersion>&, const FeedCursor&);
std::pair<std::vector<ObjectVersion>, FeedCursor> selectedInputsRequestFromJson(const std::string&);

std::string regionQueryRequestToJson(const RegionSlice&, const RegionOptions&);
std::pair<RegionSlice, RegionOptions> regionQueryRequestFromJson(const std::string&);
std::string regionQueryResultToJson(const StoreStatus&, const RegionResult&);
RegionResult regionQueryResultFromJson(const std::string&, StoreStatus* asOf);

std::string proximityQueryRequestToJson(const SurfaceProximityQuery&);
SurfaceProximityQuery proximityQueryRequestFromJson(const std::string&);
std::string proximityQueryResultToJson(const StoreStatus&, const std::vector<SurfaceProximityHit>&);
std::vector<SurfaceProximityHit> proximityQueryResultFromJson(const std::string&, StoreStatus* asOf);

} // namespace vc::objects
