// Typed transaction requests: canonical encoding, digest, and execution
// through the ordinary ObjectStore mutation APIs.
#include "vc_object_store/objects/TransactionJson.hpp"
#include "internal/Anchors.hpp"
#include "internal/History.hpp"
#include "internal/RecordJson.hpp"
#include "internal/RecordSql.hpp"
#include "internal/Versions.hpp"
#include "vc_object_store/objects/GeometryEdits.hpp"
#include "geometry/GeometryIngestBatch.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <mutex>

namespace vc::objects {
using namespace detail;
using Json = nlohmann::json;

namespace {
template<class... Ts> struct Overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

std::string sha256(const uint8_t* data, size_t size)
{
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    if (EVP_Digest(data, size, digest, &length, EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("object store: SHA-256 failed");
    std::string result;
    for (unsigned int i = 0; i < length; ++i) {
        result += "0123456789abcdef"[digest[i] >> 4];
        result += "0123456789abcdef"[digest[i] & 15];
    }
    return result;
}

std::string canonical(const std::string& uuid) { return util::canonicalUuid(uuid); }

Json selectionJson(const VersionSelection& selection)
{
    auto result = Json::object();
    for (const auto& [uuid, version] : selection) result[canonical(uuid)] = version ? Json(*version) : Json(nullptr);
    return result;
}

VersionSelection selectionFrom(const Json& j)
{
    VersionSelection result;
    for (const auto& [uuid, version] : j.items()) {
        if (!version.is_null() && (!version.is_number_integer() || version.get<int64_t>() <= 0))
            throw std::invalid_argument("transaction: versions are positive integers");
        result.emplace(canonical(uuid), version.is_null() ? std::nullopt : std::optional(version.get<int64_t>()));
    }
    return result;
}

class Encoder {
public:
    explicit Encoder(TransactionPayloads* payloads) : payloads_(payloads) {}
    Json payload(const std::vector<uint8_t>& bytes)
    {
        const auto hash = payloadHash(bytes);
        if (payloads_) payloads_->emplace(hash, bytes);
        return Json{{"sha256", hash}, {"length", bytes.size()}};
    }
    Json relationships(const std::vector<GeometryRelationshipEdit>& edits)
    {
        auto result = Json::array();
        for (const auto& edit : edits)
            result.push_back({{"relationship", canonical(edit.relationshipUuid)}, {"action", edit.action},
                              {"a_anchor", Json::parse(edit.aAnchorJson)}, {"b_anchor", Json::parse(edit.bAnchorJson)}});
        return result;
    }
    Json operator()(const CreateObjectOp& op) { return {{"op", "create_object"}, {"record", Json::parse(op.recordJson)}}; }
    Json operator()(const UpdateObjectOp& op) { return {{"op", "update_object"}, {"record", Json::parse(op.recordJson)}}; }
    Json operator()(const SetMetaOp& op)
    {
        return {{"op", "set_meta"}, {"uuid", canonical(op.uuid)}, {"key", op.key}, {"value", Json::parse(op.jsonValue)}};
    }
    Json operator()(const RemoveMetaOp& op) { return {{"op", "remove_meta"}, {"uuid", canonical(op.uuid)}, {"key", op.key}}; }
    Json operator()(const SetApprovalOp& op) { return {{"op", "set_approval"}, {"uuid", canonical(op.uuid)}, {"state", op.state}}; }
    Json operator()(const RetireObjectOp& op) { return {{"op", "retire_object"}, {"uuid", canonical(op.uuid)}}; }
    Json operator()(const RestoreObjectOp& op)
    {
        Json j{{"op", "restore_object"}, {"uuid", canonical(op.uuid)}, {"source", nullptr}};
        if (op.source)
            j["source"] = {{"store", canonical(op.source->storeUuid)}, {"generation", canonical(op.source->generation)},
                           {"object", canonical(op.source->objectUuid)}, {"version", op.source->version}};
        return j;
    }
    Json operator()(const DeleteObjectOp& op) { return {{"op", "delete_object"}, {"uuid", canonical(op.uuid)}}; }
    Json operator()(const AddEdgeOp& op) { return {{"op", "add_relationship"}, {"record", Json::parse(op.recordJson)}}; }
    Json operator()(const UpdateEdgeOp& op) { return {{"op", "update_relationship"}, {"record", Json::parse(op.recordJson)}}; }
    Json operator()(const RemoveEdgeOp& op) { return {{"op", "remove_relationship"}, {"uuid", canonical(op.uuid)}}; }
    Json operator()(const RetainRelationshipOp& op) { return {{"op", "retain_relationship"}, {"uuid", canonical(op.uuid)}}; }
    Json operator()(const KeepRecordedRelationshipOp& op)
    {
        return {{"op", "keep_recorded_relationship"}, {"uuid", canonical(op.uuid)}};
    }
    Json operator()(const CheckVolumeMappingOp& op)
    {
        return {{"op", "check_volume_mapping"}, {"object", canonical(op.objectUuid)},
                {"target_volume", op.targetVolume}, {"target_scale", op.targetScale}, {"mapping", op.mapping.json()}};
    }
    Json operator()(const PutGeometryOp& op)
    {
        return keep(op, {{"op", "put_geometry"}, {"object", canonical(op.objectUuid)}, {"data", data(op.data)},
                {"relationships", relationships(op.relationships)}});
    }
    Json operator()(const EditGeometryOp& op)
    {
        auto edits = Json::array();
        for (const auto& edit : op.edits)
            edits.push_back({{"array", edit.array}, {"element_offset", edit.elementOffset}, {"bytes", payload(edit.bytes)}});
        return keep(op, {{"op", "edit_geometry"}, {"object", canonical(op.objectUuid)}, {"edits", edits},
                {"relationships", relationships(op.relationships)}});
    }
    Json operator()(const EditGeometryRectangleOp& op)
    {
        return keep(op, {{"op", "edit_geometry_rectangle"}, {"object", canonical(op.objectUuid)}, {"array", op.arrayName},
                {"row", op.row}, {"column", op.column}, {"rows", op.rows}, {"columns", op.columns},
                {"bytes", payload(op.bytes)}, {"relationships", relationships(op.relationships)}});
    }
    Json operator()(const SetSurfaceChannelsOp& op)
    {
        auto channels = Json::array();
        for (const auto& array : op.channels)
            channels.push_back({{"name", array.name}, {"dtype", array.dtype}, {"shape", array.shape},
                                {"bytes", payload(array.bytes)}});
        return keep(op, {{"op", "set_surface_channels"}, {"object", canonical(op.objectUuid)}, {"channels", channels},
                         {"remove", op.remove}, {"relationships", relationships(op.relationships)}});
    }
    Json operator()(const UpdateGeometryArraysOp& op)
    {
        auto annotations = Json::object();
        for (const auto& [name, value] : op.annotations) annotations[name] = Json::parse(value);
        return {{"op", "update_geometry_arrays"}, {"object", canonical(op.objectUuid)}, {"data", data(op.data)},
                {"annotations", annotations}, {"affected", op.affected}};
    }
    Json operator()(const IngestGeometryOp& op)
    {
        auto items = Json::array();
        for (const auto& item : op.items)
            items.push_back({{"record", Json::parse(item.recordJson)}, {"data", data(item.data)},
                             {"source", Json::parse(item.sourceJson)}});
        return {{"op", "ingest_geometry"}, {"items", items}, {"coordinates", Json::parse(op.coordinateContext)}};
    }
    Json operator()(const RecordCatalogTransformsOp& op)
    {
        auto registrations = Json::array();
        for (const auto& r : op.registrations)
            registrations.push_back({{"id", r.id}, {"revision", r.revision}, {"source", r.source}, {"target", r.target},
                                     {"matrix", r.matrix.json()}, {"provenance", r.provenance}});
        return {{"op", "record_catalog_transforms"}, {"registrations", registrations}};
    }
    Json operator()(const LinkObjectsOp& op)
    {
        auto plan = Json::parse(op.planJson);
        if (!plan.is_object()) throw std::invalid_argument("transaction: link_objects plan is a JSON object");
        return {{"op", kLinkObjectsOperation}, {"plan", std::move(plan)}};
    }
private:
    Json data(const GeometryData& data)
    {
        auto arrays = Json::array();
        for (const auto& array : data.arrays)
            arrays.push_back({{"name", array.name}, {"dtype", array.dtype}, {"shape", array.shape},
                              {"bytes", payload(array.bytes)}});
        return {{"kind", data.kind}, {"format_version", data.formatVersion}, {"coordinate_system", data.coordinateSystem},
                {"attributes", Json::parse(data.attributes)}, {"arrays", arrays}};
    }
    // Present only when set, so requests without it keep their digests.
    template <class Op> static Json keep(const Op& op, Json j)
    {
        if (op.keepAffectedRecorded) j["keep_affected_recorded"] = true;
        return j;
    }
    TransactionPayloads* payloads_;
};

bool keepFrom(const Json& j)
{
    const auto found = j.find("keep_affected_recorded");
    if (found == j.end()) return false;
    if (!found->is_boolean() || !found->get<bool>())
        throw std::invalid_argument("transaction: keep_affected_recorded is true or absent");
    return true;
}

Json requestJson(const TransactionRequest& request, TransactionPayloads* payloads, bool withRequestUuid)
{
    Encoder encode(payloads);
    auto operations = Json::array();
    for (const auto& op : request.operations) operations.push_back(std::visit(encode, op));
    Json j{{"command_version", kTransactionCommandVersion}, {"store", canonical(request.storeUuid)},
           {"generation", canonical(request.generation)}, {"author", request.author},
           {"expected_objects", selectionJson(request.expectedObjects)},
           {"expected_relationships", selectionJson(request.expectedRelationships)},
           {"read_dependencies", selectionJson(request.readDependencies)},
           {"expected_commit_sequence", request.expectedCommitSequence ? Json(*request.expectedCommitSequence) : Json(nullptr)},
           {"operations", operations}};
    if (withRequestUuid) j["request"] = canonical(request.requestUuid);
    return j;
}

std::vector<uint8_t> payloadFrom(const Json& j, const TransactionPayloads& payloads)
{
    const auto hash = j.at("sha256").get<std::string>();
    const auto found = payloads.find(hash);
    if (found == payloads.end()) throw std::invalid_argument("transaction: missing payload " + hash);
    if (found->second.size() != j.at("length").get<uint64_t>() || payloadHash(found->second) != hash)
        throw std::invalid_argument("transaction: payload does not match its identity " + hash);
    return found->second;
}

std::vector<GeometryRelationshipEdit> relationshipsFrom(const Json& j)
{
    std::vector<GeometryRelationshipEdit> result;
    for (const auto& edit : j)
        result.push_back({canonical(edit.at("relationship").get<std::string>()), edit.at("action").get<std::string>(),
                          edit.value("a_anchor", Json::object()).dump(), edit.value("b_anchor", Json::object()).dump()});
    return result;
}

GeometryData dataFrom(const Json& data, const TransactionPayloads& payloads)
{
    GeometryData result;
    result.kind = data.at("kind").get<std::string>();
    result.formatVersion = data.at("format_version").get<int64_t>();
    result.coordinateSystem = data.at("coordinate_system").get<std::string>();
    result.attributes = data.at("attributes").dump();
    for (const auto& array : data.at("arrays"))
        result.arrays.push_back({array.at("name").get<std::string>(), array.at("dtype").get<std::string>(),
                                 array.at("shape").get<std::vector<int64_t>>(), payloadFrom(array.at("bytes"), payloads)});
    return result;
}

TransactionOperation operationFrom(const Json& j, const TransactionPayloads& payloads)
{
    const auto op = j.at("op").get<std::string>();
    auto uuid = [&] { return canonical(j.at("uuid").get<std::string>()); };
    if (op == "create_object") return CreateObjectOp{j.at("record").dump()};
    if (op == "check_volume_mapping")
        return CheckVolumeMappingOp{canonical(j.at("object").get<std::string>()),
            j.at("target_volume").get<std::string>(), j.at("target_scale").get<double>(),
            vcr::catalog::Affine::fromJson(j.at("mapping"))};
    if (op == "update_object") return UpdateObjectOp{j.at("record").dump()};
    if (op == "set_meta") return SetMetaOp{uuid(), j.at("key").get<std::string>(), j.at("value").dump()};
    if (op == "remove_meta") return RemoveMetaOp{uuid(), j.at("key").get<std::string>()};
    if (op == "set_approval") return SetApprovalOp{uuid(), j.at("state").get<std::string>()};
    if (op == "retire_object") return RetireObjectOp{uuid()};
    if (op == "restore_object") {
        RestoreObjectOp result{uuid(), std::nullopt};
        if (const auto& s = j.at("source"); !s.is_null())
            result.source = ObjectVersion{canonical(s.at("store").get<std::string>()), canonical(s.at("generation").get<std::string>()),
                                          canonical(s.at("object").get<std::string>()), s.at("version").get<int64_t>()};
        return result;
    }
    if (op == "delete_object") return DeleteObjectOp{uuid()};
    if (op == "add_relationship") return AddEdgeOp{j.at("record").dump()};
    if (op == "update_relationship") return UpdateEdgeOp{j.at("record").dump()};
    if (op == "remove_relationship") return RemoveEdgeOp{uuid()};
    if (op == "retain_relationship") return RetainRelationshipOp{uuid()};
    if (op == "keep_recorded_relationship") return KeepRecordedRelationshipOp{uuid()};
    if (op == "put_geometry") {
        PutGeometryOp result;
        result.objectUuid = canonical(j.at("object").get<std::string>());
        result.data = dataFrom(j.at("data"), payloads);
        result.relationships = relationshipsFrom(j.at("relationships"));
        result.keepAffectedRecorded = keepFrom(j);
        return result;
    }
    if (op == "edit_geometry") {
        EditGeometryOp result;
        result.objectUuid = canonical(j.at("object").get<std::string>());
        for (const auto& edit : j.at("edits"))
            result.edits.push_back({edit.at("array").get<std::string>(), edit.at("element_offset").get<int64_t>(),
                                    payloadFrom(edit.at("bytes"), payloads)});
        result.relationships = relationshipsFrom(j.at("relationships"));
        result.keepAffectedRecorded = keepFrom(j);
        return result;
    }
    if (op == "edit_geometry_rectangle") {
        EditGeometryRectangleOp result;
        result.objectUuid = canonical(j.at("object").get<std::string>());
        result.arrayName = j.at("array").get<std::string>();
        result.row = j.at("row").get<int64_t>(); result.column = j.at("column").get<int64_t>();
        result.rows = j.at("rows").get<int64_t>(); result.columns = j.at("columns").get<int64_t>();
        result.bytes = payloadFrom(j.at("bytes"), payloads);
        result.relationships = relationshipsFrom(j.at("relationships"));
        result.keepAffectedRecorded = keepFrom(j);
        return result;
    }
    if (op == "set_surface_channels") {
        SetSurfaceChannelsOp result;
        result.objectUuid = canonical(j.at("object").get<std::string>());
        for (const auto& array : j.at("channels"))
            result.channels.push_back({array.at("name").get<std::string>(), array.at("dtype").get<std::string>(),
                                       array.at("shape").get<std::vector<int64_t>>(), payloadFrom(array.at("bytes"), payloads)});
        result.remove = j.at("remove").get<std::vector<std::string>>();
        result.relationships = relationshipsFrom(j.at("relationships"));
        result.keepAffectedRecorded = keepFrom(j);
        return result;
    }
    if (op == "update_geometry_arrays") {
        UpdateGeometryArraysOp result;
        result.objectUuid = canonical(j.at("object").get<std::string>());
        result.data = dataFrom(j.at("data"), payloads);
        for (const auto& [name, value] : j.at("annotations").items()) result.annotations.emplace(name, value.dump());
        result.affected = j.at("affected").get<std::string>();
        return result;
    }
    if (op == "ingest_geometry") {
        IngestGeometryOp result;
        for (const auto& item : j.at("items"))
            result.items.push_back({item.at("record").dump(), dataFrom(item.at("data"), payloads), item.at("source").dump()});
        result.coordinateContext = j.at("coordinates").dump();
        return result;
    }
    if (op == "record_catalog_transforms") {
        RecordCatalogTransformsOp result;
        for (const auto& r : j.at("registrations"))
            result.registrations.push_back({r.at("id").get<std::string>(), r.at("revision").get<std::string>(),
                r.at("source").get<std::string>(), r.at("target").get<std::string>(),
                vcr::catalog::Affine::fromJson(r.at("matrix")), r.at("provenance").get<std::string>()});
        return result;
    }
    if (op == kLinkObjectsOperation) {
        const auto& plan = j.at("plan");
        if (!plan.is_object()) throw std::invalid_argument("transaction: link_objects plan is a JSON object");
        return LinkObjectsOp{plan.dump()};
    }
    throw std::invalid_argument("transaction: unknown operation " + op);
}

Json pageJson(const ChangePage& page)
{
    auto commits = Json::array();
    for (const auto& commit : page.commits) {
        auto changes = Json::array();
        for (const auto& change : commit.changes)
            changes.push_back({{"kind", change.recordKind}, {"uuid", change.uuid}, {"version", change.version},
                               {"deleted", change.deleted}});
        commits.push_back({{"sequence", commit.sequence},
                           {"request_uuid", commit.requestUuid ? Json(*commit.requestUuid) : Json(nullptr)},
                           {"author", commit.author}, {"host", commit.host}, {"published_at_us", commit.publishedAtUs},
                           {"changes", changes}, {"complete", commit.complete}});
    }
    return {{"cursor_valid", page.cursorValid}, {"commits", commits}, {"more", page.more},
            {"next", {{"store", page.next.storeUuid}, {"generation", page.next.generation},
                      {"sequence", page.next.sequence}, {"offset", page.next.offset}}}};
}
} // namespace

std::string payloadHash(const std::vector<uint8_t>& bytes) { return sha256(bytes.data(), bytes.size()); }

PayloadHasher::PayloadHasher() : context_(EVP_MD_CTX_new())
{
    if (!context_ || EVP_DigestInit_ex(static_cast<EVP_MD_CTX*>(context_), EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(static_cast<EVP_MD_CTX*>(context_));
        throw std::runtime_error("object store: SHA-256 failed");
    }
}

PayloadHasher::~PayloadHasher() { EVP_MD_CTX_free(static_cast<EVP_MD_CTX*>(context_)); }

void PayloadHasher::update(const void* data, std::size_t size)
{
    if (size && EVP_DigestUpdate(static_cast<EVP_MD_CTX*>(context_), data, size) != 1)
        throw std::runtime_error("object store: SHA-256 failed");
}

std::string PayloadHasher::finish()
{
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(static_cast<EVP_MD_CTX*>(context_), digest, &length) != 1)
        throw std::runtime_error("object store: SHA-256 failed");
    std::string result;
    for (unsigned int i = 0; i < length; ++i) {
        result += "0123456789abcdef"[digest[i] >> 4];
        result += "0123456789abcdef"[digest[i] & 15];
    }
    return result;
}

std::string transactionRequestToJson(const TransactionRequest& request, TransactionPayloads* payloads)
{
    return requestJson(request, payloads, true).dump();
}

TransactionRequest transactionRequestFromJson(const std::string& text, const TransactionPayloads& payloads)
{
    const auto j = Json::parse(text);
    if (j.at("command_version").get<int>() != kTransactionCommandVersion)
        throw std::invalid_argument("transaction: unsupported command version");
    TransactionRequest request;
    request.requestUuid = canonical(j.at("request").get<std::string>());
    request.storeUuid = canonical(j.at("store").get<std::string>());
    request.generation = canonical(j.at("generation").get<std::string>());
    request.author = j.at("author").get<std::string>();
    request.expectedObjects = selectionFrom(j.at("expected_objects"));
    request.expectedRelationships = selectionFrom(j.at("expected_relationships"));
    request.readDependencies = selectionFrom(j.value("read_dependencies", Json::object()));
    if (const auto& s = j.at("expected_commit_sequence"); !s.is_null()) request.expectedCommitSequence = s.get<int64_t>();
    for (const auto& op : j.at("operations")) request.operations.push_back(operationFrom(op, payloads));
    return request;
}

std::string transactionRequestDigest(const TransactionRequest& request)
{
    const auto text = requestJson(request, nullptr, false).dump();
    return sha256(reinterpret_cast<const uint8_t*>(text.data()), text.size());
}

std::vector<std::string> transactionPayloadHashes(const TransactionRequest& request)
{
    TransactionPayloads payloads;
    (void)requestJson(request, &payloads, false);
    std::vector<std::string> result;
    for (const auto& [hash, bytes] : payloads) result.push_back(hash);
    return result;
}

namespace {
// Library-managed fields are ignored on write; leaving them out keeps a
// request's digest independent of the stamps its author happened to read.
Json writableFields(Json j)
{
    for (const auto* key : {"generation", "created_by", "created_at", "created_host", "changed_by", "changed_at",
                            "changed_host", "approval_state", "approved_by", "approved_at", "approved_host",
                            "approved_generation", "retired", "lifecycle_epoch"})
        j.erase(key);
    return j;
}
}

std::string objectRecordJson(const ObjectRecord& record) { return writableFields(objectToJson(record)).dump(); }
std::string edgeRecordJson(const EdgeRecord& record) { return writableFields(edgeToJson(record, nullptr, false)).dump(); }

std::string transactionReceiptToJson(const TransactionReceipt& receipt)
{
    auto j = receiptToJson(receipt);
    j["replayed"] = receipt.replayed;
    return j.dump();
}

TransactionReceipt transactionReceiptFromJson(const std::string& text)
{
    const auto j = Json::parse(text);
    auto receipt = receiptFromJson(j);
    receipt.replayed = j.value("replayed", false);
    return receipt;
}

std::string storeStatusToJson(const StoreStatus& status)
{
    return Json{{"store", status.storeUuid}, {"generation", status.generation},
                {"commit_sequence", status.commitSequence}, {"feed_start", status.feedStart}}.dump();
}

std::string changePageToJson(const ChangePage& page) { return pageJson(page).dump(); }

// ---- Execution -------------------------------------------------------------

namespace {
EdgeRecord edgeInput(const Json& j)
{
    EdgeRecord e;
    e.uuid = j.value("uuid", ""); e.kind = j.at("kind").get<std::string>();
    e.aUuid = canonical(j.at("a").get<std::string>()); e.bUuid = canonical(j.at("b").get<std::string>());
    if (j.contains("a_anchor")) e.aAnchor = anchorFromJson(j.at("a_anchor"));
    if (j.contains("b_anchor")) e.bAnchor = anchorFromJson(j.at("b_anchor"));
    if (j.contains("tags")) e.tags = j.at("tags").get<std::vector<std::string>>();
    if (j.contains("meta")) e.meta = j.at("meta").dump();
    return e;
}

struct Executor {
    ObjectStore& store;
    WriteTxn& txn;
    const std::map<std::size_t, PreparedOperation>& prepared;
    std::size_t index = 0; // of the operation being applied

    template <class Op> std::vector<GeometryEdgeEdit> edgeEdits(const Op& op)
    {
        auto result = edgeEdits(op.relationships);
        if (op.keepAffectedRecorded) {
            GeometryEdgeEdit rest;
            rest.edgeUuid = kKeepAffectedRecorded; rest.action = "keep_recorded";
            result.push_back(std::move(rest));
        }
        return result;
    }
    std::vector<GeometryEdgeEdit> edgeEdits(const std::vector<GeometryRelationshipEdit>& edits)
    {
        // Observed state comes from the checked snapshot: the transaction's
        // endpoint versions already guarantee these are the expected heads.
        std::vector<GeometryEdgeEdit> result;
        for (const auto& edit : edits) {
            const auto edge = store.edgeByUuid(txn, edit.relationshipUuid);
            if (!edge) throw std::invalid_argument("transaction: unknown relationship " + edit.relationshipUuid);
            GeometryEdgeEdit e;
            e.edgeUuid = edge->uuid; e.expectedGeneration = edge->generation; e.action = edit.action;
            const auto a = store.currentGeometry(txn, edge->aUuid), b = store.currentGeometry(txn, edge->bUuid);
            if (a) e.aRevision = a->uuid;
            if (b) e.bRevision = b->uuid;
            if (edit.action == "update") {
                e.aAnchor = anchorFromJson(Json::parse(edit.aAnchorJson));
                e.bAnchor = anchorFromJson(Json::parse(edit.bAnchorJson));
            } else {
                e.aAnchor = edge->aAnchor; e.bAnchor = edge->bAnchor;
            }
            result.push_back(std::move(e));
        }
        return result;
    }

    std::string geometryHead(const std::string& object)
    {
        const auto head = store.currentGeometry(txn, object);
        if (!head) throw std::invalid_argument("transaction: object has no geometry " + object);
        return head->uuid;
    }

    void operator()(const CreateObjectOp& op)
    {
        const auto j = Json::parse(op.recordJson);
        auto record = objectFromJson(j);
        if (record.uuid.empty()) throw std::invalid_argument("transaction: create_object requires a client-minted UUID");
        record.uuid = canonical(record.uuid);
        if (store.objectByUuid(txn, record.uuid))
            throw TransactionConflict(TransactionConflict::Reason::Version, {{"object", record.uuid, std::nullopt, std::nullopt}},
                "object history: create_object expects absence of " + record.uuid);
        store.upsertObject(txn, record);
    }
    void operator()(const UpdateObjectOp& op)
    {
        auto record = objectFromJson(Json::parse(op.recordJson));
        record.uuid = canonical(record.uuid);
        if (!store.objectByUuid(txn, record.uuid)) throw std::invalid_argument("transaction: update_object of unknown object");
        store.upsertObject(txn, record);
    }
    void operator()(const SetMetaOp& op) { store.setMeta(txn, op.uuid, op.key, op.jsonValue); }
    void operator()(const RemoveMetaOp& op) { store.removeMeta(txn, op.uuid, op.key); }
    void operator()(const SetApprovalOp& op) { store.setApproval(txn, op.uuid, op.state); }
    void operator()(const RetireObjectOp& op) { store.retireObject(txn, op.uuid); }
    void operator()(const RestoreObjectOp& op) { store.restoreObject(txn, op.uuid, op.source); }
    void operator()(const DeleteObjectOp& op) { store.deleteObject(txn, op.uuid); }
    void operator()(const AddEdgeOp& op)
    {
        auto record = edgeInput(Json::parse(op.recordJson));
        if (record.uuid.empty()) throw std::invalid_argument("transaction: add_relationship requires a client-minted UUID");
        record.uuid = canonical(record.uuid);
        if (store.edgeByUuid(txn, record.uuid))
            throw TransactionConflict(TransactionConflict::Reason::Version, {{"relationship", record.uuid, std::nullopt, std::nullopt}},
                "object history: add_relationship expects absence of " + record.uuid);
        store.addEdge(txn, record);
    }
    void operator()(const UpdateEdgeOp& op)
    {
        auto record = edgeInput(Json::parse(op.recordJson));
        const auto current = store.edgeByUuid(txn, record.uuid);
        if (!current) throw std::invalid_argument("transaction: update_relationship of unknown relationship");
        record.generation = current->generation;
        store.updateEdge(txn, record);
    }
    void operator()(const RemoveEdgeOp& op) { store.removeEdge(txn, op.uuid); }
    void operator()(const RetainRelationshipOp& op) { store.retainRelationship(txn, op.uuid); }
    void operator()(const KeepRecordedRelationshipOp& op) { store.keepRecordedRelationship(txn, op.uuid); }
    void operator()(const CheckVolumeMappingOp& op)
    {
        checkVolumeMapping(store, txn, op.objectUuid, op.targetVolume, op.targetScale, op.mapping);
    }
    void operator()(const RecordCatalogTransformsOp& op) { store.recordCatalogTransforms(txn, op.registrations); }
    void operator()(const IngestGeometryOp& op)
    {
        std::vector<GeometrySource> batch;
        for (const auto& item : op.items)
            batch.push_back({objectFromJson(Json::parse(item.recordJson)), item.data, Json::parse(item.sourceJson)});
        detail::ingestGeometryBatch(store, txn, std::move(batch), op.coordinateContext, false);
    }
    void operator()(const PutGeometryOp& op)
    {
        const auto head = store.currentGeometry(txn, op.objectUuid);
        store.putGeometry(txn, op.objectUuid, op.data, head ? std::optional(head->uuid) : std::nullopt, edgeEdits(op));
    }
    void operator()(const EditGeometryOp& op)
    {
        const auto head = geometryHead(op.objectUuid);
        store.editGeometry(txn, op.objectUuid, op.edits, head, edgeEdits(op));
    }
    void operator()(const EditGeometryRectangleOp& op)
    {
        const auto head = geometryHead(op.objectUuid);
        store.editGeometryRectangle(txn, op.objectUuid, op.arrayName, op.row, op.column, op.rows, op.columns, op.bytes, head,
            edgeEdits(op));
    }
    void operator()(const SetSurfaceChannelsOp& op)
    {
        store.updateSurfaceChannels(txn, op.objectUuid, op.channels, op.remove, geometryHead(op.objectUuid), edgeEdits(op));
    }
    void operator()(const LinkObjectsOp&)
    {
        const auto found = prepared.find(index);
        if (found == prepared.end() || !found->second.apply)
            throw std::invalid_argument("transaction: link_objects was not prepared (ObjectStore::prepare)");
        found->second.apply(store, txn);
    }
    void operator()(const UpdateGeometryArraysOp& op)
    {
        if (op.affected != "retain" && op.affected != "keep_recorded")
            throw std::invalid_argument("transaction: update_geometry_arrays decides affected relationships by retain or keep_recorded");
        // Each step of the update decides the relationships it affects.
        updateGeometryArrays(store, txn, op.objectUuid, op.data, geometryHead(op.objectUuid), op.annotations,
            [&](const std::vector<EdgeRecord>& affected) {
                std::vector<GeometryRelationshipEdit> edits;
                for (const auto& edge : affected) edits.push_back({edge.uuid, op.affected});
                return edgeEdits(edits);
            });
    }
};
}

namespace {
std::mutex& preparerMutex() { static std::mutex m; return m; }
std::map<std::string, OperationPreparer>& preparers() { static std::map<std::string, OperationPreparer> p; return p; }

std::optional<std::string> preparedOperationName(const TransactionOperation& op)
{
    if (std::holds_alternative<LinkObjectsOp>(op)) return kLinkObjectsOperation;
    return std::nullopt;
}

void validateRequest(const TransactionRequest& request)
{
    if (!util::isUuid(request.requestUuid)) throw std::invalid_argument("transaction: request UUID required");
    if (request.author.empty()) throw std::invalid_argument("transaction: author required");
    if (!util::isUuid(request.storeUuid) || !util::isUuid(request.generation))
        throw std::invalid_argument("transaction: store UUID and generation required");
}

// Merge a preparation's versions into the request's: the same record with
// another version is the request contradicting what its preparation read.
void mergeVersions(VersionSelection& into, const VersionSelection& from, const char* what)
{
    for (const auto& [uuid, version] : from) {
        const auto id = canonical(uuid);
        const auto [found, inserted] = into.emplace(id, version);
        if (!inserted && found->second != version)
            throw std::invalid_argument(std::string("transaction: ") + what + " of " + id + " differ between the request and its preparation");
    }
}
} // namespace

void registerOperationPreparer(const std::string& operationName, OperationPreparer preparer)
{
    std::lock_guard lock(preparerMutex());
    preparers()[operationName] = std::move(preparer);
}

VersionSelection PreparedOperations::expectedObjects() const
{
    VersionSelection result;
    for (const auto& [index, preparation] : operations_) mergeVersions(result, preparation.expectedObjects, "expected versions");
    return result;
}

VersionSelection PreparedOperations::readDependencies() const
{
    VersionSelection result;
    for (const auto& [index, preparation] : operations_) mergeVersions(result, preparation.readDependencies, "read dependencies");
    return result;
}

PreparedOperations ObjectStore::prepare(const TransactionRequest& request, const PrepareOptions& options)
{
    validateRequest(request);
    PreparedOperations result;
    bool any = false;
    for (const auto& op : request.operations) any = any || preparedOperationName(op).has_value();
    if (!any) return result;
    {
        // Already committed: submit() replays the receipt, so nothing to prepare.
        auto txn = read();
        if (savedReceipt(static_cast<sqlite3*>(connection(txn)), canonical(request.requestUuid), nullptr)) return result;
        const auto current = status(txn);
        result.storeUuid_ = current.storeUuid;
        result.generation_ = current.generation;
    }
    result.requestUuid_ = canonical(request.requestUuid);
    result.requestDigest_ = transactionRequestDigest(request);
    for (std::size_t i = 0; i < request.operations.size(); ++i) {
        const auto name = preparedOperationName(request.operations[i]);
        if (!name) continue;
        OperationPreparer preparer;
        {
            std::lock_guard lock(preparerMutex());
            if (const auto found = preparers().find(*name); found != preparers().end()) preparer = found->second;
        }
        if (!preparer)
            throw std::invalid_argument("transaction: " + *name + " is not available here (its library is not linked)");
        result.operations_.emplace(i, preparer(*this, request.operations[i], options));
    }
    return result;
}

TransactionReceipt ObjectStore::submit(const TransactionRequest& request)
{
    return submit(request, prepare(request));
}

TransactionReceipt ObjectStore::submit(const TransactionRequest& request, const PreparedOperations& prepared)
{
    validateRequest(request);
    const auto requestUuid = canonical(request.requestUuid);
    const auto digest = transactionRequestDigest(request);
    auto txn = write();
    auto* db = static_cast<sqlite3*>(connection(txn));
    // A saved receipt answers before any version check: a lost reply must not
    // turn into a conflict with the transaction's own changes.
    std::string savedDigest;
    if (auto saved = savedReceipt(db, requestUuid, &savedDigest)) {
        if (savedDigest != digest)
            throw std::invalid_argument("transaction: request " + requestUuid + " was already used with different contents");
        saved->replayed = true;
        return *saved; // txn rolls back
    }
    auto& pool = *activeStatementPools.at(db);
    pool.requestUuid = requestUuid; pool.requestDigest = digest; pool.actingAuthor = request.author;
    TransactionContext context;
    context.storeUuid = request.storeUuid; context.generation = request.generation;
    context.expectedObjects = request.expectedObjects; context.expectedRelationships = request.expectedRelationships;
    context.readDependencies = request.readDependencies; context.expectedCommitSequence = request.expectedCommitSequence;
    if (!prepared.empty()) {
        // A preparation publishes only for the request whose detection it
        // ran, on the store it read.
        if (prepared.requestUuid_ != requestUuid || prepared.requestDigest_ != digest)
            throw std::invalid_argument("transaction: the preparation belongs to another request");
        const auto current = status(txn);
        if (prepared.storeUuid_ != current.storeUuid || prepared.generation_ != current.generation)
            throw std::invalid_argument("transaction: the preparation was made on another store");
    }
    for (const auto& [index, preparation] : prepared.operations_) {
        if (index >= request.operations.size() || !preparedOperationName(request.operations[index]))
            throw std::invalid_argument("transaction: preparation " + std::to_string(index) + " names no prepared operation");
        mergeVersions(context.expectedObjects, preparation.expectedObjects, "expected versions");
        mergeVersions(context.readDependencies, preparation.readDependencies, "read dependencies");
    }
    // A record both written and read-guarded at the same version is just
    // expected; a read dependency at another version is a stale read the
    // check below reports (a preparation that read the record later than
    // the request's own read).
    for (const auto& [uuid, version] : context.expectedObjects)
        if (const auto guard = context.readDependencies.find(uuid);
            guard != context.readDependencies.end() && guard->second == version)
            context.readDependencies.erase(guard);
    RevisionSelection expected;
    checkTransactionContext(db, context, expected);
    expectHistory(db, expected);
    txn.expectedRevisions_ = expected;
    txn.expectedObjectVersions_ = context.expectedObjects;
    txn.expectedRelationshipVersions_ = request.expectedRelationships;
    Executor apply{*this, txn, prepared.operations_};
    for (apply.index = 0; apply.index < request.operations.size(); ++apply.index)
        std::visit(apply, request.operations[apply.index]);
    txn.commit();
    return *txn.receipt();
}

} // namespace vc::objects

namespace vc::objects {
namespace {
void collectPayloads(const Json& op, std::vector<std::string>& hashes)
{
    const auto name = op.at("op").get<std::string>();
    auto add = [&](const Json& ref) { hashes.push_back(ref.at("sha256").get<std::string>()); };
    if (name == "put_geometry" || name == "update_geometry_arrays")
        for (const auto& array : op.at("data").at("arrays")) add(array.at("bytes"));
    else if (name == "set_surface_channels")
        for (const auto& array : op.at("channels")) add(array.at("bytes"));
    else if (name == "ingest_geometry")
        for (const auto& item : op.at("items"))
            for (const auto& array : item.at("data").at("arrays")) add(array.at("bytes"));
    else if (name == "edit_geometry")
        for (const auto& edit : op.at("edits")) add(edit.at("bytes"));
    else if (name == "edit_geometry_rectangle")
        add(op.at("bytes"));
}

std::vector<std::string> payloadHashesIn(const Json& request)
{
    std::vector<std::string> hashes;
    for (const auto& op : request.at("operations")) collectPayloads(op, hashes);
    std::sort(hashes.begin(), hashes.end());
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
    return hashes;
}
} // namespace

std::vector<std::string> transactionJsonPayloadHashes(const std::string& text) { return payloadHashesIn(Json::parse(text)); }

bool transactionJsonHasPreparedOperations(const std::string& text)
{
    const auto request = Json::parse(text); // the loop reads into it
    for (const auto& op : request.at("operations"))
        if (op.at("op").get<std::string>() == kLinkObjectsOperation) return true;
    return false;
}

TransactionReceipt ObjectStore::submit(const std::string& text, const PayloadLookup& lookup)
{
    auto j = Json::parse(text);
    TransactionPayloads payloads;
    std::vector<std::string> missing;
    for (const auto& hash : payloadHashesIn(j)) {
        if (auto bytes = lookup ? lookup(hash) : std::nullopt) payloads.emplace(hash, std::move(*bytes));
        else missing.push_back(hash);
    }
    if (missing.empty()) return submit(transactionRequestFromJson(text, payloads));
    // Without the bytes only the saved receipt of this exact request can
    // answer. The digest of canonical JSON text equals the digest of its
    // decoded request; a non-canonical text just has to re-stage.
    const auto requestUuid = canonical(j.at("request").get<std::string>());
    j.erase("request");
    const auto canonicalText = j.dump();
    const auto textDigest = sha256(reinterpret_cast<const uint8_t*>(canonicalText.data()), canonicalText.size());
    auto txn = read();
    std::string savedDigest;
    if (auto saved = savedReceipt(static_cast<sqlite3*>(connection(txn)), requestUuid, &savedDigest);
        saved && savedDigest == textDigest) {
        saved->replayed = true;
        return *saved;
    }
    throw MissingPayloads(std::move(missing));
}

} // namespace vc::objects
