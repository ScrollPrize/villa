// JSON forms of conflicts, status, feed pages and historical reads, shared by
// the hosted service and its clients.
#include "vc_object_store/objects/TransactionJson.hpp"
#include "internal/RecordJson.hpp"
#include "internal/ObjectVersionJson.hpp"

#include <set>
#include <limits>

namespace vc::objects {
using namespace detail;
using Json = nlohmann::json;
static void versionToJson(const RecordCommon&, Json&);
static void versionFromJson(const Json&, RecordCommon&);

namespace {
void requireKeys(const Json&, std::initializer_list<const char*>, const char*);
const char* reasonName(TransactionConflict::Reason reason)
{
    switch (reason) {
    case TransactionConflict::Reason::Store: return "store";
    case TransactionConflict::Reason::Generation: return "generation";
    case TransactionConflict::Reason::Version: return "version";
    case TransactionConflict::Reason::ReadDependency: return "read_dependency";
    case TransactionConflict::Reason::CommitSequence: return "commit_sequence";
    }
    throw std::logic_error("transaction conflict: unknown reason");
}

TransactionConflict::Reason reasonFrom(const std::string& name)
{
    using R = TransactionConflict::Reason;
    for (auto reason : {R::Store, R::Generation, R::Version, R::ReadDependency, R::CommitSequence})
        if (name == reasonName(reason)) return reason;
    throw std::invalid_argument("transaction conflict: unknown reason " + name);
}

Json optionalInt(const std::optional<int64_t>& value) { return value ? Json(*value) : Json(nullptr); }
std::optional<int64_t> intFrom(const Json& value)
{
    return value.is_null() ? std::nullopt : std::optional(value.get<int64_t>());
}
Json optionalText(const std::optional<std::string>& value) { return value ? Json(*value) : Json(nullptr); }
std::optional<std::string> textFrom(const Json& value)
{
    return value.is_null() ? std::nullopt : std::optional(value.get<std::string>());
}

// Record metadata travels as its exact stored text (key order and number
// spelling included), never re-serialized through a parsed object.
Json withExactMeta(Json j, const RecordCommon& r)
{
    if (j.contains("meta")) j["meta"] = r.meta;
    return j;
}
template<class Record> Record exactMetaFrom(const Json& j, Record r)
{
    if (j.contains("meta")) r.meta = j.at("meta").get<std::string>();
    return r;
}

FeedCursor cursorFrom(const Json& j)
{
    return {j.at("store").get<std::string>(), j.at("generation").get<std::string>(), j.at("sequence").get<int64_t>(),
            j.at("offset").get<int64_t>()};
}
} // namespace

std::string transactionConflictToJson(const TransactionConflict& conflict)
{
    auto details = Json::array();
    for (const auto& d : conflict.details())
        details.push_back({{"kind", d.recordKind}, {"uuid", d.uuid}, {"expected", optionalInt(d.expected)},
                           {"actual", optionalInt(d.actual)}});
    return Json{{"reason", reasonName(conflict.reason())}, {"message", conflict.what()}, {"details", details}}.dump();
}

TransactionConflict transactionConflictFromJson(const std::string& text)
{
    const auto j = Json::parse(text);
    std::vector<ConflictDetail> details;
    for (const auto& d : j.at("details"))
        details.push_back({d.at("kind").get<std::string>(), d.at("uuid").get<std::string>(), intFrom(d.at("expected")),
                           intFrom(d.at("actual"))});
    return TransactionConflict(reasonFrom(j.at("reason").get<std::string>()), std::move(details),
        j.at("message").get<std::string>());
}

StoreStatus storeStatusFromJson(const std::string& text)
{
    return storeStatusFrom(Json::parse(text));
}

ChangePage changePageFromJson(const std::string& text)
{
    const auto j = Json::parse(text);
    ChangePage page;
    page.cursorValid = j.at("cursor_valid").get<bool>();
    page.more = j.at("more").get<bool>();
    page.next = cursorFrom(j.at("next"));
    for (const auto& c : j.at("commits")) {
        FeedCommit commit;
        commit.sequence = c.at("sequence").get<int64_t>();
        commit.requestUuid = textFrom(c.at("request_uuid"));
        commit.author = c.at("author").get<std::string>();
        commit.host = c.at("host").get<std::string>();
        commit.publishedAtUs = c.at("published_at_us").get<int64_t>();
        commit.complete = c.at("complete").get<bool>();
        for (const auto& change : c.at("changes"))
            commit.changes.push_back({change.at("kind").get<std::string>(), change.at("uuid").get<std::string>(),
                                      change.at("version").get<int64_t>(), change.at("deleted").get<bool>()});
        page.commits.push_back(std::move(commit));
    }
    return page;
}

std::string feedCursorQuery(const FeedCursor& cursor)
{
    return "store=" + cursor.storeUuid + "&generation=" + cursor.generation + "&sequence=" + std::to_string(cursor.sequence) +
           "&offset=" + std::to_string(cursor.offset);
}

namespace {
Json revisionJson(const ObjectRevision& r)
{
    return Json{{"uuid", r.uuid}, {"object", r.objectUuid}, {"parent", optionalText(r.parentUuid)},
                {"geometry_revision", optionalText(r.geometryRevision)}, {"sequence", r.sequence},
                {"metadata_complete", r.metadataComplete}, {"published_at_us", optionalInt(r.publishedAtUs)},
                {"deleted", r.deleted}, {"version", r.version}, {"generation", r.generation}, {"store", r.storeUuid},
                {"commit_sequence", optionalInt(r.commitSequence)}, {"record", withExactMeta(objectToJson(r.record), r.record)}};
}

ObjectRevision revisionFrom(const Json& j)
{
    ObjectRevision r;
    r.uuid = j.at("uuid").get<std::string>();
    r.objectUuid = j.at("object").get<std::string>();
    r.parentUuid = textFrom(j.at("parent"));
    r.geometryRevision = textFrom(j.at("geometry_revision"));
    r.sequence = j.at("sequence").get<int64_t>();
    r.metadataComplete = j.at("metadata_complete").get<bool>();
    r.publishedAtUs = intFrom(j.at("published_at_us"));
    r.deleted = j.at("deleted").get<bool>();
    r.version = j.at("version").get<int64_t>();
    r.generation = j.at("generation").get<std::string>();
    r.storeUuid = j.at("store").get<std::string>();
    r.commitSequence = intFrom(j.at("commit_sequence"));
    const auto& record = j.at("record");
    r.record = exactMetaFrom(record, objectFromJson(record));
    commonFromJson(record, r.record);
    r.record.versionStoreUuid=r.storeUuid;
    return r;
}
} // namespace

std::string objectRevisionToJson(const ObjectRevision& r) { return revisionJson(r).dump(); }

ObjectRevision objectRevisionFromJson(const std::string& text) { return revisionFrom(Json::parse(text)); }

std::string geometryRevisionToJson(const GeometryRevision& g)
{
    auto arrays = Json::array();
    for (const auto& a : g.arrays)
        arrays.push_back({{"name", a.name}, {"dtype", a.dtype}, {"shape", a.shape}, {"byte_length", a.byteLength}});
    return Json{{"uuid", g.uuid}, {"object", g.objectUuid}, {"parent", optionalText(g.parentUuid)},
                {"content_hash", g.contentHash}, {"kind", g.kind}, {"coordinate_system", g.coordinateSystem},
                {"attributes", g.attributes}, {"format_version", g.formatVersion},
                {"object_version", g.objectVersion}, {"created_by", g.createdBy}, {"created_at", g.createdAt},
                {"created_host", g.createdHost}, {"arrays", arrays}, {"grid_origin", g.gridOrigin}}
        .dump();
}

GeometryRevision geometryRevisionFromJson(const std::string& text)
{
    const auto j = Json::parse(text);
    GeometryRevision g;
    g.uuid = j.at("uuid").get<std::string>();
    g.objectUuid = j.at("object").get<std::string>();
    g.parentUuid = textFrom(j.at("parent"));
    g.contentHash = j.at("content_hash").get<std::string>();
    g.kind = j.at("kind").get<std::string>();
    g.coordinateSystem = j.at("coordinate_system").get<std::string>();
    g.attributes = j.at("attributes").get<std::string>(); // exact stored text
    g.formatVersion = j.at("format_version").get<int64_t>();
    g.objectVersion = j.at("object_version").get<int64_t>();
    g.createdBy = j.at("created_by").get<std::string>();
    g.createdAt = j.at("created_at").get<std::string>();
    g.createdHost = j.at("created_host").get<std::string>();
    for (const auto& a : j.at("arrays"))
        g.arrays.push_back({a.at("name").get<std::string>(), a.at("dtype").get<std::string>(),
                            a.at("shape").get<std::vector<int64_t>>(), a.at("byte_length").get<int64_t>()});
    g.gridOrigin = j.at("grid_origin").get<std::array<int64_t, 2>>();
    return g;
}

// ---- Hosted queries ---------------------------------------------------------

namespace {
void requireKeys(const Json& j, std::initializer_list<const char*> allowed, const char* what)
{
    if (!j.is_object()) throw std::invalid_argument(std::string(what) + " must be a JSON object");
    for (const auto& item : j.items()) {
        bool known = false;
        for (const char* key : allowed) known = known || item.key() == key;
        if (!known) throw std::invalid_argument(std::string(what) + ": unknown key " + item.key());
    }
}

// Absent keys keep the C++ default; null clears an optional.
template<class T> void read(const Json& j, const char* key, T& field)
{
    if (j.contains(key)) j.at(key).get_to(field);
}
template<class T> void read(const Json& j, const char* key, std::optional<T>& field)
{
    if (!j.contains(key) || j.at(key).is_null()) field.reset();
    else field = j.at(key).get<T>();
}
template<class T> Json opt(const std::optional<T>& value) { return value ? Json(*value) : Json(nullptr); }

std::optional<std::set<std::string>> fieldSet(const std::optional<std::vector<std::string>>& fields)
{
    if (!fields) return std::nullopt;
    return std::set<std::string>(fields->begin(), fields->end());
}

Json envelope(const StoreStatus& status, const char* key, Json value)
{
    return Json{{"status", storeStatusJson(status)}, {key, std::move(value)}};
}

// The `key` member of an answer; `asOf` receives its snapshot status.
Json unwrap(const std::string& text, StoreStatus* asOf, const char* key)
{
    auto j = Json::parse(text);
    if (asOf) *asOf = storeStatusFrom(j.at("status"));
    return std::move(j.at(key));
}

void refuseTransforms(bool explicitTransforms, const char* what)
{
    if (explicitTransforms)
        throw std::invalid_argument(std::string("hosted ") + what +
                                    ": transforms are not sent; the store's recorded transforms apply");
}

Json selectionJson(const ObjectSelection& s)
{
    refuseTransforms(bool(s.transforms), "object selection");
    return Json{{"kind", opt(s.kind)}, {"tag", opt(s.tag)}, {"connected_to", opt(s.connectedTo)}, {"uuids", s.uuids},
                {"edge_kinds", s.edgeKinds}, {"limit", opt(s.limit)}, {"offset", s.offset}, {"bbox", opt(s.bbox)},
                {"family", opt(s.family)}, {"min_length", opt(s.minLength)}, {"order_by", s.orderBy},
                {"after_uuid", opt(s.afterUuid)}, {"after_length", opt(s.afterLength)},
                {"include_retired", s.includeRetired}, {"coordinates", opt(s.coordinates)}, {"volume", opt(s.volume)},
                {"target_scale", s.targetScale}, {"include_unbound", s.includeUnbound}, {"tag_groups", s.tagGroups},
                {"without_tags", s.withoutTags}};
}

ObjectSelection selectionFrom(const Json& j)
{
    requireKeys(j, {"kind", "tag", "connected_to", "uuids", "edge_kinds", "limit", "offset", "bbox", "family",
                    "min_length", "order_by", "after_uuid", "after_length", "include_retired", "coordinates", "volume",
                    "target_scale", "include_unbound", "tag_groups", "without_tags"}, "object selection");
    ObjectSelection s;
    read(j, "kind", s.kind); read(j, "tag", s.tag); read(j, "connected_to", s.connectedTo);
    read(j, "uuids", s.uuids); read(j, "edge_kinds", s.edgeKinds); read(j, "limit", s.limit);
    read(j, "offset", s.offset); read(j, "bbox", s.bbox); read(j, "family", s.family);
    read(j, "min_length", s.minLength); read(j, "order_by", s.orderBy); read(j, "after_uuid", s.afterUuid);
    read(j, "after_length", s.afterLength); read(j, "include_retired", s.includeRetired);
    read(j, "coordinates", s.coordinates); read(j, "volume", s.volume); read(j, "target_scale", s.targetScale);
    read(j, "include_unbound", s.includeUnbound); read(j, "tag_groups", s.tagGroups);
    read(j, "without_tags", s.withoutTags);
    return s;
}

Json edgeSelectionJson(const EdgeSelection& s)
{
    return Json{{"kind", opt(s.kind)}, {"tag", opt(s.tag)}, {"approval_states", s.approvalStates},
                {"endpoint_uuids", opt(s.endpointUuids)}, {"uuids", opt(s.uuids)}, {"both_endpoints", s.bothEndpoints},
                {"limit", opt(s.limit)}, {"after_uuid", opt(s.afterUuid)}, {"include_retired", s.includeRetired},
                {"adjacent", opt(s.adjacent)}, {"applicable_only", s.applicableOnly}, {"attachments", s.attachments}};
}

EdgeSelection edgeSelectionFrom(const Json& j)
{
    requireKeys(j, {"kind", "tag", "approval_states", "endpoint_uuids", "uuids", "both_endpoints", "limit", "after_uuid",
                    "include_retired", "adjacent", "applicable_only", "attachments"}, "edge selection");
    EdgeSelection s;
    read(j, "kind", s.kind); read(j, "tag", s.tag); read(j, "approval_states", s.approvalStates);
    read(j, "endpoint_uuids", s.endpointUuids); read(j, "uuids", s.uuids); read(j, "both_endpoints", s.bothEndpoints);
    read(j, "limit", s.limit); read(j, "after_uuid", s.afterUuid); read(j, "include_retired", s.includeRetired);
    read(j, "adjacent", s.adjacent); read(j, "applicable_only", s.applicableOnly); read(j, "attachments", s.attachments);
    return s;
}

Json fieldsJson(const std::optional<std::vector<std::string>>& fields) { return opt(fields); }

Json relationshipJson(const RelationshipRevision& r)
{
    auto record = withExactMeta(edgeToJson(r.record, nullptr, false), r.record);
    versionToJson(r.record, record);
    return Json{{"uuid", r.uuid}, {"record", std::move(record)}, {"a_revision", r.aRevision},
                {"b_revision", r.bRevision}, {"a_geometry_revision", optionalText(r.aGeometryRevision)},
                {"b_geometry_revision", optionalText(r.bGeometryRevision)}, {"version", r.version},
                {"generation", r.generation}, {"a_version", r.aReference.version}, {"b_version", r.bReference.version},
                {"a_reference", objectVersionJson(r.aReference)}, {"b_reference", objectVersionJson(r.bReference)}};
}

RelationshipRevision relationshipFrom(const Json& j)
{
    RelationshipRevision r;
    r.uuid = j.at("uuid").get<std::string>();
    r.record = exactMetaFrom(j.at("record"), edgeFromJson(j.at("record")));
    versionFromJson(j.at("record"), r.record);
    r.aRevision = j.at("a_revision").get<std::string>();
    r.bRevision = j.at("b_revision").get<std::string>();
    r.aGeometryRevision = textFrom(j.at("a_geometry_revision"));
    r.bGeometryRevision = textFrom(j.at("b_geometry_revision"));
    r.version = j.at("version").get<int64_t>();
    r.generation = j.at("generation").get<std::string>();
    r.aReference = objectVersionFrom(j.at("a_reference"));
    r.bReference = objectVersionFrom(j.at("b_reference"));
    // Older servers emitted encoded exact refs in these archive UUID fields.
    // Complete endpoint refs above already carry that identity.
    if (!r.aRevision.empty() && r.aRevision.front() == '[') r.aRevision.clear();
    if (!r.bRevision.empty() && r.bRevision.front() == '[') r.bRevision.clear();
    return r;
}

RegionStop regionStopFrom(const std::string& name)
{
    using S = RegionStop;
    for (auto stop : {S::Exhausted, S::ObjectLimit, S::OutputBudget, S::ProcessingBudget, S::CandidateBudget,
                      S::SqlBudget, S::Cancelled})
        if (name == regionStopName(stop)) return stop;
    throw std::invalid_argument("region query: unknown stop reason " + name);
}
} // namespace

std::string objectQueryRequestToJson(const ObjectSelection& selection, const std::optional<std::vector<std::string>>& fields)
{
    return Json{{"selection", selectionJson(selection)}, {"fields", fieldsJson(fields)}}.dump();
}

ObjectSelection objectQueryRequestFromJson(const std::string& text, std::optional<std::vector<std::string>>* fields)
{
    const auto j = Json::parse(text);
    requireKeys(j, {"selection", "fields"}, "object query");
    if (fields) read(j, "fields", *fields);
    return selectionFrom(j.at("selection"));
}

// The pinned head version a selection read (RecordCommon::version).
static void versionToJson(const RecordCommon& record, Json& j)
{
    if (!record.version) return;
    j["version"] = *record.version;
    j["version_generation"] = record.versionGeneration;
    j["version_store"] = record.versionStoreUuid;
}

static void versionFromJson(const Json& j, RecordCommon& record)
{
    if (!j.contains("version")) return;
    record.version = j.at("version").get<int64_t>();
    record.versionGeneration = j.at("version_generation").get<std::string>();
    record.versionStoreUuid = j.at("version_store").get<std::string>();
}

std::string objectQueryResultToJson(const StoreStatus& status, const std::vector<ObjectRecord>& objects,
                                    const std::optional<std::vector<std::string>>& fields)
{
    const auto selected = fieldSet(fields);
    auto array = Json::array();
    for (const auto& object : objects) {
        auto j = withExactMeta(objectToJson(object, selected ? &*selected : nullptr), object);
        if (object.volumeLength) j["volume_length"] = *object.volumeLength;
        versionToJson(object, j);
        array.push_back(std::move(j));
    }
    return envelope(status, "objects", std::move(array)).dump();
}

std::vector<ObjectRecord> objectQueryResultFromJson(const std::string& text, StoreStatus* asOf)
{
    std::vector<ObjectRecord> result;
    for (const auto& j : unwrap(text, asOf, "objects")) {
        auto record = exactMetaFrom(j, objectFromJson(j));
        presentCommonFromJson(j, record);
        read(j, "volume_length", record.volumeLength);
        versionFromJson(j, record);
        result.push_back(std::move(record));
    }
    return result;
}

std::string edgeQueryRequestToJson(const EdgeSelection& selection, const std::optional<std::vector<std::string>>& fields)
{
    return Json{{"selection", edgeSelectionJson(selection)}, {"fields", fieldsJson(fields)}}.dump();
}

EdgeSelection edgeQueryRequestFromJson(const std::string& text, std::optional<std::vector<std::string>>* fields)
{
    const auto j = Json::parse(text);
    requireKeys(j, {"selection", "fields"}, "edge query");
    if (fields) read(j, "fields", *fields);
    return edgeSelectionFrom(j.at("selection"));
}

std::string edgeQueryResultToJson(const StoreStatus& status, const std::vector<EdgeRecord>& edges,
                                  const std::optional<std::vector<std::string>>& fields)
{
    const auto selected = fieldSet(fields);
    auto array = Json::array();
    for (const auto& edge : edges) {
        auto j = withExactMeta(edgeToJson(edge, selected ? &*selected : nullptr, false), edge);
        versionToJson(edge, j);
        array.push_back(std::move(j));
    }
    return envelope(status, "edges", std::move(array)).dump();
}

std::vector<EdgeRecord> edgeQueryResultFromJson(const std::string& text, StoreStatus* asOf)
{
    std::vector<EdgeRecord> result;
    for (const auto& j : unwrap(text, asOf, "edges")) {
        auto edge = exactMetaFrom(j, edgeFromJson(j, true));
        versionFromJson(j, edge);
        result.push_back(std::move(edge));
    }
    return result;
}

std::string kindQueryRequestToJson(const KindQuery& query)
{
    Json j = Json::object();
    if (query.kind) j["kind"] = *query.kind;
    if (query.includeRetired) j["include_retired"] = true;
    if (query.unboundVolume) j["unbound_volume"] = *query.unboundVolume;
    return j.dump();
}

KindQuery kindQueryRequestFromJson(const std::string& text)
{
    const auto j = Json::parse(text);
    requireKeys(j, {"kind", "include_retired", "unbound_volume"}, "kind query");
    KindQuery query;
    if (j.contains("kind")) query.kind = j.at("kind").get<std::string>();
    query.includeRetired = j.value("include_retired", false);
    if (j.contains("unbound_volume")) query.unboundVolume = j.at("unbound_volume").get<std::string>();
    if (query.unboundVolume && query.includeRetired)
        throw std::invalid_argument("kind query: unbound counts are of live objects");
    return query;
}

std::string objectCountResultToJson(const StoreStatus& status, int64_t count)
{
    return envelope(status, "count", count).dump();
}

int64_t objectCountResultFromJson(const std::string& text, StoreStatus* asOf)
{
    return unwrap(text, asOf, "count").get<int64_t>();
}

std::string objectTagsResultToJson(const StoreStatus& status, const std::vector<std::string>& tags)
{
    return envelope(status, "tags", tags).dump();
}

std::vector<std::string> objectTagsResultFromJson(const std::string& text, StoreStatus* asOf)
{
    return unwrap(text, asOf, "tags").get<std::vector<std::string>>();
}

std::string graphQueryRequestToJson(const GraphQuery& query)
{
    Json j{{"traversal", query.kind == GraphQuery::Kind::Component ? "component" : "neighbours"},
           {"object", query.objectUuid}};
    if (!query.edgeKinds.empty()) j["edge_kinds"] = query.edgeKinds;
    if (query.neighbourKind) j["neighbour_kind"] = *query.neighbourKind;
    if (query.includeRetired) j["include_retired"] = true;
    return j.dump();
}

GraphQuery graphQueryRequestFromJson(const std::string& text)
{
    const auto j = Json::parse(text);
    requireKeys(j, {"traversal", "object", "edge_kinds", "neighbour_kind", "include_retired"}, "graph query");
    GraphQuery query;
    const auto traversal = j.at("traversal").get<std::string>();
    if (traversal == "neighbours") query.kind = GraphQuery::Kind::Neighbours;
    else if (traversal != "component") throw std::invalid_argument("graph query: unknown traversal " + traversal);
    query.objectUuid = j.at("object").get<std::string>();
    query.edgeKinds = j.value("edge_kinds", std::vector<std::string>{});
    if (j.contains("neighbour_kind")) query.neighbourKind = j.at("neighbour_kind").get<std::string>();
    query.includeRetired = j.value("include_retired", false);
    if (query.kind == GraphQuery::Kind::Neighbours ? query.edgeKinds.size() > 1 : bool(query.neighbourKind))
        throw std::invalid_argument("graph query: neighbours take at most one edge kind; components no neighbour kind");
    return query;
}

std::string uuidListResultToJson(const StoreStatus& status, const std::vector<std::string>& uuids)
{
    return envelope(status, "uuids", uuids).dump();
}

std::vector<std::string> uuidListResultFromJson(const std::string& text, StoreStatus* asOf)
{
    return unwrap(text, asOf, "uuids").get<std::vector<std::string>>();
}

std::string objectHistoryToJson(const StoreStatus& status, const std::vector<ObjectRevision>& revisions)
{
    auto array = Json::array();
    for (const auto& revision : revisions) array.push_back(revisionJson(revision));
    return envelope(status, "revisions", std::move(array)).dump();
}

std::vector<ObjectRevision> objectHistoryFromJson(const std::string& text, StoreStatus* asOf)
{
    std::vector<ObjectRevision> result;
    for (const auto& j : unwrap(text, asOf, "revisions")) result.push_back(revisionFrom(j));
    return result;
}

std::string objectVersionToJson(const ObjectVersion& reference) { return objectVersionJson(reference).dump(); }
std::string relationshipVersionToJson(const RelationshipVersion& reference) { return relationshipVersionJson(reference).dump(); }
RelationshipVersion relationshipVersionFromJson(const std::string& text) { return relationshipVersionFrom(Json::parse(text)); }
ObjectVersion objectVersionFromJson(const std::string& text) { return objectVersionFrom(Json::parse(text)); }

std::string objectVersionsRequestToJson(const std::vector<ObjectVersion>& versions)
{
    return Json{{"versions", objectVersionsJson(versions)}}.dump();
}

std::vector<ObjectVersion> objectVersionsRequestFromJson(const std::string& text)
{
    const auto j = Json::parse(text);
    requireKeys(j, {"versions"}, "object versions request");
    return objectVersionsFrom(j.at("versions"));
}

std::string objectsAtVersionsResultToJson(const StoreStatus& status, const std::vector<std::optional<ObjectRevision>>& revisions)
{
    auto array = Json::array();
    for (const auto& revision : revisions) array.push_back(revision ? revisionJson(*revision) : Json(nullptr));
    return envelope(status, "revisions", std::move(array)).dump();
}

std::vector<std::optional<ObjectRevision>> objectsAtVersionsResultFromJson(const std::string& text, StoreStatus* asOf)
{
    std::vector<std::optional<ObjectRevision>> result;
    for (const auto& j : unwrap(text, asOf, "revisions"))
        result.push_back(j.is_null() ? std::nullopt : std::optional(revisionFrom(j)));
    return result;
}

std::string relationshipQueryResultToJson(const StoreStatus& status, const std::vector<SelectedRelationship>& selected)
{
    auto array = Json::array();
    for (const auto& s : selected)
        array.push_back({{"relationship", relationshipJson(s.relationship)}, {"applicable", s.applicable}});
    return envelope(status, "relationships", std::move(array)).dump();
}

std::vector<SelectedRelationship> relationshipQueryResultFromJson(const std::string& text, StoreStatus* asOf)
{
    std::vector<SelectedRelationship> result;
    for (const auto& j : unwrap(text, asOf, "relationships"))
        result.push_back({relationshipFrom(j.at("relationship")), j.at("applicable").get<bool>()});
    return result;
}

std::string selectedInputsToJson(const StoreStatus& status, const SelectedInputs& inputs)
{
    auto records = Json::array(), relationships = Json::array();
    for (const auto& object : inputs.objects) records.push_back(revisionJson(object));
    for (const auto& edge : inputs.relationships)
        relationships.push_back({{"relationship", relationshipJson(edge.relationship)}, {"applicable", edge.applicable}});
    return envelope(status, "inputs", Json{{"objects", records}, {"relationships", relationships}}).dump();
}
std::string selectedInputsRequestToJson(const std::vector<ObjectVersion>& refs, const StoreStatus& position)
{
    return Json{{"versions", objectVersionsJson(refs)}, {"position", storePositionJson(position)}}.dump();
}
std::pair<std::vector<ObjectVersion>, StoreStatus> selectedInputsRequestFromJson(const std::string& text)
{
    auto request = Json::parse(text);
    requireKeys(request, {"versions", "position"}, "selected inputs request");
    const auto position = request.at("position");
    requireKeys(position, {"store", "generation", "commit_sequence"}, "selected inputs position");
    return {objectVersionsFrom(request.at("versions")), storePositionFrom(position)};
}
SelectedInputs selectedInputsFromJson(const std::string& text)
{
    const auto data = unwrap(text, nullptr, "inputs");
    SelectedInputs result;
    for (const auto& object : data.at("objects")) result.objects.push_back(revisionFrom(object));
    for (const auto& edge : data.at("relationships"))
        result.relationships.push_back({relationshipFrom(edge.at("relationship")), edge.at("applicable").get<bool>()});
    return result;
}

std::string regionQueryRequestToJson(const RegionSlice& slice, const RegionOptions& o)
{
    refuseTransforms(o.transforms || o.toTarget, "region query");
    auto result = Json{{"slice", {{"origin", slice.origin}, {"u", slice.u}, {"v", slice.v}, {"normal", slice.normal},
                           {"local_bounds", slice.localBounds}}},
                {"options", {{"kind", o.kind}, {"maximum_objects", o.maximumObjects}, {"selected", opt(o.selected)},
                             {"family", opt(o.family)}, {"min_length", o.minLength}, {"min_points", o.minPoints},
                             {"output_byte_budget", o.outputByteBudget},
                             {"processing_byte_budget", o.processingByteBudget},
                             {"candidate_work_budget", o.candidateWorkBudget}, {"sql_step_budget", o.sqlStepBudget},
                             {"max_unit_bytes", o.maxUnitBytes}, {"uuids", opt(o.uuids)},
                             {"coordinates", opt(o.coordinates)}, {"volume", opt(o.volume)},
                             {"target_scale", o.targetScale}}}};
    if (!o.excludedUuids.empty()) result["options"]["excluded_uuids"] = o.excludedUuids;
    return result.dump();
}

std::pair<RegionSlice, RegionOptions> regionQueryRequestFromJson(const std::string& text)
{
    const auto j = Json::parse(text);
    requireKeys(j, {"slice", "options"}, "region query");
    RegionSlice slice;
    const auto& s = j.at("slice");
    requireKeys(s, {"origin", "u", "v", "normal", "local_bounds"}, "region slice");
    read(s, "origin", slice.origin); read(s, "u", slice.u); read(s, "v", slice.v); read(s, "normal", slice.normal);
    read(s, "local_bounds", slice.localBounds);
    RegionOptions o;
    if (j.contains("options")) {
        const auto& p = j.at("options");
        requireKeys(p, {"kind", "maximum_objects", "selected", "family", "min_length", "min_points",
                        "output_byte_budget", "processing_byte_budget", "candidate_work_budget", "sql_step_budget",
                        "max_unit_bytes", "uuids", "excluded_uuids", "coordinates", "volume", "target_scale"}, "region options");
        read(p, "kind", o.kind); read(p, "maximum_objects", o.maximumObjects); read(p, "selected", o.selected);
        read(p, "family", o.family); read(p, "min_length", o.minLength); read(p, "min_points", o.minPoints);
        read(p, "output_byte_budget", o.outputByteBudget); read(p, "processing_byte_budget", o.processingByteBudget);
        read(p, "candidate_work_budget", o.candidateWorkBudget); read(p, "sql_step_budget", o.sqlStepBudget);
        read(p, "max_unit_bytes", o.maxUnitBytes); read(p, "uuids", o.uuids); read(p, "coordinates", o.coordinates);
        read(p, "excluded_uuids", o.excludedUuids);
        read(p, "volume", o.volume); read(p, "target_scale", o.targetScale);
    }
    return {slice, o};
}

std::string regionQueryResultToJson(const StoreStatus& status, const RegionResult& r)
{
    auto runs = Json::array();
    for (const auto& run : r.runs)
        runs.push_back({{"object", run.objectUuid}, {"revision", run.revisionUuid}, {"array", run.arrayName},
                        {"family", run.family}, {"first_segment", run.firstSegment}, {"points", run.points},
                        {"grid", run.grid},
                        {"object_reference", objectVersionJson(run.objectReference)},
                        {"object_to_volume", run.objectToVolume ? run.objectToVolume->json() : Json(nullptr)}});
    const auto& s = r.stats;
    Json stats{{"output_bytes", s.outputBytes}, {"processing_bytes", s.processingBytes},
               {"payload_bytes", s.payloadBytes}, {"verified_bytes", s.verifiedBytes},
               {"metadata_bytes", s.metadataBytes}, {"reconstructed_bytes", s.reconstructedBytes},
               {"cache_bytes", s.cacheBytes}, {"candidate_work", s.candidateWork},
               {"candidate_visits", s.candidateVisits}, {"segment_tests", s.segmentTests}, {"sql_steps", s.sqlSteps}};
    return envelope(status, "result",
        {{"runs", std::move(runs)}, {"remote", r.remote}, {"stats", std::move(stats)},
         {"intersecting_objects", r.intersectingObjects}, {"stop", regionStopName(r.stop)}})
        .dump();
}

RegionResult regionQueryResultFromJson(const std::string& text, StoreStatus* asOf)
{
    const auto j = unwrap(text, asOf, "result");
    RegionResult r;
    for (const auto& run : j.at("runs"))
        r.runs.push_back({run.at("object").get<std::string>(), run.at("revision").get<std::string>(),
                          run.at("array").get<std::string>(), run.at("family").get<std::string>(),
                          run.at("first_segment").get<int64_t>(),
                          run.at("points").get<std::vector<std::array<double, 3>>>(),
                          run.at("grid").get<std::vector<std::array<double, 2>>>(),
                          objectVersionFrom(run.at("object_reference")),
                          run.at("object_to_volume").is_null() ? std::nullopt : std::optional(VolumeAffine::fromJson(run.at("object_to_volume")))});
    r.remote = j.at("remote").get<std::vector<std::string>>();
    const auto& s = j.at("stats");
    auto& t = r.stats;
    s.at("output_bytes").get_to(t.outputBytes); s.at("processing_bytes").get_to(t.processingBytes);
    s.at("payload_bytes").get_to(t.payloadBytes); s.at("verified_bytes").get_to(t.verifiedBytes);
    s.at("metadata_bytes").get_to(t.metadataBytes); s.at("reconstructed_bytes").get_to(t.reconstructedBytes);
    s.at("cache_bytes").get_to(t.cacheBytes); s.at("candidate_work").get_to(t.candidateWork);
    s.at("candidate_visits").get_to(t.candidateVisits); s.at("segment_tests").get_to(t.segmentTests);
    s.at("sql_steps").get_to(t.sqlSteps);
    r.intersectingObjects = j.at("intersecting_objects").get<uint64_t>();
    r.stop = regionStopFrom(j.at("stop").get<std::string>());
    return r;
}

std::string proximityQueryRequestToJson(const SurfaceProximityQuery& q)
{
    refuseTransforms(q.transforms || q.toTarget, "proximity query");
    return Json{{"point", q.point}, {"tolerance", q.tolerance}, {"uuids", opt(q.uuids)},
                {"coordinates", opt(q.coordinates)}, {"volume", opt(q.volume)}, {"target_scale", q.targetScale}}
        .dump();
}

SurfaceProximityQuery proximityQueryRequestFromJson(const std::string& text)
{
    const auto j = Json::parse(text);
    requireKeys(j, {"point", "tolerance", "uuids", "coordinates", "volume", "target_scale"}, "proximity query");
    SurfaceProximityQuery q;
    read(j, "point", q.point); read(j, "tolerance", q.tolerance); read(j, "uuids", q.uuids);
    read(j, "coordinates", q.coordinates); read(j, "volume", q.volume); read(j, "target_scale", q.targetScale);
    return q;
}

std::string proximityQueryResultToJson(const StoreStatus& status, const std::vector<SurfaceProximityHit>& hits)
{
    auto array = Json::array();
    for (const auto& h : hits)
        array.push_back({{"object", h.objectUuid}, {"revision", h.revisionUuid}, {"grid", h.grid}, {"point", h.point},
                         {"distance", h.distance}});
    return envelope(status, "hits", std::move(array)).dump();
}

std::vector<SurfaceProximityHit> proximityQueryResultFromJson(const std::string& text, StoreStatus* asOf)
{
    std::vector<SurfaceProximityHit> result;
    for (const auto& h : unwrap(text, asOf, "hits"))
        result.push_back({h.at("object").get<std::string>(), h.at("revision").get<std::string>(),
                          h.at("grid").get<std::array<double, 2>>(), h.at("point").get<std::array<double, 3>>(),
                          h.at("distance").get<double>()});
    return result;
}

} // namespace vc::objects
