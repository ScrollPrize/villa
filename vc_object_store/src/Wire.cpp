// JSON forms of conflicts, status, feed pages and historical reads, shared by
// the hosted service and its clients.
#include "vc_object_store/objects/TransactionJson.hpp"
#include "internal/RecordJson.hpp"

namespace vc::objects {
using namespace detail;
using Json = nlohmann::json;

namespace {
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
    const auto j = Json::parse(text);
    return {j.at("store").get<std::string>(), j.at("generation").get<std::string>(),
            j.at("commit_sequence").get<int64_t>(), j.at("feed_start").get<int64_t>()};
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

std::string objectRevisionToJson(const ObjectRevision& r)
{
    return Json{{"uuid", r.uuid}, {"object", r.objectUuid}, {"parent", optionalText(r.parentUuid)},
                {"geometry_revision", optionalText(r.geometryRevision)}, {"sequence", r.sequence},
                {"metadata_complete", r.metadataComplete}, {"published_at_us", optionalInt(r.publishedAtUs)},
                {"deleted", r.deleted}, {"version", r.version}, {"generation", r.generation},
                {"commit_sequence", optionalInt(r.commitSequence)}, {"record", objectToJson(r.record)}}
        .dump();
}

ObjectRevision objectRevisionFromJson(const std::string& text)
{
    const auto j = Json::parse(text);
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
    r.commitSequence = intFrom(j.at("commit_sequence"));
    const auto& record = j.at("record");
    r.record = objectFromJson(record);
    commonFromJson(record, r.record);
    return r;
}

std::string geometryRevisionToJson(const GeometryRevision& g)
{
    auto arrays = Json::array();
    for (const auto& a : g.arrays)
        arrays.push_back({{"name", a.name}, {"dtype", a.dtype}, {"shape", a.shape}, {"byte_length", a.byteLength}});
    return Json{{"uuid", g.uuid}, {"object", g.objectUuid}, {"parent", optionalText(g.parentUuid)},
                {"content_hash", g.contentHash}, {"kind", g.kind}, {"coordinate_system", g.coordinateSystem},
                {"attributes", g.attributes}, {"format_version", g.formatVersion},
                {"object_generation", g.objectGeneration}, {"created_by", g.createdBy}, {"created_at", g.createdAt},
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
    g.objectGeneration = j.at("object_generation").get<int64_t>();
    g.createdBy = j.at("created_by").get<std::string>();
    g.createdAt = j.at("created_at").get<std::string>();
    g.createdHost = j.at("created_host").get<std::string>();
    for (const auto& a : j.at("arrays"))
        g.arrays.push_back({a.at("name").get<std::string>(), a.at("dtype").get<std::string>(),
                            a.at("shape").get<std::vector<int64_t>>(), a.at("byte_length").get<int64_t>()});
    g.gridOrigin = j.at("grid_origin").get<std::array<int64_t, 2>>();
    return g;
}

} // namespace vc::objects
