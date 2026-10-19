#include "vc/core/graph/GraphDb.hpp"
#include "vc/core/objects/ObjectStore.hpp"
#include "GraphDbInternal.hpp"
#include "ObjectFiberInput.hpp"
namespace vc::graph {
GraphDb::GraphDb(std::unique_ptr<Impl> impl) : RecordDb(std::move(impl)) {}
GraphDb GraphDb::create(const std::filesystem::path& p, const OpenOptions& o) { return GraphDb(openImpl(p,o,true,Kind::Graph)); }
GraphDb GraphDb::open(const std::filesystem::path& p, const OpenOptions& o) { return GraphDb(openImpl(p,o,false,Kind::Graph)); }
void GraphDb::setApproval(WriteTxn& txn, const std::string& uuid, const std::string& state) {
    (void)connection(txn);
    if (state == "approved" && edgeByUuid(txn, uuid) && edgeGeometryState(txn, uuid).stale)
        throw std::runtime_error("graph db: revalidate edge geometry before approving a stale edge");
    RecordDb::setApproval(txn, uuid, state);
}
std::string GraphDb::addEdge(WriteTxn& txn, const EdgeRecord& record)
{
    return addEdgeImpl(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname, record, false);
}

void GraphDb::updateEdge(WriteTxn& txn, const EdgeRecord& record)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto existing = edgeByUuid(txn, record.uuid);
    if (!existing) throw std::runtime_error("graph db: edge not found: " + record.uuid);
    auto rec = record;
    rec.aUuid = formatUuid(parseUuid(rec.aUuid));
    rec.bUuid = formatUuid(parseUuid(rec.bUuid));
    if (rec.aUuid == existing->bUuid && rec.bUuid == existing->aUuid) {
        std::swap(rec.aUuid, rec.bUuid); std::swap(rec.aAnchor, rec.bAnchor);
    }
    if (rec.kind != existing->kind || rec.aUuid != existing->aUuid || rec.bUuid != existing->bUuid)
        throw std::runtime_error("graph db: edge kind and endpoints cannot change");
    auto sameAnchor = [](const Anchor& a, const Anchor& b) {
        return a.index == b.index && a.position == b.position && a.direction == b.direction;
    };
    std::sort(rec.tags.begin(), rec.tags.end());
    rec.tags.erase(std::unique(rec.tags.begin(), rec.tags.end()), rec.tags.end());
    if (sameAnchor(rec.aAnchor, existing->aAnchor) && sameAnchor(rec.bAnchor, existing->bAnchor) &&
        rec.tags == existing->tags && nlohmann::json::parse(rec.meta) == nlohmann::json::parse(existing->meta)) return;
    const auto id = *edgeIdByUuid(db, parseUuid(rec.uuid));
    exec(db, "SAVEPOINT edge_refresh;");
    try {
        Statement s(db, "UPDATE edges SET a_anchor_index=?,a_anchor_x=?,a_anchor_y=?,a_anchor_z=?,a_dir_x=?,a_dir_y=?,a_dir_z=?,"
            "b_anchor_index=?,b_anchor_x=?,b_anchor_y=?,b_anchor_z=?,b_dir_x=?,b_dir_y=?,b_dir_z=?,meta=? WHERE id=?");
        bindAnchor(s, 1, rec.aAnchor); bindAnchor(s, 8, rec.bAnchor);
        s.bind(15, rec.meta); s.bind(16, id); s.stepDone();
        writeTags(db, "edge_tags", "edge_id", id, rec.tags);
        stampContentChange(db, author(), hostname(), "edges", id, parseUuid(rec.uuid), "update_edge");
        exec(db, "RELEASE edge_refresh;");
    } catch (...) {
        exec(db, "ROLLBACK TO edge_refresh; RELEASE edge_refresh;");
        throw;
    }
}

std::optional<EdgeRecord> GraphDb::edgeByUuid(const Txn& txn, const std::string& uuid) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE uuid=?");
    s.bindBlob(1, parseUuid(uuid));
    if (!s.row()) return std::nullopt;
    return readEdgeRow(db, s);
}

void GraphDb::removeEdge(WriteTxn& txn, const std::string& uuid)
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    const auto bytes = parseUuid(uuid);
    auto id = edgeIdByUuid(db, bytes);
    if (!id) throw std::runtime_error("graph db: edge not found: " + uuid);
    Statement s(db, "DELETE FROM edges WHERE id=?");
    s.bind(1, *id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "remove_edge");
}

std::vector<EdgeRecord> GraphDb::edgesOf(const Txn& txn, const std::string& objectUuid) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    auto id = objectIdByUuid(db, parseUuid(objectUuid));
    if (!id) throw std::runtime_error("graph db: object not found: " + objectUuid);
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE a_id=? OR b_id=? ORDER BY id");
    s.bind(1, *id);
    s.bind(2, *id);
    std::vector<EdgeRecord> out;
    while (s.row()) out.push_back(readEdgeRow(db, s));
    return out;
}

std::vector<EdgeRecord> GraphDb::edgesByKind(const Txn& txn, const std::string& edgeKind) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE kind=? ORDER BY id");
    s.bind(1, edgeKind);
    std::vector<EdgeRecord> out;
    while (s.row()) out.push_back(readEdgeRow(db, s));
    return out;
}

std::vector<std::string> GraphDb::neighbours(const Txn& txn, const std::string& objectUuid,
                                             const std::optional<std::string>& edgeKind,
                                             const std::optional<std::string>& neighbourKind) const
{
    std::vector<std::string> out;
    std::set<std::string> seen;
    const auto canonical = formatUuid(parseUuid(objectUuid));
    for (const auto& e : edgesOf(txn, canonical)) {
        if (edgeKind && e.kind != *edgeKind) continue;
        const std::string& other = (e.aUuid == canonical) ? e.bUuid : e.aUuid;
        if (!seen.insert(other).second) continue;
        if (neighbourKind) {
            auto obj = objectByUuid(txn, other);
            if (!obj || obj->kind != *neighbourKind) continue;
        }
        out.push_back(other);
    }
    return out;
}

namespace {

struct ProgressGuard {
    sqlite3* db;
    ~ProgressGuard() { sqlite3_progress_handler(db, 0, nullptr, nullptr); }
};

std::vector<std::string> traverse(sqlite3* db, const std::string& startUuid, int maxDepth,
                                  const std::vector<std::string>& edgeKindFilter, const std::atomic_bool* cancel)
{
    auto checkCancelled = [&] {
        if (cancel && cancel->load())
            throw std::runtime_error("graph db: traversal cancelled");
    };
    checkCancelled();
    ProgressGuard guard{db};
    if (cancel)
        sqlite3_progress_handler(
            db, 1000, [](void* p) { return static_cast<const std::atomic_bool*>(p)->load() ? 1 : 0; },
            const_cast<std::atomic_bool*>(cancel));
    const auto startBytes = parseUuid(startUuid);
    if (!objectIdByUuid(db, startBytes))
        throw std::runtime_error("graph db: object not found: " + startUuid);
    // Probe the endpoint indexes only for nodes reached by this traversal.
    // ORDER BY id preserves the previous breadth-first visitation order.
    std::string sql = "SELECT id,CASE WHEN a_id=? THEN b_id ELSE a_id END FROM edges WHERE (a_id=? OR b_id=?)";
    if (!edgeKindFilter.empty()) {
        sql += " AND kind IN (";
        for (size_t i = 0; i < edgeKindFilter.size(); ++i) sql += (i ? ",?" : "?");
        sql += ")";
    }
    sql += " ORDER BY id";
    Statement adjacent(db, sql);
    const int64_t startId = *objectIdByUuid(db, startBytes);
    std::vector<std::string> out;
    std::unordered_set<int64_t> visited{startId};
    std::queue<std::pair<int64_t, int>> queue;
    queue.emplace(startId, 0);
    while (!queue.empty()) {
        checkCancelled();
        auto [id, depth] = queue.front();
        queue.pop();
        out.push_back(uuidTextByObjectId(db, id));
        if (depth >= maxDepth) continue;
        adjacent.reset();
        adjacent.bind(1, id); adjacent.bind(2, id); adjacent.bind(3, id);
        for (size_t i = 0; i < edgeKindFilter.size(); ++i) adjacent.bind(int(i) + 4, edgeKindFilter[i]);
        try {
            while (adjacent.row()) {
                checkCancelled();
                const auto next = adjacent.i64(1);
                if (visited.insert(next).second) queue.emplace(next, depth + 1);
            }
        } catch (const std::runtime_error&) {
            checkCancelled(); // translate SQLite's interrupt into the cancellation contract
            throw;
        }
    }
    return out;
}

}  // namespace

std::vector<std::string> GraphDb::componentOf(const Txn& txn, const std::string& objectUuid,
                                              const std::vector<std::string>& edgeKindFilter,
                                              const std::atomic_bool* cancel) const
{
    return traverse(static_cast<sqlite3*>(connection(txn)), objectUuid, INT_MAX, edgeKindFilter, cancel);
}

std::vector<std::string> GraphDb::reachableWithin(const Txn& txn, const std::string& objectUuid, int maxDepth,
                                                  const std::vector<std::string>& edgeKindFilter,
                                                  const std::atomic_bool* cancel) const
{
    if (maxDepth < 0) throw std::runtime_error("graph db: maxDepth must be >= 0");
    return traverse(static_cast<sqlite3*>(connection(txn)), objectUuid, maxDepth, edgeKindFilter, cancel);
}

namespace {

nlohmann::json anchorToJson(const Anchor& a)
{
    nlohmann::json j = nlohmann::json::object();
    if (a.index) j["index"] = *a.index;
    if (a.position) j["position"] = {(*a.position)[0], (*a.position)[1], (*a.position)[2]};
    if (a.direction) j["direction"] = {(*a.direction)[0], (*a.direction)[1], (*a.direction)[2]};
    return j;
}

Anchor anchorFromJson(const nlohmann::json& j)
{
    Anchor a;
    if (!j.is_object()) return a;
    if (j.contains("index") && j.at("index").is_number_integer()) a.index = j.at("index").get<int64_t>();
    for (const char* key : {"position", "direction"}) {
        if (j.contains(key) && j.at(key).is_array() && j.at(key).size() == 3) {
            std::array<double, 3> v{};
            for (int i = 0; i < 3; ++i) v[size_t(i)] = j.at(key).at(size_t(i)).get<double>();
            if (std::string(key) == "position") a.position = v; else a.direction = v;
        }
    }
    return a;
}

std::optional<std::array<double, 6>> bboxFromJson(const nlohmann::json& j)
{
    if (!j.is_array() || j.size() != 6) return std::nullopt;
    std::array<double, 6> box{};
    for (int i = 0; i < 6; ++i) box[size_t(i)] = j.at(size_t(i)).get<double>();
    return box;
}

nlohmann::json bboxToJson(const std::optional<std::array<double, 6>>& box)
{
    if (!box) return nullptr;
    return {(*box)[0], (*box)[1], (*box)[2], (*box)[3], (*box)[4], (*box)[5]};
}

std::array<double, 2> scaleFromJson(const nlohmann::json& value)
{
    std::array<double, 2> scale{};
    if (value.is_number()) {
        scale.fill(value.get<double>());  // v1 scalar scales were isotropic
    } else if (value.is_array() && value.size() == 2) {
        scale = {value.at(0).get<double>(), value.at(1).get<double>()};
    } else {
        throw std::runtime_error("graph db: patch scale must be a number or a two-element array");
    }
    for (double axis : scale)
        if (!std::isfinite(axis) || axis < 0)
            throw std::runtime_error("graph db: patch scale must be finite and nonnegative");
    return scale;
}

nlohmann::json commonToJson(const RecordCommon& r)
{
    nlohmann::json j{{"uuid", r.uuid}, {"kind", r.kind}, {"generation", r.generation},
        {"created_by", r.createdBy}, {"created_at", r.createdAt}, {"created_host", r.createdHost},
        {"changed_by", r.changedBy}, {"changed_at", r.changedAt}, {"changed_host", r.changedHost},
        {"tags", r.tags}, {"meta", nlohmann::json::parse(r.meta)}, {"approval_state", r.approvalState}};
    j["approved_by"] = r.approvedBy ? nlohmann::json(*r.approvedBy) : nlohmann::json(nullptr);
    j["approved_at"] = r.approvedAt ? nlohmann::json(*r.approvedAt) : nlohmann::json(nullptr);
    j["approved_host"] = r.approvedHost ? nlohmann::json(*r.approvedHost) : nlohmann::json(nullptr);
    j["approved_generation"] = r.approvedGeneration ? nlohmann::json(*r.approvedGeneration) : nlohmann::json(nullptr);
    return j;
}

void commonFromJson(const nlohmann::json& j, RecordCommon& r)
{
    r.generation = j.at("generation").get<int64_t>();
    if (r.generation < 1) throw std::runtime_error("graph db: snapshot generation must be positive");
    r.createdBy = j.at("created_by").get<std::string>();
    r.createdAt = j.at("created_at").get<std::string>();
    r.createdHost = j.at("created_host").get<std::string>();
    r.changedBy = j.at("changed_by").get<std::string>();
    r.changedAt = j.at("changed_at").get<std::string>();
    r.changedHost = j.at("changed_host").get<std::string>();
    auto optionalText = [&](const char* key) -> std::optional<std::string> {
        const auto& value = j.at(key);
        return value.is_null() ? std::nullopt : std::optional(value.get<std::string>());
    };
    r.approvedBy = optionalText("approved_by");
    r.approvedAt = optionalText("approved_at");
    r.approvedHost = optionalText("approved_host");
    if (!j.at("approved_generation").is_null()) r.approvedGeneration = j.at("approved_generation").get<int64_t>();
    if (r.approvalState != "pending" && r.approvalState != "approved" && r.approvalState != "rejected")
        throw std::runtime_error("graph db: invalid snapshot approval state");
    if (r.approvedGeneration && *r.approvedGeneration != r.generation)
        throw std::runtime_error("graph db: snapshot approval does not cover its generation");
    if (r.approvalState == "pending" && (r.approvedBy || r.approvedAt || r.approvedHost || r.approvedGeneration))
        throw std::runtime_error("graph db: pending snapshot has approval stamps");
}

nlohmann::json objectToJson(const ObjectRecord& r)
{
    nlohmann::json j = commonToJson(r);
    j["name"] = r.name;
    j["format_version"] = r.formatVersion;
    j["storage_kind"] = r.storageKind;
    j["storage_ref"] = r.storageRef;
    j["data_generation"] = r.dataGeneration;
    if (r.dataChecksum) j["data_checksum"] = *r.dataChecksum;
    if (r.fiber) {
        j["fiber"] = {{"family", r.fiber->family}, {"manual_family", r.fiber->manualFamily},
                      {"control_point_count", r.fiber->controlPointCount}, {"line_point_count", r.fiber->linePointCount},
                      {"length", r.fiber->length}, {"width", r.fiber->width},
                      {"width_gap_fraction", r.fiber->widthGapFraction}, {"optimization_mode", r.fiber->optimizationMode},
                      {"bbox", bboxToJson(r.fiber->bbox)}};
    }
    if (r.patch) {
        j["patch"] = {{"grid_width", r.patch->gridWidth}, {"grid_height", r.patch->gridHeight},
                      {"scale", r.patch->scale}, {"bbox", bboxToJson(r.patch->bbox)}};
    }
    if (r.track) {
        j["track"] = {{"dataset_ref", r.track->datasetRef}, {"external_key", r.track->externalKey},
                      {"family", r.track->family}, {"point_count", r.track->pointCount},
                      {"length", r.track->length}, {"bbox", bboxToJson(r.track->bbox)}};
    }
    return j;
}

ObjectRecord objectFromJson(const nlohmann::json& j)
{
    ObjectRecord r;
    r.uuid = j.value("uuid", "");
    r.kind = j.value("kind", "");
    r.name = j.value("name", "");
    r.formatVersion = j.value("format_version", int64_t(0));
    r.storageKind = j.value("storage_kind", "");
    r.storageRef = j.value("storage_ref", "");
    r.dataGeneration = j.value("data_generation", int64_t(0));
    if (j.contains("data_checksum") && j.at("data_checksum").is_string()) r.dataChecksum = j.at("data_checksum").get<std::string>();
    if (j.contains("tags")) r.tags = j.at("tags").get<std::vector<std::string>>();
    if (j.contains("meta")) r.meta = j.at("meta").dump();
    r.approvalState = j.value("approval_state", "pending");
    if (j.contains("fiber")) {
        const auto& f = j.at("fiber");
        FiberDetail d;
        d.family = f.value("family", "");
        d.manualFamily = f.value("manual_family", false);
        d.controlPointCount = f.value("control_point_count", int64_t(0));
        d.linePointCount = f.value("line_point_count", int64_t(0));
        d.length = f.value("length", 0.0);
        d.width = f.value("width", 0.0);
        d.widthGapFraction = f.value("width_gap_fraction", 0.2);
        d.optimizationMode = f.value("optimization_mode", "");
        if (f.contains("bbox")) d.bbox = bboxFromJson(f.at("bbox"));
        r.fiber = d;
    }
    if (j.contains("patch")) {
        const auto& p = j.at("patch");
        PatchDetail d;
        d.gridWidth = p.value("grid_width", int64_t(0));
        d.gridHeight = p.value("grid_height", int64_t(0));
        d.scale = scaleFromJson(p.value("scale", nlohmann::json(0.0)));
        if (p.contains("bbox")) d.bbox = bboxFromJson(p.at("bbox"));
        r.patch = d;
    }
    if (j.contains("track")) {
        const auto& t = j.at("track");
        TrackDetail d;
        d.datasetRef = t.value("dataset_ref", "");
        d.externalKey = t.value("external_key", "");
        d.family = t.value("family", "");
        d.pointCount = t.value("point_count", int64_t(0));
        d.length = t.value("length", 0.0);
        if (t.contains("bbox")) d.bbox = bboxFromJson(t.at("bbox"));
        r.track = d;
    }
    return r;
}

nlohmann::json edgeToJson(EdgeRecord e)
{
    // Stable across databases whose internal endpoint IDs differ.
    if (e.bUuid < e.aUuid) {
        std::swap(e.aUuid, e.bUuid);
        std::swap(e.aAnchor, e.bAnchor);
    }
    auto j = commonToJson(e);
    j["a"] = e.aUuid;
    j["b"] = e.bUuid;
    j["a_anchor"] = anchorToJson(e.aAnchor);
    j["b_anchor"] = anchorToJson(e.bAnchor);
    return j;
}

}  // namespace

std::string GraphDb::exportJson(const Txn& txn, const std::optional<std::string>& objectKind,
                                const std::optional<std::string>& tag) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    nlohmann::json out;
    out["format"] = "vc_graph_export";
    out["version"] = 2;
    out["objects"] = nlohmann::json::array();
    out["edges"] = nlohmann::json::array();
    std::unordered_set<std::string> included;
    const bool filtered = objectKind.has_value() || tag.has_value();
    std::vector<ObjectRecord> objects;
    if (tag) {
        objects = objectsByTag(txn, *tag);
    } else if (objectKind) {
        objects = objectsByKind(txn, *objectKind);
    } else {
        Statement s(db, std::string("SELECT ") + kObjectCols + " FROM objects ORDER BY id");
        while (s.row()) objects.push_back(readObjectRow(db, s));
    }
    if (tag && objectKind)
        std::erase_if(objects, [&](const ObjectRecord& o) { return o.kind != *objectKind; });
    for (const auto& o : objects) {
        if (selectedObject(txn, o.uuid))
            throw std::runtime_error("graph db: JSON export does not include object-store revision bindings; use a SQLite backup of the graph and retain its object stores");
        included.insert(o.uuid);
        out["objects"].push_back(objectToJson(o));
    }
    Statement edges(db, std::string("SELECT ") + kEdgeCols + " FROM edges ORDER BY id");
    while (edges.row()) {
        EdgeRecord e = readEdgeRow(db, edges);
        if (filtered && (!included.contains(e.aUuid) || !included.contains(e.bUuid))) continue;
        out["edges"].push_back(edgeToJson(e));
    }
    return out.dump(2);
}

void GraphDb::importJson(WriteTxn& txn, const std::string& jsonText)
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    const auto root = nlohmann::json::parse(jsonText);
    if (root.value("format", "") != "vc_graph_export")
        throw std::runtime_error("graph db: not a vc_graph_export document");
    const int version = root.value("version", 0);
    if (version != 2)
        throw std::runtime_error("graph db: unsupported export version");
    // A failed import must not leave half a snapshot, even if the caller
    // catches the error and commits other work in its enclosing transaction.
    exec(db, "SAVEPOINT graph_import;");
    try {
        std::set<std::string> seen;
        auto checkIdentity = [&](std::string& uuid) {
            uuid = formatUuid(parseUuid(uuid));
            if (!seen.insert(uuid).second)
                throw std::runtime_error("graph db: duplicate snapshot UUID: " + uuid);
        };
        for (const auto& j : root.at("objects")) {
            auto object = objectFromJson(j);
            checkIdentity(object.uuid);
            commonFromJson(j, object);
            if (edgeByUuid(txn, object.uuid))
                throw std::runtime_error("graph db: snapshot UUID belongs to an edge: " + object.uuid);
            if (auto existing = objectByUuid(txn, object.uuid)) {
                if (objectToJson(*existing) == objectToJson(object)) continue;
                throw std::runtime_error("graph db: snapshot conflicts with existing object: " + object.uuid);
            }
            upsertObjectImpl(db, impl_->options.author, impl_->hostname, object, true, true);
        }
        for (const auto& j : root.at("edges")) {
            EdgeRecord e;
            e.uuid = j.at("uuid").get<std::string>();
            checkIdentity(e.uuid);
            e.kind = j.at("kind").get<std::string>();
            e.aUuid = formatUuid(parseUuid(j.at("a").get<std::string>()));
            e.bUuid = formatUuid(parseUuid(j.at("b").get<std::string>()));
            if (j.contains("a_anchor")) e.aAnchor = anchorFromJson(j.at("a_anchor"));
            if (j.contains("b_anchor")) e.bAnchor = anchorFromJson(j.at("b_anchor"));
            if (j.contains("tags")) e.tags = j.at("tags").get<std::vector<std::string>>();
            if (j.contains("meta")) e.meta = j.at("meta").dump();
            e.approvalState = j.value("approval_state", "pending");
            commonFromJson(j, e);
            if (objectByUuid(txn, e.uuid))
                throw std::runtime_error("graph db: snapshot UUID belongs to an object: " + e.uuid);
            if (auto existing = edgeByUuid(txn, e.uuid)) {
                if (edgeToJson(*existing) == edgeToJson(e)) continue;
                throw std::runtime_error("graph db: snapshot conflicts with existing edge: " + e.uuid);
            }
            addEdgeImpl(db, impl_->options.author, impl_->hostname, e, true, true);
        }
        exec(db, "RELEASE graph_import;");
    } catch (...) {
        exec(db, "ROLLBACK TO graph_import; RELEASE graph_import;");
        throw;
    }
}

namespace {

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read file: " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string fnvHashHex(const std::string& text)
{
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : text) h = (h ^ c) * 1099511628211ULL;
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

std::string relativeToDb(GraphDb& db, const std::filesystem::path& file)
{
    std::error_code ec;
    auto rel = std::filesystem::relative(file, db.path().parent_path(), ec);
    if (ec) return file.filename().string();
    return rel.string();
}

struct ParsedBranch {
    int localCp = -1;
    std::string targetFile;
    int targetCp = -1;
    bool pending = false;
    bool adjacent = false;
    Anchor local, target;
};

Anchor branchAnchor(const nlohmann::json& j, const char* indexKey, const char* posKey, const char* dirKey)
{
    Anchor a;
    if (j.contains(indexKey) && j.at(indexKey).is_number_integer()) a.index = j.at(indexKey).get<int64_t>();
    for (int i = 0; i < 2; ++i) {
        const char* key = i == 0 ? posKey : dirKey;
        if (j.contains(key) && j.at(key).is_array() && j.at(key).size() == 3) {
            std::array<double, 3> v{};
            for (int k = 0; k < 3; ++k) v[size_t(k)] = j.at(key).at(size_t(k)).get<double>();
            if (i == 0) a.position = v; else a.direction = v;
        }
    }
    return a;
}

}  // namespace

ImportReport importFibers(GraphDb& db, WriteTxn& txn, const std::filesystem::path& fiberDir,
    const util::ProgressCallback& progress)
{
    ImportReport report;
    util::reportProgress(progress, "scanning fiber directory");
    struct Parsed {
        std::filesystem::path path;
        std::string fileName;
        ObjectRecord object;
        std::vector<ParsedBranch> branches;
        std::string checksum;
    };
    std::vector<Parsed> parsed;
    std::unordered_map<std::string, size_t> byFileName;

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(fiberDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    size_t fileIndex = 0;
    for (const auto& file : files) {
        util::reportProgress(progress, "reading fibers", fileIndex++, files.size(), file.filename().string());
        nlohmann::json root;
        std::string text;
        try {
            text = readFile(file);
            root = nlohmann::json::parse(text);
        } catch (const std::exception& e) {
            report.problems.push_back(file.filename().string() + ": unreadable/unparseable JSON: " + e.what());
            continue;
        }
        if (!root.is_object() || !root.contains("uuid") ||
            !(root.contains("control_points") || root.contains("line_points")))
            continue;  // not a vc3d_fiber file; other JSON in the dir is not our business
        const std::string uuid = root.value("uuid", "");
        if (!util::isUuid(uuid)) {
            report.problems.push_back(file.filename().string() + ": missing/invalid uuid, skipped");
            continue;
        }
        if (db.selectedObject(txn, uuid)) {
            report.problems.push_back(file.filename().string() +
                ": geometry is database-owned; use ingestGeometrySource to publish a revision (legacy branch import skipped)");
            continue;
        }
        Parsed p;
        p.path = file;
        p.fileName = file.filename().string();
        p.checksum = fnvHashHex(text);
        ObjectRecord& o = p.object;
        o.uuid = formatUuid(parseUuid(uuid));
        o.kind = "fiber";
        o.name = file.stem().string();
        o.formatVersion = root.value("version", int64_t(1));
        o.storageKind = "file";
        o.storageRef = relativeToDb(db, file);
        o.dataGeneration = 1;
        o.dataChecksum = p.checksum;
        if (root.contains("tags") && root.at("tags").is_array())
            for (const auto& t : root.at("tags"))
                if (t.is_string()) o.tags.push_back(t.get<std::string>());
        FiberDetail f;
        try {
            auto points = objects::detail::fiberPoints(root, "line_points");
            if (points.empty()) points = objects::detail::fiberPoints(root, "control_points");
            f.length = objects::detail::fiberPolylineLength(points);
        } catch (const std::exception& error) {
            report.problems.push_back(p.fileName + ": invalid fiber geometry: " + error.what());
            continue;
        }
        if (root.contains("control_points") && root.at("control_points").is_array())
            f.controlPointCount = int64_t(root.at("control_points").size());
        if (root.contains("line_points") && root.at("line_points").is_array())
            f.linePointCount = int64_t(root.at("line_points").size());
        f.width = root.value("width", 0.0);
        f.widthGapFraction = root.value("width_gap_fraction", 0.2);
        f.optimizationMode = root.value("optimization_mode", "");
        if (root.contains("hv_classification") && root.at("hv_classification").is_object()) {
            const auto& hv = root.at("hv_classification");
            const std::string manual = hv.value("manual_tag", "");
            const std::string automatic = hv.value("automatic_tag", "");
            f.manualFamily = !manual.empty();
            f.family = f.manualFamily ? manual : automatic;
        }
        o.fiber = f;
        for (const auto* arrayKey : {"branches", "adjacent_branches"}) {
            if (!root.contains(arrayKey) || !root.at(arrayKey).is_array()) continue;
            for (const auto& bj : root.at(arrayKey)) {
                if (!bj.is_object()) continue;
                ParsedBranch b;
                b.adjacent = std::string(arrayKey) == "adjacent_branches";
                b.localCp = bj.value("control_point_index", -1);
                b.targetCp = bj.value("branch_control_point_index", -1);
                b.targetFile = std::filesystem::path(bj.value("branch_file", "")).filename().string();
                b.pending = bj.value("pending", false);
                b.local = branchAnchor(bj, "control_point_index", "control_point_position", "control_point_direction");
                b.target = branchAnchor(bj, "branch_control_point_index", "branch_control_point_position", "branch_control_point_direction");
                p.branches.push_back(std::move(b));
            }
        }
        byFileName[p.fileName] = parsed.size();
        parsed.push_back(std::move(p));
    }

    util::reportProgress(progress, "reading fibers", files.size(), files.size());

    // Objects first (edges need their endpoints to exist). Unchanged
    // files (same uuid + content hash) are skipped: re-import is a no-op.
    size_t objectIndex = 0;
    for (const auto& p : parsed) {
        util::reportProgress(progress, "registering fibers", objectIndex++, parsed.size());
        auto refreshed = p.object;
        if (auto existing = db.objectByUuid(txn, p.object.uuid)) {
            if (existing->dataChecksum && *existing->dataChecksum == p.checksum) {
                db.setFiberLength(txn, p.object.uuid, p.object.fiber->length);
                continue;
            }
            // Files seed annotations once. Subsequent imports own only the
            // source descriptor and measurements, not graph names/tags/meta.
            refreshed.name = existing->name;
            refreshed.tags = existing->tags;
            refreshed.meta = existing->meta;
            refreshed.dataGeneration = existing->dataGeneration + 1;
            for (const auto& edge : db.edgesOf(txn, existing->uuid))
                stampContentChange(RecordDbImplAccess::conn(txn), db.author(), db.hostname(), "edges",
                    *edgeIdByUuid(RecordDbImplAccess::conn(txn), parseUuid(edge.uuid)), parseUuid(edge.uuid), "geometry_invalidated");
        }
        db.upsertObject(txn, refreshed);
        ++report.objectsImported;
    }

    util::reportProgress(progress, "registering fibers", parsed.size(), parsed.size());

    auto hasReciprocal = [&](const Parsed& owner, const ParsedBranch& b) {
        auto it = byFileName.find(b.targetFile);
        if (it == byFileName.end()) return false;
        for (const auto& other : parsed[it->second].branches)
            if (other.targetFile == owner.fileName && other.localCp == b.targetCp && other.targetCp == b.localCp &&
                other.adjacent == b.adjacent)
                return true;
        return false;
    };

    std::set<std::string> seenPairs, retainedBranches, importedObjects;
    for (const auto& p : parsed) importedObjects.insert(p.object.uuid);
    size_t branchIndex = 0;
    for (const auto& p : parsed) {
        util::reportProgress(progress, "importing fiber branches", branchIndex++, parsed.size());
        for (const auto& b : p.branches) {
            auto it = byFileName.find(b.targetFile);
            if (it == byFileName.end()) {
                report.problems.push_back(p.fileName + ": branch target not in imported set: " + b.targetFile);
                continue;
            }
            const Parsed& target = parsed[it->second];
            std::string keyA = p.fileName + "#" + std::to_string(b.localCp);
            std::string keyB = b.targetFile + "#" + std::to_string(b.targetCp);
            const std::string pairKey = keyA < keyB ? keyA + "|" + keyB : keyB + "|" + keyA;
            const bool reciprocal = hasReciprocal(p, b);
            if (!reciprocal)
                report.problems.push_back(p.fileName + ": one-sided branch to " + b.targetFile +
                                          " (edge imported from this side only)");
            if (!seenPairs.insert(pairKey).second)
                continue;
            std::optional<EdgeRecord> existing;
            for (const auto& edge : db.edgesOf(txn, p.object.uuid)) {
                const bool forward = edge.aUuid == p.object.uuid;
                if ((forward ? edge.bUuid : edge.aUuid) == target.object.uuid && edge.kind == "fiber_branch" &&
                    (forward ? edge.aAnchor : edge.bAnchor).index == b.local.index &&
                    (forward ? edge.bAnchor : edge.aAnchor).index == b.target.index) {
                    existing = edge;
                    break;
                }
            }
            if (existing) {
                retainedBranches.insert(existing->uuid);
                auto meta = nlohmann::json::parse(existing->meta);
                // Unmarked legacy/manual edges have ambiguous ownership. Do
                // not silently claim or delete them based on external files.
                if (meta.value("source_import", "") != "fiber-json") continue;
                auto refreshed = *existing;
                refreshed.aUuid = p.object.uuid; refreshed.bUuid = target.object.uuid;
                refreshed.aAnchor = b.local; refreshed.bAnchor = b.target;
                meta["adjacent"] = b.adjacent;
                refreshed.meta = meta.dump();
                db.updateEdge(txn, refreshed);
                continue;
            }
            EdgeRecord edge;
            edge.kind = "fiber_branch";
            edge.aUuid = p.object.uuid; edge.bUuid = target.object.uuid;
            edge.aAnchor = b.local; edge.bAnchor = b.target;
            edge.meta = nlohmann::json{{"adjacent", b.adjacent}, {"source_import", "fiber-json"}}.dump();
            edge.approvalState = b.pending ? "pending" : "approved";
            retainedBranches.insert(addEdgeImpl(RecordDbImplAccess::conn(txn), db.author(), db.hostname(), edge, true));
            ++report.edgesImported;
        }
    }
    // Only reconcile this importer's edges when both files were parsed.
    // Missing, unreadable, or omitted files cannot establish branch removal.
    for (const auto& edge : db.edgesByKind(txn, "fiber_branch")) {
        if (!importedObjects.contains(edge.aUuid) || !importedObjects.contains(edge.bUuid) || retainedBranches.contains(edge.uuid)) continue;
        const auto meta = nlohmann::json::parse(edge.meta);
        if (meta.value("source_import", "") == "fiber-json") db.removeEdge(txn, edge.uuid);
    }
    util::reportProgress(progress, "importing fiber branches", parsed.size(), parsed.size());
    return report;
}

ImportReport importPatches(GraphDb& db, WriteTxn& txn, const std::filesystem::path& segmentsDir,
    const util::ProgressCallback& progress)
{
    ImportReport report;
    util::reportProgress(progress, "scanning patch directory");
    std::vector<std::filesystem::path> dirs;
    for (const auto& entry : std::filesystem::directory_iterator(segmentsDir))
        if (entry.is_directory()) dirs.push_back(entry.path());
    std::sort(dirs.begin(), dirs.end());
    size_t index = 0;
    for (const auto& dir : dirs) {
        util::reportProgress(progress, "registering patches", index++, dirs.size(), dir.filename().string());
        const auto metaPath = dir / "meta.json";
        if (!std::filesystem::exists(metaPath)) continue;
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(readFile(metaPath));
        } catch (const std::exception& e) {
            report.problems.push_back(dir.filename().string() + ": unparseable meta.json: " + e.what());
            continue;
        }
        const std::string uuid = root.value("uuid", "");
        if (!util::isUuid(uuid)) {
            report.problems.push_back(dir.filename().string() + ": meta.json has no valid uuid, skipped");
            continue;
        }
        if (db.objectByUuid(txn, uuid)) continue;  // already registered
        ObjectRecord o;
        o.uuid = uuid;
        o.kind = "patch";
        o.name = dir.filename().string();
        o.storageKind = "file";
        o.storageRef = relativeToDb(db, dir);
        o.dataGeneration = 1;
        PatchDetail p;
        try {
            p.gridWidth = root.value("grid_width", int64_t(0));
            p.gridHeight = root.value("grid_height", int64_t(0));
            p.scale = scaleFromJson(root.value("scale", nlohmann::json(0.0)));
            if (root.contains("bbox")) {
                const auto& box = root.at("bbox");
                if (box.is_array() && box.size() == 2 && box.at(0).is_array() && box.at(1).is_array() &&
                    box.at(0).size() == 3 && box.at(1).size() == 3) {
                    p.bbox = std::array<double, 6>{box[0][0].get<double>(), box[0][1].get<double>(), box[0][2].get<double>(),
                        box[1][0].get<double>(), box[1][1].get<double>(), box[1][2].get<double>()};
                } else {
                    p.bbox = bboxFromJson(box);
                    if (!p.bbox) throw std::runtime_error("invalid patch bbox");
                }
            }
        } catch (const std::exception& e) {
            report.problems.push_back(dir.filename().string() + ": invalid patch metadata: " + e.what());
            continue;
        }
        o.patch = p;
        db.upsertObject(txn, o);
        ++report.objectsImported;
    }
    util::reportProgress(progress, "registering patches", dirs.size(), dirs.size());
    return report;
}

ImportReport importTrackList(GraphDb& db, WriteTxn& txn, const std::filesystem::path& trackListJson,
    const util::ProgressCallback& progress)
{
    ImportReport report;
    util::reportProgress(progress, "reading track list");
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(readFile(trackListJson));
    } catch (const std::exception& e) {
        throw std::runtime_error("graph db: cannot parse track list " + trackListJson.string() + ": " + e.what());
    }
    if (!root.is_array())
        throw std::runtime_error("graph db: track list must be a JSON array: " + trackListJson.string());

    // Index already-registered tracks by identity so re-registration
    // reuses the minted UUID instead of colliding with the UNIQUE index.
    std::map<std::pair<std::string, std::string>, ObjectRecord> registered;
    for (auto& existing : db.objectsByKind(txn, "track"))
        if (existing.track)
            registered[{existing.track->datasetRef, existing.track->externalKey}] = existing;

    for (size_t i = 0; i < root.size(); ++i) {
        util::reportProgress(progress, "registering tracks", i, root.size());
        const auto& entry = root.at(i);
        const std::string label = "entry " + std::to_string(i);
        if (!entry.is_object()) {
            report.problems.push_back(label + ": not an object, skipped");
            continue;
        }
        const std::string datasetRef = entry.value("dataset_ref", "");
        const std::string externalKey = entry.value("external_key", "");
        if (datasetRef.empty() || externalKey.empty()) {
            report.problems.push_back(label + ": missing dataset_ref/external_key, skipped");
            continue;
        }
        ObjectRecord o;
        o.kind = "track";
        o.name = entry.value("name", externalKey);
        o.storageKind = "dataset";
        o.storageRef = datasetRef;
        o.dataGeneration = 1;
        if (entry.contains("tags") && entry.at("tags").is_array())
            for (const auto& t : entry.at("tags"))
                if (t.is_string()) o.tags.push_back(t.get<std::string>());
        if (entry.contains("meta")) o.meta = entry.at("meta").dump();
        TrackDetail t;
        t.datasetRef = datasetRef;
        t.externalKey = externalKey;
        t.family = entry.value("family", "");
        t.pointCount = entry.value("point_count", int64_t(0));
        t.length = entry.value("length", 0.0);
        if (entry.contains("bbox")) t.bbox = bboxFromJson(entry.at("bbox"));
        o.track = t;

        auto it = registered.find({datasetRef, externalKey});
        if (it != registered.end()) {
            const ObjectRecord& cur = it->second;
            o.name = cur.name; o.tags = cur.tags; o.meta = cur.meta;
            if (db.selectedObject(txn, cur.uuid)) {
                report.problems.push_back(label + ": geometry is database-owned; use ingestGeometrySource to publish a revision");
                continue;
            }
            const bool unchanged = cur.track &&
                                   cur.track->family == t.family && cur.track->pointCount == t.pointCount &&
                                   cur.track->length == t.length && cur.track->bbox == t.bbox;
            if (unchanged)
                continue;  // idempotent re-registration
            o.uuid = cur.uuid;
            db.upsertObject(txn, o);
            ++report.objectsImported;
        } else {
            const std::string uuid = db.upsertObject(txn, o);
            o.uuid = uuid;
            registered[{datasetRef, externalKey}] = o;
            ++report.objectsImported;
        }
    }
    util::reportProgress(progress, "registering tracks", root.size(), root.size());
    return report;
}

EdgeGeometryState GraphDb::edgeGeometryState(const Txn& txn, const std::string& edgeUuid) const
{
    (void)connection(txn);
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto edge = edgeByUuid(txn, edgeUuid);
    if (!edge) throw std::invalid_argument("geometry: edge not found");
    Statement binding(db, "SELECT b.a_revision,b.b_revision FROM edge_geometry_bindings b JOIN edges e ON e.id=b.edge_id WHERE e.uuid=?");
    binding.bindBlob(1, parseUuid(edgeUuid));
    EdgeGeometryState result;
    if (binding.row()) {
        if (!binding.isNull(0)) result.aRevision = formatUuid(binding.blob16(0));
        if (!binding.isNull(1)) result.bRevision = formatUuid(binding.blob16(1));
    }
    result.stale = result.aRevision != geometryHead(db, edge->aUuid) || result.bRevision != geometryHead(db, edge->bUuid);
    return result;
}

void GraphDb::revalidateEdgeGeometry(WriteTxn& txn, const std::string& edgeUuid)
{
    const auto state = edgeGeometryState(txn, edgeUuid);
    if (!state.stale) return;
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto edge = *edgeByUuid(txn, edgeUuid);
    const auto id = *edgeIdByUuid(db, parseUuid(edgeUuid));
    exec(db, "SAVEPOINT geometry_revalidate;");
    try {
        Statement s(db, "INSERT INTO edge_geometry_bindings VALUES(?,?,?) ON CONFLICT(edge_id) DO UPDATE SET a_revision=excluded.a_revision,b_revision=excluded.b_revision");
        s.bind(1, id);
        const auto a = geometryHead(db, edge.aUuid), b = geometryHead(db, edge.bUuid);
        if (a) s.bindBlob(2, parseUuid(*a)); else s.bindNull(2);
        if (b) s.bindBlob(3, parseUuid(*b)); else s.bindNull(3);
        s.stepDone();
        stampContentChange(db, author(), hostname(), "edges", id, parseUuid(edgeUuid), "geometry_revalidated");
        exec(db, "RELEASE geometry_revalidate;");
    } catch (...) {
        exec(db, "ROLLBACK TO geometry_revalidate; RELEASE geometry_revalidate;");
        throw;
    }
}


} // namespace vc::graph
