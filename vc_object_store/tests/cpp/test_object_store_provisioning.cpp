#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include <sqlite3.h>
#include <cstring>

using namespace vc::objects;

int64_t scalar(ObjectStore& store, const Txn& txn, const char* sql)
{
    sqlite3_stmt* s = nullptr;
    auto* db = ObjectStoreImplAccess::conn(txn);
    if (sqlite3_prepare_v2(db, sql, -1, &s, nullptr) != SQLITE_OK) throw std::runtime_error(sqlite3_errmsg(db));
    if (sqlite3_step(s) != SQLITE_ROW) { sqlite3_finalize(s); throw std::runtime_error("missing scalar"); }
    const auto value = sqlite3_column_int64(s, 0);
    sqlite3_finalize(s); return value;
}

void directStore(bool spatial)
{
    TemporaryDirectory dir;
    OpenOptions options; options.author = "test"; options.history = HistoryMode::Off;
    options.spatialIndex = spatial;
    options.sqliteCacheBytes = 128 * 1024 * 1024;
    const auto path = dir.path / "direct.vcobj";
    auto store = ObjectStore::create(path, options);
    DatabaseIdentity physical;
    const auto a = util::newUuid(), b = util::newUuid();
    auto data = vc::geometry::encode(vc::geometry::Fiber{{}, {{1,2,3},{4,5,6}}, {}});
    std::string original, other, edgeId;
    {
        auto t = store.read();
        require(scalar(store,t,"PRAGMA cache_size")==-131072,"configured SQLite reader page cache");
        require(store.provisioning(t).generation == 1, "initial generation");
        physical = store.databaseIdentity(t);
        require(physical.uuid != store.uuid(t) && physical.schemaVersion == 18, "physical database and logical store identities are separate");
        std::set<std::string> tableIds;
        for (const auto& table : physical.tables) {
            require(tableIds.insert(table.uuid).second, "table UUIDs are unique");
            require(table.uuid.at(14) == '4', "table UUID v4");
            if (table.name == "objects") require(table.primaryKey == std::vector<std::string>{"id"}, "object primary key");
            if (table.name == "object_tags") require(table.primaryKey == std::vector<std::string>{"object_id", "tag"}, "ordered composite primary key");
            require(table.name != "geometry_revisions", "current geometry compatibility view has no independent table identity");
        }
        require(!physical.tables.empty(), "table directory populated");
        require(store.provisioning(t).spatialIndex == spatial, "persistent spatial capability");
        require(scalar(store, t, "SELECT count(*) FROM sqlite_schema WHERE name='object_bounds'") == int64_t(spatial), "optional bounds table");
        if (!spatial)
            require(scalar(store, t, "SELECT count(*) FROM sqlite_schema WHERE name LIKE 'spatial_%'") == 0, "no spatial tables, triggers or R-tree shadow tables");
        require(scalar(store, t, "SELECT count(*) FROM sqlite_schema WHERE type='table' AND (name LIKE 'history_%' OR name IN ('journal','object_revisions','object_heads','relationship_revisions','geometry_revisions','geometry_deltas'))") == 0,
                "no application history tables");
    }
    {
        auto w = store.write();
        require(scalar(store,w,"PRAGMA cache_size")==-131072,"configured SQLite writer page cache");
        for (const auto& id : {a, b}) {
            ObjectRecord record; record.uuid = id; record.kind = "fiber"; record.fiber = FiberDetail{};
            store.upsertObject(w, record);
            const auto token = store.putGeometry(w, id, data);
            if (id == a) original = token; else other = token;
        }
        EdgeRecord edge; edge.kind = "perpendicular"; edge.aUuid = a; edge.bUuid = b;
        edge.aAnchor.points = {{"line_points", {0}}}; edge.bAnchor.points = {{"line_points", {0}}};
        edgeId = store.addEdge(w, edge);
        w.commit();
    }
    {
        auto t = store.read();
        require(store.provisioning(t).generation == 2, "one generation per committed write");
        require(store.objectByUuid(t, a)->generation == 2 && store.objectByUuid(t, b)->generation == 2 &&
                store.edgeByUuid(t, edgeId)->generation == 2 && store.currentGeometry(t,a)->objectGeneration == 2,
                "objects, geometry and relationships share publication generation");
        const auto expected = std::find_if(data.arrays.begin(), data.arrays.end(), [](const auto& a) { return a.name == "line_points"; });
        require(store.readGeometryArray(t, original, "line_points") == expected->bytes, "current geometry reads");
        require(!store.edgeGeometryState(t, edgeId).stale, "initial relationship valid");
        require(store.edgesOf(t, a).size() == 1, "graph reads independent of spatial index");
        if (!spatial) {
            const auto absent = [&](auto call) {
                bool failed = false;
                try { call(); }
                catch (const std::runtime_error& e) {
                    failed = std::string(e.what()).find("spatial index is absent") != std::string::npos;
                }
                require(failed, "spatial query reports absent capability, not missing SQL table or empty result");
            };
            ObjectSelection box; box.bbox = std::array<double,6>{0,0,0,10,10,10};
            absent([&] { store.selectObjects(t, box); });
            absent([&] { store.spatialRegions(t, box); });
            box.volume = "no-matching-volume";
            absent([&] { store.selectObjects(t, box); });
            absent([&] { store.geometryRegion(t, RegionSlice{}); });
            absent([&] { store.nearestSurfaces(t, SurfaceProximityQuery{}); });
        }
        mustFail([&] { store.objectHistory(t, a); });
        mustFail([&] { store.geometryHistory(t, a); });
        mustFail([&] { store.journalPosition(t); });
    }
    { auto w = store.write(); w.commit(); }
    { auto w = store.write(); store.setMeta(w, a, "rolled_back", "true"); }
    {
        auto t = store.read();
        require(store.provisioning(t).generation == 2, "no-op and rollback preserve generation");
    }
    std::string replacement;
    {
        auto w = store.write();
        GeometryArrayEdit edit; edit.array = "line_points"; edit.elementOffset = 3;
        const double x = 7; edit.bytes.resize(sizeof(x)); std::memcpy(edit.bytes.data(), &x, sizeof(x));
        replacement = store.editGeometry(w, a, {edit}, original);
        w.commit();
    }
    {
        auto t = store.read();
        require(!store.geometryRevision(t, original), "old geometry is not retained");
        require(store.currentGeometry(t, a)->uuid == replacement, "new current geometry");
        require(!store.edgeGeometryState(t, edgeId).stale, "unaffected relationship rebound to current geometry");
        require(scalar(store, t, "SELECT count(*) FROM current_geometry") == 2, "one descriptor per object");
        require(scalar(store, t, "SELECT count(*) FROM geometry_chunks c WHERE NOT EXISTS (SELECT 1 FROM current_geometry_chunks m WHERE m.hash=c.hash) AND NOT EXISTS (SELECT 1 FROM current_shared_payloads p WHERE p.hash=c.hash)") == 0, "no superseded unreferenced payloads retained");
    }
    // Another writable handle must observe sealing even if opened beforehand.
    auto second = ObjectStore::open(path, OpenOptions{.author="other"});
    { auto w = store.write(); store.seal(w); w.commit(); }
    mustFail([&] { auto w = second.write(); });
    mustFail([&] { ObjectStore::open(path, OpenOptions{.author="other"}); });
    auto readonly = ObjectStore::open(path, OpenOptions{.readOnly=true});
    {
        auto t = readonly.read();
        require(readonly.provisioning(t).history == HistoryMode::Off, "persistent history mode survives open defaults");
        require(readonly.provisioning(t).spatialIndex == spatial, "persistent spatial policy survives open defaults");
        require(readonly.objectCount(t, "fiber") == 2, "sealed current data readable");
        const auto reopened = readonly.databaseIdentity(t);
        require(reopened.uuid == physical.uuid && reopened.tables.size() == physical.tables.size(), "physical identity preserved through seal and reopen");
        for (size_t i = 0; i < physical.tables.size(); ++i)
            require(reopened.tables[i].uuid == physical.tables[i].uuid, "table identity preserved through seal and reopen");
    }
}

void surfaceWithoutSpatial(HistoryMode history)
{
    TemporaryDirectory dir;
    OpenOptions options; options.author = "test"; options.history = history; options.spatialIndex = false;
    auto store = ObjectStore::create(dir.path / "surface.vcobj", options);
    vc::geometry::Surface surface{2, 2, {1,1}, {{0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}}};
    std::string id;
    {
        auto w = store.write();
        ObjectRecord record; record.kind = "surface"; record.surface = SurfaceDetail{};
        id = store.upsertObject(w, record);
        store.putGeometry(w, id, vc::geometry::encode(surface));
        require(store.objectByUuid(w, id)->surface->area == 1, "surface area maintained without spatial index");
        w.commit();
    }
    {
        auto w = store.write();
        for (auto& point : surface.points) point[0] *= 2;
        store.putGeometry(w, id, vc::geometry::encode(surface), store.currentGeometry(w, id)->uuid);
        require(store.objectByUuid(w, id)->surface->area == 2, "surface geometry replacement updates area without spatial index");
        w.commit();
    }
}

int main()
{
    directStore(true);
    directStore(false);
    surfaceWithoutSpatial(HistoryMode::Off);
    surfaceWithoutSpatial(HistoryMode::On);
    TemporaryDirectory dir;
    OpenOptions options; options.author = "test"; options.spatialIndex = false;
    auto store = ObjectStore::create(dir.path / "history-no-spatial.vcobj", options);
    std::string id;
    {
        auto w = store.write();
        ObjectRecord record; record.kind = "fiber"; record.fiber = FiberDetail{};
        id = store.upsertObject(w, record);
        store.putGeometry(w, id, vc::geometry::encode(vc::geometry::Fiber{{}, {{1,2,3},{4,5,6}}, {}}));
        w.commit();
    }
    auto t = store.read();
    require(!store.objectHistory(t, id).empty(), "history retention independent of spatial index");
    require(scalar(store, t, "SELECT count(*) FROM sqlite_schema WHERE name LIKE 'spatial_%'") == 0, "history-on can omit spatial index");
    auto independent = ObjectStore::create(dir.path / "independent.vcobj", options);
    auto r = independent.read();
    require(store.databaseIdentity(t).uuid != independent.databaseIdentity(r).uuid, "independently created databases have distinct identities");
    require(store.databaseIdentity(t).tables.front().uuid != independent.databaseIdentity(r).tables.front().uuid, "independent table instances have distinct identities");
    t.commit(); r.commit();
    {
        auto w = independent.write();
        detail::exec(ObjectStoreImplAccess::conn(w), "CREATE TABLE extension_data(key TEXT PRIMARY KEY,value BLOB);");
        w.commit();
    }
    {
        auto r = independent.read();
        const auto directory = independent.databaseIdentity(r);
        const auto extra = std::find_if(directory.tables.begin(), directory.tables.end(), [](const auto& t) { return t.name == "extension_data"; });
        require(extra != directory.tables.end() && extra->primaryKey == std::vector<std::string>{"key"}, "schema changes register extension tables atomically");
        require(independent.provisioning(r).generation == 2, "provisioning change advances store generation");
    }
    {
        auto w = independent.write();
        detail::exec(ObjectStoreImplAccess::conn(w), "CREATE TABLE rolled_back_extension(id INTEGER PRIMARY KEY);");
    }
    {
        auto r = independent.read();
        require(scalar(independent, r, "SELECT count(*) FROM table_directory WHERE name='rolled_back_extension'") == 0, "rolled-back schema does not publish a table identity");
    }
}
