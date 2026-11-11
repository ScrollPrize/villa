#include "TestSupport.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/objects/ObjectCopy.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include "internal/History.hpp"
#include "internal/RecordJson.hpp"
#include "geometry/GeometryReader.hpp"
#include <bit>
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;
namespace {
GeometryArrayEdit coordinate(int64_t offset, double value)
{
    GeometryArrayEdit edit{"line_points", offset, std::vector<uint8_t>(8)};
    const auto bits = std::bit_cast<uint64_t>(value);
    for (int i = 0; i < 8; ++i) edit.bytes[i] = uint8_t(bits >> (i * 8));
    return edit;
}

int64_t scalar(const Txn& txn, const std::string& sql)
{
    Statement row(ObjectStoreImplAccess::conn(txn), sql);
    require(row.row(), "scalar row"); return row.i64(0);
}

void exercise(bool spatial)
{
    TemporaryDirectory dir;
    const auto path = dir.path / "policy.vcobj";
    OpenOptions options; options.author = "test"; options.spatialIndex = spatial;
    options.maxGeometryDeltaBytes = 1024;
    const auto a = util::newUuid(), b = util::newUuid();
    std::string edge;
    nlohmann::json annotations;
    std::map<std::string, std::vector<uint8_t>> expected;
    {
        auto store = ObjectStore::create(path, options);
        vc::geometry::Fiber data;
        data.controlPoints = {{0,0,0}, {5000,0,0}, {9999,0,0}};
        data.controlPointIds = {10,20,30};
        for (int i = 0; i < 10000; ++i) data.linePoints.push_back({double(i), 0, 0});
        std::string head;
        {
            auto txn = store.write();
            for (const auto& id : {a,b}) {
                ObjectRecord record; record.uuid = id; record.kind = "fiber";
                record.meta = R"({"source":{"control_points_attributes":[{"note":"kept"},{},{}]}})";
                store.upsertObject(txn, record);
                const auto revision = store.putGeometry(txn, id, vc::geometry::encode(data));
                if (id == a) head = revision;
            }
            EdgeRecord relation; relation.kind = "perpendicular"; relation.aUuid = a; relation.bUuid = b;
            relation.aAnchor.points = {{"control_points", {0}}}; relation.bAnchor.points = relation.aAnchor.points;
            edge = store.addEdge(txn, relation);
            // This extension remains valid under either current-geometry
            // representation. Policy changes must not silently delete it.
            exec(ObjectStoreImplAccess::conn(txn),
                "CREATE TABLE extension_events(n INTEGER);"
                "CREATE VIEW extension_current AS SELECT uuid FROM objects WHERE geometry_manifest IS NOT NULL;"
                "CREATE TRIGGER extension_geometry_note AFTER UPDATE OF name ON objects BEGIN "
                "INSERT INTO extension_events SELECT count(*) FROM objects WHERE geometry_manifest IS NOT NULL; END;");
            txn.commit();
        }
        {
            auto txn = store.write(); head = store.editGeometry(txn, a, {coordinate(0, 1)}, head); txn.commit();
        }
        { auto txn = store.read(); require(scalar(txn, "SELECT count(*) FROM geometry_deltas") == 1, "fixture has sparse delta"); }
        for (auto& point : data.linePoints) point[0] += 2;
        const auto encoded = vc::geometry::encode(data);
        const auto line = std::find_if(encoded.arrays.begin(), encoded.arrays.end(), [](const auto& x) { return x.name == "line_points"; });
        {
            auto txn = store.write();
            head = store.editGeometry(txn, a, {{"line_points", 0, line->bytes}}, head);
            txn.commit();
        }
        {
            auto txn = store.write(); head = store.editGeometry(txn, a, {coordinate(3, 42)}, head);
            // Policy removal requires explicit revalidation of old pins even
            // when dependency-level checks show their points are unchanged.
            store.retainRelationship(txn, edge); txn.commit();
        }
        auto txn = store.read();
        require(scalar(txn,"SELECT count(*) FROM sqlite_schema WHERE name IN "
            "('geometry_heads','current_geometry','current_geometry_chunks','current_shared_payloads')")==0,
            "history-on also uses direct current object geometry");
        require(scalar(txn,"SELECT count(*) FROM objects WHERE geometry_manifest IS NOT NULL")==2,
            "both current objects directly contain their geometry descriptors");
        {
            Statement current(ObjectStoreImplAccess::conn(txn),"SELECT storage_ref,geometry_manifest FROM objects WHERE uuid=?",parseUuid(a));
            require(current.row()&&current.text(0)==head,"object contains the current geometry token");
            require(nlohmann::json::parse(current.text(1))==GeometryReader(txn,head).manifest(),
                "current descriptor is complete after sparse and full edits");
        }
        require(scalar(txn, "SELECT count(*) FROM geometry_deltas") == 2, "fixture has delta after large shared-array revision");
        for (const auto& array : encoded.arrays) expected[array.name] = store.readGeometryArray(txn, head, array.name);
        annotations = store.geometryPointAnnotations(txn, head, "control_points");
        require(annotations.dump().find("kept") != std::string::npos, "fixture has sequence annotations");
        require(!store.edgeGeometryState(txn, edge).stale, "unaffected dependency rebinding before transition");
    }
    int64_t iteration = 0;
    for (const auto mode : {HistoryMode::Off, HistoryMode::On, HistoryMode::Off, HistoryMode::On}) {
        const auto geometryIdentity=[&]{auto before=ObjectStore::open(path,options);auto txn=before.read();
            return before.currentGeometry(txn,a)->uuid;}();
        changeHistoryMode(path, mode);
        auto store = ObjectStore::open(path, options);
        std::string head;
        {
            auto txn = store.read();
            require(store.provisioning(txn).history == mode, "persistent policy changed");
            head = store.currentGeometry(txn, a)->uuid;
            require(head==geometryIdentity,"history policy changes retention, not geometry identity");
            for (const auto& [name, bytes] : expected)
                require(store.readGeometryArray(txn, head, name) == bytes, "transition preserves exact geometry bytes");
            const auto copy=readObjectCopy(LocalStoreReader(store,txn),a);
            require(copy.geometry.has_value(),"current object copy works under either history policy");
            for(const auto& array:copy.geometry->arrays)
                require(array.bytes==expected.at(array.name),"current copy preserves geometry bytes without synthetic history");
            require(store.geometryPointAnnotations(txn, head, "control_points") == annotations, "transition preserves annotation payload graph");
            require(!store.edgeGeometryState(txn, edge).stale, "transition preserves current relationship applicability");
            require(scalar(txn, "SELECT count(*) FROM sqlite_schema WHERE name='extension_geometry_note'") == 1,
                    "transition preserves compatible extension triggers");
            require(scalar(txn, "SELECT count(*) FROM extension_current") >= 2, "compatible extension view remains readable");
            if (mode == HistoryMode::Off)
                require(scalar(txn, "SELECT count(*) FROM sqlite_schema WHERE type='table' AND name='geometry_deltas'") == 0,
                        "history-off has no hidden sparse history");
        }
        {
            auto txn = store.write();
            head = store.editGeometry(txn, a, {coordinate(6, double(++iteration + 100))}, head);
            auto record = *store.objectByUuid(txn, a); record.name = "updated" + std::to_string(iteration);
            store.upsertObject(txn, record);
            store.retainRelationship(txn, edge);
            txn.commit();
        }
        {
            auto txn = store.read();
            expected["line_points"] = store.readGeometryArray(txn, head, "line_points");
            require(!store.edgeGeometryState(txn, edge).stale, "edit after transition keeps unaffected relationships valid");
            require(store.geometryPointAnnotations(txn, head, "control_points") == annotations, "subsequent edits preserve annotations");
            require(scalar(txn, "SELECT count(*) FROM extension_events") == iteration, "preserved extension remains operational");
        }
    }
}

void rejectsUnavailableHistory()
{
    TemporaryDirectory dir;
    const auto path=dir.path/"dependent.vcobj";
    {
        auto store=ObjectStore::create(path,OpenOptions{.author="test"});
        auto txn=store.write();
        exec(ObjectStoreImplAccess::conn(txn),"CREATE VIEW extension_history AS SELECT uuid FROM object_revisions;");
        txn.commit();
    }
    mustFail([&] { changeHistoryMode(path,HistoryMode::Off); });
    auto store=ObjectStore::open(path,OpenOptions{.readOnly=true});
    auto txn=store.read();
    require(store.provisioning(txn).history==HistoryMode::On && store.provisioning(txn).commitSequence==1,
            "incompatible extension rejects history removal without touching original");
    require(scalar(txn,"SELECT count(*) FROM extension_history")==0,"original extension still readable after refusal");
}

void leanDependencies(HistoryMode history)
{
    TemporaryDirectory dir;
    OpenOptions options;options.author="test";options.history=history;
    auto store=ObjectStore::create(dir.path/"dependencies.vcobj",options);
    std::string a,b,anchored,empty;
    {
        auto txn=store.write();
        ObjectRecord record;record.kind="note";
        record.meta=nlohmann::json{{"important",{{"nested",{1,2,3}}}},{"null",nullptr},{"unused",std::string(256*1024,'x')}}.dump();
        a=store.upsertObject(txn,record);
        record.meta=R"({"important":false,"null":"other endpoint"})";
        b=store.upsertObject(txn,record);
        // Reverse insertion-ID order and use disjoint dependencies: the native
        // canonical endpoint swap must also swap validation metadata correctly.
        EdgeRecord edge;edge.kind="related";edge.aUuid=b;edge.bUuid=a;
        edge.aAnchor.metadataKeys={"null"};edge.bAnchor.metadataKeys={"important","missing"};
        anchored=store.addEdge(txn,edge);
        edge.kind="unanchored";edge.aAnchor.metadataKeys.clear();edge.bAnchor.metadataKeys.clear();empty=store.addEdge(txn,edge);
        txn.commit();
    }
    {
        auto txn=store.read();auto* db=ObjectStoreImplAccess::conn(txn);
        const auto record=objectToJson(*store.objectByUuid(txn,a));
        for(const auto& keys:std::vector<std::vector<std::string>>{{},{"important"},{"missing"},{"null","missing"},{"important","unused"}}) {
            Anchor anchor;anchor.metadataKeys=keys;
            require(currentDependencyMetadata(db,a,anchor)==dependencyMetadata(record,anchor),"lean dependency projection equals full-record projection");
        }
        mustFail([&]{currentDependencyMetadata(db,util::newUuid(),Anchor{});});
        // Prove the no-key path never even reads the (potentially large) meta
        // column, while still checking the endpoint exists and its lifecycle.
        struct Authorizer {
            sqlite3* db;
            ~Authorizer(){sqlite3_set_authorizer(db,nullptr,nullptr);}
        } guard{db};
        sqlite3_set_authorizer(db,[](void*,int action,const char* table,const char* column,const char*,const char*){
            return action==SQLITE_READ && table && column && std::string_view(table)=="objects" && std::string_view(column)=="meta" ? SQLITE_DENY : SQLITE_OK;
        },nullptr);
        require(currentDependencyMetadata(db,a,Anchor{})==dependencyMetadata(record,Anchor{}),"unanchored validation does not access object metadata");
    }
    {auto txn=store.write();store.setMeta(txn,a,"unrelated","true");txn.commit();}
    mustFail([&]{auto txn=store.write();store.setMeta(txn,a,"important",R"({"nested":[4]})");txn.commit();});
    {auto txn=store.write();store.setMeta(txn,a,"important",R"({"nested":[4]})");store.retainRelationship(txn,anchored);txn.commit();}
    const auto checked=[&] {
        VersionSelection pins;
        {auto txn=store.read();for(const auto& id:{a,b})pins[id]=store.currentVersion(txn,id);}
        return pins;
    };
    const auto write=[&]{return store.writeNumbered(checked());};
    {auto txn=write();store.retireObject(txn,a);txn.commit();}
    {auto txn=store.read();const auto full=objectToJson(*store.objectByUuid(txn,a));
     require(currentDependencyMetadata(ObjectStoreImplAccess::conn(txn),a,Anchor{})==dependencyMetadata(full,Anchor{}),"retired state and lifecycle epoch retained even without metadata keys");}
    {auto txn=write();store.restoreObject(txn,a);txn.commit();}
    {auto txn=store.read();require(!store.edgeByUuid(txn,anchored)&&!store.edgeByUuid(txn,empty),
        "restoring an endpoint never revives retired current relationships");}
}
}

int main() try
{
    exercise(false);
    exercise(true);
    rejectsUnavailableHistory();
    leanDependencies(HistoryMode::Off);
    leanDependencies(HistoryMode::On);
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
}
