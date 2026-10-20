#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/ArrayBytes.hpp"

using namespace vc::objects;

TEST_CASE("lifecycle-only publication reuses geometry without payload work")
{
    const auto path = std::filesystem::temp_directory_path() / ("vc-lifecycle-" + util::newUuid());
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { for (const auto& suffix : {"", "-wal", "-shm"}) {
            std::error_code error; std::filesystem::remove(path.string() + suffix, error);
        } }
    } cleanup{path};
    OpenOptions options; options.author = "test";
    auto store = ObjectStore::create(path, options);
    GeometryData geometry; geometry.kind = "point_collection";
    geometry.arrays.push_back(detail::scalarArray<double>("points", "<f8", {2,3}, std::vector<double>{1,2,3,4,5,6}));
    geometry.arrays.push_back(detail::scalarArray<int64_t>("point_ids", "<i8", {2}, std::vector<int64_t>{42,99}));
    std::string a, b, edge, payload;
    {
        auto txn = store.write();
        ObjectRecord collection; collection.kind = "point_collection";
        a = store.upsertObject(txn, collection);
        ObjectRecord note; note.kind = "note";
        b = store.upsertObject(txn, note);
        payload = store.putGeometry(txn, a, geometry);
        EdgeRecord reference; reference.kind = "declared_reference";
        reference.aUuid = a; reference.bUuid = b;
        reference.aAnchor.points.push_back({"point_ids", {42}});
        reference.bAnchor.wholeGeometry = true;
        edge = store.addEdge(txn, reference);
        txn.commit();
    }
    auto expected = [&] {
        auto txn = store.read();
        return RevisionSelection{{a, store.currentObjectRevision(txn, a)->uuid},
                                 {b, store.currentObjectRevision(txn, b)->uuid}};
    };
    detail::GeometryWork work;
    detail::GeometryWorkScope scope(work);
    {
        auto txn = store.write(expected());
        store.retireObject(txn, a);
        store.keepRecordedRelationship(txn, edge);
        txn.commit();
    }
    CHECK(work.readBytes.empty()); CHECK(work.hashedBytes.empty());
    {
        auto txn = store.read();
        ObjectSelection selection; selection.uuids = {a}; selection.includeRetired = true;
        CHECK(store.selectObjects(txn, selection).front().retired);
        const auto projected = store.selectObjects(txn, selection, std::vector<std::string>{}).front();
        CHECK_FALSE(projected.retired);
        CHECK(projected.lifecycleEpoch.empty());
    }
    {
        auto txn = store.write(expected());
        store.restoreObject(txn, a);
        store.keepRecordedRelationship(txn, edge);
        txn.commit();
    }
    CHECK(work.readBytes.empty()); CHECK(work.hashedBytes.empty());
    auto txn = store.read();
    CHECK(store.geometryHistory(txn, a).size() == 1);
    CHECK(store.currentObjectRevision(txn, a)->geometryRevision == payload);
    CHECK_FALSE(store.objectByUuid(txn, a)->retired);
    CHECK(store.edgesOf(txn, a).empty());
    CHECK(store.edgesOf(txn, a, true).size() == 1);
    CHECK(store.readGeometryArray(txn, payload, "points") == geometry.arrays[0].bytes);
}
