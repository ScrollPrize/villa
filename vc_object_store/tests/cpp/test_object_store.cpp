#define VC_TEST_MAIN
#include "vc_test.hpp"
#include "TempDir.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include <atomic>
#include <thread>
using namespace vc::objects;
using vc::test::TempDir;
namespace {
OpenOptions testOptions() { return OpenOptions{.author="tester", .hostnameOverride="testhost"}; }
std::string addObject(ObjectStore& db, const std::string& kind) {
    auto w = db.write();
    ObjectRecord record; record.kind = kind;
    auto uuid = db.upsertObject(w, record);
    w.commit();
    return uuid;
}
}

TEST_CASE("object history publishes a whole checked action and preserves reader snapshots")
{
    TempDir tmp("unified_history");
    auto db = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
    const auto a = addObject(db, "note"), b = addObject(db, "note");
    auto before = db.read();
    const auto a0 = db.currentObjectRevision(before, a)->uuid;
    const auto b0 = db.currentObjectRevision(before, b)->uuid;
    std::string edge;
    {
        auto write = db.write(RevisionSelection{{a, a0}, {b, b0}});
        db.setMeta(write, a, "winding", "2");
        EdgeRecord link;
        link.kind = "connected"; link.aUuid = a; link.bUuid = b;
        edge = db.addEdge(write, link);
        // Revision reads inside a write describe the last published state.
        CHECK(db.currentObjectRevision(write, a)->uuid == a0);
        write.commit();
    }
    CHECK(db.currentObjectRevision(before, a)->uuid == a0);
    CHECK(db.currentObjectRevision(before, b)->uuid == b0);
    CHECK(db.recordedRelationships(before, a0).empty());
    auto after = db.read();
    const auto a1 = *db.currentObjectRevision(after, a);
    const auto b1 = *db.currentObjectRevision(after, b);
    CHECK(a1.parentUuid == a0);
    CHECK(b1.parentUuid == b0);
    CHECK(a1.record.meta == R"({"winding":2})");
    const auto links = db.relationshipsAt(after, {{a, a1.uuid}, {b, b1.uuid}});
    REQUIRE(links.size() == 1);
    CHECK(links.front().applicable);
    CHECK(db.recordedRelationships(after, a1.uuid).front().uuid ==
          db.recordedRelationships(after, b1.uuid).front().uuid);
    CHECK_THROWS(db.write(RevisionSelection{{a, a0}, {b, b0}}));
    {
        auto write = db.write(RevisionSelection{{a, a1.uuid}});
        db.setMeta(write, a, "abort", "true");
        db.removeEdge(write, edge); // shared owner b lacks an expectation
        CHECK_THROWS(write.commit());
        CHECK_FALSE(write.active());
    }
    auto unchanged = db.read();
    CHECK(db.currentObjectRevision(unchanged, a)->uuid == a1.uuid);
    CHECK(db.currentObjectRevision(unchanged, b)->uuid == b1.uuid);
    CHECK(db.edgeByUuid(unchanged, edge).has_value());
}


TEST_CASE("object store serializes independent writer transactions")
{
    TempDir tmp("object_writers");
    auto store = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
    std::atomic_bool failed{false};
    std::vector<std::thread> writers;
    for (int worker = 0; worker < 3; ++worker) {
        writers.emplace_back([&] {
            try {
                for (int i = 0; i < 10; ++i) addObject(store, "note");
            } catch (...) { failed.store(true); }
        });
    }
    for (auto& writer : writers) writer.join();
    CHECK_FALSE(failed.load());
    auto read = store.read();
    CHECK(store.objectCount(read, "note") == 30);
}
