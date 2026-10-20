#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/ArrayBytes.hpp"

using namespace vc::objects;
TEST_CASE("channel mutations read and hash only the affected payload")
{
    const auto path = std::filesystem::temp_directory_path() / ("vc-channels-" + util::newUuid());
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { for (const auto& suffix : {"", "-wal", "-shm"}) { std::error_code e; std::filesystem::remove(path.string()+suffix, e); } }
    } cleanup{path};
    OpenOptions options; options.author = "test";
    auto store = ObjectStore::create(path, options);
    constexpr int64_t rows = 257, cols = 259;
    GeometryData data; data.kind = "patch"; data.attributes = R"({"scale":[1,1]})";
    std::vector<float> points(rows*cols*3, 1.0f);
    data.arrays.push_back(detail::scalarArray<float>("points", "<f4", {rows,cols,3}, points));
    std::vector<uint16_t> generations(rows*cols, 65535);
    data.arrays.push_back(detail::scalarArray<uint16_t>("generations", "<u2", {rows,cols}, generations));
    auto write = store.write();
    ObjectRecord object; object.kind = "patch";
    const auto id = store.upsertObject(write, object);
    auto revision = store.putGeometry(write, id, data, std::nullopt, {}, GeometryStorageOptions{32});
    ObjectRecord note; note.kind = "note";
    const auto other = store.upsertObject(write, note);
    EdgeRecord edge; edge.kind = "connected"; edge.aUuid = id; edge.bUuid = other;
    edge.aAnchor.points.push_back({"points", {0, 0}}); edge.bAnchor.wholeGeometry = true;
    store.addEdge(write, edge);
    write.commit();
    detail::GeometryWork work;
    detail::GeometryWorkScope scope(work);
    GeometryArray approval{"approval", "|u1", {rows,cols,3}, std::vector<uint8_t>(rows*cols*3)};
    {
        auto txn = store.write();
        revision = store.updateSurfaceChannels(txn, id, {approval}, {}, revision);
        txn.commit();
    }
    CHECK(work.readBytes["approval"] == approval.bytes.size());
    CHECK(work.readBytes["points"] == 0);
    CHECK(work.readBytes["generations"] == 0);
    CHECK(work.hashedBytes["points"] == 0);
    CHECK(work.hashedBytes["generations"] == 0);
    CHECK(work.hashedBytes["approval"] == approval.bytes.size());
    work = {};
    {
        auto txn = store.write();
        revision = store.editGeometryRectangle(txn, id, "approval", 31, 31, 2, 2, std::vector<uint8_t>(12, 7), revision);
        txn.commit();
    }
    CHECK(work.readBytes["points"] == 0);
    CHECK(work.readBytes["generations"] == 0);
    CHECK(work.hashedBytes["points"] == 0);
    CHECK(work.hashedBytes["generations"] == 0);
    CHECK(work.readBytes["approval"] == 4*32*32*3);
    CHECK(work.hashedBytes["approval"] == 4*4*32*32*3);
    work = {};
    // Preserve the previously edited approval by reference when another
    // channel is added, including when its head is a sparse delta.
    {
        auto txn = store.write();
        revision = store.updateSurfaceChannels(txn, id, {{"extra", "|u1", {rows,cols}, std::vector<uint8_t>(rows*cols)}}, {}, revision);
        txn.commit();
    }
    CHECK(work.readBytes["extra"] == rows*cols);
    CHECK(work.readBytes["approval"] == 0);
    CHECK(work.readBytes["points"] == 0);
    CHECK(work.readBytes["generations"] == 0);
    CHECK(work.hashedBytes["approval"] == 0);
    CHECK(work.hashedBytes["points"] == 0);
    CHECK(work.hashedBytes["generations"] == 0);
    work = {};
    {
        auto txn = store.write();
        revision = store.updateSurfaceChannels(txn, id, {}, {"approval"}, revision);
        txn.commit();
    }
    CHECK(work.readBytes.empty());
    CHECK(work.hashedBytes.empty());
    work = {};
    auto read = store.read();
    CHECK(store.readGeometryRectangle(read, revision, "generations", 31, 31, 2, 2) == std::vector<uint8_t>(8, 255));
    CHECK(work.readBytes["generations"] == 4*32*32*2);
    CHECK(work.readBytes["points"] == 0);
    CHECK(work.readBytes["extra"] == 0);
}
