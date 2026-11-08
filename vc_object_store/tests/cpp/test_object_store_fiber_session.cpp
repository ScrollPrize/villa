#include "TestSupport.hpp"
#include "StoreTestSupport.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <bit>
#include <iostream>

using namespace vc::objects;
using Json = nlohmann::json;

int main() try
{
    TemporaryDirectory directory;
    OpenOptions options; options.author = "fiber-session-test";
    const auto path = directory.path / "fibers.vcobj";
    { auto store = ObjectStore::create(path, options); }
    auto db = ObjectStore::open(path, options);
    const auto a = util::newUuid(), b = util::newUuid(), c = util::newUuid();
    vc::geometry::Fiber geometry;
    geometry.controlPoints = {{0,0,0}, {10,0,0}, {20,0,0}};
    geometry.controlPointIds = {10,20,30};
    storeAct(db, {{a,{}}, {b,{}}, {c,{}}}, [&](auto& store, auto& txn) {
        for (const auto& uuid : {a,b,c}) {
            ObjectRecord record; record.uuid = uuid; record.kind = "fiber";
            record.name = "duplicate display name";
            record.fiber.emplace(); record.fiber->family = "H"; record.fiber->width = 3.5;
            record.fiber->controlPointCount = geometry.controlPoints.size(); record.fiber->length = geometry.length();
            record.meta = R"({"source":{"control_points_attributes":[{"label":"a"},{"label":"b"},{"label":"c"}]}})";
            store.upsertObject(txn, record);
            store.putGeometry(txn, uuid, vc::geometry::encode(geometry));
        }
    });
    auto heads = [&] {
        VersionSelection result;
        storeRead(db, [&](const auto& store, const auto& txn, const auto&) {
            for (const auto& uuid : {a,b,c}) result[uuid] = store.currentVersion(txn, uuid);
        });
        return result;
    };
    auto expected = heads();
    std::string originalGeometry;
    storeRead(db, [&](const auto& store, const auto& txn, const auto&) {
        ObjectSelection selection; selection.kind = "fiber";
        require(store.selectObjects(txn, selection).size() == 3, "same names retain distinct UUID identities");
        originalGeometry = store.currentGeometry(txn, a)->uuid;
    });
    const auto ordinary = util::newUuid(), adjacent = util::newUuid();
    storeAct(db, expected, [&](auto& store, auto& txn) {
        EdgeRecord edge; edge.uuid = ordinary; edge.kind = "fiber_branch"; edge.aUuid = a; edge.bUuid = b;
        edge.aAnchor.points = {{"control_points", {0}}}; edge.bAnchor.points = edge.aAnchor.points;
        store.addEdge(txn, edge);
        edge.uuid = adjacent; edge.bUuid = c; edge.meta = R"({"adjacent":true})";
        store.addEdge(txn, edge);
        store.setApproval(txn, ordinary, "approved");
    });
    storeRead(db, [&](const auto& store, const auto& txn, const auto&) {
        ObjectSelection objects; objects.kind = "fiber";
        EdgeSelection edges; edges.kind = "fiber_branch"; edges.adjacent = false; edges.applicableOnly = true;
        require(store.graphReadView(txn, objects, edges).edgeUuids == std::vector<std::string>{ordinary}, "same-winding excludes adjacent edges");
        edges.adjacent = true;
        require(store.graphReadView(txn, objects, edges).edgeUuids == std::vector<std::string>{adjacent}, "adjacent graph explicit");
    });

    expected = heads();
    const auto before = expected;
    geometry.controlPoints[0][0] = -5;
    const auto bits = std::bit_cast<uint64_t>(-5.0);
    GeometryArrayEdit edit{"control_points", 0, std::vector<uint8_t>(8)};
    for (int i = 0; i < 8; ++i) edit.bytes[i] = uint8_t(bits >> (i*8));
    storeAct(db, expected, [&](auto& store, auto& txn) {
        store.editGeometry(txn, a, {edit}, originalGeometry, [&](const std::vector<EdgeRecord>& affected) {
            std::vector<GeometryEdgeEdit> decisions;
            for (const auto& edge : affected) {
                GeometryEdgeEdit decision; decision.edgeUuid = edge.uuid; decision.expectedGeneration = edge.generation;
                decision.aRevision = store.currentGeometry(txn, edge.aUuid)->uuid;
                decision.bRevision = store.currentGeometry(txn, edge.bUuid)->uuid;
                decision.action = "keep_recorded";
                decisions.push_back(decision);
            }
            return decisions;
        });
        auto record = *store.objectByUuid(txn, a);
        record.fiber->length = geometry.length();
        store.upsertObject(txn, record);
    });
    mustFail([&] { storeAct(db, before, [](auto&, auto&) {}); });
    storeRead(db, [&](const auto& store, const auto& txn, const auto&) {
        require(store.objectByUuid(txn, a)->fiber->length == 25, "summary committed with geometry");
        const auto current = store.currentGeometry(txn, a)->uuid;
        require(store.readGeometryArray(txn, current, "control_points", 0, 8) == edit.bytes, "bounded current read");
        require(store.readGeometryArray(txn, originalGeometry, "control_points", 0, 8) == std::vector<uint8_t>(8), "historical read");
    });

    // Split/merge uses the existing sequence API and explicit source retirement.
    // Start from an unlinked fiber to isolate stable ID and annotation handling.
    const auto source = util::newUuid(), left = util::newUuid(), right = util::newUuid();
    storeAct(db, {{source,{}}}, [&](auto& store, auto& txn) {
        const auto original = *store.objectByUuid(txn, a);
        ObjectRecord record; record.uuid = source; record.kind = "fiber";
        record.name = original.name; record.fiber = original.fiber;
        record.meta = Json{{"source", {{"control_points_attributes", store.geometryPointAnnotations(
            txn, store.currentGeometry(txn, a)->uuid, "control_points")}}}}.dump();
        store.upsertObject(txn, record);
        store.putGeometry(txn, source, vc::geometry::encode(geometry));
    });
    std::string sourceGeometry;
    VersionSelection splitHeads{{left,{}}, {right,{}}};
    storeRead(db, [&](const auto& store, const auto& txn, const auto&) {
        splitHeads[source] = store.currentVersion(txn, source);
        sourceGeometry = store.currentGeometry(txn, source)->uuid;
    });
    storeAct(db, splitHeads, [&](auto& store, auto& txn) {
        for (const auto& uuid : {left,right}) {
            ObjectRecord record; record.uuid = uuid; record.kind = "fiber"; record.fiber.emplace();
            record.fiber->controlPointCount = uuid == left ? 2 : 1;
            record.fiber->length = uuid == left ? 15 : 0;
            store.upsertObject(txn, record);
            const auto empty = store.putGeometry(txn, uuid, vc::geometry::encode(vc::geometry::Fiber{}));
            GeometrySequenceEdit sequence; sequence.array = "control_points";
            GeometrySequencePart part; part.revision = sourceGeometry;
            part.start = uuid == left ? 0 : 2; part.count = record.fiber->controlPointCount;
            sequence.parts.push_back(part);
            store.editGeometrySequence(txn, uuid, sequence, empty);
        }
        store.retireObject(txn, source);
    });
    VersionSelection mergeHeads;
    std::string leftGeometry, rightGeometry;
    storeRead(db, [&](const auto& store, const auto& txn, const auto&) {
        require(store.objectByUuid(txn, source)->retired, "atomic source retirement");
        for (const auto& uuid : {left,right}) mergeHeads[uuid] = store.currentVersion(txn, uuid);
        leftGeometry = store.currentGeometry(txn, left)->uuid; rightGeometry = store.currentGeometry(txn, right)->uuid;
        require(store.geometryPointAnnotations(txn, rightGeometry, "control_points")[0]["label"] == "c", "split annotations");
    });
    storeAct(db, mergeHeads, [&](auto& store, auto& txn) {
        GeometrySequenceEdit sequence; sequence.array = "control_points"; sequence.start = 2;
        GeometrySequencePart part; part.revision = rightGeometry; part.count = 1; sequence.parts.push_back(part);
        store.editGeometrySequence(txn, left, sequence, leftGeometry);
        auto record = *store.objectByUuid(txn, left);
        record.fiber->controlPointCount = 3; record.fiber->length = geometry.length();
        store.upsertObject(txn, record);
        store.retireObject(txn, right);
    });
    auto reopened = ObjectStore::open(path, options);
    storeRead(reopened, [&](const auto& store, const auto& txn, const auto&) {
        require(store.objectByUuid(txn, left)->fiber->controlPointCount == 3, "immediate persistence and reload");
        const auto revision = store.currentGeometry(txn, left)->uuid;
        require(store.readGeometryArray(txn, revision, "control_point_ids") ==
                store.readGeometryArray(txn, sourceGeometry, "control_point_ids"), "stable IDs through split and merge");
    });
    std::cout << "direct fiber API tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
