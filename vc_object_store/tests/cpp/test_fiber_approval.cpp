#include "TestSupport.hpp"
#include "StoreTestSupport.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/FiberApprovalExtension.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <cmath>
#include <iostream>

using namespace vc::objects;
namespace {
using Aspect = FiberApprovalAspect;
using State = FiberStretchState;

std::vector<vc::geometry::Point> line(size_t points = 101)
{
    std::vector<vc::geometry::Point> out;
    for (size_t x = 0; x < points; ++x) out.push_back({double(x), 0.0, 0.0});
    return out;
}
} // namespace

int main() try
{
    // Painting states: cut, clear, join; aspects are independent.
    {
        const auto points = line();
        std::vector<FiberStretch> stretches;
        setFiberStretchState(stretches, Aspect::Geometry, 10, 40, State::Approved, points);
        setFiberStretchState(stretches, Aspect::Normals, 20, 30, State::Approved, points);
        require(stretches.size() == 2, "one stretch per aspect");
        setFiberStretchState(stretches, Aspect::Geometry, 20.5, 25.25, State::Broken, points);
        require(stretches.size() == 4, "a state inside a stretch cuts it in two");
        require(fiberStretchStateAt(stretches, Aspect::Geometry, 15) == State::Approved, "left part stays approved");
        require(fiberStretchStateAt(stretches, Aspect::Geometry, 22) == State::Broken, "the painted part is broken");
        require(fiberStretchStateAt(stretches, Aspect::Geometry, 30) == State::Approved, "right part stays approved");
        require(fiberStretchStateAt(stretches, Aspect::Normals, 22) == State::Approved, "the other aspect is untouched");
        require(fiberStretchStateAt(stretches, Aspect::Geometry, 50) == State::Undefined, "outside is undefined");
        setFiberStretchState(stretches, Aspect::Geometry, 20.5, 25.25, State::Approved, points);
        require(stretches.size() == 2 && stretches[0].from == 10 && stretches[0].to == 40,
                "the same state over the gap joins the parts again");
        setFiberStretchState(stretches, Aspect::Geometry, 0, 100, State::Undefined, points);
        require(stretches.size() == 1 && stretches[0].aspect == Aspect::Normals, "undefined clears");
        mustFail([&] { setFiberStretchState(stretches, Aspect::Geometry, 30, 30, State::Approved, points); });
    }
    // Staleness: only where the line changed. Normals: the saved ones, on approved stretches and at normal points.
    {
        auto points = line();
        FiberApprovalLayer layer;
        setFiberStretchState(layer.stretches, Aspect::Normals, 10, 20, State::Approved, points);
        setFiberStretchState(layer.stretches, Aspect::Normals, 60, 70, State::Approved, points);
        setFiberStretchState(layer.stretches, Aspect::Geometry, 60, 70, State::Broken, points);
        require(setFiberNormalPoint(layer.normalPoints, 40.2, {0, 0, 2}, points) == 0, "first normal point");
        require(setFiberNormalPoint(layer.normalPoints, 40.4, {0, 1, 0}, points) == 0 && layer.normalPoints.size() == 1,
                "a normal point at the same place replaces it");
        require(setFiberNormalPoint(layer.normalPoints, 5, {0, 0, 3}, points) == 0 && layer.normalPoints.size() == 2 &&
                    std::abs(layer.normalPoints[0].normal[2] - 1.0) < 1e-12, "normal points are sorted and unit");
        layer.lineNormals.assign(points.size(), {0, 0, 1});
        auto status = fiberApprovalStatus(layer, points);
        require(status.staleStretches == 0 && status.staleNormalPoints == 0 && status.lineNormalsUsable, "all current");
        auto normals = approvedFiberLineNormals(layer, status, points.size());
        require(normals.size() == points.size() && normals[15][2] == 1 && normals[30][2] == 0 && normals[40][1] == 1 &&
                    normals[5][2] == 1 && normals[65][2] == 1, "saved normals on approved stretches and at normal points");
        require(fiberStretchRanges(layer, status, Aspect::Geometry, State::Broken).size() == 1, "one broken stretch");
        points[65][1] = 3; // the fiber was edited inside the second stretch
        status = fiberApprovalStatus(layer, points);
        require(status.staleStretches == 2 && status.staleNormalPoints == 0, "only the stretches on the changed part are stale");
        normals = approvedFiberLineNormals(layer, status, points.size());
        require(normals[15][2] == 1 && normals[65][2] == 0, "a stale stretch approves nothing");
        require(fiberStretchRanges(layer, status, Aspect::Geometry, State::Broken).empty(), "a stale stretch is not broken");
        points.pop_back();
        require(!fiberApprovalStatus(layer, points).lineNormalsUsable, "normals of another line are not used");
    }
    // Through a store: the newest generation is the layer.
    {
        TemporaryDirectory directory;
        OpenOptions options; options.author = "approval-test";
        const auto path = directory.path / "approval.vcobj";
        { auto store = ObjectStore::create(path, options); }
        const auto uuid = util::newUuid(), other = util::newUuid();
        const auto points = line();
        ObjectStoreSession session;
        const auto attachment = session.attach(path, options);
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
            require(!readFiberApproval(reader, uuid) && fiberApprovalGenerations(reader).empty(), "no layer before the tables exist");
        });
        FiberApprovalLayer layer;
        layer.fiberUuid = uuid; layer.generation = 1; layer.fiberVersion = 3; layer.author = "alice";
        layer.savedAt = "2026-11-25T10:00:00Z"; layer.host = "host-a"; layer.baselineRun = "run-1";
        setFiberStretchState(layer.stretches, Aspect::Geometry, 10, 40, State::Approved, points);
        setFiberStretchState(layer.stretches, Aspect::Normals, 12.5, 30, State::Broken, points);
        setFiberNormalPoint(layer.normalPoints, 20, {0, 0.5, 0.5}, points);
        layer.lineNormals.assign(points.size(), {0, 0.5, -0.25});
        const auto write = [&](const FiberApprovalLayer& value) {
            auto request = session.newRequest(attachment);
            for (const auto& table : fiberApprovalTables()) request.operations.push_back(CreateExtensionTableOp{table});
            auto rows = fiberApprovalRows(value);
            request.operations.push_back(AppendExtensionRowsOp{kFiberApprovalTable, {rows.header}});
            if (!rows.stretches.empty())
                request.operations.push_back(AppendExtensionRowsOp{kFiberApprovalStretchTable, rows.stretches});
            if (!rows.normalPoints.empty())
                request.operations.push_back(AppendExtensionRowsOp{kFiberApprovalNormalPointTable, rows.normalPoints});
            session.submit(attachment, request);
        };
        write(layer);
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
            const auto read = readFiberApproval(reader, uuid);
            require(read && *read == layer, "the layer reads back");
            require(!readFiberApproval(reader, other), "another fiber has none");
        });
        auto second = layer;
        second.generation = 2;
        second.stretches.clear();
        second.normalPoints.clear();
        write(second);
        auto third = layer; third.fiberUuid = other;
        write(third);
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
            const auto read = readFiberApproval(reader, uuid);
            require(read && *read == second && read->empty(), "the newest generation replaces the layer, also by an empty one");
            const auto generations = fiberApprovalGenerations(reader);
            require(generations.size() == 2 && generations.at(uuid) == 2 && generations.at(other) == 1, "generations per fiber");
        });
    }
    // Dropping an extension table removes its rows and its registration; the others stay.
    {
        TemporaryDirectory directory;
        OpenOptions options; options.author = "drop-test";
        const auto path = directory.path / "drop.vcobj";
        { auto store = ObjectStore::create(path, options); }
        auto store = ObjectStore::open(path, options);
        FiberApprovalLayer layer;
        layer.fiberUuid = util::newUuid(); layer.generation = 1; layer.author = "a"; layer.savedAt = "t"; layer.host = "h";
        const auto points = line();
        setFiberStretchState(layer.stretches, Aspect::Geometry, 10, 40, State::Approved, points);
        const auto rows = fiberApprovalRows(layer);
        {
            auto txn = store.write();
            for (const auto& table : fiberApprovalTables()) store.createExtensionTable(txn, table);
            store.appendExtensionRows(txn, kFiberApprovalTable, std::vector<ExtensionRow>{rows.header});
            store.appendExtensionRows(txn, kFiberApprovalStretchTable, rows.stretches);
            txn.commit();
        }
        {
            auto txn = store.write();
            store.dropExtensionTable(txn, kFiberApprovalStretchTable);
            mustFail([&] { store.dropExtensionTable(txn, "ext_never_registered"); });
            txn.commit();
        }
        auto txn = store.read();
        const auto tables = store.extensionTables(txn);
        require(tables.size() == 2, "two of the three tables remain");
        for (const auto& table : tables) require(table.name != kFiberApprovalStretchTable, "the dropped table is not registered");
        std::int64_t headers = 0;
        store.scanExtensionRows(txn, kFiberApprovalTable, [&](const ExtensionRow&) { ++headers; return true; });
        require(headers == 1, "the other tables keep their rows");
        mustFail([&] { store.scanExtensionRows(txn, kFiberApprovalStretchTable, [](const ExtensionRow&) { return true; }); });
    }
    std::cout << "fiber approval tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
