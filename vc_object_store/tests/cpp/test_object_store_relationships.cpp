// Relationship decisions across endpoint edits, retirement and restoration.
#include "TestSupport.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"

#include <cstring>
#include <iostream>

using namespace vc::objects;

int main()
try {
    TemporaryDirectory directory;
    OpenOptions options; options.author = "relationships-test";
    auto db = ObjectStore::create(directory.path / "relationships.vcobj", options);
    std::string a, b, edge, first;
    {
        auto txn = db.write();
        auto add = [&](const vc::geometry::Track& track) {
            ObjectRecord record; record.kind = "track"; record.track.emplace();
            record.track->datasetRef = "test"; record.track->externalKey = util::newUuid();
            const auto uuid = db.upsertObject(txn, record);
            return std::pair{uuid, db.putGeometry(txn, uuid, vc::geometry::encode(track))};
        };
        vc::geometry::Track two; two.points = {{1, 2, 3}, {4, 5, 6}};
        vc::geometry::Track one; one.points = {{1, 2, 3}};
        std::tie(a, first) = add(two);
        b = add(one).first;
        EdgeRecord link; link.kind = "fiber_branch"; link.aUuid = a; link.bUuid = b;
        link.aAnchor.points = link.bAnchor.points = {{"points", {0}}};
        link.aAnchor.metadataKeys = link.bAnchor.metadataKeys = {"coordinates"};
        edge = db.addEdge(txn, link);
        txn.commit();
    }
    const auto heads = [&] {
        auto txn = db.read();
        return VersionSelection{{a, db.currentVersion(txn, a)}, {b, db.currentVersion(txn, b)}};
    };
    const auto component = [&](const std::string& uuid, std::optional<bool> adjacent, bool applicableOnly) {
        auto txn = db.read();
        return db.componentOf(txn, uuid, {}, nullptr, false, adjacent, applicableOnly);
    };
    const auto selected = heads();
    const auto pinned = [&] {
        auto txn = db.read();
        return std::vector<ObjectVersion>{db.versionReference(txn, *db.currentObjectRevision(txn, a)),
                                          db.versionReference(txn, *db.currentObjectRevision(txn, b))};
    }();
    require(component(a, false, true) == std::vector<std::string>{a, b}, "same-winding component");
    require(component(a, true, false) == std::vector<std::string>{a}, "adjacent filter");
    require([&] { auto txn = db.read(); return db.neighbours(txn, a); }() == std::vector<std::string>{b}, "neighbours");

    // An action that changes an endpoint it did not pin rolls back entirely.
    mustFail([&] {
        auto txn = db.write({{a, selected.at(a)}});
        db.setMeta(txn, a, "rollback", "true");
        db.removeEdge(txn, edge);
        txn.commit();
    });
    require([&] { auto txn = db.read(); return db.edgeByUuid(txn, edge).has_value(); }(), "edge survives the rollback");

    // Editing an anchored point needs a decision for the relationship.
    std::vector<uint8_t> nine(4); const int32_t value = 9; std::memcpy(nine.data(), &value, 4);
    const std::vector<GeometryArrayEdit> edits{{"points", 0, nine}};
    mustFail([&] { auto txn = db.write(); db.editGeometry(txn, a, edits, first); txn.commit(); });
    std::string second;
    {
        auto txn = db.write();
        second = db.editGeometry(txn, a, edits, first, [&](const std::vector<EdgeRecord>& affected) {
            std::vector<GeometryEdgeEdit> decisions;
            for (const auto& record : affected) {
                GeometryEdgeEdit decision; decision.edgeUuid = record.uuid; decision.expectedVersion = record.version.value();
                decision.action = "keep_recorded";
                decision.aRevision = db.currentGeometry(txn, record.aUuid)->uuid;
                decision.bRevision = db.currentGeometry(txn, record.bUuid)->uuid;
                decisions.push_back(decision);
            }
            return decisions;
        });
        txn.commit();
    }
    require(component(a, std::nullopt, true) == std::vector<std::string>{a}, "kept as recorded: not applicable now");
    {
        auto txn = db.read();
        const auto at = db.relationshipsAt(txn, pinned);
        require(at.size() == 1 && at[0].applicable, "applicable at the pinned revisions");
    }
    { auto txn = db.write(heads());
      db.retainRelationship(txn, edge);txn.commit(); }
    require(component(a, std::nullopt, true) == std::vector<std::string>{a, b}, "explicitly retained: applicable again");
    for (const bool retire : {true, false}) {
        auto txn = db.write(heads());
        if (retire) db.retireObject(txn, a);
        else db.restoreObject(txn, a);
        txn.commit();
        auto read = db.read();
        require(db.currentGeometry(read, a)->uuid == second, "lifecycle keeps the geometry");
        require(db.componentOf(read, b, {}, nullptr, false, std::nullopt, true) == std::vector<std::string>{b},
                "retirement removes current relations and restoration does not revive them");
        require(!db.edgeByUuid(read,edge),"retired relationship is absent from current graph");
    }
    auto txn = db.read();
    require(!db.objectByUuid(txn, a)->retired, "restored");
    require(db.objectAtVersion(txn, pinned[0])->geometryRevision == first, "history keeps the pinned geometry");
    std::cout << "relationship decisions passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
}
