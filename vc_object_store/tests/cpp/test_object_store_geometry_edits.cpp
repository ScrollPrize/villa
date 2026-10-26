#include "TestSupport.hpp"
#include "vc_object_store/objects/GeometryEdits.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
using namespace vc::objects;
using Json = nlohmann::json;
int main() try {
    TemporaryDirectory directory;
    const auto path = directory.path / "edits.vcobj";
    OpenOptions options; options.author = "generic-edit-test";
    { auto store = ObjectStore::create(path, options); }
    ObjectStoreSession session; const auto attachment = session.attach(path, options);
    const std::string volume = "20260411134726";
    require(geometryCoordinates(Json{{"volume_uuid",volume}})["volume_uuid"] == volume,
            "catalog volume ID is stored verbatim");
    require(geometryCoordinates(Json{{"volume_uuid","MixedCase-ID"}})["volume_uuid"] == "MixedCase-ID",
            "volume identities are not canonicalized as object UUIDs");
    mustFail([&] { geometryCoordinates(Json{{"volume_uuid",""}}); });
    mustFail([&] { geometryCoordinates(Json{{"volume_uuid",20260411134726LL}}); });
    require(!geometryScaleToVolume(Json{{"coordinates",{{"volume_uuid",volume},{"scale_factor",4}}}},
                                  "2d2a9b21-9c82-5343-9443-57838a5fa5a1"),
            "derived UUID does not alias authoritative volume ID");
    require(geometryScaleToVolume(Json{{"coordinates",{{"volume_uuid",volume},{"scale_factor",4}}}},volume) == .25,
            "known coordinates convert exactly");
    require(!geometryScaleToVolume(Json{{"coordinates",{{"volume_uuid",volume},{"scale_factor",4}}}},util::newUuid()), "different volume stays unbound");
    require(!geometryScaleToVolume(Json::object(),volume), "unknown coordinates stay unbound");
    mustFail([&] { geometryCoordinates(Json{{"scale_factor",0}}); });
    mustFail([&] { mergeGeometryCoordinates(Json{{"scale_factor",(uint64_t(1)<<60)+1}},Json{{"scale_factor",double(uint64_t(1)<<60)}}); });
    for (const auto& kind : {"fiber", "point_collection", "track", "patch"}) {
        const auto uuid = util::newUuid();
        GeometryData first, target;
        if (std::string(kind) == "fiber") {
            first = vc::geometry::encode(vc::geometry::Fiber{{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}},{10,20}});
            target = vc::geometry::encode(vc::geometry::Fiber{{{4,5,7},{8,9,10},{1,2,3}},{{4,5,7},{1,2,3}},{20,30,10}});
        } else if (std::string(kind) == "point_collection") {
            first = vc::geometry::encode(vc::geometry::PointCollection{{10,20},{{1,2,3},{4,5,6}}});
            target = vc::geometry::encode(vc::geometry::PointCollection{{20,30,10},{{4,5,7},{8,9,10},{1,2,3}}});
        } else if (std::string(kind) == "track") {
            first = vc::geometry::encode(vc::geometry::Track{{{1,2,3},{4,5,6}}});
            target = vc::geometry::encode(vc::geometry::Track{{{1,2,3},{4,5,7},{8,9,10}}});
        } else {
            first = vc::geometry::encode(vc::geometry::Patch{1,2,{1,1},{{1,2,3},{4,5,6}}});
            target = vc::geometry::encode(vc::geometry::Patch{1,2,{1,1},{{1,2,3},{4,5,7}}});
        }
        const auto annotated = std::string(kind) == "fiber" || std::string(kind) == "point_collection";
        const auto array = std::string(kind) == "fiber" ? "control_points" : "points";
        std::string initialGeometry, editedGeometry;
        auto pins = session.act(attachment, {{uuid,{}}}, [&](auto& db, auto& txn) {
            ObjectRecord object; object.uuid = uuid; object.kind = kind;
            if (annotated) object.meta = Json{{"source",{{std::string(kind)=="fiber"?"control_points_attributes":"point_attributes",
                Json::array({Json{{"note","first"}},Json{{"note","second"}}})}}}}.dump();
            db.upsertObject(txn,object); initialGeometry = db.putGeometry(txn,uuid,first);
        });
        pins = session.act(attachment,pins,[&](auto& db,auto& txn) {
            editedGeometry = updateGeometryArrays(db,txn,uuid,target,initialGeometry);
            require(updateGeometryArrays(db,txn,uuid,target,editedGeometry) == editedGeometry, "identical update is a no-op");
        });
        session.read(attachment,[&](const auto& db,const auto& txn,const auto&) {
            for (const auto& desired : target.arrays)
                require(db.readGeometryArray(txn,editedGeometry,desired.name) == desired.bytes,"generic edit preserves exact bytes");
            for (const auto& original : first.arrays)
                require(db.readGeometryArray(txn,initialGeometry,original.name) == original.bytes,"historical arrays are immutable");
            if (annotated) require(Json::parse(db.geometryPointAnnotations(txn,editedGeometry,array)) ==
                Json::array({Json{{"note","second"}},Json::object(),Json{{"note","first"}}}),"annotations follow stable IDs through splice");
        });
        mustFail([&] { session.act(attachment,pins,[&](auto& db,auto& txn) {
            updateGeometryArrays(db,txn,uuid,target,initialGeometry);
        }); });
    }
    // Auxiliary channels and non-finite payload bits belong to the generic
    // store schema even when owned, typed geometry cannot represent them.
    {
        const auto id = util::newUuid();
        auto first = vc::geometry::encode(vc::geometry::Patch{1,2,{1,1},{{1,2,3},{4,5,6}}});
        first.arrays.push_back({"review", "<u2", {1,2}, {1,0,2,0}});
        first.arrays[0].bytes[0] = 0x45; first.arrays[0].bytes[1] = 0x23;
        first.arrays[0].bytes[2] = 0xc1; first.arrays[0].bytes[3] = 0x7f; // NaN payload
        std::string initial, updated;
        auto pins = session.act(attachment, {{id,{}}}, [&](auto& db, auto& txn) {
            ObjectRecord object; object.uuid = id; object.kind = "patch";
            db.upsertObject(txn, object); initial = db.putGeometry(txn, id, first);
        });
        auto target = first; target.arrays.back().bytes = {3,0,4,0};
        pins = session.act(attachment, pins, [&](auto& db, auto& txn) {
            updated = updateGeometryArrays(db, txn, id, target, initial);
            require(db.readGeometryArray(txn, updated, "review") == target.arrays.back().bytes, "auxiliary edits use sparse store updates");
            require(db.readGeometryArray(txn, updated, "points") == first.arrays[0].bytes, "exact non-finite bits survive channel edits");
            require(db.readGeometryArray(txn, initial, "review") == first.arrays.back().bytes, "channel history remains immutable");
        });
        auto malformed = target; malformed.arrays.back().bytes.pop_back();
        mustFail([&] { session.act(attachment, pins, [&](auto& db, auto& txn) {
            updateGeometryArrays(db, txn, id, malformed, updated);
        }); });
    }
    // Structural safety must survive the convenience edit API, including a
    // caller catching the error and trying to commit the surrounding action.
    const auto uuid = util::newUuid();
    vc::geometry::Fiber original{{{0,0,0},{1,0,0}},{{0,0,0},{1,0,0}},{10,20}};
    std::string initial;
    auto pins = session.act(attachment, {{uuid,{}}}, [&](auto& db, auto& txn) {
        ObjectRecord object; object.uuid = uuid; object.kind = "fiber";
        object.meta = R"({"source":{"control_points_attributes":[{"segment_to_next":{"direction":[1,0,0]}},{}],"line_points_attributes":[{"label":"sample"},{}]}})";
        db.upsertObject(txn, object); initial = db.putGeometry(txn, uuid, vc::geometry::encode(original));
        require(!Json::parse(db.objectByUuid(txn, uuid)->meta).at("source").contains("control_points_attributes"), "creation consumes initial annotations");
    });
    auto reverse = original;
    std::reverse(reverse.controlPoints.begin(), reverse.controlPoints.end());
    std::reverse(reverse.controlPointIds.begin(), reverse.controlPointIds.end());
    bool rejected = false;
    mustFail([&] { session.act(attachment, pins, [&](auto& db, auto& txn) {
        try { updateGeometryArrays(db, txn, uuid, vc::geometry::encode(reverse), initial); }
        catch (const std::exception&) { rejected = true; }
    }); });
    require(rejected, "directional reorder requires an explicit transform");
    auto partial = original;
    partial.controlPoints[0][0] = 9; // This sparse update precedes the failing resize.
    partial.linePoints.push_back({2,0,0});
    rejected = false;
    mustFail([&] { session.act(attachment, pins, [&](auto& db, auto& txn) {
        try { updateGeometryArrays(db, txn, uuid, vc::geometry::encode(partial), initial); }
        catch (const std::exception&) { rejected = true; }
        require(db.currentGeometry(txn, uuid)->uuid == initial, "failed helper rolls back every array");
    }); });
    require(rejected, "annotated sample resize requires an explicit transform");
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        require(db.currentGeometry(txn, uuid)->uuid == initial, "caught failures cannot partially commit");
    });
    session.act(attachment, pins, [&](auto& db, auto& txn) {
        const auto head = updateGeometryArrays(db, txn, uuid, vc::geometry::encode(reverse), initial,
            {{"control_points", R"([{"segment_to_next":{"direction":[-1,0,0]}},{}])"}});
        require(Json::parse(db.geometryPointAnnotations(txn, head, "control_points"))[0]["segment_to_next"]["direction"][0] == -1,
            "explicit directional transform is preserved");
        require(Json::parse(db.geometryPointAnnotations(txn, initial, "control_points"))[0]["segment_to_next"]["direction"][0] == 1,
            "initial annotations remain immutable");
    });
    {
        // A linked seed fiber is reoptimized: its linked control point gets a new
        // ID and the line is resampled. Once the control-point edit keeps the
        // edge as recorded, the line-point edit must not ask about it again.
        const auto seed = util::newUuid(), parent = util::newUuid();
        const vc::geometry::Fiber single{{{5,5,5}},{{5,5,5}},{0}};
        std::string seedHead;
        auto linked = session.act(attachment, {{seed,{}},{parent,{}}}, [&](auto& db, auto& txn) {
            for (const auto& id : {seed, parent}) {
                ObjectRecord object; object.uuid = id; object.kind = "fiber"; db.upsertObject(txn, object);
                const auto head = db.putGeometry(txn, id, vc::geometry::encode(single));
                if (id == seed) seedHead = head;
            }
            EdgeRecord edge; edge.kind = "fiber_branch"; edge.aUuid = seed; edge.bUuid = parent;
            edge.aAnchor.points = {{"control_point_ids", {0}}}; edge.bAnchor.points = {{"control_point_ids", {0}}};
            db.addEdge(txn, edge);
        });
        int asked = 0;
        session.act(attachment, linked, [&](auto& db, auto& txn) {
            const auto head = updateGeometryArrays(db, txn, seed,
                vc::geometry::encode(vc::geometry::Fiber{{{4,5,5},{5,5,6},{6,5,5}},{{4,5,5},{6,5,5}},{1,3,2}}), seedHead, {},
                [&](const std::vector<EdgeRecord>& affected) {
                    std::vector<GeometryEdgeEdit> result;
                    for (const auto& edge : affected) {
                        GeometryEdgeEdit edit; edit.edgeUuid = edge.uuid; edit.expectedGeneration = edge.generation;
                        edit.action = "keep_recorded";
                        edit.aRevision = db.currentGeometry(txn, edge.aUuid)->uuid;
                        edit.bRevision = db.currentGeometry(txn, edge.bUuid)->uuid;
                        result.push_back(std::move(edit)); ++asked;
                    }
                    return result;
                });
            require(db.readGeometryData(txn, head).arrays.size() == 3, "reoptimized seed is published");
        });
        require(asked == 1, "the replaced linked point asks once");
    }
    std::cout << "generic geometry editing and coordinates passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
