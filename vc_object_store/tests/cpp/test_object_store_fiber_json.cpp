#include "TestSupport.hpp"
#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
using namespace vc::objects;
using Json = nlohmann::json;
int main() try {
    const auto uuid = util::newUuid();
    Json root{{"type","vc3d_fiber"},{"uuid",uuid},{"version",4},{"name","same name"},
        {"control_point_ids",{0,1}},{"width",5.5},{"tags",{"reviewed"}},
        {"control_points",Json::array({Json{{"position",{1.25,2,3}},{"direction",{1,0,0}},{"tags",{"break"}},{"segment_to_next",{{"goal","global"}}}},
                                      Json{{"position",{5.25,2,3}}}})},
        {"line_points",{{1.25,2,3},{5.25,2,3}}}, {"hv_classification",{{"manual_tag","H"}}}};
    const auto doc = decodeFiberJson(root.dump());
    for (const int version : {1, 2}) {
        auto legacy = root; legacy["version"] = version;
        mustFail([&] { decodeFiberJson(legacy.dump()); });
    }
    auto version3 = root; version3["version"] = 3;
    version3["vc_open_data_coordinate_space"] = "PHercParis4/20260411134726@L0";
    version3["vc_open_data_source_coordinate_scale_factor"] = 1;
    version3["vc_open_data_source_path"] = "https://example.invalid/volume.zarr";
    const auto fromVersion3 = decodeFiberJson(version3.dump());
    require(fromVersion3.source["coordinates"] == Json({{"volume_uuid","20260411134726"},{"scale_factor",1}}) &&
            !fromVersion3.source.contains("vc_open_data_source_path") && fromVersion3.object.formatVersion == 4,
            "version 3 level-0 coordinate space becomes the volume binding");
    version3["vc_open_data_coordinate_space"] = "PHercParis4/20260411134726@L2";
    version3["vc_open_data_source_coordinate_scale_factor"] = 4;
    mustFail([&] { decodeFiberJson(version3.dump()); });
    auto versionless = root; versionless.erase("version");
    mustFail([&] { decodeFiberJson(versionless.dump()); });
    auto oldControls = root; oldControls["control_points"] = {{1.25,2,3},{5.25,2,3}};
    mustFail([&] { decodeFiberJson(oldControls.dump()); });
    const auto controlOnlyGeometry = vc::geometry::encode(vc::geometry::Fiber{{{1,2,3},{4,5,6}}, {}});
    const auto controlOnly = decodeFiberJson(encodeFiberJson(uuid, controlOnlyGeometry, "{}"));
    require(controlOnly.object.formatVersion == 4 && controlOnly.object.fiber->linePointCount == 0,
            "typed control-only geometry exports current transfer format");
    require(doc.object.uuid == uuid && doc.object.fiber->length == 4, "native summaries");
    require(doc.object.fiber->family == "H" && doc.object.tags == std::vector<std::string>{"reviewed"}, "native metadata");
    const auto source = doc.source;
    require(!source.contains("control_points") && !source.contains("line_points"), "no duplicate geometry in annotations");
    require(Json::parse(encodeFiberJson(uuid, doc.geometry, doc.source.dump())) == root, "lossless shared native codec");
    auto invalid = root; invalid["control_points"][0]["position"] = {1,2,3,4};
    mustFail([&] { decodeFiberJson(invalid.dump()); });
    invalid = source; invalid["control_points_attributes"] = Json::array();
    mustFail([&] { encodeFiberJson(uuid, doc.geometry, invalid.dump()); });
    TemporaryDirectory dir;
    const auto path = dir.path / "receipt.vcobj";
    OpenOptions options; options.author = "receipt-test";
    { auto db = ObjectStore::create(path, options); }
    ObjectStoreSession session; const auto attachment = session.attach(path, options);
    const auto receipt = session.act(attachment, {{uuid,{}}}, [&](auto& db, auto& txn) {
        auto object = doc.object; object.meta = Json{{"source",source}}.dump();
        db.upsertObject(txn, object); db.putGeometry(txn, uuid, doc.geometry);
    });
    require(receipt.size() == 1 && receipt.at(uuid).has_value(), "commit returns exact new pin");
    auto external = ObjectStore::open(path, options);
    { auto txn = external.write(receipt); external.setMeta(txn,uuid,"external","true");
      auto changed = source; changed["control_points_attributes"][0]["tags"] = {"changed"};
      mustFail([&] { external.setMeta(txn, uuid, "source", changed.dump()); }); txn.commit();
      require(txn.committedRevisions().at(uuid) != receipt.at(uuid), "external commit produces different pin"); }
    session.read(attachment,[&](const auto& db,const auto& txn,const auto&) {
        require(db.objectRevision(txn,*receipt.at(uuid))->record.meta.find("external") == std::string::npos, "receipt is immutable after external commit");
        const auto historical=Json::parse(fiberRevisionJson(db,txn,*receipt.at(uuid)));
        require(historical["control_points"]==root["control_points"] && historical["line_points"]==root["line_points"], "pinned wire geometry and annotations");
        require(!historical.contains("optimization_mode"), "absent legacy mode stays absent on the wire");
        mustFail([&] {fiberRevisionJson(db,txn,*receipt.at(uuid),1);});
    });
    mustFail([&] { session.act(attachment,receipt,[](auto&,auto&) {}); });
    const auto a = util::newUuid(), b = util::newUuid(), isolated = util::newUuid();
    std::string ordinary, adjacent;
    const auto networkPins = session.act(attachment, {{a,{}},{b,{}},{isolated,{}}}, [&](auto& db, auto& txn) {
        vc::geometry::Fiber geometry{{{1,2,3},{5,2,3}}, {{1,2,3},{5,2,3}}, {90,10}};
        for (const auto& objectUuid : {a,b,isolated}) {
            ObjectRecord record; record.uuid = objectUuid; record.kind = "fiber";
            record.name = "same name"; record.formatVersion = 4;
            record.meta = Json{{"coordinates",{{"volume_uuid",uuid},{"scale_factor",4.0}}},
                {"source",{{"coordinates",{{"scale_factor",8.0}}},
                           {"coordinate_binding",{{"active_volume_id","stale"}}}}}}.dump();
            db.upsertObject(txn, record); db.putGeometry(txn, objectUuid, vc::geometry::encode(geometry));
        }
        EdgeRecord edge; edge.kind = "fiber_branch"; edge.aUuid = a; edge.bUuid = b;
        edge.aAnchor.points = {{"control_point_ids", {10}}}; edge.bAnchor.points = {{"control_point_ids", {90}}};
        edge.aAnchor.position = {5,2,3}; edge.bAnchor.position = {1,2,3};
        edge.aAnchor.direction = {1,0,0}; edge.bAnchor.direction = {0,1,0};
        // A stale automatic mark on a reviewed edge is not transported.
        edge.meta = R"({"auto":true})";
        ordinary = db.addEdge(txn, edge); db.setApproval(txn, ordinary, "approved");
        edge.meta = R"({"adjacent":true,"auto":true})";
        edge.aAnchor.points = {{"control_point_ids", {90}}}; edge.aAnchor.position = {1,2,3};
        edge.bAnchor.points = {{"control_point_ids", {10}}}; edge.bAnchor.position = {5,2,3};
        adjacent = db.addEdge(txn, edge); db.setApproval(txn, adjacent, "pending");
    });
    std::map<std::string,std::string> exported;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        exported = fiberNetworkJson(db, txn, networkPins, {a});
        require(exported.size() == 2 && exported.contains(b), "export follows linked component only");
        const auto first = Json::parse(exported.at(a)), second = Json::parse(exported.at(b));
        require(first["name"] == second["name"] && first["filename"] != second["filename"], "UUID wire names preserve duplicate labels");
        const auto& link = first["branches"].at(0);
        require(link["control_point_index"] == 1 && link["branch_control_point_index"] == 0, "stable control IDs resolve to selected indices");
        require(link["branch_file"] == b+".json" && link["edge_uuid"] == ordinary && !link["pending"].get<bool>(), "ordinary branch exported from edge");
        require(second["branches"][0]["branch_file"] == a+".json" && second["branches"][0]["control_point_index"] == 0, "reciprocal orientation exported");
        require(first["adjacent_branches"][0]["edge_uuid"] == adjacent && first["adjacent_branches"][0]["pending"].get<bool>(), "adjacent pending edge preserved separately");
        require(!link.contains("auto") && !first["adjacent_branches"][0].contains("auto"),
                "automatic mark only on pending ordinary links");
        require(first["coordinates"]["scale_factor"] == 4.0, "coordinate binding exported");
        require(!first.contains("coordinate_binding"), "source annotations cannot restore an absent binding");
        mustFail([&] { fiberNetworkJson(db,txn,{{a,networkPins.at(a)}}); });
        mustFail([&] { fiberNetworkJson(db,txn,{{a,{}}}); });
        mustFail([&] { fiberNetworkJson(db,txn,networkPins,{a},1); });
    });
    session.act(attachment, networkPins, [&](auto& db, auto& txn) { db.removeEdge(txn,ordinary); });
    RevisionSelection currentNetworkPins;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        require(fiberNetworkJson(db,txn,networkPins,{a}) == exported, "pinned topology survives later edge deletion");
        auto current = networkPins;
        for (auto& [objectUuid, pin] : current) pin = db.currentObjectRevision(txn,objectUuid)->uuid;
        currentNetworkPins = current;
        const auto now = fiberNetworkJson(db,txn,current,{a});
        require(Json::parse(now.at(a))["branches"].empty(), "current export observes edge deletion");
        require(Json::parse(now.at(a))["adjacent_branches"].size() == 1, "parallel adjacent edge survives");
    });
    session.act(attachment, currentNetworkPins, [&](auto& db, auto& txn) {
        db.retireObject(txn, b);
        db.keepRecordedRelationship(txn, adjacent);
    });
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        for (auto& [objectUuid, pin] : currentNetworkPins) pin = db.currentObjectRevision(txn,objectUuid)->uuid;
        const auto now = fiberNetworkJson(db,txn,currentNetworkPins,{a});
        require(now.size() == 1 && Json::parse(now.at(a))["adjacent_branches"].empty(), "retired endpoints and inactive edges are not transported");
        require(fiberNetworkJson(db,txn,networkPins,{a}) == exported, "historical network remains independent of current lifecycle");
    });
    std::cout << "fiber codec and checked commit receipt passed\n";
} catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
